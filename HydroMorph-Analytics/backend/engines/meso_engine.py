"""MESO-SCALE: Exner Morphodynamics Engine, as six sub-modules coupled in a time march.

  2.1 Hydrodynamic State Solver (MacCormack Saint-Venant)   2.4 Bed-Evolution Divergence Solver (Exner, finite volume)
  2.2 Boundary Condition Manager (ghost cells, hydrograph)  2.5 Morphological Stability Controller (CFL, dynamic MorFac)
  2.3 Basal Shear & Transport Capacity (Shields, MPM)       2.6 State-Space Persister (decimation, Parquet export)

All spatial operators are NumPy slice operations. The only Python loop is the explicit
time march t -> t + dt inside `MesoExnerModel.solve`.
"""
from __future__ import annotations

import io
from time import perf_counter

import numpy as np

from core.base_model import BaseModel, InputField, SubModule, timing_key
from core.physics_constants import GRAVITY, SEDIMENT_DENSITY, WATER_DENSITY

_MANNING_EXPONENT = 3.0 / 5.0          # h_n = (n q / sqrt(S0))^(3/5)
_FRICTION_DEPTH_EXPONENT = 7.0 / 3.0   # friction slope ~ q|q| / h^(7/3)
_BED_STRESS_DEPTH_EXPONENT = 1.0 / 3.0  # tau_b = rho g n^2 u|u| / h^(1/3)
_MPM_EXPONENT = 1.5
_MAX_STEPS = 2_000_000
_TINY = np.finfo(float).tiny


def normal_depth(q: float, n: float, S0: float) -> float:
    return (n * q / np.sqrt(S0)) ** _MANNING_EXPONENT


class HydrodynamicStateSolver(SubModule):
    id, title = "2.1", "Hydrodynamic State Solver"

    def fields(self):
        return [InputField.from_constant(GRAVITY),
                InputField("L", "Reach Length L (m)", 500.0, 50.0, 20000.0),
                InputField("nx", "Grid Cells N_x (dimensionless)", 250, 20, 3000, integer=True),
                InputField("n_man", "Manning Roughness n (s/m^(1/3))", 0.03, 0.005, 0.2),
                InputField("S0", "Mean Bed Slope S₀ (m/m)", 1.0e-3, 1.0e-5, 0.05, "%.3e"),
                InputField("h_min", "Dry-Depth Tolerance h_min (m)", 1.0e-3, 1.0e-6, 0.1, "%.3e"),
                InputField("art_visc", "Artificial Viscosity Coefficient ε (dimensionless)", 0.05, 0.0, 0.25)]

    def equations(self):
        return [("Saint-Venant mass", r"\frac{\partial h}{\partial t}+\frac{\partial q}{\partial x}=0"),
                ("Saint-Venant momentum",
                 r"\frac{\partial q}{\partial t}+\frac{\partial}{\partial x}\left(\frac{q^{2}}{h}+\frac{g h^{2}}{2}\right)=-g h\frac{\partial z_b}{\partial x}-\frac{g n^{2} q|q|}{h^{7/3}}"),
                ("MacCormack predictor-corrector",
                 r"U^{*}_i=U^n_i-\tfrac{\Delta t}{\Delta x}\left(F^n_{i+1}-F^n_i\right)+\Delta t\,S^n_i,\quad U^{n+1}_i=\tfrac12\left[U^n_i+U^{*}_i-\tfrac{\Delta t}{\Delta x}\left(F^{*}_i-F^{*}_{i-1}\right)+\Delta t\,S^{*}_i\right]")]

    def initial_state(self, v, q0):
        x = np.linspace(0.0, v["L"], int(v["nx"]))
        h_n = normal_depth(q0, v["n_man"], v["S0"])
        return x, x[1] - x[0], np.stack((np.full(x.size, h_n), np.full(x.size, q0)))

    @staticmethod
    def _flux(U, g):
        h, q = U
        return np.stack((q, q * q / h + 0.5 * g * h * h))

    @staticmethod
    def _source(U, dzdx, g, n):
        h, q = U
        return np.stack((np.zeros_like(h), -g * h * dzdx - g * n * n * q * np.abs(q) / h**_FRICTION_DEPTH_EXPONENT))

    def step(self, Ue, de, dt, dx, v):
        """One MacCormack step on the ghost-padded state Ue (2, nx+4); returns the (2, nx) interior."""
        g, n, eps, h_min, lam = v["g"], v["n_man"], v["art_visc"], v["h_min"], dt / dx
        F, S = self._flux(Ue, g), self._source(Ue, de, g, n)
        Up = Ue.copy()
        Up[:, :-1] = Ue[:, :-1] - lam * (F[:, 1:] - F[:, :-1]) + dt * S[:, :-1]            # predictor (forward)
        Up[0] = np.maximum(Up[0], h_min)
        Fp, Sp = self._flux(Up, g), self._source(Up, de, g, n)
        Un = Ue.copy()
        Un[:, 1:-1] = 0.5 * (Ue[:, 1:-1] + Up[:, 1:-1]
                             - lam * (Fp[:, 1:-1] - Fp[:, :-2]) + dt * Sp[:, 1:-1])        # corrector (backward)
        Un[:, 1:-1] += eps * (Un[:, 2:] - 2.0 * Un[:, 1:-1] + Un[:, :-2])                   # Jameson-type smoothing
        out = Un[:, 2:-2]
        out[0] = np.maximum(out[0], h_min)
        return out


class BoundaryConditionManager(SubModule):
    id, title = "2.2", "Boundary Condition Manager"

    def fields(self):
        return [InputField.from_constant(GRAVITY),
                InputField("q", "Base Unit Discharge q (m²/s)", 2.0, 0.01, 100.0),
                InputField("q_peak", "Hydrograph Peak Unit Discharge q_peak (m²/s)", 3.0, 0.01, 100.0,
                           help="Set equal to the base discharge for steady inflow."),
                InputField("t_peak", "Hydrograph Peak Time t_peak (s)", 600.0, 0.0, 1.0e6),
                InputField("sigma_t", "Hydrograph Width σ_t (s)", 150.0, 1.0, 1.0e6)]

    def equations(self):
        return [("Inflow hydrograph (Gaussian pulse)",
                 r"q_{in}(t)=q+\left(q_{peak}-q\right)\exp\!\left[-\tfrac12\left(\frac{t-t_{peak}}{\sigma_t}\right)^{2}\right]"),
                ("Ghost cells: Riemann invariants held from the interior",
                 r"\text{inflow: } u-2\sqrt{gh}=\text{const},\ q=q_{in};\qquad \text{outflow: } u+2\sqrt{gh}=\text{const},\ h=h_n(q_{in})")]

    @staticmethod
    def q_in(t, v):
        return v["q"] + (v["q_peak"] - v["q"]) * np.exp(-0.5 * ((t - v["t_peak"]) / v["sigma_t"]) ** 2)

    def equilibrium_depth(self, t, v):
        return normal_depth(self.q_in(t, v), v["n_man"], v["S0"])

    def pad(self, U, t, v):
        """Attach two ghost cells per side carrying the boundary data (inflow q, rating-curve outflow h)."""
        g, q_in = v["g"], self.q_in(t, v)
        h_out = self.equilibrium_depth(t, v)
        h1, q1, hN, qN = U[0, 0], U[1, 0], U[0, -1], U[1, -1]
        hL = h1 + (q_in - q1) / (q1 / h1 + np.sqrt(g * h1))
        qR = qN + (qN / hN - np.sqrt(g * hN)) * (h_out - hN)
        return np.concatenate((np.array([[hL, hL], [q_in, q_in]]), U, np.array([[h_out, h_out], [qR, qR]])), axis=1)

    def cross_validate(self, v):
        q_max = max(v["q"], v["q_peak"])
        h_n = normal_depth(q_max, v["n_man"], v["S0"])
        Fr = q_max / h_n / np.sqrt(v["g"] * h_n)
        return [f"Peak normal-flow Froude number is {Fr:.2f}; this solver is subcritical (q imposed at inflow, h at outflow). "
                "Reduce q or S₀, or increase n."] if Fr >= 0.95 else []


class BasalShearTransportEngine(SubModule):
    id, title = "2.3", "Basal Shear & Transport Capacity"

    def fields(self):
        return [InputField.from_constant(GRAVITY), InputField.from_constant(WATER_DENSITY),
                InputField.from_constant(SEDIMENT_DENSITY),
                InputField("D", "Median Grain Diameter D (m)", 1.0e-3, 1.0e-5, 0.1, "%.3e"),
                InputField("theta_c", "Critical Shields Parameter θ_c (dimensionless)", 0.047, 0.01, 0.2),
                InputField("mpm_coef", "Meyer-Peter & Müller Coefficient (dimensionless)", 8.0, 1.0, 20.0)]

    def equations(self):
        return [("Basal shear (Manning closure) and Shields parameter",
                 r"\tau_b=\frac{\rho_w g n^{2}u|u|}{h^{1/3}},\qquad \theta=\frac{|\tau_b|}{(\rho_s-\rho_w)\,g\,D}"),
                ("Meyer-Peter & Müller transport capacity",
                 r"q_s=8\,(\theta-\theta_c)_+^{3/2}\sqrt{R\,g\,D^{3}},\qquad R=\frac{\rho_s}{\rho_w}-1")]

    @staticmethod
    def shields(h, u, v):
        tau_b = v["rho_w"] * v["g"] * v["n_man"] ** 2 * u * np.abs(u) / h**_BED_STRESS_DEPTH_EXPONENT
        return np.abs(tau_b) / ((v["rho_s"] - v["rho_w"]) * v["g"] * v["D"])

    def capacity(self, h, u, v):
        R = v["rho_s"] / v["rho_w"] - 1.0
        excess = np.clip(self.shields(h, u, v) - v["theta_c"], 0.0, None)
        return v["mpm_coef"] * excess**_MPM_EXPONENT * np.sqrt(R * v["g"] * v["D"] ** 3)


class BedEvolutionDivergenceSolver(SubModule):
    id, title = "2.4", "Bed-Evolution Divergence Solver"

    def fields(self):
        return [InputField("porosity", "Bed Porosity p (dimensionless)", 0.4, 0.0, 0.95),
                InputField("bump_h", "Initial Bed Perturbation Height Δz (m)", 0.3, 0.0, 5.0),
                InputField("bump_x", "Perturbation Centre x_c (m)", 150.0, 0.0, 20000.0),
                InputField("bump_sigma", "Perturbation Width σ (m)", 20.0, 0.5, 5000.0)]

    def equations(self):
        return [("Exner equation (finite-volume, upwind face fluxes, Forward Euler)",
                 r"(1-p)\frac{\partial z_b}{\partial t}+\psi\,\frac{\partial q_s}{\partial x}=0\;\Rightarrow\;z_{b,i}^{n+1}=z_{b,i}^{n}-\frac{\psi\,\Delta t}{(1-p)}\,\frac{q_{s,i}-q_{s,i-1}}{\Delta x}")]

    def cross_validate(self, v):
        return [] if 0.0 < v["bump_x"] < v["L"] else ["Perturbation centre x_c must lie inside the reach (0 < x_c < L)."]

    def initial_bed(self, x, v):
        zb0 = v["S0"] * (v["L"] - x)
        return zb0, zb0 + v["bump_h"] * np.exp(-0.5 * ((x - v["bump_x"]) / v["bump_sigma"]) ** 2)

    @staticmethod
    def flux_divergence(qs, qs_feed, dx):
        return np.diff(np.concatenate(([qs_feed], qs))) / dx        # dq_s/dx with the feed as upstream face

    @staticmethod
    def apply(zb, div, dz_scale, p):
        return zb - (dz_scale / (1.0 - p)) * div


class MorphologicalStabilityController(SubModule):
    id, title = "2.5", "Morphological Stability Controller"

    def fields(self):
        return [InputField("T", "Simulated Hydraulic Duration T (s)", 1200.0, 10.0, 1.0e6),
                InputField("t_spin", "Hydrodynamic Spin-up Time t_spin (s)", 100.0, 0.0, 1.0e5,
                           help="Bed is frozen until the initial transient wave has left the reach."),
                InputField("cfl", "CFL Number (dimensionless)", 0.4, 0.05, 0.9),
                InputField("morph_factor", "Morphological Acceleration Factor ψ (dimensionless)", 50.0, 1.0, 5000.0),
                InputField("dz_max", "Max Bed Change per Step Δz_max (m)", 5.0e-3, 1.0e-6, 0.5, "%.2e")]

    def equations(self):
        return [("Hydrodynamic time step", r"\Delta t=\mathrm{CFL}\,\frac{\Delta x}{\max\left(|u|+\sqrt{g h}\right)}"),
                ("Dynamic MorFac", r"\psi_{\mathrm{eff}}=\min\!\left(\psi,\ \frac{\Delta z_{\max}}{\Delta t\,\max|\partial z_b/\partial t|}\right),\qquad \psi_{\mathrm{eff}}=0\ \ (t<t_{spin})")]

    def cross_validate(self, v):
        errs = []
        if v["t_spin"] >= v["T"]:
            errs.append("Spin-up time t_spin must be smaller than the duration T.")
        q_max = max(v["q"], v["q_peak"])
        h_n = normal_depth(q_max, v["n_man"], v["S0"])
        dx = v["L"] / (v["nx"] - 1)
        est = 1.5 * v["T"] * (q_max / h_n + np.sqrt(v["g"] * h_n)) / (v["cfl"] * dx)
        if est > _MAX_STEPS:
            errs.append(f"Estimated {est:.2e} time steps exceeds the {_MAX_STEPS:.0e} limit; shorten T, coarsen N_x or raise the CFL number.")
        return errs

    @staticmethod
    def time_step(U, dx, t, t_target, v):
        smax = np.max(np.abs(U[1] / U[0]) + np.sqrt(v["g"] * U[0]))
        return min(v["cfl"] * dx / smax, t_target - t)

    @staticmethod
    def morfac(t, dt, bed_rate_max, v):
        if t < v["t_spin"]:
            return 0.0
        return min(v["morph_factor"], v["dz_max"] / (dt * bed_rate_max + _TINY))


class StatePersister(SubModule):
    id, title = "2.6", "State-Space Persister"

    def fields(self):
        return [InputField("n_frames", "Output Frames N_f (dimensionless)", 80, 2, 500, integer=True)]

    def equations(self):
        return [("Temporal decimation", r"U(x,t_k),\quad t_k=\frac{k\,T}{N_f-1},\quad k=0,\dots,N_f-1\;\Rightarrow\;\text{one tensor of shape }(N_f,N_x)")]

    @staticmethod
    def allocate(nf, nx):
        return {k: np.empty((nf, nx), np.float32) for k in ("zb", "h", "u")}

    @staticmethod
    def store(buf, k, zb, U):
        buf["zb"][k], buf["h"][k], buf["u"][k] = zb, U[0], U[1] / U[0]

    @staticmethod
    def to_parquet(arrays: dict):
        """Tidy long table (t, x, z_b, h, u) as zstd Parquet bytes, or None if pyarrow is unavailable."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return None
        nf, nx = arrays["zb"].shape
        table = pa.table({"t_s": np.repeat(arrays["t"], nx), "x_m": np.tile(arrays["x"], nf),
                          "zb_m": arrays["zb"].ravel(), "h_m": arrays["h"].ravel(), "u_ms": arrays["u"].ravel()})
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="zstd")
        return buf.getvalue()


class MesoExnerModel(BaseModel):
    key = "meso_exner"
    title = "Meso-scale · Exner Morphodynamics"
    scale = "Meso"
    summary = ("Coupled Saint-Venant / Exner time march with an inflow hydrograph, MPM transport capacity and "
               "dynamic morphological acceleration; decimated output feeds the animation.")

    def submodules(self):
        return [HydrodynamicStateSolver(), BoundaryConditionManager(), BasalShearTransportEngine(),
                BedEvolutionDivergenceSolver(), MorphologicalStabilityController(), StatePersister()]

    def solve(self, v):
        s = {sm.id: sm for sm in self.submodules()}
        hyd, bcm, trn, bed, ctl, per = s["2.1"], s["2.2"], s["2.3"], s["2.4"], s["2.5"], s["2.6"]
        clock = dict.fromkeys(s, 0.0)
        last = perf_counter()

        def lap(sid: str) -> None:
            nonlocal last
            now = perf_counter()
            clock[sid] += now - last
            last = now

        nf, p, T = int(v["n_frames"]), v["porosity"], v["T"]
        x, dx, U = hyd.initial_state(v, v["q"])
        zb0, zb = bed.initial_bed(x, v)
        zb_init = zb.copy()
        h_n = float(U[0, 0])
        frame_t = np.linspace(0.0, T, nf)
        buf = per.allocate(nf, x.size)
        q_hist = np.empty(nf)
        per.store(buf, 0, zb, U)
        q_hist[0] = bcm.q_in(0.0, v)

        t, k, steps, flux_balance, psi_min = 0.0, 1, 0, 0.0, v["morph_factor"]
        lap("2.1")
        while k < nf:                                                   # explicit time march (only loop)
            dt = ctl.time_step(U, dx, t, frame_t[k], v);                              lap("2.5")
            Ue = bcm.pad(U, t, v);                                                     lap("2.2")
            U = hyd.step(Ue, np.pad(np.gradient(zb, dx), 2, mode="edge"), dt, dx, v);  lap("2.1")
            h_eq = bcm.equilibrium_depth(t, v)
            qs_feed = trn.capacity(h_eq, bcm.q_in(t, v) / h_eq, v)                     # equilibrium sediment feed
            qs = trn.capacity(U[0], U[1] / U[0], v);                                   lap("2.3")
            div = bed.flux_divergence(qs, qs_feed, dx)
            psi = ctl.morfac(t, dt, float(np.max(np.abs(div))) / (1.0 - p), v);        lap("2.5")
            zb = bed.apply(zb, div, psi * dt, p)
            flux_balance += psi * dt * (qs_feed - qs[-1])
            psi_min = min(psi_min, psi) if psi > 0.0 else psi_min;                     lap("2.4")
            t += dt
            steps += 1
            if not (np.isfinite(U).all() and np.isfinite(zb).all()):
                raise RuntimeError("Numerical blow-up. Lower the CFL number, the morphological factor ψ, "
                                   "or raise the artificial viscosity.")
            if t >= frame_t[k] - 1.0e-9:
                per.store(buf, k, zb, U)
                q_hist[k] = bcm.q_in(t, v)
                k += 1
            lap("2.6")

        d_volume = float((1.0 - p) * np.sum(zb - zb_init) * dx)
        scale = max(abs(d_volume), abs(flux_balance), _TINY)
        pos0, pos1 = np.clip(zb_init - zb0, 0.0, None), np.clip(zb - zb0, 0.0, None)
        centroid = lambda w: float(np.sum(x * w) / max(np.sum(w), _TINY))   # noqa: E731
        scalars = {
            "h_normal": h_n, "Fr_normal": float(v["q"] / h_n / np.sqrt(v["g"] * h_n)),
            "steps": float(steps), "max_bed_change": float(np.max(np.abs(buf["zb"][-1] - buf["zb"][0]))),
            "centroid_shift": centroid(pos1) - centroid(pos0),
            "Fr_max": float(np.max(np.abs(buf["u"]) / np.sqrt(v["g"] * buf["h"]))),
            "psi_min": float(psi_min), "sediment_mass_error": float(abs(d_volume - flux_balance) / scale),
            "q_peak_used": float(q_hist.max()),
        }
        scalars.update({timing_key(sid): c * 1.0e3 for sid, c in clock.items()})
        arrays = {"x": x, "t": frame_t, "zb": buf["zb"], "h": buf["h"], "u": buf["u"],
                  "zb_initial": zb_init, "q_in": q_hist}
        return scalars, arrays
