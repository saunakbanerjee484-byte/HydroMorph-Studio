"""MACRO-SCALE: MorphoTensor, as six sub-modules.

  3.1 Mixture Rheology Configurator        3.4 Transverse Gravity-Shear Coupler
  3.2 3D Stress Tensor Generator           3.5 Yield-Surface Tracker
  3.3 Non-Newtonian Flux Router            3.6 3D Morphological Updater

Data dependencies fix the execution order 3.1 -> 3.4 -> 3.3 -> 3.2 -> 3.5 -> 3.6 (the topology and
gravity direction from 3.4 are needed before the flux router can build the velocity field, and the
stress tensor needs that field). Every stage broadcasts over the (y, z) grid: no Python loops.
"""
from __future__ import annotations

import numpy as np

from core.base_model import BaseModel, InputField, SubModule
from core.physics_constants import GRAVITY, SEDIMENT_DENSITY, WATER_DENSITY

_GAMMA_FLOOR = np.finfo(float).eps      # numerical floor so that 0**(n-1) never occurs at rest


def _cumtrapz_z(f: np.ndarray, dz: float) -> np.ndarray:
    return np.concatenate((np.zeros((f.shape[0], 1)), np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * dz, axis=1)), axis=1)


class MixtureRheologyConfigurator(SubModule):
    id, title = "3.1", "Mixture Rheology Configurator"

    def fields(self):
        return [InputField.from_constant(WATER_DENSITY), InputField.from_constant(SEDIMENT_DENSITY),
                InputField("phi", "Solid Volume Fraction φ (dimensionless)", 0.55, 0.0, 0.8),
                InputField("tau_y", "Yield Stress τ_y (Pa)", 1500.0, 0.0, 1.0e5),
                InputField("K", "Consistency Index K (Pa·sⁿ)", 300.0, 1.0e-3, 1.0e5),
                InputField("n_idx", "Flow Behaviour Index n (dimensionless)", 1.0, 0.2, 2.0,
                           help="n = 1 recovers a Bingham fluid; n < 1 shear-thinning; n > 1 shear-thickening."),
                InputField("m_reg", "Papanastasiou Stress-Growth Exponent m (s)", 100.0, 1.0, 1.0e5)]

    def equations(self):
        return [("Mixture density", r"\rho_m=\varphi\,\rho_s+(1-\varphi)\,\rho_w"),
                ("Herschel-Bulkley law (Bingham when n = 1)", r"\tau=\tau_y+K\dot\gamma^{\,n}\ \ (\tau>\tau_y),\qquad \dot\gamma=\sqrt{2\,\mathbf D:\mathbf D}"),
                ("Papanastasiou-regularised apparent viscosity", r"\eta=K\dot\gamma^{\,n-1}+\tau_y\,\frac{1-e^{-m\dot\gamma}}{\dot\gamma}")]

    def run(self, v, state):
        rho_m = v["phi"] * v["rho_s"] + (1.0 - v["phi"]) * v["rho_w"]
        return {"rho_m": float(rho_m), "is_bingham": float(abs(v["n_idx"] - 1.0) < 1.0e-9)}, {}


class TransverseGravityShearCoupler(SubModule):
    id, title = "3.4", "Transverse Gravity-Shear Coupler"

    def fields(self):
        return [InputField.from_constant(GRAVITY),
                InputField("theta", "Down-Slope Angle θ (°)", 10.0, 0.1, 60.0),
                InputField("phi_cross", "Cross-Slope Angle φ (°)", 2.0, -45.0, 45.0,
                           help="Non-zero values drive lateral flow; wall blockage is not modelled."),
                InputField("H0", "Maximum Flow Depth H (m)", 1.0, 0.01, 50.0),
                InputField("W", "Channel Width W (m)", 4.0, 0.05, 500.0),
                InputField("shape", "Cross-Section Shape Exponent s (dimensionless)", 2.0, 1.0, 10.0,
                           help="Depth H(y) = H·(1 − |2y/W − 1|^s); s = 2 is parabolic."),
                InputField("ny", "Lateral Grid Points N_y (dimensionless)", 61, 11, 401, integer=True),
                InputField("nz", "Vertical Grid Points N_z (dimensionless)", 61, 11, 401, integer=True)]

    def equations(self):
        return [("Gravity components in the bed plane and normal to it",
                 r"g_x=g\sin\theta,\quad g_y=g\sin\varphi,\quad g_t=\sqrt{g_x^2+g_y^2},\quad g_n=g\cos\theta\cos\varphi"),
                ("Stress and shear-rate vectors share the direction of the tangential gravity",
                 r"\mathbf e=\frac{(g_x,\,g_y)}{g_t},\qquad \boldsymbol\tau(z)=\rho_m g_t\,(H-z)\,\mathbf e"),
                ("Cross-section topography", r"H(y)=H\left[1-\left|\tfrac{2y}{W}-1\right|^{s}\right]_+")]

    def run(self, v, state):
        ny, nz = int(v["ny"]), int(v["nz"])
        y, z = np.linspace(0.0, v["W"], ny), np.linspace(0.0, v["H0"], nz)
        H = v["H0"] * np.clip(1.0 - np.abs(2.0 * y / v["W"] - 1.0) ** v["shape"], 0.0, None)
        th, ph = np.deg2rad(v["theta"]), np.deg2rad(v["phi_cross"])
        gx, gy = v["g"] * np.sin(th), v["g"] * np.sin(ph)
        g_t = np.hypot(gx, gy)
        return ({"g_t": float(g_t), "g_n": float(v["g"] * np.cos(th) * np.cos(ph)), "e_x": float(gx / g_t),
                 "e_y": float(gy / g_t), "flow_azimuth_deg": float(np.degrees(np.arctan2(gy, gx)))},
                {"y": y, "z": z, "H": H, "fluid": (z[None, :] <= H[:, None]).astype(np.float32)})


class NonNewtonianFluxRouter(SubModule):
    id, title = "3.3", "Non-Newtonian Flux Router"

    def equations(self):
        return [("Herschel-Bulkley velocity magnitude (z ≤ z_y; rigid plug above)",
                 r"|\mathbf u|(z)=\frac{n}{n+1}\left(\frac{\rho_m g_t}{K}\right)^{1/n}\left[z_y^{\frac{n+1}{n}}-(z_y-z)^{\frac{n+1}{n}}\right],\quad z_y=H-\frac{\tau_y}{\rho_m g_t}"),
                ("Incompressibility closes the vertical velocity", r"\nabla\!\cdot\mathbf u=0\;\Rightarrow\;w(y,z)=-\int_0^{z}\frac{\partial v}{\partial y}\,dz'"),
                ("Mass and momentum flux divergence",
                 r"Q=\iint \mathbf u\,dA,\qquad \nabla\!\cdot(\rho_m\,\mathbf u\otimes\mathbf u)\ \text{vs}\ \rho_m g_t")]

    def run(self, v, state):
        n, K, tau_y, rho = v["n_idx"], v["K"], v["tau_y"], state["rho_m"]
        y, z, H, fluid = state["y"], state["z"], state["H"], state["fluid"] > 0.5
        dy, dz = y[1] - y[0], z[1] - z[0]
        a = rho * state["g_t"]
        z_y = np.clip(H - tau_y / a, 0.0, None)[:, None]
        Z = z[None, :]
        p1 = (n + 1.0) / n
        mag = (n / (n + 1.0)) * (a / K) ** (1.0 / n) * (z_y**p1 - (z_y - np.minimum(Z, z_y)) ** p1)
        u, vv = mag * state["e_x"], mag * state["e_y"]
        w = -_cumtrapz_z(np.gradient(vv, dy, axis=0), dz)                  # continuity-derived vertical velocity
        div = np.gradient(vv, dy, axis=0) + np.gradient(w, dz, axis=1)
        conv = np.gradient(rho * vv * u, dy, axis=0) + np.gradient(rho * w * u, dz, axis=1)
        Qx, Qy = float(np.sum(np.where(fluid, u, 0.0)) * dy * dz), float(np.sum(np.where(fluid, vv, 0.0)) * dy * dz)
        area = float(np.sum(fluid) * dy * dz)
        u_mean = Qx / area if area > 0 else 0.0
        ref = max(float(np.max(np.abs(np.gradient(vv, dy, axis=0)))), _GAMMA_FLOOR)
        return ({"Q": Qx, "Q_lateral": Qy, "u_mean": float(u_mean), "u_max": float(mag.max()),
                 "mass_residual": float(np.max(np.abs(np.where(fluid, div, 0.0))) / ref),
                 "inertia_ratio": float(np.max(np.abs(np.where(fluid, conv, 0.0))) / a),
                 "Froude": float(u_mean / np.sqrt(state["g_n"] * v["H0"])),
                 "no_flow": float(z_y.max() <= 0.0), "z_y_analytic_max": float(z_y.max())},
                {"u": u, "v": vv, "w": w, "z_y": z_y[:, 0]})


class StressTensorGenerator(SubModule):
    id, title = "3.2", "3D Stress Tensor Generator"

    def equations(self):
        return [("Velocity-gradient, strain-rate and spin tensors",
                 r"\mathbf L=\nabla\mathbf u,\qquad \mathbf D=\tfrac12\left(\mathbf L+\mathbf L^{\mathsf T}\right),\qquad \mathbf W=\tfrac12\left(\mathbf L-\mathbf L^{\mathsf T}\right)"),
                ("Cauchy stress (non-Newtonian partition into pressure and deviator)",
                 r"\boldsymbol\sigma=-p\,\mathbf I+2\eta(\dot\gamma)\,\mathbf D,\qquad p=\rho_m g_n\,(H-z)"),
                ("Momentum balance", r"\rho_m\frac{D\mathbf u}{Dt}=\nabla\!\cdot\boldsymbol\sigma+\rho_m\mathbf g"),
                ("Equivalent stress and principal stresses", r"\tau_{eq}=\sqrt{\tfrac12\,\mathbf s:\mathbf s},\qquad \det(\boldsymbol\sigma-\sigma_i\mathbf I)=0")]

    def run(self, v, state):
        n, K, tau_y, m, rho = v["n_idx"], v["K"], v["tau_y"], v["m_reg"], state["rho_m"]
        y, z, H, fluid = state["y"], state["z"], state["H"], state["fluid"] > 0.5
        dy, dz = y[1] - y[0], z[1] - z[0]
        ny, nz = y.size, z.size
        F = np.stack((state["u"], state["v"], state["w"]))                         # (3, ny, nz)
        dFdy, dFdz = np.gradient(F, dy, dz, axis=(1, 2))
        L = np.zeros((ny, nz, 3, 3))
        L[..., :, 1] = np.moveaxis(dFdy, 0, -1)
        L[..., :, 2] = np.moveaxis(dFdz, 0, -1)
        D = 0.5 * (L + np.swapaxes(L, -1, -2))
        gdot = np.sqrt(2.0 * np.einsum("...ij,...ij->...", D, D))
        gdot_z = np.hypot(dFdz[0], dFdz[1])                                        # vertical (bed-normal) shear rate
        gs = np.maximum(gdot, _GAMMA_FLOOR)
        eta = K * gs ** (n - 1.0) + tau_y * (1.0 - np.exp(-m * gs)) / gs
        p = np.where(fluid, rho * state["g_n"] * (H[:, None] - z[None, :]), 0.0)
        dev = 2.0 * eta[..., None, None] * D
        sigma = dev - p[..., None, None] * np.eye(3)
        tau_eq = np.sqrt(0.5 * np.einsum("...ij,...ij->...", dev, dev))
        principal = np.linalg.eigvalsh(sigma)
        a = rho * state["g_t"]
        sheared = fluid & (z[None, :] <= state["z_y"][:, None]) & (gdot > 0.0)
        partition = np.abs(tau_eq - a * (H[:, None] - z[None, :]))                # constitutive vs equilibrium stress
        return ({"tau_eq_max": float(tau_eq.max()), "gamma_dot_max": float(gdot.max()),
                 "eta_min": float(eta[fluid].min()) if fluid.any() else 0.0,
                 "partition_error": float(np.mean(partition[sheared]) / (a * v["H0"])) if sheared.any() else 0.0},
                {"gamma_dot": gdot, "gamma_dot_z": gdot_z, "eta_app": eta, "tau_eq": tau_eq, "pressure": p,
                 "sigma3": principal[..., 0], "sigma2": principal[..., 1], "sigma1": principal[..., 2]})


class YieldSurfaceTracker(SubModule):
    id, title = "3.5", "Yield-Surface Tracker"

    def fields(self):
        return [InputField("gamma_tol", "Yielded Shear-Rate Threshold γ̇_tol (1/s)", 1.0e-4, 1.0e-12, 1.0, "%.2e")]

    def equations(self):
        return [("Yield interface (plug base) and detected sheared region",
                 r"z_y(y)=\left[H(y)-\frac{\tau_y}{\rho_m g_t}\right]_+,\qquad \mathcal Y=\left\{\left|\partial\mathbf u/\partial z\right|>\dot\gamma_{tol}\right\}"),
                ("Rigid plug thickness", r"h_{plug}(y)=H(y)-z_y(y)")]

    def run(self, v, state):
        z, H, fluid = state["z"], state["H"], state["fluid"] > 0.5
        dz = z[1] - z[0]
        yielded = fluid & (state["gamma_dot_z"] > v["gamma_tol"])   # the plug still shears laterally, so test vertical shear
        z_y_num = np.maximum(np.sum(yielded, axis=1) - 1, 0) * dz                  # contiguous sheared height from the bed
        plug = H - state["z_y"]
        wet = H > 0
        return ({"yielded_fraction": float(np.sum(yielded) / max(np.sum(fluid), 1)),
                 "plug_thickness_centre": float(plug[H.size // 2]),
                 "yield_surface_error": float(np.max(np.abs(z_y_num - state["z_y"])[wet]) / dz) if wet.any() else 0.0,
                 "Bingham_number": float(v["tau_y"] / (v["K"] * max(state["u_max"] / v["H0"], _GAMMA_FLOOR) ** v["n_idx"]))},
                {"z_y_num": z_y_num, "plug_thickness": plug, "yielded": yielded.astype(np.float32)})


class MorphologicalUpdater(SubModule):
    id, title = "3.6", "3D Morphological Updater"

    def fields(self):
        return [InputField("tau_c", "Critical Bed Shear Stress τ_c (Pa)", 1500.0, 0.0, 1.0e5),
                InputField("k_e", "Entrainment Coefficient k_e (m/(s·Pa))", 1.0e-6, 0.0, 1.0e-2, "%.2e"),
                InputField("t_m", "Morphological Event Duration t_m (s)", 60.0, 0.0, 1.0e5)]

    def equations(self):
        return [("Basal shear from stress partition", r"\tau_b(y)=\rho_m g_t\,H(y)"),
                ("Excess-shear entrainment and bed deformation",
                 r"\dot e=k_e\,(\tau_b-\tau_c)_+,\qquad \Delta z_b(y)=-\dot e\,t_m,\qquad H^{+}=H-\Delta z_b"),
                ("Eroded volume per unit stream length", r"V'=\int\left|\Delta z_b\right|\,dy")]

    def run(self, v, state):
        y, H = state["y"], state["H"]
        dy = y[1] - y[0]
        tau_b = state["rho_m"] * state["g_t"] * H
        flowing = state["z_y"] > 0.0                                                 # a static column does not entrain
        scour = np.where(flowing, np.clip(tau_b - v["tau_c"], 0.0, None), 0.0) * v["k_e"] * v["t_m"]   # m of bed lowering
        bed_before = v["H0"] - H
        return ({"max_scour": float(scour.max()), "eroded_area": float(np.sum(scour) * dy),
                 "scoured_width_fraction": float(np.sum(scour > 0) / max(np.sum(H > 0), 1))},
                {"tau_b": tau_b, "d_bed": -scour, "bed_before": bed_before, "bed_after": bed_before - scour, "H_after": H + scour})


class MacroTensorModel(BaseModel):
    key = "macro_tensor"
    title = "Macro-scale · MorphoTensor Debris Flow"
    scale = "Macro"
    summary = ("Non-Newtonian debris flow on a cross-sloped channel: rheology, transverse coupling, flux routing, "
               "full stress tensors, yield surface and bed deformation.")
    execution_order = ("3.1", "3.4", "3.3", "3.2", "3.5", "3.6")

    def submodules(self):
        return [MixtureRheologyConfigurator(), StressTensorGenerator(), NonNewtonianFluxRouter(),
                TransverseGravityShearCoupler(), YieldSurfaceTracker(), MorphologicalUpdater()]
