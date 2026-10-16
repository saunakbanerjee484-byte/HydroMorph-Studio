"""MICRO-SCALE: Benthic-Boundary Engine, as six sequential sub-modules.

  1.1 Near-Bed Kinematics            1.4 Turbulent Eddy Diffusivity Solver
  1.2 Viscous Sublayer Diagnostics   1.5 Suspension ODE Integrator
  1.3 Particle Settling & Entrainment  1.6 Macroscopic Flux Aggregator

Every stage is vectorised; there is no loop of any kind in this file.
"""
from __future__ import annotations

import numpy as np

from core.base_model import BaseModel, InputField, SubModule
from core.physics_constants import (GRAVITY, KINEMATIC_VISCOSITY, SEDIMENT_DENSITY,
                                    VON_KARMAN, WATER_DENSITY)

# Model-form constants (structural, not physical constants of the fluid)
_RE_SMOOTH, _RE_ROUGH = 5.0, 70.0          # Nikuradse roughness-Reynolds regime limits
_Z0_SMOOTH_COEF, _Z0_ROUGH_DIV = 0.11, 30.0  # z0 = 0.11 nu/u* + ks/30 (valid in all regimes)
_SUBLAYER_COEF = 11.6                      # viscous sublayer edge: z+ = 11.6
_GP_RP_EXPONENT, _GP_Z_EXPONENT, _GP_RP_MIN = 0.6, 5, 2.36
_SURFACE_CLIP = 1.0e-3                     # numerical: stop the grid just below z = h


def _trapz(f: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * np.sum((f[1:] + f[:-1]) * np.diff(x)))


def _z0(v: dict, u_star: float) -> float:
    return _Z0_SMOOTH_COEF * v["nu"] / u_star + v["ks"] / _Z0_ROUGH_DIV


class NearBedKinematics(SubModule):
    id, title = "1.1", "Near-Bed Kinematics"

    def fields(self):
        return [InputField.from_constant(GRAVITY), InputField.from_constant(KINEMATIC_VISCOSITY),
                InputField.from_constant(VON_KARMAN),
                InputField("h", "Flow Depth h (m)", 2.0, 0.01, 100.0),
                InputField("S_f", "Energy Slope S_f (m/m)", 2.0e-4, 1.0e-7, 0.1, "%.3e"),
                InputField("ks", "Nikuradse Roughness Height k_s (m)", 7.5e-4, 1.0e-6, 1.0, "%.3e"),
                InputField("nz", "Vertical Grid Points N_z (dimensionless)", 400, 50, 5000, integer=True)]

    def equations(self):
        return [("Shear velocity", r"u_*=\sqrt{g\,h\,S_f}"),
                ("Law of the Wall", r"u(z)=\frac{u_*}{\kappa}\ln\frac{z}{z_0},\qquad z_0=0.11\frac{\nu}{u_*}+\frac{k_s}{30}")]

    def cross_validate(self, v):
        u_star = np.sqrt(v["g"] * v["h"] * v["S_f"])
        z0, a = _z0(v, u_star), v["a_ratio"] * v["h"]
        return [f"Roughness length z₀ = {z0:.3g} m must be smaller than the reference level a = (a/h)·h = {a:.3g} m."] if z0 >= a else []

    def run(self, v, state):
        u_star = np.sqrt(v["g"] * v["h"] * v["S_f"])
        z0 = _z0(v, u_star)
        z_u = np.geomspace(z0, v["h"], int(v["nz"]))
        u = u_star / v["kappa"] * np.log(z_u / z0)
        u_mean = u_star / v["kappa"] * (np.log(v["h"] / z0) - 1.0)
        return ({"u_star": float(u_star), "z0": float(z0), "u_mean": float(u_mean),
                 "c_f": float((u_star / u_mean) ** 2)}, {"z_u": z_u, "u": u})


class ViscousSublayerDiagnostics(SubModule):
    id, title = "1.2", "Viscous Sublayer Diagnostics"

    def equations(self):
        return [("Roughness Reynolds number and regime",
                 r"Re_*=\frac{u_*k_s}{\nu}:\quad Re_*<5\ \text{smooth},\ \ 5\le Re_*\le70\ \text{transitional},\ \ Re_*>70\ \text{rough}"),
                ("Viscous sublayer thickness", r"\delta_v=11.6\,\frac{\nu}{u_*}")]

    def run(self, v, state):
        re_star = state["u_star"] * v["ks"] / v["nu"]
        regime = float(np.digitize(re_star, (_RE_SMOOTH, _RE_ROUGH + np.finfo(float).eps)))   # 0 / 1 / 2
        delta_v = _SUBLAYER_COEF * v["nu"] / state["u_star"]
        return ({"Re_star": float(re_star), "regime": regime, "delta_v": float(delta_v),
                 "ks_over_delta": float(v["ks"] / delta_v),
                 "z0_viscous_share": float(_Z0_SMOOTH_COEF * v["nu"] / state["u_star"] / state["z0"])}, {})


class ParticleSettlingEntrainment(SubModule):
    id, title = "1.3", "Particle Settling & Entrainment"

    def fields(self):
        return [InputField.from_constant(GRAVITY), InputField.from_constant(WATER_DENSITY),
                InputField.from_constant(SEDIMENT_DENSITY), InputField.from_constant(KINEMATIC_VISCOSITY),
                InputField("D", "Median Grain Diameter D (m)", 3.0e-4, 1.0e-5, 0.05, "%.3e"),
                InputField("C1", "Ferguson-Church (Stokes) Coefficient C₁ (dimensionless)", 18.0, 10.0, 60.0),
                InputField("C2", "Ferguson-Church Drag Coefficient C₂ (dimensionless)", 0.4, 0.1, 2.0),
                InputField("A_gp", "Garcia-Parker Coefficient A (dimensionless)", 1.3e-7, 1.0e-9, 1.0e-4, "%.3e"),
                InputField("E_max", "Entrainment Saturation Cap E_max (dimensionless)", 0.3, 0.01, 1.0)]

    def equations(self):
        return [("Settling velocity (Stokes drag limit C₁ν, turbulent-wake limit C₂)",
                 r"w_s=\frac{R g D^{2}}{C_1\nu+\sqrt{0.75\,C_2\,R g D^{3}}},\quad w_{s,\mathrm{Stokes}}=\frac{R g D^{2}}{C_1\nu},\quad R=\frac{\rho_s}{\rho_w}-1"),
                ("Garcia-Parker entrainment (C_a = E_s)",
                 r"E_s=\frac{A Z^{5}}{1+\frac{A}{E_{\max}}Z^{5}},\quad Z=\frac{u_*}{w_s}R_p^{0.6},\quad R_p=\frac{\sqrt{RgD}\,D}{\nu}")]

    def run(self, v, state):
        g, nu, D = v["g"], v["nu"], v["D"]
        R = v["rho_s"] / v["rho_w"] - 1.0
        w_s = R * g * D**2 / (v["C1"] * nu + np.sqrt(0.75 * v["C2"] * R * g * D**3))
        w_stokes = R * g * D**2 / (v["C1"] * nu)
        Rp = np.sqrt(R * g * D) * D / nu
        Z5 = (state["u_star"] / w_s * Rp**_GP_RP_EXPONENT) ** _GP_Z_EXPONENT
        E_s = v["A_gp"] * Z5 / (1.0 + (v["A_gp"] / v["E_max"]) * Z5)
        return ({"w_s": float(w_s), "w_stokes": float(w_stokes), "stokes_ratio": float(w_s / w_stokes),
                 "R_p": float(Rp), "Z_u": float(Z5 ** (1.0 / _GP_Z_EXPONENT)), "E_s": float(E_s),
                 "shields": float(state["u_star"] ** 2 / (R * g * D)),
                 "u_star_over_ws": float(state["u_star"] / w_s), "rp_valid": float(Rp > _GP_RP_MIN)}, {})


class EddyDiffusivitySolver(SubModule):
    id, title = "1.4", "Turbulent Eddy Diffusivity Solver"

    def fields(self):
        return [InputField.from_constant(VON_KARMAN),
                InputField("beta", "Reynolds-Analogy Ratio β = ε_s/ε_m (dimensionless)", 1.0, 0.2, 3.0),
                InputField("a_ratio", "Reference Level Ratio a/h (dimensionless)", 0.05, 0.001, 0.5)]

    def equations(self):
        return [("Parabolic momentum diffusivity", r"\varepsilon_m(z)=\kappa\,u_*\,z\left(1-\frac{z}{h}\right)"),
                ("Reynolds analogy (scalar vs momentum)", r"\varepsilon_s=\beta\,\varepsilon_m,\qquad \beta=\frac{1}{\mathrm{Sc}_t}")]

    def run(self, v, state):
        h, a = v["h"], v["a_ratio"] * v["h"]
        z = np.linspace(a, h * (1.0 - _SURFACE_CLIP), int(v["nz"]))
        eps_m = v["kappa"] * state["u_star"] * z * (1.0 - z / h)
        return {"a": float(a)}, {"z": z, "eps_m": eps_m, "eps_s": v["beta"] * eps_m}


class SuspensionODEIntegrator(SubModule):
    id, title = "1.5", "Suspension ODE Integrator"

    def equations(self):
        return [("Steady diffusion-settling balance", r"w_s\,C+\varepsilon_s\,\frac{dC}{dz}=0\;\Rightarrow\;\ln\frac{C}{C_a}=-\int_a^{z}\frac{w_s}{\varepsilon_s}\,dz'"),
                ("Analytic Rouse profile (verification)", r"\frac{C}{C_a}=\left[\frac{h-z}{z}\,\frac{a}{h-a}\right]^{P},\quad P=\frac{w_s}{\beta\kappa u_*}")]

    def run(self, v, state):
        z, eps, w_s, E_s, h, a = state["z"], state["eps_s"], state["w_s"], state["E_s"], v["h"], state["a"]
        P = w_s / (v["beta"] * v["kappa"] * state["u_star"])
        C_exact = E_s * (((h - z) / z) * (a / (h - a))) ** P
        f = w_s / eps
        I = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(z))))   # cumulative trapezoid
        C_ode = E_s * np.exp(-I)
        return ({"rouse_P": float(P), "ode_error": float(np.max(np.abs(C_ode - C_exact)) / E_s)},
                {"C_exact": C_exact, "C_ode": C_ode})


class FluxAggregator(SubModule):
    id, title = "1.6", "Macroscopic Flux Aggregator"

    def equations(self):
        return [("Suspended sediment flux", r"q_s=\int_a^{h}u(z)\,C(z)\,dz,\qquad \bar C=\frac{1}{h-a}\int_a^{h}C\,dz")]

    def run(self, v, state):
        z, C = state["z"], state["C_ode"]
        u_z = state["u_star"] / v["kappa"] * np.log(z / state["z0"])
        integrand = u_z * C
        q_s, c_int = _trapz(integrand, z), _trapz(C, z)
        return ({"q_s": q_s, "C_mean": float(c_int / (z[-1] - z[0])), "u_conc_weighted": float(q_s / c_int)},
                {"flux_integrand": integrand})


class MicroBenthicModel(BaseModel):
    key = "micro_benthic"
    title = "Micro-scale · Benthic Boundary Layer"
    scale = "Micro"
    summary = ("Six-stage pipeline: near-bed kinematics, sublayer regime, settling and entrainment, eddy diffusivity, "
               "Rouse ODE and suspended flux.")

    def submodules(self):
        return [NearBedKinematics(), ViscousSublayerDiagnostics(), ParticleSettlingEntrainment(),
                EddyDiffusivitySolver(), SuspensionODEIntegrator(), FluxAggregator()]
