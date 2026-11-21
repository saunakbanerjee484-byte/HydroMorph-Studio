from typing import Any

import streamlit as st

from backend.engines.micro_engine import MicroBenthicModel
from frontend.components import plots
from frontend.components.workflow import metrics, run_workflow

_REGIMES = {0: "hydraulically smooth", 1: "transitional", 2: "hydraulically rough"}


def _results(s: dict[str, Any], a: dict[str, Any], inputs: dict[str, Any]) -> None:
    if not s["rp_valid"]:
        st.warning("Particle Reynolds number is below 2.36, outside the validity range of the Garcia-Parker closure.")
    st.caption(f"Bed regime: **{_REGIMES[int(float(s['regime']))]}** (Re* = {float(s['Re_star']):.1f}, "
               f"viscous sublayer δ_v = {s['delta_v']*1e3:.2f} mm, k_s/δ_v = {s['ks_over_delta']:.1f}). "
               f"u*/w_s = {s['u_star_over_ws']:.2f}: "
               + ("sediment is carried in suspension." if s["u_star_over_ws"] > 1 else "transport stays mostly near the bed."))
    metrics([
        ("Shear velocity u* (m/s)", f"{s['u_star']:.4f}"), ("Roughness length z₀ (m)", f"{s['z0']:.2e}"),
        ("Depth-mean velocity (m/s)", f"{s['u_mean']:.3f}"), ("Settling velocity w_s (m/s)", f"{s['w_s']:.4f}"),
        ("w_s / Stokes limit", f"{s['stokes_ratio']:.2f}"), ("Entrainment E_s = C_a", f"{s['E_s']:.3e}"),
        ("Rouse number P", f"{s['rouse_P']:.3f}"), ("Shields number θ", f"{s['shields']:.3f}"),
        ("Suspended flux q_s (m²/s)", f"{s['q_s']:.3e}"), ("Mean concentration C̄", f"{s['C_mean']:.2e}"),
        ("Conc.-weighted velocity (m/s)", f"{s['u_conc_weighted']:.3f}"), ("Rouse ODE error (norm.)", f"{s['ode_error']:.1e}"),
    ], per_row=3)
    figs = plots.micro_figures(a)
    for tab, (_, fig) in zip(st.tabs([n for n, _ in figs]), figs):
        with tab:
            plots.show_plot(fig)


def render() -> None:
    run_workflow(MicroBenthicModel(), _results)
