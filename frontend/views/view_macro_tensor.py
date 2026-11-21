import streamlit as st

from backend.engines.macro_engine import MacroTensorModel
from frontend.components import plots
from frontend.components.workflow import metrics, run_workflow


def _results(s: dict, a: dict, inputs: dict) -> None:
    if s["no_flow"]:
        st.info("The driving stress at the bed does not exceed the yield stress anywhere: the material stays at rest.")
    metrics([
        ("Mixture density ρ_m (kg/m³)", f"{s['rho_m']:.0f}"), ("Discharge Q (m³/s)", f"{s['Q']:.3f}"),
        ("Lateral discharge (m³/s)", f"{s['Q_lateral']:.3f}"), ("Mean velocity (m/s)", f"{s['u_mean']:.3f}"),
        ("Peak velocity (m/s)", f"{s['u_max']:.3f}"), ("Flow azimuth (°)", f"{s['flow_azimuth_deg']:.1f}"),
        ("Centre plug thickness (m)", f"{s['plug_thickness_centre']:.3f}"), ("Sheared fraction of flow", f"{s['yielded_fraction']:.2f}"),
        ("Bingham number Bi", f"{s['Bingham_number']:.2f}"), ("Max bed scour (m)", f"{s['max_scour']:.3f}"),
        ("Eroded area (m²)", f"{s['eroded_area']:.3f}"), ("Inertia / gravity ratio", f"{s['inertia_ratio']:.2f}"),
    ], per_row=3)
    st.caption(f"Checks: continuity residual {s['mass_residual']:.1e}, stress-partition error {s['partition_error']:.1e}, "
               f"yield-surface detection error {s['yield_surface_error']:.2f} cells.")
    figs = plots.macro_figures(a)
    for tab, (_, fig) in zip(st.tabs([n for n, _ in figs]), figs):
        with tab:
            plots.show_plot(fig)


def render() -> None:
    run_workflow(MacroTensorModel(), _results)
