import streamlit as st

from backend.engines.meso_engine import StatePersister, MesoExnerModel
from frontend.components import plots
from frontend.components.workflow import metrics, run_workflow


def _results(s: dict, a: dict, inputs: dict) -> None:
    metrics([
        ("Normal depth h_n (m)", f"{s['h_normal']:.3f}"), ("Base Froude number", f"{s['Fr_normal']:.3f}"),
        ("Peak Froude number", f"{s['Fr_max']:.3f}"), ("Max bed change (m)", f"{s['max_bed_change']:.3f}"),
        ("Perturbation centroid shift (m)", f"{s['centroid_shift']:.1f}"), ("Lowest effective MorFac ψ", f"{s['psi_min']:.1f}"),
        ("Time steps", f"{int(s['steps']):,}"), ("Peak inflow q (m²/s)", f"{s['q_peak_used']:.2f}"),
    ], per_row=4)
    st.caption(f"Sediment mass-balance error: {s['sediment_mass_error']:.1e} (relative)")
    plots.show_plot(plots.meso_animation(a, inputs["morph_factor"], inputs["t_spin"]))
    plots.show_plot(plots.meso_final(a))
    plots.show_plot(plots.meso_hydrograph(a))


def _exports(res: dict):
    blob = StatePersister.to_parquet(res["arrays"])
    if blob is None:
        return []
    return [("Export time series (Parquet)", blob, "hydromorph_meso_timeseries.parquet", "application/octet-stream")]


def render() -> None:
    run_workflow(MesoExnerModel(), _results, exports=_exports)
