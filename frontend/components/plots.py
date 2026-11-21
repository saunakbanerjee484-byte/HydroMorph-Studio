"""Plotly figure builders. Charts are fully transparent so the glass gradient shows through, animation
runs from Plotly frames in the browser (Python never loops over time to redraw), and its axes are locked."""
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

PINK, BLUE, VIOLET, TEAL = "#ff4f9a", "#3d7bff", "#9b5cff", "#18b5a4"
GLASS_SCALE = [[0.0, "#eaf2ff"], [0.35, "#8fb4ff"], [0.7, "#c58bff"], [1.0, "#ff4f9a"]]
_WEBGL_THRESHOLD = 1500        # switch to WebGL traces for very long profiles
_CLEAR = "rgba(0,0,0,0)"


def show_plot(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                                   # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height, template="plotly_white", paper_bgcolor=_CLEAR, plot_bgcolor=_CLEAR,
        font=dict(color="#1b2447"), colorway=[PINK, BLUE, VIOLET, TEAL],
        margin=dict(l=60, r=20, t=50, b=50), legend=dict(orientation="h", y=1.12, bgcolor=_CLEAR),
    )
    fig.update_xaxes(gridcolor="rgba(74,100,200,.18)", zerolinecolor="rgba(74,100,200,.25)")
    fig.update_yaxes(gridcolor="rgba(74,100,200,.18)", zerolinecolor="rgba(74,100,200,.25)")
    return fig


def _sci_log(fig: go.Figure, axis: str) -> None:
    """Log axis labelled only at decades in 10^n notation (no mixed micro-formatting)."""
    update = fig.update_xaxes if axis == "x" else fig.update_yaxes
    update(type="log", exponentformat="power", showexponent="all", dtick=1, minor=dict(showgrid=True, ticks="outside"))


# ---- micro ------------------------------------------------------------------
def micro_figures(a: dict[str, Any]) -> list[tuple[str, go.Figure]]:
    f1 = go.Figure(go.Scatter(x=a["u"], y=a["z_u"], mode="lines", line=dict(color=BLUE, width=3), name="u(z)"))
    f1.update_layout(title="Law of the Wall", xaxis_title="Velocity u (m/s)", yaxis_title="Height z (m)")
    _sci_log(f1, "y")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=a["eps_m"], y=a["z"], name="ε_m (momentum)", line=dict(color=VIOLET, width=3)))
    f2.add_trace(go.Scatter(x=a["eps_s"], y=a["z"], name="ε_s (sediment)", line=dict(color=PINK, width=2, dash="dash")))
    f2.update_layout(title="Parabolic eddy diffusivity", xaxis_title="Diffusivity (m²/s)", yaxis_title="Height z (m)")
    ca = a["C_exact"][0]
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=a["C_exact"] / ca, y=a["z"], name="Rouse analytic", line=dict(color=PINK, width=3)))
    f3.add_trace(go.Scatter(x=a["C_ode"] / ca, y=a["z"], name="ODE (numerical)", line=dict(color=BLUE, width=2, dash="dash")))
    f3.update_layout(title="Suspended-sediment profile", xaxis_title="C / C_a (dimensionless)", yaxis_title="Height z (m)")
    _sci_log(f3, "x")
    f4 = go.Figure(go.Scatter(x=a["flux_integrand"], y=a["z"], fill="tozerox", name="u·C",
                              line=dict(color=TEAL, width=3), fillcolor="rgba(24,181,164,.25)"))
    f4.update_layout(title="Suspended-flux integrand", xaxis_title="u·C (m/s)", yaxis_title="Height z (m)")
    return [("Velocity", _style(f1)), ("Diffusivity", _style(f2)), ("Suspension", _style(f3)), ("Flux", _style(f4))]


# ---- meso -------------------------------------------------------------------
def meso_animation(a: dict[str, Any], morph_factor: float, t_spin: float) -> go.Figure:
    x, t, zb, h = a["x"], a["t"], a["zb"].astype(float), a["h"].astype(float)
    surface = zb + h
    min_elev, max_elev = float(zb.min()) - 0.05, float(surface.max()) + 0.15          # locked once, from the full tensor
    Trace = go.Scattergl if x.size > _WEBGL_THRESHOLD else go.Scatter
    bed = Trace(x=x, y=zb[0], name="Bed", fill="tozeroy", line=dict(color=VIOLET, width=2), fillcolor="rgba(155,92,255,.35)")
    wat = Trace(x=x, y=surface[0], name="Water surface", fill="tonexty", line=dict(color=BLUE, width=2), fillcolor="rgba(61,123,255,.28)")
    morph_h = np.clip(t - t_spin, 0.0, None) * morph_factor / 3600.0
    frames = [go.Frame(name=str(i), data=[go.Scatter(x=x, y=zb[i]), go.Scatter(x=x, y=surface[i])], traces=[0, 1])
              for i in range(t.size)]
    fig = go.Figure(data=[bed, wat], frames=frames)
    step_args = {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
    fig.update_layout(
        title="Bed and free-surface evolution", xaxis_title="Streamwise distance x (m)", yaxis_title="Elevation (m)",
        xaxis=dict(range=[float(x[0]), float(x[-1])], fixedrange=True),
        yaxis=dict(range=[min_elev, max_elev], fixedrange=True),
        updatemenus=[dict(type="buttons", x=0.0, y=-0.18, direction="left", buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 60, "redraw": True},
                                                              "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
        sliders=[dict(x=0.18, len=0.8, y=-0.1, currentvalue={"prefix": "Morphological time: ", "suffix": " h"},
                      steps=[dict(method="animate", args=[[str(i)], step_args], label=f"{morph_h[i]:.2f}") for i in range(t.size)])],
    )
    return _style(fig, 480)


def meso_final(a: dict[str, Any]) -> go.Figure:
    """Initial vs final bed; fill='tonexty' shades the scoured / deposited sediment volume between them."""
    x = a["x"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=a["zb"][0], name="Initial bed", line=dict(color=BLUE, dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=x, y=a["zb"][-1], name="Final bed", line=dict(color=PINK, width=3),
                             fill="tonexty", fillcolor="rgba(255,79,154,.22)"))
    fig.update_layout(title="Initial vs final bed (shaded: scoured / deposited volume)",
                      xaxis_title="Streamwise distance x (m)", yaxis_title="Bed elevation z_b (m)")
    return _style(fig, 360)


def meso_hydrograph(a: dict[str, Any]) -> go.Figure:
    fig = go.Figure(go.Scatter(x=a["t"], y=a["q_in"], line=dict(color=BLUE, width=3), fill="tozeroy",
                               fillcolor="rgba(61,123,255,.2)", name="q_in"))
    fig.update_layout(title="Inflow hydrograph", xaxis_title="Hydraulic time t (s)", yaxis_title="Unit discharge q_in (m²/s)")
    return _style(fig, 300)


# ---- macro ------------------------------------------------------------------
def _heat(a: dict[str, Any], field: np.ndarray, title: str, unit: str, log10: bool = False) -> go.Figure:
    data = np.where(a["fluid"] > 0.5, field, np.nan)
    if log10:
        data = np.log10(np.maximum(data, np.finfo(float).tiny))
        unit = f"log₁₀ {unit}"
    fig = go.Figure(go.Heatmap(x=a["y"], y=a["z"], z=data.T, colorscale=GLASS_SCALE, hoverongaps=False, colorbar=dict(title=unit)))
    fig.update_layout(title=title, xaxis_title="Lateral coordinate y (m)", yaxis_title="Height z (m)")
    return _style(fig, 400)


def _yield_fig(a: dict[str, Any]) -> go.Figure:
    y = a["y"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=a["z_y"], name="Yield surface z_y (analytic)", line=dict(color=VIOLET, width=3)))
    fig.add_trace(go.Scatter(x=y, y=a["z_y_num"], name="Detected from shear rate", line=dict(color=TEAL, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=y, y=a["H"], name="Free surface H(y)", line=dict(color=BLUE, width=3),
                             fill="tonexty", fillcolor="rgba(61,123,255,.22)"))
    fig.update_layout(title="Yield interface (shaded: rigid plug)", xaxis_title="Lateral coordinate y (m)", yaxis_title="Height z (m)")
    return _style(fig)


def _bed_fig(a: dict[str, Any]) -> go.Figure:
    y = a["y"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=a["bed_before"], name="Bed before event", line=dict(color=BLUE, dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=y, y=a["bed_after"], name="Bed after event", line=dict(color=PINK, width=3),
                             fill="tonexty", fillcolor="rgba(255,79,154,.22)"))
    fig.update_layout(title="Cross-section bed deformation (shaded: eroded volume)",
                      xaxis_title="Lateral coordinate y (m)", yaxis_title="Bed elevation (m)")
    return _style(fig)


def macro_figures(a: dict[str, Any]) -> list[tuple[str, go.Figure]]:
    return [
        ("Velocity", _heat(a, a["u"], "Streamwise velocity u", "m/s")),
        ("Lateral v", _heat(a, a["v"], "Lateral velocity v (cross-slope driven)", "m/s")),
        ("Shear rate", _heat(a, a["gamma_dot"], "Shear rate γ̇", "1/s")),
        ("Viscosity", _heat(a, a["eta_app"], "Apparent viscosity η", "Pa·s", log10=True)),
        ("Stress", _heat(a, a["tau_eq"], "Equivalent shear stress τ_eq", "Pa")),
        ("Principal σ", _heat(a, a["sigma1"], "Major principal stress σ₁ (tension positive)", "Pa")),
        ("Yield surface", _yield_fig(a)),
        ("Bed update", _bed_fig(a)),
    ]
