"""Shared page workflow: equations -> inputs -> validation -> confirm -> results -> save/load."""
from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from core.base_model import BaseModel, InputField, timing_key
from frontend.components import theme
from frontend.components.api_client import run_model
from frontend.components.persistence import build_snapshot, load_snapshot


def _widget(model_key: str, f: InputField) -> Any:
    key = f"{model_key}::{f.key}"
    if f.integer:
        return st.number_input(f.label, value=int(f.default), step=1, format="%d", key=key, help=f.help or None)
    return st.number_input(f.label, value=float(f.default), step=float(f.step), format=f.fmt,
                           key=key, help=f.help or None)


def render_inputs(model: BaseModel) -> dict[str, Any]:
    """number_input only (no sliders). No min/max are passed to the widget, so out-of-range
    entries reach validation and are reported instead of being silently clamped."""
    groups: dict[str, list[InputField]] = {}
    for f in model.fields():
        groups.setdefault(f.group, []).append(f)
    values = {}
    for group, fields in groups.items():
        with st.expander(group, expanded=True):
            cols = st.columns(2)
            for i, f in enumerate(fields):
                with cols[i % 2]:
                    values[f.key] = _widget(model.key, f)
    return values


def metrics(items: list[tuple[str, str]], per_row: int = 3) -> None:
    for start in range(0, len(items), per_row):
        row = items[start:start + per_row]
        for col, (label, value) in zip(st.columns(len(row)), row):
            col.metric(label, value)


def _handle_upload(model: BaseModel, state_key: str) -> None:
    up = st.file_uploader("Load saved state (.zip), skips the solver", type=["zip"], key=f"upload::{model.key}")
    if up is None:
        return
    uid, seen_key = f"{up.name}:{up.size}", f"seen::{model.key}"
    if st.session_state.get(seen_key) == uid:
        return
    try:
        snap = load_snapshot(up.getvalue())
    except Exception as exc:                                      # noqa: BLE001 - show any parse failure
        st.error(f"Could not read the archive: {exc}")
        return
    if snap["model"] != model.key:
        st.error(f"This archive belongs to '{snap['model']}', not '{model.key}'.")
        return
    spec = {f.key: f for f in model.fields()}
    for k, val in snap["inputs"].items():                         # echo saved inputs into the widgets
        if k in spec:
            st.session_state[f"{model.key}::{k}"] = int(val) if spec[k].integer else float(val)
    st.session_state[state_key] = {"scalars": snap["scalars"], "arrays": snap["arrays"],
                                   "inputs": snap["inputs"], "source": "snapshot"}
    st.session_state[seen_key] = uid
    st.success("State loaded. Results were rendered without running the solver.")


def _download(label: str, **kw) -> None:
    try:
        st.download_button(label, width="stretch", **kw)
    except TypeError:                                             # older Streamlit
        st.download_button(label, use_container_width=True, **kw)


def _pipeline_report(model: BaseModel, scalars: dict[str, Any]) -> None:
    rows = [f"| {sm.id} | {sm.title} | {scalars[timing_key(sm.id)]:.1f} |"
            for sm in model.submodules() if timing_key(sm.id) in scalars]
    if rows:
        with st.expander("Pipeline stages and compute time"):
            st.markdown("| Stage | Sub-module | Time (ms) |\n|---|---|---|\n" + "\n".join(rows))


def run_workflow(
    model: BaseModel,
    render_results: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None],
    exports: Callable[[dict[str, Any]], list[tuple[str, bytes, str, str]]] | None = None,
) -> None:
    state_key = f"result::{model.key}"
    theme.hero(model.title, model.summary)

    session = st.sidebar.container(key="session_mgmt")          # anchored, visually distinct export/import area
    with session:
        st.markdown("#### Session Management")
        _handle_upload(model, state_key)

    with st.expander("📐 View Governing Equations", expanded=False):
        st.subheader("Governing equations")
        for caption, tex in model.equations():
            st.caption(caption)
            st.latex(tex)

    st.subheader("Inputs")
    values = render_inputs(model)

    errors = model.validate(values)
    for msg in errors:
        st.error(msg)

    if st.button("Confirm Inputs & Execute", type="primary", disabled=bool(errors), key=f"run::{model.key}"):
        try:
            with st.spinner("Solving…"):
                scalars, arrays, source = run_model(model, values)
        except ValueError as exc:
            for msg in (exc.args[0] if exc.args and isinstance(exc.args[0], list) else [str(exc)]):
                st.error(msg)
        except Exception as exc:                                   # noqa: BLE001
            st.error(f"Solver failed: {exc}")
        else:
            st.session_state[state_key] = {"scalars": scalars, "arrays": arrays,
                                           "inputs": dict(values), "source": source}

    res = st.session_state.get(state_key)
    if not res:
        with session:
            st.caption("Run a calculation to enable state export.")
        st.info("Set the inputs, then press **Confirm Inputs & Execute**. Nothing runs until you do.")
        return

    st.subheader("Results")
    origin = {"api": "FastAPI engine", "local": "in-process engine (API offline)",
              "snapshot": "saved state, solver bypassed"}[res["source"]]
    st.caption(f"Source: {origin}")
    if res["source"] != "snapshot" and any(abs(float(res["inputs"].get(k, v)) - float(v)) > 0 for k, v in values.items()):
        st.warning("Inputs changed since this result was computed. Press Confirm Inputs & Execute to update it.")
    _pipeline_report(model, res["scalars"])
    render_results(res["scalars"], res["arrays"], res["inputs"])

    with session:
        _download("Download system state (.zip)",
                  data=build_snapshot(model.key, {k: float(v) for k, v in res["inputs"].items()},
                                      {k: float(v) for k, v in res["scalars"].items()}, res["arrays"]),
                  file_name=f"hydromorph_{model.key}_state.zip", mime="application/zip", key=f"dl::{model.key}")
        for label, blob, fname, mime in (exports(res) if exports else []):
            _download(label, data=blob, file_name=fname, mime=mime, key=f"ex::{model.key}::{fname}")
