"""Streamlit entry point. Run from the project root:  python -m streamlit run frontend/app.py

This file is a thin shell: it never contains physics and never needs editing when a model is added.
Any failure inside a page is caught and reported without taking the app down.
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="HydroMorph-Analytics", page_icon="🌊", layout="wide")
    try:
        from frontend.components.api_client import API_URL, backend_online
        from frontend.components.theme import inject_theme
        from frontend.registry import MODELS, load_renderer

        inject_theme()
        with st.sidebar:
            st.markdown("### HydroMorph-Analytics")
            key = st.radio("Model", list(MODELS), format_func=lambda k: MODELS[k].label, key="nav",
                           label_visibility="collapsed")
            st.caption("Compute engine: " + ("FastAPI online" if backend_online() else "API offline, using in-process engine"))
            st.caption(f"Endpoint: {API_URL}")
        load_renderer(key)()
    except Exception as exc:                                    # noqa: BLE001 - the shell must never fail
        st.error(f"This page failed to load: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


main()
