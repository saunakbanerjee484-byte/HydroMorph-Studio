"""Client for the FastAPI engine with an in-process fallback.

If the server is unreachable the same engine code runs locally, so the UI never
dead-ends. Both paths return identical (scalars, arrays, source) tuples.
"""
from __future__ import annotations

import base64
import os

import requests
import streamlit as st

from typing import Any, cast

from core.base_model import BaseModel, unpack_arrays

API_URL = os.environ.get("HYDROMORPH_API_URL", "http://127.0.0.1:8000")


@st.cache_data(ttl=5, show_spinner=False)
def backend_online() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=0.4).ok
    except requests.exceptions.RequestException:
        return False


def run_model(model: BaseModel, values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    try:
        r = requests.post(f"{API_URL}/api/models/{model.key}/run", json=values, timeout=(1.0, 900.0))
    except requests.exceptions.ConnectionError:
        from backend.api_registry import MODEL_REGISTRY      # local fallback
        scalars_loc, arrays_loc = MODEL_REGISTRY[model.key].run(values)
        return scalars_loc, arrays_loc, "local"
    if r.status_code == 422:
        raise ValueError(r.json()["detail"]["errors"])
    if not r.ok:
        err_msg: str = r.text[:300]
        if r.headers.get("content-type", "").startswith("application/json"):
            try:
                data: dict[str, Any] = cast(dict[str, Any], r.json())
                detail_val = data.get("detail")
                if isinstance(detail_val, dict):
                    detail_dict: dict[str, Any] = cast(dict[str, Any], detail_val)
                    errors_val = detail_dict.get("errors")
                    if isinstance(errors_val, list):
                        raw_list: list[Any] = cast(list[Any], errors_val)
                        err_msg = "; ".join(str(item) for item in raw_list)
            except Exception:
                pass
        raise RuntimeError(err_msg)
    body: dict[str, Any] = cast(dict[str, Any], r.json())
    arrays: dict[str, Any] = unpack_arrays(base64.b64decode(body["arrays_npz_b64"]))
    scalars: dict[str, Any] = cast(dict[str, Any], body["scalars"])
    return scalars, arrays, "api"

