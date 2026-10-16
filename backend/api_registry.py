"""Backend router registry: one place that maps model keys to engines and HTTP routes."""
from __future__ import annotations

import base64
from dataclasses import asdict
from typing import Dict

from fastapi import APIRouter, Body, HTTPException

from backend.engines.macro_engine import MacroTensorModel
from backend.engines.meso_engine import MesoExnerModel
from backend.engines.micro_engine import MicroBenthicModel
from core.base_model import BaseModel, pack_arrays

MODEL_REGISTRY: dict[str, BaseModel] = {m.key: m for m in (MicroBenthicModel(), MesoExnerModel(), MacroTensorModel())}


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/models")
    def list_models():
        return [{"key": m.key, "title": m.title, "scale": m.scale} for m in MODEL_REGISTRY.values()]

    @router.get("/models/{key}/spec")
    def spec(key: str):
        model = _get(key)
        return {"fields": [asdict(f) for f in model.fields()], "equations": model.equations()}

    @router.post("/models/{key}/run")
    def run(key: str, inputs: Dict[str, float] = Body(...)):
        model = _get(key)
        try:
            scalars, arrays = model.run(inputs)
        except ValueError as exc:                       # failed validation -> 422 with every message
            raise HTTPException(status_code=422, detail={"errors": exc.args[0]})
        except RuntimeError as exc:                     # numerical failure (e.g. blow-up)
            raise HTTPException(status_code=500, detail={"errors": [str(exc)]})
        return {"scalars": scalars, "arrays_npz_b64": base64.b64encode(pack_arrays(arrays)).decode("ascii")}

    return router


def _get(key: str) -> BaseModel:
    if key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail={"errors": [f"Unknown model '{key}'."]})
    return MODEL_REGISTRY[key]
