"""Central model dictionary and routing. Adding a model = one entry here + one view + one backend engine."""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    key: str
    label: str
    view_module: str


MODELS: dict[str, ModelEntry] = {e.key: e for e in (
    ModelEntry("micro_benthic", "Micro · Benthic Boundary", "frontend.views.view_micro_benthic"),
    ModelEntry("meso_exner", "Meso · Exner Morphodynamics", "frontend.views.view_meso_exner"),
    ModelEntry("macro_tensor", "Macro · MorphoTensor", "frontend.views.view_macro_tensor"),
)}


def load_renderer(key: str):
    return importlib.import_module(MODELS[key].view_module).render
