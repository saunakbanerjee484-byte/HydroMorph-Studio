"""Abstract base class that every HydroMorph engine must implement.

A model declares (1) its input fields with SI units and admissible ranges,
(2) the governing equations to display, and (3) a pure `solve` function.
Validation lives here so the frontend (instant feedback) and the backend
(authoritative check) share exactly one set of rules.
"""
from __future__ import annotations

import io
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from core.physics_constants import Constant


@dataclass(frozen=True)
class InputField:
    key: str
    label: str                 # must carry its SI unit, e.g. "Discharge (m³/s)"
    default: float
    min_value: float
    max_value: float
    fmt: str = "%.4g"
    integer: bool = False
    help: str = ""
    group: str = "Parameters"

    @classmethod
    def from_constant(cls, c: Constant, group: str = "Physical constants") -> "InputField":
        return cls(c.key, c.label, c.default, c.min_value, c.max_value, c.fmt, False, c.help, group)

    @property
    def step(self) -> float:
        if self.integer:
            return 1
        d = abs(self.default)
        return 0.01 if d == 0 else 10.0 ** (math.floor(math.log10(d)) - 1)


CONSTANTS_GROUP = "Physical constants"


class SubModule(ABC):
    """One numbered stage of an engine pipeline (e.g. 1.3 Particle Settling & Entrainment).

    A sub-module declares the inputs it owns, the equations it implements and a `run`
    that reads upstream results from `state` and returns (scalars, arrays) for downstream stages.
    """
    id: str = ""
    title: str = ""

    @property
    def label(self) -> str:
        return f"{self.id} · {self.title}"

    def fields(self) -> list[InputField]:
        return []

    def equations(self) -> list[tuple[str, str]]:
        return []

    def cross_validate(self, v: dict[str, Any]) -> list[str]:
        return []

    def run(self, v: dict[str, Any], state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError(f"{self.label} has no stand-alone run(); the engine drives it step by step.")


class BaseModel(ABC):
    key: str = ""
    title: str = ""
    scale: str = ""
    summary: str = ""
    execution_order: tuple[str, ...] | None = None     # ids, when data dependencies differ from numbering

    # ---- declarative interface (aggregated from the sub-modules) --------------------
    @abstractmethod
    def submodules(self) -> list[SubModule]:
        """Sub-modules in numbering order."""

    def fields(self) -> list[InputField]:
        seen: set[str] = set()
        out: list[InputField] = []
        for sm in self.submodules():
            for f in sm.fields():
                if f.key in seen:                         # shared inputs (g, kappa, ...) appear once
                    continue
                seen.add(f.key)
                out.append(f if f.group == CONSTANTS_GROUP else replace(f, group=sm.label))
        out.sort(key=lambda f: f.group != CONSTANTS_GROUP)   # constants first, order otherwise kept
        return out

    def equations(self) -> list[tuple[str, str]]:
        return [(f"{sm.id} · {cap}", tex) for sm in self.submodules() for cap, tex in sm.equations()]

    def cross_validate(self, v: dict[str, Any]) -> list[str]:
        return [e for sm in self.submodules() for e in sm.cross_validate(v)]

    def solve(self, v: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Default engine: run the sub-modules once, in dependency order."""
        by_id = {sm.id: sm for sm in self.submodules()}
        sequence = [by_id[i] for i in (self.execution_order or tuple(by_id))]
        state: dict[str, Any] = dict(v)
        scalars: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        for sm in sequence:
            t0 = perf_counter()
            s_out, a_out = sm.run(v, state)
            state.update(s_out)
            state.update(a_out)
            scalars.update(s_out)
            arrays.update(a_out)
            scalars[timing_key(sm.id)] = (perf_counter() - t0) * 1.0e3
        return scalars, arrays

    # ---- shared machinery -----------------------------------------------------
    def defaults(self) -> dict[str, Any]:
        return {f.key: (int(f.default) if f.integer else float(f.default)) for f in self.fields()}

    def coerce(self, values: dict[str, Any]) -> dict[str, Any]:
        return {f.key: (int(values[f.key]) if f.integer else float(values[f.key])) for f in self.fields()}

    def validate(self, values: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for f in self.fields():
            raw = values.get(f.key)
            if raw is None or isinstance(raw, bool):
                errors.append(f"{f.label}: value is missing.")
                continue
            try:
                x = float(raw)
            except (TypeError, ValueError):
                errors.append(f"{f.label}: '{raw}' is not a number.")
                continue
            if not math.isfinite(x):
                errors.append(f"{f.label}: value must be finite.")
            elif x < f.min_value or x > f.max_value:
                errors.append(f"{f.label}: {x:g} is outside the admissible range "
                              f"[{f.min_value:g}, {f.max_value:g}].")
            elif f.integer and x != int(x):
                errors.append(f"{f.label}: must be a whole number.")
        if not errors:
            errors += self.cross_validate(self.coerce(values))
        return errors

    def run(self, values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        errors = self.validate(values)
        if errors:
            raise ValueError(errors)
        return self.solve(self.coerce(values))


def timing_key(sub_id: str) -> str:
    return "time_" + sub_id.replace(".", "_")


# ---- tensor serialisation (compressed binary, no pickle) ---------------------
def pack_arrays(arrays: dict[str, Any]) -> bytes:
    """Compressed .npz blob. float64 is stored as float32 to halve size and I/O."""
    buf = io.BytesIO()
    prepared = {k: (np.asarray(a, dtype=np.float32) if np.asarray(a).dtype == np.float64 else np.asarray(a))
                for k, a in arrays.items()}
    np.savez_compressed(buf, **prepared)
    return buf.getvalue()


def unpack_arrays(blob: bytes) -> dict[str, Any]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {str(k): z[k] for k in z.files}
