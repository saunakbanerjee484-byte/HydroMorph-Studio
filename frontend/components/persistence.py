"""State-space snapshots: inputs (JSON) + result tensors (compressed npz) in one zip."""
from __future__ import annotations

import io
import json
import zipfile

from core.base_model import pack_arrays, unpack_arrays

SNAPSHOT_VERSION = 1
_MAX_UNCOMPRESSED = 300 * 1024 * 1024


def build_snapshot(model_key: str, inputs: dict, scalars: dict, arrays: dict) -> bytes:
    meta = {"version": SNAPSHOT_VERSION, "model": model_key, "inputs": inputs, "scalars": scalars}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json", json.dumps(meta, indent=2))
        z.writestr("arrays.npz", pack_arrays(arrays), compress_type=zipfile.ZIP_STORED)   # already compressed
    return buf.getvalue()


def load_snapshot(blob: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if sum(i.file_size for i in z.infolist()) > _MAX_UNCOMPRESSED:
            raise ValueError("Archive is too large to be a HydroMorph snapshot.")
        if not {"meta.json", "arrays.npz"} <= set(z.namelist()):
            raise ValueError("Archive is missing meta.json or arrays.npz.")
        meta = json.loads(z.read("meta.json"))
        arrays = unpack_arrays(z.read("arrays.npz"))
    if meta.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported snapshot version {meta.get('version')}.")
    return {"model": meta["model"], "inputs": meta["inputs"], "scalars": meta["scalars"], "arrays": arrays}
