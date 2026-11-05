"""Model registry and HF cache resolution for the M8 package."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_MODEL = {
    "repo_id": "AnanyaPathak/esmc-li-300m-v1",
    "filename": "esmc-300m-homology-maxsim-v1-f16.gguf",
    "sha256": "e87b42f1a6344e57b7a0d8a6624cc9d7e2e15f34d4a6927447b4cb3c90209bab",
}
MODELS = {
    "esmc-li-300m-v1": _MODEL,
    # historical registry name; old indexes/sidecars still resolve
    "esmc-300m-homology-maxsim-v1": _MODEL,
    "AnanyaPathak/esmc-300m-homology-maxsim-v1": _MODEL,
}
DEFAULT_MODEL = "esmc-li-300m-v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cache_root() -> Path:
    return Path(os.environ.get(
        "ESMC_RETRIEVAL_CACHE", Path.home() / ".cache" / "esmc-retrieval"))
# historical env/cache names kept for compatibility with v1 downloads


def resolve_model(model: str | Path | None) -> Path | None:
    """Resolve a local path, registry name, or registry filename to GGUF."""
    if model is None:
        return None
    p = Path(model)
    if p.exists():
        return p
    key = str(model)
    rec = MODELS.get(key)
    if rec is None:
        for name, r in MODELS.items():
            if key in (name, r["repo_id"], r["filename"]):
                rec = r
                break
    if rec is None:
        raise FileNotFoundError(
            f"model not found: {model}; use {DEFAULT_MODEL}, a registry "
            "filename, a local path, or ESMC_RETRIEVAL_GGUF")
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        repo_id=rec["repo_id"], filename=rec["filename"],
        cache_dir=str(cache_root() / "hub")))
    if _sha256(path) != rec["sha256"]:
        raise ValueError(f"model hash mismatch: {path}")
    return path
