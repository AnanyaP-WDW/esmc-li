"""Prebuilt index registry and HF cache resolution."""
from __future__ import annotations

from pathlib import Path

from esmc_li.models import cache_root

INDEXES = {
    "scope-1k": {
        "repo_id": "AnanyaPathak/esmc-retrieval-indexes",
        "folder": "scope-1k",
        "name": "esmc-300m-homology-maxsim-v1-scope-1k",
    },
    "swissprot-100k": {
        "repo_id": "AnanyaPathak/esmc-retrieval-indexes",
        "folder": "swissprot-100k",
        "name": "esmc-300m-homology-maxsim-v1-swissprot-100k",
    },
}


def resolve_index(index: str | Path) -> Path:
    p = Path(index)
    if p.exists():
        return p
    rec = INDEXES.get(str(index))
    if rec is None:
        raise FileNotFoundError(
            f"index not found: {index}; available: {', '.join(INDEXES)}")
    from huggingface_hub import hf_hub_download

    base = None
    for suffix in (".esmci", ".esmci.json", ".esmci.head.npy"):
        got = Path(hf_hub_download(
            repo_id=rec["repo_id"],
            filename=f"{rec['folder']}/{rec['name']}{suffix}",
            repo_type="dataset",
            cache_dir=str(cache_root() / "hub"),
        ))
        if suffix == ".esmci":
            base = got
    assert base is not None
    return base
