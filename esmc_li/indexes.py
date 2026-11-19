"""Prebuilt index registry and HF cache resolution."""
from __future__ import annotations

import hashlib
from pathlib import Path

from esmc_li.models import cache_root

_VERIFIED: set[tuple[str, str]] = set()
_RESOLVED: dict[str, Path] = {}

INDEXES = {
    "scope-1k": {
        "repo_id": "AnanyaPathak/esmc-li-indexes",
        "folder": "scope-1k",
        "name": "esmc-300m-homology-maxsim-v1-scope-1k",
        "sha256": {
            ".esmci": "05fe3e18b43c5c8c012b66f07b71aa912a1bd4dfd717daa130986118a167f212",
            ".esmci.json": "c4f67afe4c0fe795c28ae49a22789c3018116227043de3ec3dac02cf012842ea",
            ".esmci.head.npy": "b86b04e28a658ea9a3a209e476a6e130a9f3c83e6fc0ab7f7cd9a48694a09bc3",
        },
    },
    "swissprot-100k": {
        "repo_id": "AnanyaPathak/esmc-li-indexes",
        "folder": "swissprot-100k",
        "name": "esmc-300m-homology-maxsim-v1-swissprot-100k",
        "sha256": {
            ".esmci": "69a31248a332320552b6d7c57d2decbfec380183946a2fb83bd629a882c90d31",
            ".esmci.json": "2a9b3aa021cfd1178b27d9b88719d0144fd8267818edbee62230c7339c42f50d",
            ".esmci.head.npy": "b86b04e28a658ea9a3a209e476a6e130a9f3c83e6fc0ab7f7cd9a48694a09bc3",
        },
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve_index(index: str | Path) -> Path:
    p = Path(index)
    if p.exists():
        return p
    if str(index) in _RESOLVED and _RESOLVED[str(index)].exists():
        return _RESOLVED[str(index)]
    rec = INDEXES.get(str(index))
    if rec is None:
        raise FileNotFoundError(
            f"index not found: {index}; available: {', '.join(INDEXES)}")
    from huggingface_hub import hf_hub_download

    files = {}
    for suffix in (".esmci", ".esmci.json", ".esmci.head.npy"):
        name = f"{rec['name']}{suffix}"
        want = rec["sha256"][suffix]
        path = Path(hf_hub_download(
            repo_id=rec["repo_id"],
            filename=f"{rec['folder']}/{name}",
            repo_type="dataset",
            cache_dir=str(cache_root() / "hub"),
        ))
        files[suffix] = path
        key = (str(path), want)
        if key not in _VERIFIED:
            if _sha256(path) != want:
                raise ValueError(f"index checksum mismatch for {name}")
            _VERIFIED.add(key)
    base = files[".esmci"]
    _RESOLVED[str(index)] = base
    return base
