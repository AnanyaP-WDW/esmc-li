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
    },
    "swissprot-100k": {
        "repo_id": "AnanyaPathak/esmc-li-indexes",
        "folder": "swissprot-100k",
        "name": "esmc-300m-homology-maxsim-v1-swissprot-100k",
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
        files[suffix] = Path(hf_hub_download(
            repo_id=rec["repo_id"],
            filename=f"{rec['folder']}/{rec['name']}{suffix}",
            repo_type="dataset",
            cache_dir=str(cache_root() / "hub"),
        ))
    sums_path = Path(hf_hub_download(
        repo_id=rec["repo_id"], filename=f"{rec['folder']}/SHA256SUMS",
        repo_type="dataset", cache_dir=str(cache_root() / "hub")))
    expected = {}
    for line in sums_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            expected[parts[1]] = parts[0]
    for suffix, path in files.items():
        name = f"{rec['name']}{suffix}"
        want = expected.get(name)
        if want is None:
            raise ValueError(f"index checksum missing for {name}")
        key = (str(path), want)
        if key not in _VERIFIED:
            got = _sha256(path)
            if got != want:
                raise ValueError(f"index checksum mismatch for {name}")
            _VERIFIED.add(key)
    base = files[".esmci"]
    _RESOLVED[str(index)] = base
    return base
