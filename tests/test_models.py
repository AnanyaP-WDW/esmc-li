"""M8 model/index resolution without network."""
import hashlib

import pytest

from esmc_li import indexes
from esmc_li.indexes import resolve_index
from esmc_li.models import (DEFAULT_MODEL, MODELS, is_registry_model,
                            resolve_model)


def test_model_rename_alias():
    assert DEFAULT_MODEL == "esmc-li-300m-homology-v1"
    for key in ("esmc-li-300m-v1", "AnanyaPathak/esmc-li-300m-v1",
                "esmc-300m-homology-maxsim-v1",
                "AnanyaPathak/esmc-300m-homology-maxsim-v1"):
        assert MODELS[key]["repo_id"] == "AnanyaPathak/esmc-li-300m-homology-v1"


def test_resolve_local_model(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x")
    assert resolve_model(p) == p
    assert resolve_model(str(p)) == p
    assert resolve_model(None) is None


def test_resolve_missing_model():
    with pytest.raises(FileNotFoundError):
        resolve_model("not-a-real-model")


def test_resolve_local_index(tmp_path):
    p = tmp_path / "x.esmci"
    p.write_bytes(b"x")
    assert resolve_index(p) == p


def test_registry_model_allowlist():
    assert is_registry_model(DEFAULT_MODEL)
    assert is_registry_model(MODELS[DEFAULT_MODEL]["filename"])
    assert not is_registry_model("/tmp/evil.gguf")
    assert not is_registry_model(None)


def test_resolve_index_uses_pinned_hashes(tmp_path, monkeypatch):
    files = {}
    for suffix in (".esmci", ".esmci.json", ".esmci.head.npy"):
        p = tmp_path / f"idx{suffix}"
        p.write_bytes(suffix.encode())
        files[suffix] = p
    rec = {
        "repo_id": "test/repo",
        "folder": "folder",
        "name": "idx",
        "sha256": {s: hashlib.sha256(p.read_bytes()).hexdigest()
                   for s, p in files.items()},
    }
    monkeypatch.setitem(indexes.INDEXES, "test-index", rec)
    monkeypatch.setattr(indexes, "_VERIFIED", set())
    monkeypatch.setattr(indexes, "_RESOLVED", {})

    def fake_download(repo_id, filename, **kwargs):
        suffix = next(s for s in files if filename.endswith(s))
        return str(files[suffix])

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert resolve_index("test-index") == files[".esmci"]

    files[".esmci"].write_bytes(b"tampered")
    indexes._VERIFIED.clear()
    indexes._RESOLVED.clear()
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_index("test-index")
