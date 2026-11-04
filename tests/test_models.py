"""M8 model/index resolution without network."""
import pytest

from esmc_retrieval.indexes import resolve_index
from esmc_retrieval.models import resolve_model


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
