"""M8 model/index resolution without network."""
import pytest

from esmc_li.indexes import resolve_index
from esmc_li.models import DEFAULT_MODEL, MODELS, resolve_model


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
