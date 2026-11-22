"""Smoke test for the EC .esmci index + metadata contract."""
import numpy as np

from esmc_li.index import Esmci, write
from esmc_li.search import maxsim_topk_numpy


def test_ec_index_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((6, 8)).astype(np.float16)
    off = np.array([0, 2, 4, 6], dtype=np.uint64)
    p = write(tmp_path / "ec.esmci", vecs, off,
              {"ids": ["a", "b", "c"],
               "ec4": ["1.1.1.1", "1.1.1.1", "2.1.1.1"]})
    idx = Esmci(p)
    assert idx.ids == ["a", "b", "c"]
    assert idx.meta["ec4"][1] == "1.1.1.1"
    q = vecs[0:1].astype(np.float32)
    s, i = maxsim_topk_numpy(q, idx.vecs, idx.offsets, k=1)
    assert s.shape == (1,) and i.shape == (1,)
