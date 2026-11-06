"""M8 exit: numpy exact MaxSim matches the naive reference."""
import numpy as np

from esmc_li.protein_index import ProteinIndex
from esmc_li.scoring import l2_normalize, naive_maxsim
from esmc_li.search import maxsim_topk_numpy


def test_numpy_maxsim_matches_naive():
    rng = np.random.default_rng(0)
    lens = rng.integers(3, 12, size=17)
    off = np.concatenate(([0], np.cumsum(lens)))
    db = l2_normalize(rng.standard_normal((int(off[-1]), 8))).astype(
        np.float16)
    q = l2_normalize(rng.standard_normal((6, 8))).astype(np.float32)
    got_s, got_i = maxsim_topk_numpy(q, db, off, k=5, doc_chunk=3)
    want = np.array([naive_maxsim(q, db[off[i]:off[i + 1]].astype(np.float32))
                     for i in range(len(lens))])
    order = np.argsort(-want, kind="stable")[:5]
    np.testing.assert_allclose(got_s, want[order], rtol=1e-5)
    np.testing.assert_array_equal(got_i, order)


def test_auto_falls_back_when_metal_init_fails(monkeypatch):
    import esmc_li.metal_search as ms

    class FakeIdx:
        vecs = l2_normalize(np.eye(4, dtype=np.float32)).astype(np.float16)
        offsets = np.array([0, 2, 4], dtype=np.uint64)

    idx = object.__new__(ProteinIndex)
    idx.device = "auto"
    idx._idx = FakeIdx()

    def boom(*args, **kwargs):
        raise ms.MetalInitError("no device")

    monkeypatch.setattr(ms, "maxsim_topk_metal", boom)
    scores, rows = idx._search([np.eye(4, dtype=np.float32)[:2]], k=1)
    assert scores.shape == (1, 1)
    assert rows.shape == (1, 1)
    assert idx.device == "numpy"
