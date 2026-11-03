"""M8 exit: numpy exact MaxSim matches the naive reference."""
import numpy as np

from esmc_retrieval.scoring import l2_normalize, naive_maxsim
from esmc_retrieval.search import maxsim_topk_numpy


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
