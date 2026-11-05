"""T7.2 exits: coarse candidates + rerank consistency."""
import numpy as np
import pytest

pytest.importorskip("torch")

from esmc_li.coarse import (candidate_recall, pool_docs, pool_query,
                                   rerank, top_c)


def make_index(n=50, d=8, lens=None, seed=0):
    rng = np.random.default_rng(seed)
    lens = lens or rng.integers(5, 20, size=n)
    off = np.concatenate(([0], np.cumsum(lens)))
    x = rng.standard_normal((int(off[-1]), d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float16), off.astype(np.int64)


def test_pool_shapes():
    x, off = make_index()
    p = pool_docs(x, off)
    assert p.shape == (50, 8)
    np.testing.assert_allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-5)
    assert pool_query(x[off[0] : off[1]]).shape == (8,)


def test_top_c_is_exact_pooled_topk():
    x, off = make_index()
    p = pool_docs(x, off)
    q = [p[3], p[7]]
    cand = top_c(np.stack(q), p, 5)
    S = np.stack(q) @ p.T
    for i in range(2):
        expect = np.argsort(-S[i], kind="stable")[:5]
        np.testing.assert_array_equal(cand[i], expect)


def test_candidate_recall_perfect_when_c_full():
    x, off = make_index()
    p = pool_docs(x, off)
    cand = top_c(p[:4], p, 50)
    top = np.argsort(-(p[:4] @ p.T), axis=1)[:, :10]
    assert candidate_recall(cand, top) == 1.0
    cand5 = top_c(p[:4], p, 5)
    assert candidate_recall(cand5, top) == 0.5  # top-5 of top-10


def test_rerank_matches_direct_maxsim():
    from esmc_li.search import maxsim_topk

    x, off = make_index(n=20, d=8, seed=1)
    p = pool_docs(x, off)
    q = [np.asarray(x[off[4] : off[5]])]
    cand = top_c(p[:1], p, 10)
    s1, i1 = rerank(q, x, off, cand, k=5, device="cpu")
    ids_c = np.unique(cand[0])
    pieces = [np.asarray(x[off[j] : off[j + 1]]) for j in ids_c]
    sub_off = np.concatenate(([0], np.cumsum([len(p) for p in pieces])))
    sub = np.concatenate(pieces)
    s2, i2 = maxsim_topk(q, sub, sub_off, 5, device="cpu",
                         normalize=False)
    np.testing.assert_allclose(s1[0], s2[0], rtol=1e-5)
    np.testing.assert_array_equal(i1[0], ids_c[i2[0]])


def test_rerank_chunk_merge_matches_full():
    x, off = make_index(n=20, d=8, seed=1)
    p = pool_docs(x, off)
    q = [np.asarray(x[off[4] : off[5]])]
    cand = top_c(p[:1], p, 10)
    s1, i1 = rerank(q, x, off, cand, k=5, device="cpu", chunk_c=3)
    s2, i2 = rerank(q, x, off, cand, k=5, device="cpu", chunk_c=100)
    np.testing.assert_allclose(s1, s2, rtol=1e-5)
    np.testing.assert_array_equal(i1, i2)


def test_candidate_recall_excludes_self():
    cand = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    top = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3]])
    # q0: {1,2}∩{1,2,3}=2; q1: {2,3}∩{0,2,3}=2; q2: {3,4}∩{0,1,3}=1
    assert np.isclose(candidate_recall(cand, top, self_ids=np.arange(3)),
                      (2 + 2 + 1) / 3 / 3)


def test_rerank_excludes_self():
    from esmc_li.search import maxsim_topk

    x, off = make_index(n=20, d=8, seed=1)
    p = pool_docs(x, off)
    q = [np.asarray(x[off[i] : off[i + 1]]) for i in range(5)]
    cand = top_c(p[:5], p, 10)
    assert 4 in cand[4]
    s1, i1 = rerank(q, x, off, cand, k=5, device="cpu",
                    self_ids=np.arange(5))
    assert 4 not in i1[4]
    ids_c = np.unique(cand[4])
    ids_c = ids_c[ids_c != 4]
    pieces = [np.asarray(x[off[j] : off[j + 1]]) for j in ids_c]
    sub_off = np.concatenate(([0], np.cumsum([len(p) for p in pieces])))
    sub = np.concatenate(pieces)
    s2, i2 = maxsim_topk([q[4]], sub, sub_off, 5, device="cpu",
                         normalize=False)
    np.testing.assert_allclose(s1[4], s2[0], rtol=1e-5)
    np.testing.assert_array_equal(i1[4], ids_c[i2[0]])
