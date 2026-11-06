"""T7.2 (M7): coarse candidate stage -> exact MaxSim rerank.

Pipeline (plan §9, published trivial baseline — mean pooling, not an
invention): mean-pool each document's residue rows, L2-normalize, take the
top-C documents by exact cosine to the pooled query (= FAISS IndexFlatIP
semantics; computed with plain numpy GEMM — faiss and torch cannot share a
process here, and at 10⁶ x 128 exact IP is a single BLAS call either way),
then rerank the candidate set with the frozen exhaustive scorer.

search_coarse() returns the same (scores, doc_ids) contract as
search.maxsim_topk, restricted to candidates.
"""
from __future__ import annotations

import numpy as np


def pool_docs(vecs: np.ndarray, offsets: np.ndarray,
              chunk: int = 1_000_000) -> np.ndarray:
    """Mean-pooled, L2-normalized doc vectors [n_docs, d] f32.

    ``chunk`` is a row budget: docs are processed in contiguous blocks that
    stay at or below it, so a 1M-doc mmap index is never materialized at once.
    """
    off = np.asarray(offsets, dtype=np.int64)
    n = len(off) - 1
    d = vecs.shape[1]
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    out = np.empty((n, d), dtype=np.float32)
    d0 = 0
    while d0 < n:
        r0 = int(off[d0])
        d1 = d0 + 1
        while d1 < n and int(off[d1 + 1]) - r0 <= chunk:
            d1 += 1
        r1 = int(off[d1])
        blk = np.asarray(vecs[r0:r1], dtype=np.float32)
        lens = off[d0 + 1 : d1 + 1] - off[d0:d1]
        if np.any(lens <= 0):
            raise ValueError("empty protein rows")
        starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
        sums = np.add.reduceat(blk, starts.astype(np.int64), axis=0)
        out[d0:d1] = sums / lens[:, None]
        d0 = d1
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def pool_query(rows: np.ndarray) -> np.ndarray:
    v = np.asarray(rows, dtype=np.float32).mean(axis=0)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def top_c(q_pool: np.ndarray, d_pool: np.ndarray, C: int) -> np.ndarray:
    """[nq, C] candidate doc ids by exact pooled cosine (stable top-C)."""
    S = np.asarray(q_pool, dtype=np.float32) @ np.asarray(
        d_pool, dtype=np.float32).T
    C = min(C, d_pool.shape[0])
    if C >= d_pool.shape[0]:
        return np.argsort(-S, axis=1, kind="stable")
    part = np.argpartition(-S, C - 1, axis=1)[:, :C]
    rows = np.arange(S.shape[0])[:, None]
    order = np.argsort(-S[rows, part], axis=1, kind="stable")
    return part[rows, order]


def candidate_recall(cand: np.ndarray, exhaustive_top: np.ndarray,
                     self_ids: np.ndarray | None = None) -> float:
    """Fraction of the exhaustive top-k docs present in the candidate sets.

    `self_ids` is for leave-one-out: self_ids[i] is query i's own doc id and
    must be removed from both sets before counting.
    """
    hit = []
    for i in range(len(cand)):
        cs = set(cand[i].tolist())
        ts = set(exhaustive_top[i].tolist())
        if self_ids is not None:
            cs.discard(int(self_ids[i]))
            ts.discard(int(self_ids[i]))
        hit.append(len(cs & ts))
    return float(np.mean(hit) / exhaustive_top.shape[1])


def rerank(q_rows: list[np.ndarray], vecs: np.ndarray, offsets: np.ndarray,
           cand: np.ndarray, k: int, device: str = "mps",
           self_ids: np.ndarray | None = None, chunk_c: int = 30_000
           ) -> tuple[np.ndarray, np.ndarray]:
    """Exact MaxSim over each query's candidate docs (frozen scorer).

    Candidates are reranked in `chunk_c` blocks and the per-block top-k are
    merged. This keeps the f32 score buffer bounded when C is large (the 1M
    coarse sweep needs C>=30K; a single concatenated candidate buffer would
    exceed RAM). `device="metal"` uses the fused M5 scorer, so the coarse
    comparison is against the same exact reranker as exhaustive search.
    """
    from esmc_li.search import maxsim_topk

    metal = device == "metal"
    if metal:
        from esmc_li.metal_search import maxsim_topk_metal

    off = np.asarray(offsets, dtype=np.int64)
    assert chunk_c > 0
    assert cand.shape[1] >= k + int(self_ids is not None), \
        "C must be >= k (+ self)"
    out_s = np.empty((len(q_rows), k), dtype=np.float32)
    out_i = np.empty((len(q_rows), k), dtype=np.int64)
    for qi, q in enumerate(q_rows):
        ids_c = np.unique(cand[qi])
        if self_ids is not None:
            ids_c = ids_c[ids_c != int(self_ids[qi])]
            assert len(ids_c) >= k, "not enough candidates after dropping self"
        scores, ids = [], []
        for lo in range(0, len(ids_c), chunk_c):
            cids = ids_c[lo : lo + chunk_c]
            pieces = [np.asarray(vecs[off[j] : off[j + 1]]) for j in cids]
            sub_off = np.concatenate(([0], np.cumsum([len(p) for p in pieces])))
            sub = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
            if metal:
                s, idx = maxsim_topk_metal([q], sub, sub_off,
                                           min(k, len(cids)),
                                           normalize=False)
            else:
                s, idx = maxsim_topk([q], sub, sub_off, min(k, len(cids)),
                                     device=device, normalize=False)
            scores.append(s[0])
            ids.append(cids[idx[0]])
        all_s = np.concatenate(scores)
        all_i = np.concatenate(ids)
        order = np.argsort(-all_s, kind="stable")[:k]
        out_s[qi], out_i[qi] = all_s[order], all_i[order]
    return out_s, out_i


def search_coarse(q_rows: list[np.ndarray], vecs: np.ndarray,
                  offsets: np.ndarray, d_pool: np.ndarray, C: int, k: int,
                  device: str = "mps", self_ids: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    qp = np.stack([pool_query(q) for q in q_rows])
    cand = top_c(qp, d_pool, C)
    return rerank(q_rows, vecs, offsets, cand, k, device=device,
                  self_ids=self_ids)
