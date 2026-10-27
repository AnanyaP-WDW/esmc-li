"""T7.1 (M7): index compression for the D=128 trained rows.

Schemes (encode returns codes + aux; decode restores f32 rows that the
frozen search path L2-normalizes as usual):

  int8           per-dimension absmax scale, 1 B/val            (2x vs f16)
  residual_int8  ColBERTv2-style: coarse centroid + INT8 residual
                 (2x vs f16 — the negative-result candidate)
  residual_pq    coarse centroid (k=256) + PQ residual, m subquantizers
                 x 256 centroids, 1 B/subvector                (>=16x vs f16)

kmeans is plain seeded Lloyd (numpy): faiss and torch cannot share a
process (two libopms deadlock — M5 T5.4 lesson).
"""
from __future__ import annotations

import numpy as np

COARSE_K = 256
SUB_K = 256
ITERS = 20


def kmeans(X: np.ndarray, k: int = COARSE_K, iters: int = ITERS,
           seed: int = 0) -> np.ndarray:
    """(k, d) centroids; seeded random-init Lloyd with empty-cluster repair."""
    rng = np.random.default_rng(seed)
    C = np.asarray(X[rng.choice(len(X), k, replace=False)], dtype=np.float32)
    c2 = np.einsum("ij,ij->i", C, C)
    for _ in range(iters):
        assign = np.empty(len(X), dtype=np.int64)
        for s in range(0, len(X), 250_000):
            blk = np.asarray(X[s : s + 250_000], dtype=np.float32)
            d2 = (np.einsum("ij,ij->i", blk, blk)[:, None] - 2 * blk @ C.T
                  + c2[None, :])
            assign[s : s + len(blk)] = d2.argmin(axis=1)
        empty = np.setdiff1d(np.arange(k), np.unique(assign))
        if len(empty):
            # repair: steal the points farthest from their own centroid
            mind = np.empty(len(X), dtype=np.float32)
            for s in range(0, len(X), 250_000):
                blk = np.asarray(X[s : s + 250_000], dtype=np.float32)
                mind[s : s + len(blk)] = _sqdist(blk, C).min(axis=1)
            for i, e in zip(np.argsort(-mind)[: len(empty)], empty):
                C[e] = X[i]
                assign[i] = e
        newC = np.zeros_like(C)
        np.add.at(newC, assign, np.asarray(X, dtype=np.float32))
        counts = np.bincount(assign, minlength=k).clip(min=1)
        newC /= counts[:, None]
        newC[counts == 0] = C[counts == 0]
        if np.allclose(newC, C, atol=1e-4):
            C = newC
            break
        C = newC
    return C


def _sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    blk = np.asarray(X, dtype=np.float32)
    return (np.einsum("ij,ij->i", blk, blk)[:, None] - 2 * blk @ C.T
            + np.einsum("ij,ij->i", C, C)[None, :])


def _absmax(X: np.ndarray) -> np.ndarray:
    out = None
    for s in range(0, len(X), 1_000_000):
        blk = np.abs(np.asarray(X[s : s + 1_000_000], dtype=np.float32))
        out = blk.max(axis=0) if out is None else np.maximum(out,
                                                             blk.max(axis=0))
    return out


def encode_int8(X: np.ndarray, absmax: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Codes via a per-dim pass — X may be a memmap; never fully copied."""
    if absmax is None:
        absmax = _absmax(X)
    scale = np.maximum(absmax, 1e-6) / 127.0
    codes = np.empty(X.shape, dtype=np.int8)
    for s in range(0, len(X), 1_000_000):
        blk = np.asarray(X[s : s + 1_000_000], dtype=np.float32)
        codes[s : s + len(blk)] = np.clip(
            np.round(blk / scale), -127, 127).astype(np.int8)
    return codes, scale


def decode_int8(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) * scale


def _mean(X: np.ndarray) -> np.ndarray:
    acc = np.zeros(X.shape[1], dtype=np.float64)
    for s in range(0, len(X), 1_000_000):
        acc += np.asarray(X[s : s + 1_000_000], dtype=np.float32).sum(
            axis=0, dtype=np.float64)
    return (acc / len(X)).astype(np.float32)


def encode_res_int8(X: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray]:
    mu = _mean(X)
    absmax = None
    for s in range(0, len(X), 1_000_000):
        blk = np.abs(np.asarray(X[s : s + 1_000_000], dtype=np.float32)
                     - mu)
        absmax = blk.max(axis=0) if absmax is None else np.maximum(
            absmax, blk.max(axis=0))
    scale = np.maximum(absmax, 1e-6) / 127.0
    codes = np.empty(X.shape, dtype=np.int8)
    for s in range(0, len(X), 1_000_000):
        blk = np.asarray(X[s : s + 1_000_000], dtype=np.float32) - mu
        codes[s : s + len(blk)] = np.clip(
            np.round(blk / scale), -127, 127).astype(np.int8)
    return codes, mu, scale


def decode_res_int8(codes: np.ndarray, mu: np.ndarray,
                    scale: np.ndarray) -> np.ndarray:
    return decode_int8(codes, scale) + mu


def encode_res_pq(X: np.ndarray, m: int, seed: int = 0,
                  coarse_train: int = 250_000,
                  sub_train: int = 250_000) -> dict:
    """ColBERTv2-style residual PQ: coarse centroid id + PQ of the residual.

    Returns a dict of codes/books; decode_res_pq() consumes it.
    """
    d = X.shape[1]
    assert d % m == 0, "D must be divisible by m"
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(coarse_train, len(X)), replace=False)
    sample = np.asarray(X[idx], dtype=np.float32)
    C = kmeans(sample, k=COARSE_K, seed=seed)
    assign = np.empty(len(X), dtype=np.uint8)
    for s in range(0, len(X), 250_000):
        blk = np.asarray(X[s : s + 250_000], dtype=np.float32)
        assign[s : s + len(blk)] = _sqdist(blk, C).argmin(axis=1).astype(
            np.uint8)
    sub = sample - C[assign[idx]]          # residual of the training sample
    del sample
    dm = d // m
    books = np.empty((m, SUB_K, dm), dtype=np.float32)
    for j in range(m):
        books[j] = kmeans(sub[:, j * dm : (j + 1) * dm], k=SUB_K,
                          seed=seed + j)
    del sub
    codes = np.empty((len(X), m), dtype=np.uint8)
    for s in range(0, len(X), 250_000):
        blk = np.asarray(X[s : s + 250_000], dtype=np.float32) - C[
            assign[s : s + 250_000]]
        for j in range(m):
            rj = blk[:, j * dm : (j + 1) * dm]
            codes[s : s + len(blk), j] = _sqdist(rj, books[j]).argmin(
                axis=1).astype(np.uint8)
    return {"coarse": assign, "sub": codes, "C": C, "books": books, "m": m}


def decode_res_pq(q: dict) -> np.ndarray:
    m, books, dm = q["m"], q["books"], q["C"].shape[1] // q["m"]
    out = q["C"][q["coarse"].astype(np.int64)].astype(np.float32)
    for j in range(m):
        out[:, j * dm : (j + 1) * dm] += books[j][q["sub"][:, j]]
    return out


def scheme_bytes(scheme: str, n: int, d: int, aux=None) -> int:
    """Stored size in bytes (codes + codebooks)."""
    if scheme == "int8":
        return n * d
    if scheme == "residual_int8":
        return n * d + aux  # + mu, scales (negligible, passed in)
    if scheme.startswith("residual_pq"):
        m = int(scheme.rsplit("m", 1)[1])
        return n * (1 + m) + COARSE_K * d * 2 + m * SUB_K * (d // m) * 2
    raise ValueError(scheme)
