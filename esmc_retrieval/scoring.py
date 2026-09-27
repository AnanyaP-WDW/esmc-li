"""Exact scorers for M1: pooled cosine and tiled MaxSim (§5).

MaxSim: S(A,B) = sum_i max_j (a_i . b_j) over L2-normalized residue vectors
(L2 after projection per §5; for the 960D no-projection case we normalize too,
so each term is a per-residue cosine).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / "cache" / "scope"


def load_embeddings(ids: list[str], cache: Path = CACHE) -> list[np.ndarray]:
    return [np.load(cache / f"{i}.npy").astype(np.float32) for i in ids]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def random_projection(seed: int = 0, d_in: int = 960, d_out: int = 128) -> np.ndarray:
    """Gaussian sketch, entries N(0, 1/d_in) (Kacwin)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((d_in, d_out)) / np.sqrt(float(d_in))).astype(np.float32)


def project(embs: list[np.ndarray], proj: np.ndarray | None) -> list[np.ndarray]:
    if proj is None:
        return [l2_normalize(e) for e in embs]
    return [l2_normalize(e @ proj) for e in embs]


def pooled_cosine_matrix(embs: list[np.ndarray]) -> np.ndarray:
    m = np.stack([l2_normalize(e.mean(axis=0)) for e in embs])
    return (m @ m.T).astype(np.float32)


def maxsim_matrix(embs: list[np.ndarray], device: str = "mps") -> np.ndarray:
    """S[i, j] = MaxSim(i, j) for all ordered pairs (exact, tiled by query).

    Per query: one GEMM [La, D] x [D, total_L], then a segment-wise max over
    each candidate's residue rows (scatter_reduce amax; segment_reduce isn't
    implemented on MPS) and a sum over query residues.
    """
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    C = torch.from_numpy(np.concatenate(embs, axis=0)).to(device)
    lens = [len(e) for e in embs]
    seg = torch.repeat_interleave(torch.tensor(lens, dtype=torch.int64)).to(device)
    n = len(embs)
    S = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        a = torch.from_numpy(embs[i]).to(device)
        sim = a @ C.T  # [La, total_L]
        La = sim.shape[0]
        out = torch.full((n, La), float("-inf"), device=device)
        out = out.scatter_reduce(0, seg.unsqueeze(1).expand(-1, La), sim.T,
                                 reduce="amax", include_self=True)
        S[i] = out.sum(dim=1).cpu().numpy()  # sum over query residues of segment max
        del a, sim, out
    return S


def naive_maxsim(a: np.ndarray, b: np.ndarray) -> float:
    """O(La*Lb*D) reference for the T1.3 exit check."""
    s = 0.0
    for ai in a:
        s += max(float(ai @ bj) for bj in b)
    return s


def rankings_from_scores(S: np.ndarray, ids: list[str]) -> dict[str, list[str]]:
    """Descending score, self excluded."""
    n = len(ids)
    out: dict[str, list[str]] = {}
    for i in range(n):
        order = np.argsort(-S[i], kind="stable")
        out[ids[i]] = [ids[j] for j in order if j != i]
    return out
