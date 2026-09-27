"""T1.3 exit: 50 random pair scores match the naive double loop to 1e-5."""
from __future__ import annotations

import numpy as np

from esmc_retrieval.scoring import (
    l2_normalize,
    maxsim_matrix,
    naive_maxsim,
    project,
    random_projection,
)


def _pair_scores(embs, n_pairs=50, seed=0):
    rng = np.random.default_rng(seed)
    n = len(embs)
    pairs = [(int(rng.integers(n)), int(rng.integers(n))) for _ in range(n_pairs)]
    return pairs


def test_maxsim_960_matches_naive():
    rng = np.random.default_rng(0)
    embs = [l2_normalize(rng.standard_normal((int(rng.integers(20, 80)), 960)).astype(np.float32))
            for _ in range(12)]
    S = maxsim_matrix(embs, device="cpu")
    for i, j in _pair_scores(embs):
        want = naive_maxsim(embs[i], embs[j])
        assert abs(S[i, j] - want) < 1e-5, (i, j, S[i, j], want)


def test_maxsim_128_matches_naive():
    rng = np.random.default_rng(1)
    embs = [rng.standard_normal((int(rng.integers(20, 80)), 960)).astype(np.float32)
            for _ in range(12)]
    proj = random_projection(seed=0)
    embs = project(embs, proj)
    S = maxsim_matrix(embs, device="cpu")
    for i, j in _pair_scores(embs):
        want = naive_maxsim(embs[i], embs[j])
        assert abs(S[i, j] - want) < 1e-5, (i, j, S[i, j], want)
