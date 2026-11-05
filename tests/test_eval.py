import math

from esmc_li.eval import evaluate, expected_random_cr, random_ranking

ROWS = [
    {"id": "q1", "sequence": "MKTA", "family": "A1", "superfamily": "A", "fold": "F1"},
    {"id": "a2", "sequence": "MKTB", "family": "A1", "superfamily": "A", "fold": "F1"},
    {"id": "a3", "sequence": "MKTC", "family": "A2", "superfamily": "A", "fold": "F1"},
    {"id": "b1", "sequence": "MKTD", "family": "B1", "superfamily": "B", "fold": "F2"},
    {"id": "b2", "sequence": "MKTE", "family": "B1", "superfamily": "B", "fold": "F2"},
]

# hand-computed expectations (see module docstring of eval for the protocol)
RANKINGS = {
    "q1": ["a2", "b1", "a3", "b2"],   # cR@k: hits 1,2,2,2 / min(k,2)
    "a2": ["b1", "a3", "q1", "b2"],   # cR@1: 0; first relevant at rank 2
}


def test_toy_capped_recall_and_mrr():
    m = evaluate(RANKINGS, ROWS, ks=(1, 5, 10, 100))
    assert m["n_queries"] == 2
    assert math.isclose(m["cR@1"], 0.5)      # (1.0 + 0.0) / 2
    assert math.isclose(m["cR@5"], 1.0)      # both find both relevant by k=5
    assert math.isclose(m["cR@10"], 1.0)
    assert math.isclose(m["cR@100"], 1.0)
    assert math.isclose(m["mrr"], 0.75)      # (1/1 + 1/2) / 2


def test_toy_sensitivity_to_first_fp():
    m = evaluate(RANKINGS, ROWS, ks=(1, 5, 10, 100))
    # q1: sf [a2|b1...] tp=1/2 ; fam A1 [a2|b1...] tp=1/1 ; fold [a2|b1...] 1/2
    # a2: sf [b1|...] tp=0/2  ; fam A1 [b1|...] tp=0/1  ; fold [b1|...] 0/2
    assert math.isclose(m["sens_superfamily"], 0.25)
    assert math.isclose(m["sens_family"], 0.5)
    assert math.isclose(m["sens_fold"], 0.25)


def _synthetic_rows(n_groups: int = 70, seed: int = 7) -> list[dict]:
    import random

    rng = random.Random(seed)
    rows = []
    for g in range(n_groups):
        size = 2 + (g % 9)
        for j in range(size):
            rows.append({
                "id": f"g{g}_{j}",
                "sequence": "MKTA",
                "family": f"g{g}" if j < size - 1 else f"g{g}_x",
                "superfamily": f"g{g}",
                "fold": f"f{g % 5}",
            })
    rng.shuffle(rows)
    return rows


def test_random_ranking_within_3sigma_of_analytic():
    rows = _synthetic_rows()
    n = len(rows)
    assert n >= 300
    for k in (1, 10, 100):
        mu, sigma = expected_random_cr(rows, k)
        assert sigma > 0
        rank = random_ranking(rows, seed=k)
        m = evaluate(rank, rows, ks=(k,))
        obs = m[f"cR@{k}"]
        assert abs(obs - mu) <= 3 * sigma, (
            f"cR@{k}: observed {obs:.4f} vs analytic {mu:.4f} +/- {sigma:.4f}"
        )
