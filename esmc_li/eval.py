"""Frozen evaluation harness (M0). All reported numbers come from here.

Protocol (matches ProtoCol, arXiv:2605.29158):
  corpus = the split itself; queries = the same rows (leave-one-out).
  relevant(q) = rows sharing q's group (superfamily for the headline metric),
  excluding q itself.

Metrics
  cR@k     capped recall: hits@k / min(k, R)   (ProtoCol eq. 4)
  mrr      1 / rank of first relevant
  sens     sensitivity up to the first false positive: fraction of the R
           relevant rows retrieved before the first non-relevant hit
           (reported at family / superfamily / fold level)
"""
from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

KS = (1, 5, 10, 100)


def load_split(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def evaluate(
    rankings: dict[str, list[str]],
    rows: list[dict],
    ks: tuple[int, ...] = KS,
    levels: tuple[str, ...] = ("family", "superfamily", "fold"),
) -> dict[str, float]:
    """rankings: query id -> ordered candidate ids (self must be excluded)."""
    by_id = {r["id"]: r for r in rows}
    groups: dict[str, dict[str, set]] = {
        lv: defaultdict(set) for lv in levels
    }
    for r in rows:
        for lv in levels:
            g = r.get(lv) or ""
            if g:
                groups[lv][g].add(r["id"])
    group_count = {
        lv: Counter(r.get(lv) or "" for r in rows if (r.get(lv) or ""))
        for lv in levels
    }

    cr = {k: [] for k in ks}
    rr = []
    sens = {lv: [] for lv in levels}

    for qid, ranking in rankings.items():
        q = by_id[qid]
        for lv in levels:
            g = q.get(lv) or ""
            if not g:
                continue
            R = group_count[lv][g] - 1
            if R <= 0:
                continue
            tp = 0
            for cid in ranking:
                c = by_id.get(cid)
                if c is None:
                    continue
                if (c.get(lv) or "") != g:
                    break
                tp += 1
            sens[lv].append(tp / R)

        g = q["superfamily"]
        R = group_count["superfamily"][g] - 1
        if R <= 0:
            continue
        first = None
        hits = {k: 0 for k in ks}
        max_k = max(ks)
        for i, cid in enumerate(ranking, start=1):
            c = by_id.get(cid)
            if c is None:
                continue
            if c["superfamily"] == g:
                if first is None:
                    first = i
                if i <= max_k:
                    for k in ks:
                        if i <= k:
                            hits[k] += 1
            if first is not None and i > max_k:
                break
        rr.append(1.0 / first if first else 0.0)
        for k in ks:
            cr[k].append(hits[k] / min(k, R))

    out: dict[str, float] = {"n_queries": float(len(rr))}
    for k in ks:
        out[f"cR@{k}"] = sum(cr[k]) / len(cr[k]) if cr[k] else 0.0
    out["mrr"] = sum(rr) / len(rr) if rr else 0.0
    for lv in levels:
        out[f"sens_{lv}"] = sum(sens[lv]) / len(sens[lv]) if sens[lv] else 0.0
    return out


def random_ranking(
    rows: list[dict],
    exclude_self: bool = True,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Independent random permutation per query (analytically tractable)."""
    ids = [r["id"] for r in rows]
    rng = random.Random(seed)
    out = {}
    for i, qid in enumerate(ids):
        cand = ids[:i] + ids[i + 1:] if exclude_self else ids[:]
        rng.shuffle(cand)
        out[qid] = cand
    return out


def expected_random_cr(
    rows: list[dict],
    k: int,
    level: str = "superfamily",
) -> tuple[float, float]:
    """Exact (mean, sigma) of mean cR@k for `random_ranking`.

    Each query's permutation is independent; hits ~ Hypergeom(N=n-1, K=R, draws)
    with draws = min(k, N), so E and Var of the mean follow exactly.
    """
    counts = Counter(r[level] for r in rows if r.get(level))
    n = len(rows)
    N = n - 1
    means, vars_ = [], []
    for r in rows:
        g = r.get(level)
        if not g:
            continue
        R = counts[g] - 1
        if R <= 0:
            continue
        draws = min(k, N)
        p = R / N
        mean_h = draws * p
        var_h = draws * p * (1 - p) * ((N - draws) / (N - 1)) if N > 1 else 0.0
        d = min(k, R)
        means.append(mean_h / d)
        vars_.append(var_h / (d * d))
    m = len(means)
    mu = sum(means) / m
    sigma = math.sqrt(sum(vars_)) / m  # independent queries
    return mu, sigma
