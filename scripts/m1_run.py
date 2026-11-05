#!/usr/bin/env python3
"""M1 scorers on scope_test: pooled ESM-C (T1.2), MaxSim 960D/128D (T1.3),
length-bias rho + per-normalization cR@k (T1.4).

Writes results/m1_scorers.csv, results/m1_length_bias.csv;
matrices cached in cache/m1/*.npy (reused by T1.5); logs to runs.jsonl.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from esmc_li.eval import evaluate, load_split
from esmc_li.runlog import log_run
from esmc_li.scoring import (
    load_embeddings,
    maxsim_matrix,
    pooled_cosine_matrix,
    project,
    random_projection,
    rankings_from_scores,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "splits" / "scope_test.tsv"
M1CACHE = ROOT / "cache" / "m1"
OUT_SCORERS = ROOT / "results" / "m1_scorers.csv"
OUT_BIAS = ROOT / "results" / "m1_length_bias.csv"
FIELDNAMES = ["split", "method", "n", "wall_s", "n_queries", "cR@1", "cR@5",
              "cR@10", "cR@100", "mrr", "sens_family", "sens_superfamily",
              "sens_fold"]


def get_matrix(name: str, fn) -> np.ndarray:
    M1CACHE.mkdir(exist_ok=True)
    p = M1CACHE / f"{name}.npy"
    if p.exists():
        return np.load(p)
    t = time.time()
    M = fn()
    np.save(p, M)
    print(f"{name}: computed in {time.time() - t:.1f}s -> {p}")
    return M


def spearman_len_score(S: np.ndarray, lens: np.ndarray, same_sf: np.ndarray) -> float:
    """rho(candidate length, score) over non-homolog (different-superfamily) pairs."""
    from scipy.stats import spearmanr

    mask = ~same_sf
    np.fill_diagonal(mask, False)
    qi, ci = mask.nonzero()
    rho, _ = spearmanr(lens[ci], S[qi, ci])
    return float(rho)


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        f.flush()


def main() -> int:
    t0 = time.time()
    rows = load_split(SPLIT)
    ids = [r["id"] for r in rows]
    n = len(rows)
    print(f"split scope_test n={n}")
    lens = np.array([len(r["sequence"]) for r in rows], dtype=np.int64)
    sf = np.array([r["superfamily"] for r in rows])
    same_sf = sf[:, None] == sf[None, :]
    np.fill_diagonal(same_sf, True)

    embs = load_embeddings(ids)
    assert all(e.shape[1] == 960 for e in embs)

    S_pooled = get_matrix("pooled", lambda: pooled_cosine_matrix(embs))
    S_960 = get_matrix("maxsim960", lambda: maxsim_matrix(project(embs, None)))
    proj = random_projection(seed=0)
    S_128 = get_matrix("maxsim128", lambda: maxsim_matrix(project(embs, proj)))

    scorers = {"pooled_esmc": S_pooled, "maxsim960": S_960, "maxsim128": S_128}

    # T1.2 / T1.3: raw-score rows
    for name, S in scorers.items():
        t = time.time()
        m = evaluate(rankings_from_scores(S, ids), rows)
        wall = time.time() - t
        row = {"split": "scope_test", "method": name, "n": n,
               "wall_s": round(wall, 2),
               **{k: round(v, 4) for k, v in m.items()}}
        append_row(OUT_SCORERS, row)
        print(row)
        log_run(f"T1.{'2' if name == 'pooled_esmc' else '3'}",
                {"split": "scope_test", "method": name,
                 "vector_norm": "l2", "proj": None if name != "maxsim128"
                  else "gaussian_960to128_seed0"},
                m, seed=0, wall_s=wall)

    # T1.4: length bias + normalizations
    with OUT_BIAS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scorer", "rho_len_score_nonhomolog", "normalization",
                    "cR@1", "cR@5", "cR@10", "cR@100", "mrr",
                    "sens_superfamily", "rankings_same_as_raw"])
        for name, S in scorers.items():
            rho = spearman_len_score(S, lens, same_sf)
            S_sym = 0.5 * (S + S.T)
            variants = {
                "raw": (S, True),
                "div_lenA": (S / lens[:, None], False),
                "sym_div_lenSum": (S_sym / (lens[:, None] + lens[None, :]), False),
                "sym_div_geomean": (S_sym / np.sqrt(lens[:, None] * lens[None, :]), False),
            }
            raw_rank = rankings_from_scores(S, ids)
            for vname, (Sn, is_raw) in variants.items():
                ranks = raw_rank if is_raw else rankings_from_scores(Sn, ids)
                m = evaluate(ranks, rows)
                same = ranks == raw_rank
                w.writerow([name, round(rho, 4), vname,
                            round(m["cR@1"], 4), round(m["cR@5"], 4),
                            round(m["cR@10"], 4), round(m["cR@100"], 4),
                            round(m["mrr"], 4), round(m["sens_superfamily"], 4),
                            same])
                print(name, vname, "cR@10", round(m["cR@10"], 4),
                      "rho", round(rho, 4))
                log_run("T1.4", {"split": "scope_test", "scorer": name,
                                 "normalization": vname,
                                 "rho": round(rho, 4)},
                        m, seed=0)
    print(f"total {time.time() - t0:.1f}s -> {OUT_SCORERS}, {OUT_BIAS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
