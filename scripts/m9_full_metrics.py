#!/usr/bin/env python3
"""M9.0: recover full test metrics from the cached M2/M3 score matrices.

The M2/M3 training scripts only logged cR@10. The matrices are still in
cache/m2/test_s*_d128.npy and cache/m3/eval/test_k3_s*.npy; re-evaluate them
with the frozen harness and append one T2.full / T3.full run row per seed.

This is evaluation-only: it does not retrain, and it asserts cR@10 against the
committed final CSVs before logging.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from esmc_retrieval.eval import evaluate, load_split  # noqa: E402
from esmc_retrieval.runlog import log_run  # noqa: E402
from esmc_retrieval.scoring import rankings_from_scores  # noqa: E402

SPLIT = ROOT / "data" / "splits" / "scope_test.tsv"
M2_FINAL = ROOT / "results" / "m2_final.csv"
M3_FINAL = ROOT / "results" / "m3_final.csv"
M2_OUT = ROOT / "results" / "m2_full_metrics.csv"
M3_OUT = ROOT / "results" / "m3_full_metrics.csv"
RUNS = ROOT / "results" / "runs.jsonl"

FIELDS = ["model", "seed", "n_queries", "cR@1", "cR@5", "cR@10", "cR@100",
          "mrr", "sens_family", "sens_superfamily", "sens_fold"]


def existing(task: str, seed: int) -> bool:
    if not RUNS.exists():
        return False
    for line in RUNS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task") == task and r.get("seed") == seed:
            return True
    return False


def final_cr10(path: Path) -> dict[int, float]:
    with path.open(newline="") as f:
        return {int(r["seed"]): float(r["test_cR@10"]) for r in csv.DictReader(f)}


def run(model: str, task: str, pattern: str, final: Path, out: Path) -> None:
    rows = load_split(SPLIT)
    ids = [r["id"] for r in rows]
    expected = final_cr10(final)
    written = []
    for seed in (0, 1, 2):
        p = ROOT / pattern.format(seed=seed)
        if not p.exists():
            raise FileNotFoundError(p)
        S = np.load(p)
        if S.shape != (len(rows), len(rows)):
            raise ValueError(f"{p}: shape {S.shape} != {(len(rows), len(rows))}")
        m = evaluate(rankings_from_scores(S, ids), rows)
        got = m["cR@10"]
        if abs(got - expected[seed]) > 5e-5:  # final CSVs are rounded to 4 dp
            raise AssertionError(
                f"{model} seed {seed}: recovered cR@10 {got:.8f} != "
                f"m2/m3_final {expected[seed]:.8f}")
        rec = {"model": model, "seed": seed,
               "n_queries": int(m["n_queries"]), **{
                   k: round(float(m[k]), 6) for k in FIELDS[3:]}}
        written.append(rec)
        if existing(task, seed):
            print(f"[skip] {task} seed {seed} already logged")
        else:
            log_run(task, {"model": model, "split": "scope_test",
                           "matrix": str(p.relative_to(ROOT)),
                           "source": "cached M2/M3 test score matrix"},
                    {k: rec[k] for k in FIELDS[3:]}, seed=seed)
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(written)
    print(f"wrote {out.relative_to(ROOT)} ({len(written)} rows)")


def main() -> int:
    run("m2_head_d128", "T2.full", "cache/m2/test_s{seed}_d128.npy",
        M2_FINAL, M2_OUT)
    run("m3_ft_k3_d128", "T3.full", "cache/m3/eval/test_k3_s{seed}.npy",
        M3_FINAL, M3_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
