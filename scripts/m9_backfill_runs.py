#!/usr/bin/env python3
"""M9.0: backfill per-run rows that M2/M3/T5.4 only logged as aggregate CSV paths.

Each appended row is explicitly marked ``derived: true`` with the source CSV
and its sha256. The values are copied mechanically from committed artifacts;
``source_ts`` and ``source_git_sha`` preserve the original aggregate run's
provenance. Nothing is invented.

Idempotent: re-running skips rows whose ``derived_id`` is already present.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs.jsonl"
HARDWARE = "Darwin arm64 arm"

# task, csv, aggregate task/ts/git_sha, id column, config/metrics mapping
JOBS = [
    {
        "task": "T2.4",
        "csv": "results/m2_sweep.csv",
        "source_task": "T2.4",
        "source_ts": "2026-09-27T17:36:00+0530",
        "source_git_sha": "c29f9fc84da9c39b9790abf2f2d7874a5ad078f7",
        "config": ["tau", "masked", "seed", "d"],
        "metrics": ["val_cR@10", "frozen_val", "final_train_loss"],
    },
    {
        "task": "T2.5",
        "csv": "results/m2_dims.csv",
        "source_task": "T2.5",
        "source_ts": "2026-09-27T18:55:06+0530",
        "source_git_sha": "c29f9fc84da9c39b9790abf2f2d7874a5ad078f7",
        "config": ["d", "masked", "tau", "seed"],
        "metrics": ["val_cR@10", "frozen_val", "B_per_res_fp32",
                    "B_per_res_fp16", "B_per_res_int8"],
    },
    {
        "task": "T3.2",
        "csv": "results/m3_sweep.csv",
        "source_task": "T3.2",
        "source_ts": "2026-09-27T21:37:25+0530",
        "source_git_sha": "2ff1a621435a7602af284330e057ccf4b072c18c",
        "config": ["k", "seed", "d", "tau"],
        "metrics": ["val_cR@10", "final_train_loss"],
    },
    {
        "task": "T5.4.row",
        "csv": "results/m5_baselines.csv",
        "source_task": "T5.4",
        "source_ts": "2026-10-04T23:09:14+0530",
        "source_git_sha": "d573009760021917c2aab3aa93fc42265def84a9",
        "config": ["method", "n_db", "n_queries"],
        "metrics": ["build_s", "index_mb", "search_s", "qps"],
    },
]


def scalar(s: str):
    s = s.strip()
    if s in ("True", "False"):
        return s == "True"
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_existing() -> set[str]:
    out = set()
    if not RUNS.exists():
        return out
    for line in RUNS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("derived_id"):
            out.add(r["derived_id"])
    return out


def main() -> int:
    existing = load_existing()
    appended = 0
    for job in JOBS:
        path = ROOT / job["csv"]
        digest = sha256(path)
        rows = list(csv.DictReader(path.open(newline="")))
        for i, row in enumerate(rows):
            derived_id = f"{job['csv']}:{i + 1}"
            if derived_id in existing:
                continue
            config = {k: scalar(row[k]) for k in job["config"]}
            config.update({
                "split": "scope_val" if job["task"] in ("T2.4", "T2.5", "T3.2")
                         else "scope_test",
                "source_task": job["source_task"],
                "source_csv": job["csv"],
            })
            metrics = {k: scalar(row[k]) for k in job["metrics"]}
            entry = {
                "task": job["task"],
                "config": config,
                "seed": int(row["seed"]) if "seed" in row else 0,
                "metrics": metrics,
                "git_sha": job["source_git_sha"],
                "wall_s": scalar(row["wall_s"]) if "wall_s" in row else None,
                "hardware": HARDWARE,
                "model_hash": None,
                "ts": job["source_ts"],
                "derived": True,
                "derived_id": derived_id,
                "source_csv": job["csv"],
                "source_csv_sha256": digest,
            }
            with RUNS.open("a") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
            appended += 1
    print(f"appended {appended} derived rows -> {RUNS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
