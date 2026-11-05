#!/usr/bin/env python3
"""Verify paper/results.lock.json against the pinned results commit."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "paper" / "results.lock.json"


def main() -> int:
    lock = json.loads(LOCK.read_text())
    commit = lock["results_commit"]
    raw = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:results/runs.jsonl"],
        capture_output=True, check=True).stdout
    if hashlib.sha256(raw).hexdigest() != lock["runs_sha256"]:
        print("lock: runs.jsonl sha256 mismatch", file=sys.stderr)
        return 1
    lines = raw.decode().splitlines()
    for rid, r in lock["results"].items():
        i = r["line"] - 1
        if i < 0 or i >= len(lines):
            print(f"lock: {rid} line out of range", file=sys.stderr)
            return 1
        if hashlib.sha256(lines[i].encode()).hexdigest() != r["line_sha256"]:
            print(f"lock: {rid} line hash mismatch", file=sys.stderr)
            return 1
        row = json.loads(lines[i])
        if row.get("task") != r["task"] or row.get("ts") != r["ts"]:
            print(f"lock: {rid} task/ts mismatch", file=sys.stderr)
            return 1
        if row.get("metrics") != r["metrics"]:
            print(f"lock: {rid} metrics mismatch", file=sys.stderr)
            return 1
    rows = [json.loads(x) for x in lines if x.strip()]
    t03 = next(r for r in rows if r["task"] == "T0.3")
    t04 = next(r for r in rows if r["task"] == "T0.4")
    scope = lock["facts"]["scope"]["values"]
    for k, v in {"domains": t03["metrics"]["astral_40_domains"],
                 "train": t04["metrics"]["scope_train"],
                 "val": t04["metrics"]["scope_val"],
                 "test": t04["metrics"]["scope_test"]}.items():
        if scope[k] != v:
            print(f"lock: scope fact {k} mismatch", file=sys.stderr)
            return 1
    fl = json.loads((ROOT / "data" / "fulllength" /
                     "manifest.json").read_text())
    facts = lock["facts"]["full_length"]["values"]
    mapping = {"n_db": "db_size", "n_queries": "n_queries",
               "query_clans": "query_clans", "n_gt_2046": "n_gt_2046",
               "max_length": "max_length", "clan_overlap": "clan_overlap"}
    for k, src in mapping.items():
        if facts[k] != fl[src]:
            print(f"lock: full-length fact {k} mismatch", file=sys.stderr)
            return 1
    print(f"lock: {len(lock['results'])} results and facts verified at "
          f"{commit[:12]}; runs sha256 ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
