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
    print(f"lock: {len(lock['results'])} results verified at "
          f"{commit[:12]}; runs sha256 ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
