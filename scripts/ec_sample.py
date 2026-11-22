#!/usr/bin/env python3
"""Deterministic stratified subset for EC head-only training.

One protein per EC4 class, then random fill.  The full split remains the
evaluation reference; this subset only controls training cost.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.split.open(newline=""), delimiter="\t"))
    by_ec: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ec = (r.get("ec4") or "").split(";")[0]
        if ec:
            by_ec[ec].append(r)

    rng = random.Random(args.seed)
    keep: list[dict] = []
    for ec in sorted(by_ec):
        if len(by_ec[ec]) >= 2:
            keep.append(rng.choice(by_ec[ec]))
    if len(keep) < args.n:
        chosen = {r["id"] for r in keep}
        pool = [r for r in rows if r["id"] not in chosen]
        rng.shuffle(pool)
        keep.extend(pool[: args.n - len(keep)])
    keep = keep[: args.n]
    fields = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(keep)
    print(f"{len(keep)} rows -> {args.out}")
    print(f"EC4 classes: {len({r['ec4'] for r in keep})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
