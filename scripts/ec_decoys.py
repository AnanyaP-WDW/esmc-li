#!/usr/bin/env python3
"""Generate sequence-only catalytic decoys for EC training.

Each active-site residue is replaced with Ala (or Gly if already Ala).  This is
the model-agnostic control from the plan; structure-based 5/10/15 A decoys are
evaluated externally with EnzymARC.  Parent/decoy pairs are split 90/10 by a
stable hash so validation decoys never enter training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mutate(seq: str, positions: list[int]) -> tuple[str, list[int]]:
    seen, changed = set(), []
    out = list(seq)
    for p in positions:
        i = p - 1  # UniProt is 1-based
        if not (0 <= i < len(seq)) or i in seen:
            continue
        seen.add(i)
        new = "A" if seq[i] != "A" else "G"
        out[i] = new
        changed.append(i + 1)
    return "".join(out), changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("train", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.train.open(newline=""), delimiter="\t"))
    rows = [r for r in rows if r.get("active_sites")]
    rows.sort(key=lambda r: hashlib.sha256(
        f"{args.seed}:{r['id']}".encode()).hexdigest())
    rows = rows[: args.n]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta, fa = [], []
    for r in rows:
        positions = [int(x) for x in r["active_sites"].split(";") if x]
        seq, changed = mutate(r["sequence"], positions)
        if not changed:
            continue
        decoy_id = f"{r['id']}|catalytic"
        fa.append((decoy_id, seq))
        meta.append({
            "decoy_id": decoy_id,
            "parent": r["id"],
            "category": "catalytic",
            "parent_ec4": r["ec4"],
            "n_mutated": len(changed),
            "mutations": ";".join(map(str, changed)),
            "sequence": seq,
        })
    split = int(0.9 * len(meta))
    with (args.out_dir / "train_catalytic.fasta").open("w") as f:
        for did, seq in fa[:split]:
            f.write(f">{did}\n{seq}\n")
    with (args.out_dir / "val_catalytic.fasta").open("w") as f:
        for did, seq in fa[split:]:
            f.write(f">{did}\n{seq}\n")
    fields = ["decoy_id", "parent", "category", "parent_ec4",
              "n_mutated", "mutations", "sequence"]
    for name, data in (("train_decoys.tsv", meta[:split]),
                       ("val_decoys.tsv", meta[split:])):
        with (args.out_dir / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t",
                               lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    print(f"{len(meta)} decoys, train {split}, val {len(meta)-split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
