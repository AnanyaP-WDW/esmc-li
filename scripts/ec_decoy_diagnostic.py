#!/usr/bin/env python3
"""Homolog-free EnzymARC decoy diagnostic.

Removes every reference protein with >=30% identity (>=80% coverage) to any
sampled decoy parent, then reruns the same head models.  Also writes a
per-query identity/FPR table so we can see whether false positives are just
homolog transfer.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict]:
    out = list(csv.DictReader(path.open(newline=""), delimiter="\t"))
    for r in out:
        r["id"] = r.get("id") or r.get("decoy_id")
    return out


def write_fa(seqs: list[tuple[str, str]], path: Path) -> None:
    with path.open("w") as f:
        for rid, seq in seqs:
            f.write(f">{rid}\n{seq}\n")


def search(parent_fa: Path, ref_fa: Path, out: Path, work: Path) -> None:
    cmd = ["mmseqs", "easy-search", str(parent_fa), str(ref_fa), str(out),
           str(work), "--min-seq-id", "0.3", "-c", "0.8",
           "--format-output", "query,target,pident", "-v", "1"]
    subprocess.run(cmd, check=True, capture_output=True)


def read_hits(path: Path) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            p = line.split("\t")
            if len(p) >= 3:
                try:
                    out[p[0]].append((p[1], float(p[2])))
                except ValueError:
                    pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=["catalytic", "5A"], required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--decoy", type=Path, required=True)
    ap.add_argument("--ref-cache", type=Path, required=True)
    ap.add_argument("--query-cache", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dec = rows(args.decoy)[: args.limit]
    parents = list(dict.fromkeys(r["parent"] for r in dec))
    train = {r["id"]: r["sequence"] for r in rows(args.train)}
    ref = rows(args.ref)
    parent_missing = [p for p in parents if p not in train]
    if parent_missing:
        print(f"parents missing from train: {len(parent_missing)}", file=sys.stderr)

    parent_fa = args.out_dir / f"{args.category}_parents.fa"
    ref_fa = args.out_dir / f"{args.category}_ref.fa"
    hits_tsv = args.out_dir / f"{args.category}_hits.tsv"
    write_fa([(p, train[p]) for p in parents if p in train], parent_fa)
    write_fa([(r["id"], r["sequence"]) for r in ref], ref_fa)
    search(parent_fa, ref_fa, hits_tsv, args.out_dir / "mm_tmp")

    hits = read_hits(hits_tsv)
    homologs = {
        t for pairs in hits.values() for t, pident in pairs
        if pident >= 30.0
    }
    filtered = [r for r in ref if r["id"] not in homologs]
    filtered_path = args.out_dir / f"ref_nohom_{args.category}.tsv"
    with filtered_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ref[0].keys()), delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(filtered)

    # Per-parent max identity to the original reference.
    max_id = {p: max((x[1] for x in hits.get(p, [])), default=0.0)
              for p in parents}
    with (args.out_dir / f"{args.category}_parent_identity.csv").open(
            "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent", "max_identity_to_ref", "n_homologs"])
        for p in parents:
            w.writerow([p, max_id[p], len(hits.get(p, []))])

    # Evaluate the same model on the filtered reference.
    pred = args.out_dir / f"{args.category}_nohom.csv"
    cmd = [sys.executable, str(ROOT / "scripts" / "ec_eval.py"),
           "--ref", str(filtered_path), "--query", str(args.decoy),
           "--ref-cache", str(args.ref_cache),
           "--query-cache", str(args.query_cache),
           "--method", "head", "--head", str(args.head), "--k", "10",
           "--query-limit", str(args.limit), "--decoy", "--out", str(pred)]
    subprocess.run(cmd, check=True)
    print(json.dumps({"category": args.category,
                      "ref_before": len(ref), "ref_after": len(filtered),
                      "parents": len(parents),
                      "parents_with_30pct_homolog": sum(
                          1 for p in parents if max_id[p] >= 30.0),
                      "out": str(pred)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
