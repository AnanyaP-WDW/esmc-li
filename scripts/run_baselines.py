#!/usr/bin/env python3
"""Dumb baselines (T0.6): random, MinHash 5-mer, MMseqs2 -s 7.5.

Protocol = ProtoCol's: corpus = the split, queries = the same rows,
leave-one-out, MinHash/MMseqs ranked by similarity/e-value.
Writes results/m0_baselines.csv and appends to results/runs.jsonl.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from esmc_retrieval.eval import evaluate, load_split, random_ranking
from esmc_retrieval.runlog import log_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m0_baselines.csv"
SPLIT = ROOT / "data" / "splits" / "scope_test.tsv"
KMER_K, NUM_PERM = 5, 256


def minhash_rankings(rows: list[dict]) -> dict[str, list[str]]:
    from datasketch import MinHash

    ids = [r["id"] for r in rows]
    sigs = np.empty((len(rows), NUM_PERM), dtype=np.uint64)
    for i, r in enumerate(rows):
        m = MinHash(num_perm=NUM_PERM)
        seq = r["sequence"]
        for j in range(len(seq) - KMER_K + 1):
            m.update(seq[j : j + KMER_K].encode())
        sigs[i] = m.hashvalues
    s64 = sigs.astype(np.int64)
    out: dict[str, list[str]] = {}
    n = len(rows)
    for i in range(n):
        eq = (s64 == s64[i]).sum(axis=1)
        eq = eq.astype(np.float64) / NUM_PERM
        eq[i] = -1.0
        order = np.argsort(-eq, kind="stable")
        out[ids[i]] = [ids[j] for j in order if j != i]
    return out


def mmseqs_rankings(rows: list[dict]) -> dict[str, list[str]]:
    if not shutil.which("mmseqs"):
        raise RuntimeError("mmseqs not found (brew install mmseqs2)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fasta = td / "seqs.fasta"
        with fasta.open("w") as f:
            for r in rows:
                f.write(f">{r['id']}\n{r['sequence']}\n")
        out_tsv = td / "hits.tsv"
        subprocess.run(
            ["mmseqs", "easy-search", str(fasta), str(fasta), str(out_tsv),
             str(td / "tmp"), "-s", "7.5", "--max-seqs", "1000",
             "-e", "10000000000",
             "--format-output", "query,target,evalue", "-v", "1"],
            check=True, capture_output=True,
        )
        hits: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for line in out_tsv.read_text().splitlines():
            p = line.split("\t")
            if len(p) < 3 or p[0] == p[1]:
                continue
            hits[p[0]].append((p[1], float(p[2])))
        out = {}
        for r in rows:
            ranked = sorted(hits.get(r["id"], []), key=lambda x: x[1])
            out[r["id"]] = [t for t, _ in ranked]  # unhit seqs = not retrieved
        return out


def main() -> int:
    t0 = time.time()
    rows = load_split(SPLIT)
    print(f"split: {SPLIT.name} n={len(rows)}")

    methods = {
        "random": lambda: random_ranking(rows, seed=0),
        "minhash5mer": lambda: minhash_rankings(rows),
        "mmseqs2_s7.5": lambda: mmseqs_rankings(rows),
    }
    fieldnames = ["split", "method", "n", "wall_s", "n_queries"] + [
        "cR@1", "cR@5", "cR@10", "cR@100", "mrr",
        "sens_family", "sens_superfamily", "sens_fold",
    ]
    OUT.parent.mkdir(exist_ok=True)
    write_header = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for name, fn in methods.items():
            t = time.time()
            rankings = fn()
            wall = time.time() - t
            m = evaluate(rankings, rows)
            row = {"split": SPLIT.stem, "method": name, "n": len(rows),
                   "wall_s": round(wall, 2), **{k: round(v, 4) for k, v in m.items()}}
            w.writerow(row)
            f.flush()
            print(row)
            log_run("T0.6", {"split": SPLIT.stem, "method": name,
                             "kmer_k": KMER_K, "num_perm": NUM_PERM,
                             "mmseqs": "-s 7.5 -e 1e10 --max-seqs 1000 "
                                       "(matches ProtoCol Table 1; released "
                                       "code defaults give cR@10 0.497)"},
                    m, seed=0, wall_s=wall)
    print(f"total {time.time() - t0:.1f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
