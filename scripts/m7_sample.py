#!/usr/bin/env python3
"""M7 T7.3 prep — deterministic UniRef50 subsets (seed 0).

Streams `uniref50.fasta.gz` once into a 1,000,000-sequence reservoir sample
(seeded), shuffles it, and writes nested subsets (10K ⊂ 100K ⊂ 1M) in the
M4 fulllength TSV shape. Queries = first 50 ids of every subset (leave-one-out
like M4, shared across sizes because subsets are nested).

Family/superfamily/fold columns are empty (UniRef50 has no clan annotation);
T7.3 measures build/index/latency/recall-vs-exhaustive, not family recall.
"""
from __future__ import annotations

import csv
import gzip
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GZ = ROOT / "data" / "raw" / "uniref50" / "uniref50.fasta.gz"
OUTDIR = ROOT / "data" / "raw" / "uniref50"
SIZES = (10_000, 100_000, 1_000_000)
N_RES = 1_000_000
SEED = 0


def stream_fasta(path: Path):
    ident, seq = None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line[0] == ">":
                if ident is not None:
                    yield ident, "".join(seq)
                ident = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
    if ident is not None:
        yield ident, "".join(seq)


def main() -> int:
    if not GZ.exists():
        print(f"missing {GZ}", file=sys.stderr)
        return 1
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    pool: list[tuple[str, str]] = []
    n_seen = 0
    for rec in stream_fasta(GZ):
        n_seen += 1
        if len(pool) < N_RES:
            pool.append(rec)
        else:
            j = int(rng.integers(0, n_seen))
            if j < N_RES:
                pool[j] = rec
        if n_seen % 5_000_000 == 0:
            print(f"  streamed {n_seen:,} (pool {len(pool):,})",
                  flush=True)
    rng.shuffle(pool)
    print(f"{n_seen:,} reps seen; sample {len(pool):,} "
          f"in {time.time() - t0:.0f}s", flush=True)

    for n in SIZES:
        p = OUTDIR / f"db_{n}.tsv"
        with p.open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "sequence", "family", "superfamily", "fold"])
            for ident, seq in pool[:n]:
                w.writerow([ident, seq, "", "", ""])
        print(f"  {p.name}: {n} rows", flush=True)
    q = OUTDIR / "queries.tsv"
    with q.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "sequence", "family", "superfamily", "fold"])
        for ident, seq in pool[:50]:
            w.writerow([ident, seq, "", "", ""])
    print(f"  queries.tsv: 50 rows (first 50 of every subset)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
