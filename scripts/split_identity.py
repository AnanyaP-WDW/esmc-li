#!/usr/bin/env python3
"""Report max train<->test sequence identity with MMseqs2 (T0.4 exit criterion)."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
OUTF = ROOT / "results" / "m0_split_identity.txt"


def write_fasta(tsv: Path, fa: Path) -> int:
    n = 0
    with fa.open("w") as out:
        for line in tsv.read_text().splitlines()[1:]:
            cols = line.split("\t")
            out.write(f">{cols[0]}\n{cols[1]}\n")
            n += 1
    return n


def max_identity(dataset: str) -> tuple[tuple[float, str, str], float, str]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        n_tr = write_fasta(SPLITS / f"{dataset}_train.tsv", td / "train.fa")
        n_te = write_fasta(SPLITS / f"{dataset}_test.tsv", td / "test.fa")
        cmd = [
            "mmseqs", "easy-search", str(td / "train.fa"), str(td / "test.fa"),
            str(td / "out.tsv"), str(td / "tmp"),
            "-s", "7.5", "--max-seqs", "1000",
            "--format-output", "query,target,fident,qcov,tcov,alnlen", "-v", "1",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        best = (0.0, "", "")
        best_full = 0.0
        for line in (td / "out.tsv").read_text().splitlines():
            p = line.split("\t")
            if len(p) < 6:
                continue
            ident, qcov, tcov = float(p[2]), float(p[3]), float(p[4])
            if ident > best[0]:
                best = (ident, f"{p[0]}->{p[1]}", f"alnlen={p[5]} qcov={qcov:.2f} tcov={tcov:.2f}")
            if min(qcov, tcov) >= 0.8:
                best_full = max(best_full, ident)
        return best, best_full, f"train={n_tr} test={n_te}"


def main() -> int:
    if not shutil.which("mmseqs"):
        print("mmseqs not found", file=sys.stderr)
        return 1
    lines = []
    for dataset in ("scope", "pfam"):
        (ident, pair, detail), full_id, sizes = max_identity(dataset)
        line = (f"{dataset}: max train<->test identity = {ident * 100:.2f}% "
                f"({pair}; {detail}) | max identity over >=80%-coverage "
                f"alignments = {full_id * 100:.2f}% | {sizes}")
        print(line)
        lines.append(line)
    OUTF.parent.mkdir(exist_ok=True)
    OUTF.write_text("\n".join(lines) + "\n")
    print(f"-> {OUTF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
