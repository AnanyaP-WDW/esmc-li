#!/usr/bin/env python3
"""Rebuild data/scope.fa from the raw ASTRAL-40 FASTA (fetch_data.py input).

Headers reduced to the ID (first whitespace token), sequences and line
wrapping verbatim; '>' is only a record separator at column 0 (the raw file
has '>' characters inside header text, e.g. "Appr>p").
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "scope" / "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
OUT = ROOT / "data" / "scope.fa"


def main() -> int:
    lines = RAW.read_text().splitlines()
    out: list[str] = []
    for line in lines:
        if line.startswith(">"):
            out.append(">" + line[1:].split()[0])
        elif line.strip():
            out.append(line)
    text = "\n".join(out) + "\n"
    if OUT.exists() and OUT.read_text() == text:
        print(f"scope.fa already up to date ({len(out)} lines)")
        return 0
    OUT.write_text(text)
    n = sum(1 for l in out if l.startswith(">"))
    print(f"wrote {OUT} ({n} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
