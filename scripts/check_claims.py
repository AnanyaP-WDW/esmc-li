#!/usr/bin/env python3
"""T9.2: verify every numeric placeholder is mapped to a generated table cell.

Hardcoded numbers are rejected. The allowlist is only for names that contain
digits (MMseqs2, knnProtT5, UniRef50, ESM-2, L2) and structural labels
(Table 1a, Figure 2).
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "paper" / "main.template.md"
CLAIMS = ROOT / "paper" / "claims.tsv"
NUMBERS = ROOT / "paper" / "numbers.json"
TABLES = ROOT / "paper" / "tables"
NUM_RE = re.compile(r"\{\{n:([^}]+)\}\}")
CITE_RE = re.compile(r"\[@[^\]]+\]")
COMMENT_RE = re.compile(r"<!--.*?-->")

ALLOW_NAMES = re.compile(
    r"\b(?:MMseqs2|knnProtT5|UniRef50|ESM-2|35M|L2|M4 Max|ASTRAL40)\b")
ALLOW_METRIC = re.compile(r"\b(?:cR@[\d/]+|top-\d+)\b")
ALLOW_TABLE = re.compile(r"\b(?:Table|Figure)\s+\d+[a-c]?\b")
HEAD_RE = re.compile(r"^#{1,4}\s+")


def main() -> int:
    import json
    text = TEMPLATE.read_text()
    numbers = json.loads(NUMBERS.read_text())
    with CLAIMS.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    keys = [r["key"] for r in rows]
    problems: list[str] = []

    placeholders = NUM_RE.findall(text)
    for k in placeholders:
        if k not in numbers:
            problems.append(f"placeholder has no numbers.json entry: {k}")
        if k not in keys:
            problems.append(f"placeholder has no claims row: {k}")
    for r in rows:
        if r["key"] not in placeholders:
            problems.append(f"claim row not used by template: {r['key']}")
        n = numbers.get(r["key"])
        if not n:
            continue
        if r["display"] != n["display"]:
            problems.append(f"claim display drift: {r['key']}")
        table = TABLES / f"{n['table']}.md"
        if not table.exists():
            problems.append(f"missing table for {r['key']}: {table}")
            continue
        needle = n["display"]
        if needle not in table.read_text():
            problems.append(
                f"{r['key']}: {needle!r} not found in {table.name}")

    # Hardcoded-number scan: strip placeholders/comments/citations/headings.
    scan = COMMENT_RE.sub("", text)
    scan = NUM_RE.sub("X", scan)
    scan = CITE_RE.sub("", scan)
    scan = "\n".join(ln for ln in scan.splitlines()
                     if not HEAD_RE.match(ln))
    scan = ALLOW_TABLE.sub("", scan)
    scan = ALLOW_NAMES.sub("", scan)
    scan = ALLOW_METRIC.sub("", scan)
    for i, line in enumerate(scan.splitlines(), 1):
        if re.search(r"\d", line):
            problems.append(f"hardcoded digit in template line {i}: {line.strip()}")

    if problems:
        print("claims check FAILED", file=sys.stderr)
        for p in problems:
            print(" -", p, file=sys.stderr)
        return 1
    print(f"numeric claims: {len(rows)}/{len(rows)} mapped; "
          f"hardcoded: 0; tables: {len(list(TABLES.glob('*.md')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
