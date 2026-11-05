#!/usr/bin/env python3
"""T9.2: every citation used in the paper is verified, with a URL."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "paper" / "main.template.md"
REFS = json.loads((ROOT / "paper" / "refs.json").read_text())
CITE_RE = re.compile(r"\[@([a-zA-Z0-9_]+)\]")


def main() -> int:
    text = TEMPLATE.read_text()
    main = (ROOT / "paper" / "main.md").read_text()
    used = list(dict.fromkeys(CITE_RE.findall(text)))
    problems = []
    for key in used:
        if key not in REFS:
            problems.append(f"missing ref: {key}")
            continue
        r = REFS[key]
        if r.get("status") != "verified":
            problems.append(f"{key}: status {r.get('status')!r}")
        if not str(r.get("url", "")).startswith("http"):
            problems.append(f"{key}: missing URL")
    for key, r in REFS.items():
        if r.get("status") not in ("verified", "corrected"):
            problems.append(f"{key}: status {r.get('status')!r}")
    for doc, label in ((text, "template"), (main, "main.md")):
        if "UNVERIFIED" in doc:
            problems.append(f"{label}: contains UNVERIFIED")
    if "[@" in main:
        problems.append("main.md still contains citation markers")
    claims = (ROOT / "docs" / "claims.md").read_text()
    if ("introduced residue-level late interaction" not in claims
            or "does **not** claim" not in claims):
        problems.append("claims.md missing prior-work framing")
    if problems:
        print("citation check FAILED", file=sys.stderr)
        for p in problems:
            print(" -", p, file=sys.stderr)
        return 1
    print(f"citations: {len(used)}/{len(REFS)} used, all verified; "
          f"UNVERIFIED: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
