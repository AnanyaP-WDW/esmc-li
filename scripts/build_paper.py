#!/usr/bin/env python3
"""Build paper/main.md from the template, generated assets, and refs.json."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEMPLATE = PAPER / "main.template.md"
OUT = PAPER / "main.md"
NUMBERS = json.loads((PAPER / "numbers.json").read_text())
REFS = json.loads((PAPER / "refs.json").read_text())

TABLE_RE = re.compile(r"<!-- TABLE:([a-zA-Z0-9_]+) -->")
FIG_RE = re.compile(r"<!-- FIGURE:([a-zA-Z0-9_]+) -->")
NUM_RE = re.compile(r"\{\{n:([^}]+)\}\}")
CITE_RE = re.compile(r"\[@([a-zA-Z0-9_]+)\]")


def build() -> str:
    text = TEMPLATE.read_text()
    cited: list[str] = []

    def cite(m: re.Match) -> str:
        key = m.group(1)
        if key not in REFS:
            raise SystemExit(f"unknown citation key: {key}")
        if key not in cited:
            cited.append(key)
        return f"[{cited.index(key) + 1}]"

    text = CITE_RE.sub(cite, text)

    def num(m: re.Match) -> str:
        key = m.group(1)
        if key not in NUMBERS:
            raise SystemExit(f"unknown number key: {key}")
        return NUMBERS[key]["display"]

    text = NUM_RE.sub(num, text)

    def table(m: re.Match) -> str:
        name = m.group(1)
        p = PAPER / "tables" / f"{name}.md"
        if not p.exists():
            raise SystemExit(f"missing table: {p}")
        return p.read_text().rstrip()

    def figure(m: re.Match) -> str:
        name = m.group(1)
        p = PAPER / "figures" / f"{name}.svg"
        if not p.exists():
            raise SystemExit(f"missing figure: {p}")
        return f"![{name}](figures/{name}.svg)"

    text = TABLE_RE.sub(table, text)
    text = FIG_RE.sub(figure, text)

    lines = ["## References", ""]
    for i, key in enumerate(cited, 1):
        r = REFS[key]
        authors = r["authors"].rstrip(".")
        lines.append(
            f"{i}. {authors}. *{r['title']}*. {r['venue']}, "
            f"{r['year']}. {r['url']}")
    lines.append("")
    text = text.replace("<!-- REFERENCES -->", "\n".join(lines))
    return text


def main() -> int:
    text = build()
    if "--check" in sys.argv:
        old = OUT.read_text() if OUT.exists() else ""
        if old != text:
            print("paper/main.md is stale; run `make paper`", file=sys.stderr)
            return 1
        print("paper/main.md up to date")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
