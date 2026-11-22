#!/usr/bin/env python3
"""Precompute base ESM-C 960-D residue rows for an EC split.

Writes one <id>.npy per sequence into --cache.  Uses esmc-embed; BOS/EOS are
stripped by the binary, matching the M2 head-only contract.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/Users/ananyapathak/Desktop/oss/esmc.cpp/models/esmc-300m-f16.gguf")
DEFAULT_BIN = Path("/Users/ananyapathak/Desktop/oss/esmc.cpp/build/esmc-embed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", type=Path)
    ap.add_argument("cache", type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--embed-bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0,
                    help="split the FASTA into chunks; 0 = one call")
    args = ap.parse_args()
    rows = list(csv.DictReader(args.split.open(newline=""), delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]
    args.cache.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["id"] = r.get("id") or r.get("decoy_id")
    todo = [r for r in rows if not (args.cache / f"{r['id']}.npy").exists()]
    print(f"{len(rows)} rows, {len(todo)} missing", flush=True)
    if not todo:
        return 0
    batch = args.batch or len(todo)
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        with tempfile.TemporaryDirectory(prefix="ece_") as td:
            td = Path(td)
            fa = td / "batch.fa"
            with fa.open("w") as f:
                for r in chunk:
                    f.write(f">{r['id']}\n{r['sequence']}\n")
            out = td / "npy"
            out.mkdir()
            cmd = [str(args.embed_bin), "-m", str(args.model),
                   "--fasta", str(fa), "--output-dir", str(out)]
            print(f"[embed] {start + len(chunk)}/{len(todo)}", flush=True)
            subprocess.run(cmd, check=True, capture_output=True)
            for r in chunk:
                src = out / f"{r['id']}.npy"
                if not src.exists():
                    raise RuntimeError(f"missing embedding for {r['id']}")
                shutil.copyfile(src, args.cache / f"{r['id']}.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
