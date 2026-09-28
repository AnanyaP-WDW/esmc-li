#!/usr/bin/env python3
"""M4: embed data/fulllength/db.tsv -> cache/m4/db.npy (packed f16 memmap).

  default        esmc-embed (base-model final layer, residue rows only)
  --prefix       HF block-27 raw hidden state -> cache/m4/db26.npy (f32,
                 CLS/EOS kept) — input to the M3 suffix for the trained scorer

Order = longest chains first so the first batch exercises >2046 chunking.
Resumable: re-running skips finished batches (embed.py progress.json).
Logs T4.2 on completion.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from esmc_retrieval.embed import embed_dataset, prefix_dataset

ROOT = Path(__file__).resolve().parents[1]
DB_TSV = ROOT / "data" / "fulllength" / "db.tsv"
OUT = ROOT / "cache" / "m4" / "db.npy"
META = ROOT / "cache" / "m4" / "db.meta.json"
OUT26 = ROOT / "cache" / "m4" / "db26.npy"
META26 = ROOT / "cache" / "m4" / "db26.meta.json"
BATCH = 10_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", action="store_true",
                    help="block-27 prefix via HF instead of esmc-embed")
    args = ap.parse_args()
    t0 = time.time()
    rows = list(csv.DictReader(DB_TSV.open(newline=""), delimiter="\t"))
    rows.sort(key=lambda r: (-len(r["sequence"]), r["id"]))
    residues = sum(len(r["sequence"]) for r in rows)
    print(f"{len(rows)} chains, {residues} residues "
          f"({'prefix' if args.prefix else 'esmc-embed'})")
    try:
        if args.prefix:
            prefix_dataset(rows, OUT26, META26)
            out, mode = OUT26, "prefix26"
        else:
            man = json.loads((ROOT / "data" / "manifest.json").read_text())
            binary = Path(man["esmc_cpp"]["esmc_embed_binary"])
            gguf = Path(man["esmc_cpp"]["gguf"]["path"])
            embed_dataset(rows, OUT, META, binary, gguf, batch=BATCH)
            out, mode = OUT, "esmc_embed"
    except Exception as e:  # surface subprocess stderr (capture_output=True)
        print(f"FAILED: {e}", file=sys.stderr)
        if getattr(e, "stderr", None):
            print(e.stderr.decode()[-3000:], file=sys.stderr)
        return 1
    wall = time.time() - t0
    sys.path.insert(0, str(ROOT))
    from esmc_retrieval.runlog import log_run

    log_run("T4.2", {"fasta": str(DB_TSV.relative_to(ROOT)),
                     "packed": str(out.relative_to(ROOT)), "mode": mode,
                     "dtype": "float32" if args.prefix else "float16",
                     "d": 960, "batch": None if args.prefix else BATCH,
                     "order": "longest_first", "resumable": True},
            {"n": len(rows), "residues": residues,
             "n_gt_2046": sum(1 for r in rows if len(r["sequence"]) > 2046)},
            wall_s=wall)
    print(f"total {wall:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
