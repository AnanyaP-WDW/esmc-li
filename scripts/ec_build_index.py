#!/usr/bin/env python3
"""Build an f16 .esmci EC retrieval index from cached 960-D rows + head."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from esmc_li.index import write  # noqa: E402


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", type=Path)
    ap.add_argument("cache", type=Path)
    ap.add_argument("head", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.split.open(newline=""), delimiter="\t"))
    head = np.load(args.head).astype(np.float32)
    lens, ids, meta, ok = [], [], [], []
    for r in rows:
        p = args.cache / f"{r['id']}.npy"
        if not p.exists():
            continue
        ok.append(r)
        lens.append(len(r["sequence"]))
        ids.append(r["id"])
        meta.append({"ec4": r["ec4"], "ec1": r["ec1"],
                     "source": r["source"]})
    off = np.cumsum([0] + lens).astype(np.uint64)
    vecs = np.empty((int(off[-1]), 128), dtype=np.float16)
    for j, r in enumerate(ok):
        x = np.load(args.cache / f"{r['id']}.npy").astype(np.float32)
        vecs[int(off[j]):int(off[j + 1])] = l2(x @ head.T).astype(np.float16)
    payload = {"ids": ids, "metadata": meta, "ec4": [r["ec4"] for r in ok],
               "d": 128, "dtype": "float16", "source": str(args.split)}
    write(args.out, vecs, off, payload)
    print(json.dumps({"n": len(ids), "rows": int(off[-1]),
                      "bytes": args.out.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
