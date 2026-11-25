#!/usr/bin/env python3
"""Build the M3 prefix cache (input to block 30-k) for an EC split.

Writes one <id>.npy [L+2, 960] f32 per sequence, including CLS/EOS.  This is
the same contract as esmc_li.train.build_prefix_cache, but for EC splits.  All
EC sequences are <= 1023 residues, so one window per sequence is enough.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from esmc_li.train import DEV, _esmc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.split.open(newline=""), delimiter="\t"))
    for r in rows:
        r["id"] = r.get("id") or r.get("decoy_id")
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (args.out / f"{r['id']}.npy").exists()]
    print(f"{len(rows)} rows, {len(todo)} missing", flush=True)
    if not todo:
        return 0

    import torch

    cut = 30 - args.k
    esmc, tok = _esmc()
    esmc = esmc.to(DEV).eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            enc = tok([r["sequence"] for r in chunk], padding=True,
                      return_tensors="pt")
            enc = {kk: vv.to(DEV) for kk, vv in enc.items()}
            x = esmc.embed(enc["input_ids"])
            mask = enc["attention_mask"].bool()
            seq_id = None if bool(mask.all()) else mask[:, None, None, :]
            for block in list(esmc.transformer.blocks)[:cut]:
                x, _ = block(x, seq_id)
            x = x.cpu()
            for j, r in enumerate(chunk):
                t = int(mask[j].sum())
                np.save(args.out / f"{r['id']}.npy", x[j, :t].numpy())
            if i % 1024 == 0:
                print(f"  prefix {i}/{len(todo)} ({time.time() - t0:.0f}s)",
                      flush=True)
    print(f"prefix k={args.k}: +{len(todo)} files in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
