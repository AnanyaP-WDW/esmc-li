#!/usr/bin/env python3
"""Apply a fine-tuned ESM-C suffix + head to an EC split.

Reads the M3 prefix cache [L+2, 960], runs the trained suffix and 128-D head,
drops the CLS/EOS rows, and writes <id>.npy [L, 128] f16.  The output is the
cache used by `ec_eval.py --method maxsim`, matching the base 960-D residue
rows (no specials).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from esmc_li.train import DEV, Projection, Suffix, _esmc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", type=Path)
    ap.add_argument("prefix", type=Path)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
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

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    esmc, _ = _esmc()
    suffix = Suffix(esmc, ck["cut"]).to(DEV).eval()
    suffix.load_state_dict(ck["suffix"])
    head = Projection(960, 128).to(DEV).eval()
    head.load_state_dict(ck["head"])

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            embs = [np.load(args.prefix / f"{r['id']}.npy") for r in chunk]
            n = max(len(e) for e in embs)
            x = np.zeros((len(chunk), n, 960), dtype=np.float32)
            m = np.zeros((len(chunk), n), dtype=bool)
            for j, e in enumerate(embs):
                x[j, :len(e)] = e
                m[j, :len(e)] = True
            h = head(suffix(torch.from_numpy(x).to(DEV),
                            torch.from_numpy(m).to(DEV))).cpu().numpy()
            for j, r in enumerate(chunk):
                t = int(m[j].sum())
                np.save(args.out / f"{r['id']}.npy",
                        h[j, 1:t - 1].astype(np.float16))
            if i % 1024 == 0:
                print(f"  suffix {i}/{len(todo)} ({time.time() - t0:.0f}s)",
                      flush=True)
    print(f"suffix rows: +{len(todo)} files in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
