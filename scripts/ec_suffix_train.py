#!/usr/bin/env python3
"""Fine-tune the last k ESM-C blocks + head on EC pairs from the prefix cache.

Mirrors the M3 recipe: suffix blocks lr 2e-5, head lr 1e-3, symmetric InfoNCE
over MaxSim, same-EC4 false-negative masking.  Input rows are the M3 prefix
cache [L+2, 960] (CLS/EOS included), so the suffix sees the same rows it was
trained on in esmc_li.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from esmc_li.train import (DEV, Projection, Suffix, _esmc,  # noqa: E402
                           false_negative_mask, maxsim, symmetric_nce)
from ec_train import build_maps, load_rows, make_batch  # noqa: E402


class PrefixStore:
    """Lazy id -> prefix rows [L+2, 960]; None if the cache file is missing."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self._d: dict[str, np.ndarray] = {}

    def get(self, rid: str) -> np.ndarray | None:
        if rid not in self._d:
            p = self.cache / f"{rid}.npy"
            if not p.exists():
                return None
            self._d[rid] = np.load(p)
        return self._d[rid]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path,
                    default=ROOT / "data/ec/splits/train_5k.tsv")
    ap.add_argument("--prefix", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-suffix", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    train_rows = load_rows(args.train)
    if args.limit:
        train_rows = train_rows[: args.limit]
    by_ec, by_ec3, by_ec1 = build_maps(train_rows)
    store = PrefixStore(args.prefix)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    cut = 30 - args.k
    esmc, _ = _esmc()
    suffix = Suffix(esmc, cut).to(DEV).train()
    head = Projection(960, 128).to(DEV)
    opt = torch.optim.AdamW(
        [{"params": suffix.parameters(), "lr": args.lr_suffix},
         {"params": head.parameters(), "lr": args.lr_head}],
        weight_decay=0.01)
    steps = max(1, len(train_rows) // args.batch)
    t0 = time.time()
    losses = []
    for epoch in range(args.epochs):
        for step in range(steps):
            batch = make_batch(train_rows, by_ec, by_ec3, by_ec1, store, rng,
                               args.batch)
            if batch is None:
                continue
            q, d, qm, dm, labels = batch
            q, d, qm, dm = (t.to(DEV) for t in (q, d, qm, dm))
            scores = maxsim(head(suffix(q, qm)), qm,
                            head(suffix(d, dm)), dm)
            loss = symmetric_nce(scores, args.tau,
                                 false_negative_mask(labels).to(DEV))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(suffix.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(f"epoch {epoch+1} step {step}/{steps} "
                      f"loss {losses[-1]:.4f}", flush=True)
            if DEV == "mps" and step % 50 == 0:
                torch.mps.empty_cache()

    suffix.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"k{args.k}_seed{args.seed}.pt"
    torch.save({"k": args.k, "cut": cut, "seed": args.seed,
                "suffix": {k: v.cpu() for k, v in suffix.state_dict().items()},
                "head": {k: v.cpu() for k, v in head.state_dict().items()},
                "final_loss": losses[-1] if losses else None,
                "train_rows": len(train_rows),
                "wall_s": round(time.time() - t0, 1)}, out)
    print(json.dumps({"out": str(out), "seed": args.seed,
                      "final_loss": losses[-1] if losses else None,
                      "wall_s": round(time.time() - t0, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
