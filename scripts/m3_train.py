#!/usr/bin/env python3
"""M3 runner (T3.2): fine-tune last k in {2,3} ESM-C blocks + projection head
on the prefix caches, same selection protocol as T2.4 (val only; test once).

  --sweep   k x 3 seeds -> results/m3_sweep.csv
  --final   selected k x 3 seeds, test evaluated once -> results/m3_final.csv

Two param groups: suffix blocks lr 2e-5 (ProtoCol's backbone lr), head 1e-3.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from esmc_li.eval import evaluate, load_split
from esmc_li.runlog import log_run
from esmc_li.scoring import maxsim_matrix, rankings_from_scores
from esmc_li.train import (DEV, EmbeddingStore, Projection, Suffix,
                                  _esmc, false_negative_mask, m3_cache_dir,
                                  maxsim, sample_pair_batch, symmetric_nce)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m3_sweep.csv"
FINAL = ROOT / "results" / "m3_final.csv"
MODELS = ROOT / "results" / "m3_models"
EVAL_CACHE = ROOT / "cache" / "m3" / "eval"
SEEDS = [0, 1, 2]
KS = [2, 3]
EPOCHS = 3
STEPS_PER_EPOCH = 618
TAU = 1.0


def eval_head(head: Projection, suffix: Suffix, rows: list[dict], k: int,
              tag: str) -> float:
    EVAL_CACHE.mkdir(parents=True, exist_ok=True)
    p = EVAL_CACHE / f"{tag}.npy"
    if p.exists():
        S = np.load(p)
    else:
        store = EmbeddingStore(m3_cache_dir(k))
        proj = []
        with torch.no_grad():
            for r in rows:
                x = torch.from_numpy(store.get(r["id"])[None]).to(DEV)
                proj.append(head(suffix(x)).squeeze(0).cpu().numpy())
        S = maxsim_matrix(proj, device=DEV)
        np.save(p, S)
    # ponytail: after a big eval the next train step sometimes wedges on an
    # MPS sync; flush the cache between phases (observed on macOS 26 / torch 2.x)
    if DEV == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    ids = [r["id"] for r in rows]
    return evaluate(rankings_from_scores(S, ids), rows)["cR@10"]


def train_run(k: int, seed: int, d_out: int = 128) -> tuple[Projection, Suffix, float]:
    tr_rows = load_split(ROOT / "data" / "splits" / "scope_train.tsv")
    store = EmbeddingStore(m3_cache_dir(k))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    esmc, _ = _esmc()
    suffix = Suffix(esmc, 30 - k).to(DEV).train()
    head = Projection(960, d_out).to(DEV)
    opt = torch.optim.AdamW(
        [{"params": suffix.parameters(), "lr": 2e-5},
         {"params": head.parameters(), "lr": 1e-3}],
        weight_decay=0.01)
    loss = float("nan")
    t0 = time.time()
    for _ in range(EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            q, d, qm, dm, labels = sample_pair_batch(tr_rows, store, 16, rng)
            q, d, qm, dm = (t.to(DEV) for t in (q, d, qm, dm))
            scores = maxsim(head(suffix(q, qm)), qm, head(suffix(d, dm)), dm)
            loss = symmetric_nce(scores, TAU, false_negative_mask(labels).to(DEV))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(suffix.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            if step % 300 == 0:
                print(f"  k{k} seed{seed} ep-step {step}: "
                      f"{float(loss.detach()):.4f}", flush=True)
    fl = float(loss.detach())
    print(f"  train k={k} seed={seed}: {time.time()-t0:.0f}s final_loss={fl:.4f}",
          flush=True)
    suffix.eval()
    return head, suffix, fl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    if not (args.sweep or args.final):
        ap.error("pass --sweep or --final")

    va_rows = load_split(ROOT / "data" / "splits" / "scope_val.tsv")
    te_rows = load_split(ROOT / "data" / "splits" / "scope_test.tsv")
    t0 = time.time()

    if args.sweep:
        have = set()
        if OUT.exists():
            have = {(int(r["k"]), int(r["seed"])) for r in csv.DictReader(OUT.open())}
        OUT.parent.mkdir(exist_ok=True)
        with OUT.open("a", newline="") as f:
            w = csv.writer(f)
            if not have:
                w.writerow(["k", "seed", "d", "tau", "val_cR@10",
                            "final_train_loss", "wall_s"])
            for k in KS:
                for seed in SEEDS:
                    if (k, seed) in have:
                        continue
                    head, suffix, fl = train_run(k, seed)
                    v10 = eval_head(head, suffix, va_rows, k, f"val_k{k}_s{seed}")
                    print(f"  k={k} seed={seed}: val cR@10 = {v10:.4f}", flush=True)
                    w.writerow([k, seed, 128, TAU, round(v10, 4), round(fl, 4),
                                round(time.time() - t0, 1)])
                    f.flush()
        log_run("T3.2", {"grid": "k x 3 seeds", "epochs": EPOCHS, "tau": TAU,
                         "lrs": {"blocks": 2e-5, "head": 1e-3},
                         "split": "scope_val", "device": DEV},
                {"n_runs": len(KS) * len(SEEDS)}, seed=0, wall_s=time.time() - t0)
        return 0

    # --final: best k by mean val, test evaluated ONCE per seed (resume-safe)
    rows = list(csv.DictReader(OUT.open()))
    means = {k: float(np.mean([float(r["val_cR@10"]) for r in rows
                               if int(r["k"]) == k])) for k in KS}
    best_k = max(means, key=means.get)
    print(f"val means: {means} -> selected k={best_k}", flush=True)
    done: dict[int, dict] = {}
    if FINAL.exists():
        done = {int(r["seed"]): r for r in csv.DictReader(FINAL.open())}
    with FINAL.open("a", newline="") as f:
        w = csv.writer(f)
        if FINAL.stat().st_size == 0:
            w.writerow(["k", "seed", "d", "tau", "val_cR@10", "test_cR@10",
                        "final_train_loss", "wall_s"])
        vals, tests = [], []
        for seed in SEEDS:
            if seed in done:
                vals.append(float(done[seed]["val_cR@10"]))
                tests.append(float(done[seed]["test_cR@10"]))
                print(f"seed{seed}: already done, skipping", flush=True)
                continue
            head, suffix, fl = train_run(best_k, seed)
            v10 = eval_head(head, suffix, va_rows, best_k, f"selval_k{best_k}_s{seed}")
            t10 = eval_head(head, suffix, te_rows, best_k, f"test_k{best_k}_s{seed}")
            vals.append(v10)
            tests.append(t10)
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"k": best_k, "cut": 30 - best_k, "seed": seed,
                        "suffix": {kk: vv.cpu() for kk, vv in
                                   suffix.state_dict().items()},
                        "head": {kk: vv.cpu() for kk, vv in
                                 head.state_dict().items()},
                        "val_cR@10": v10, "test_cR@10": t10},
                       MODELS / f"k{best_k}_seed{seed}.pt")
            w.writerow([best_k, seed, 128, TAU, round(v10, 4), round(t10, 4),
                        round(fl, 4), round(time.time() - t0, 1)])
            f.flush()
            print(f"seed{seed}: val {v10:.4f} test {t10:.4f}", flush=True)
        summary = {"k": best_k, "val_mean": round(float(np.mean(vals)), 4),
                   "val_std": round(float(np.std(vals)), 4),
                   "test_mean": round(float(np.mean(tests)), 4),
                   "test_std": round(float(np.std(tests)), 4)}
        print(summary, flush=True)
        log_run("T3.final", {"k": best_k, "tau": TAU, "epochs": EPOCHS,
                             "note": "test evaluated once after val selection"},
                summary, seed=0, wall_s=time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
