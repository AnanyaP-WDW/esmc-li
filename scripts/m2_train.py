#!/usr/bin/env python3
"""M2 runner: train the projection head on the frozen cache and select on val.

  --pilot      one config (tau=1.0, masked, seed 0, 3 epochs) vs frozen val
  --sweep      tau x {masked,unmasked} x 3 seeds -> results/m2_sweep.csv
  --final      retrain best config x 3 seeds, evaluate TEST once

Normalization: raw (M1 winner) — head input = raw 960D residues, head output =
L2, score = raw MaxSim sum. Config selection uses scope_val only; scope_test is
evaluated once, after selection. Eval matrices are cached in cache/m2 keyed by
config — delete that dir if EPOCHS/lr change.
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

from esmc_retrieval.eval import evaluate, load_split
from esmc_retrieval.runlog import log_run
from esmc_retrieval.scoring import maxsim_matrix, rankings_from_scores
from esmc_retrieval.train import (EmbeddingStore, Projection, false_negative_mask,
                                  maxsim, sample_pair_batch, symmetric_nce)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m2_sweep.csv"
FINAL = ROOT / "results" / "m2_final.csv"
CACHE_M2 = ROOT / "cache" / "m2"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
TAUS = [0.03, 0.1, 0.3, 1.0]
SEEDS = [0, 1, 2]
EPOCHS = 3
STEPS_PER_EPOCH = 618   # len(scope_train)//16


def load_embs(rows: list[dict]) -> list[np.ndarray]:
    store = EmbeddingStore()
    return [store.get(r["id"]) for r in rows]


def eval_embeddings(embs: list[np.ndarray], rows: list[dict],
                    tag: str) -> float:
    p = CACHE_M2 / f"{tag}.npy"
    if p.exists():
        S = np.load(p)
    else:
        S = maxsim_matrix(embs, device=DEV)
        np.save(p, S)
    ids = [r["id"] for r in rows]
    m = evaluate(rankings_from_scores(S, ids), rows)
    return m["cR@10"]


def train_head(tau: float, masked: bool, seed: int, epochs: int = EPOCHS,
               d_out: int = 128) -> tuple[Projection, float]:
    rows = load_split(ROOT / "data" / "splits" / "scope_train.tsv")
    store = EmbeddingStore()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    head = Projection(960, d_out).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    loss = float("nan")
    t0 = time.time()
    for _ in range(epochs):
        for step in range(STEPS_PER_EPOCH):
            q, d, qm, dm, labels = sample_pair_batch(rows, store, 16, rng)
            q, d, qm, dm = (t.to(DEV) for t in (q, d, qm, dm))
            scores = maxsim(head(q), qm, head(d), dm)
            forbidden = false_negative_mask(labels).to(DEV) if masked else None
            loss = symmetric_nce(scores, tau, forbidden)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            if step % 200 == 0:
                print(f"  seed{seed} tau{tau} masked{masked} "
                      f"ep step {step}: {float(loss.detach()):.4f}", flush=True)
    fl = float(loss.detach())
    print(f"  train {time.time()-t0:.0f}s final_loss={fl:.4f}", flush=True)
    return head, fl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--dims", action="store_true",
                    help="T2.5: D ablation at the selected config (val only)")
    ap.add_argument("--d", type=int, default=128)
    args = ap.parse_args()
    if not (args.pilot or args.sweep or args.final or args.dims):
        ap.error("pass --pilot, --sweep, --final or --dims")
    CACHE_M2.mkdir(exist_ok=True)

    tr_rows = load_split(ROOT / "data" / "splits" / "scope_train.tsv")
    va_rows = load_split(ROOT / "data" / "splits" / "scope_val.tsv")
    te_rows = load_split(ROOT / "data" / "splits" / "scope_test.tsv")
    va_embs = load_embs(va_rows)

    t0 = time.time()
    frozen_val = eval_embeddings(va_embs, va_rows, "val_frozen960")
    print(f"frozen raw MaxSim960 val cR@10 = {frozen_val:.4f}", flush=True)

    def run_one(tau: float, masked: bool, seed: int) -> dict:
        head, fl = train_head(tau, masked, seed, d_out=args.d)
        head.eval()
        with torch.no_grad():
            proj = [head(torch.from_numpy(e).to(DEV)).cpu().numpy()
                    for e in va_embs]
        c10 = eval_embeddings(proj, va_rows, f"val_t{tau}_m{int(masked)}_s{seed}_d{args.d}")
        print(f"  val cR@10 = {c10:.4f} (frozen {frozen_val:.4f})", flush=True)
        return {"tau": tau, "masked": masked, "seed": seed, "d": args.d,
                "val_cR@10": round(c10, 4), "frozen_val": round(frozen_val, 4),
                "final_train_loss": round(fl, 4),
                "wall_s": round(time.time() - t0, 1)}

    if args.pilot:
        row = run_one(1.0, True, 0)
        log_run("T2.pilot", {"tau": 1.0, "masked": True, "seed": 0,
                             "epochs": EPOCHS, "device": DEV, "d": args.d},
                row, seed=0)
        return 0

    if args.sweep:
        OUT.parent.mkdir(exist_ok=True)
        have = set()
        if OUT.exists():
            have = {(float(r["tau"]), r["masked"] in ("True", "true", "1"),
                     int(r["seed"]), int(r["d"]))
                    for r in csv.DictReader(OUT.open())}
        with OUT.open("a", newline="") as f:
            w = csv.writer(f)
            if not have:
                w.writerow(["tau", "masked", "seed", "d", "val_cR@10",
                            "frozen_val", "final_train_loss", "wall_s"])
            for tau in TAUS:
                for masked in (True, False):
                    for seed in SEEDS:
                        if (tau, masked, seed, args.d) in have:
                            continue
                        w.writerow(run_one(tau, masked, seed).values())
                        f.flush()
        log_run("T2.4", {"grid": "tau x masked x 3seeds", "epochs": EPOCHS,
                         "device": DEV, "d": args.d, "split": "scope_val"},
                {"n_runs": len(TAUS) * 2 * len(SEEDS), "out": str(OUT)},
                seed=0, wall_s=time.time() - t0)
        return 0

    # --final: best config from the sweep CSV, test evaluated ONCE per seed
    best = select_best()
    if args.dims:
        return run_dims(best, t0)
    print(f"selected config: {best}", flush=True)
    with FINAL.open("a", newline="") as f:
        w = csv.writer(f)
        if FINAL.stat().st_size == 0:
            w.writerow(["tau", "masked", "seed", "d", "val_cR@10",
                        "test_cR@10", "final_train_loss", "wall_s"])
        vals, tests = [], []
        for seed in SEEDS:
            head, fl = train_head(best["tau"], best["masked"], seed,
                                  d_out=args.d)
            head.eval()
            with torch.no_grad():
                proj = [head(torch.from_numpy(e).to(DEV)).cpu().numpy()
                        for e in va_embs]
            v10 = eval_embeddings(proj, va_rows, f"selval_s{seed}")
            te_embs = load_embs(te_rows)
            with torch.no_grad():
                tproj = [head(torch.from_numpy(e).to(DEV)).cpu().numpy()
                         for e in te_embs]
            t10 = eval_embeddings(tproj, te_rows, f"test_s{seed}_d{args.d}")
            vals.append(v10)
            tests.append(t10)
            w.writerow([best["tau"], best["masked"], seed, args.d,
                        round(v10, 4), round(t10, 4), round(fl, 4),
                        round(time.time() - t0, 1)])
            f.flush()
            print(f"seed{seed}: val {v10:.4f} test {t10:.4f}", flush=True)
        summary = {"val_mean": round(float(np.mean(vals)), 4),
                   "val_std": round(float(np.std(vals)), 4),
                   "test_mean": round(float(np.mean(tests)), 4),
                   "test_std": round(float(np.std(tests)), 4),
                   "frozen_val": round(frozen_val, 4)}
        print(summary, flush=True)
        log_run("T2.final", {**best, "epochs": EPOCHS, "device": DEV,
                             "note": "test evaluated once after val selection"},
                summary, seed=0, wall_s=time.time() - t0)
    return 0


def run_dims(best: dict, t0: float) -> int:
    """T2.5: D in {64,128,256} at the selected config, val only (test stays
    once-only from --final), with index bytes per residue (fp32/fp16/int8)."""
    va_rows = load_split(ROOT / "data" / "splits" / "scope_val.tsv")
    va_embs = load_embs(va_rows)
    frozen_val = eval_embeddings(va_embs, va_rows, "val_frozen960")
    sweep = list(csv.DictReader(OUT.open())) if OUT.exists() else []
    dims_out = ROOT / "results" / "m2_dims.csv"
    with dims_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d", "masked", "tau", "seed", "val_cR@10", "frozen_val",
                    "B_per_res_fp32", "B_per_res_fp16", "B_per_res_int8",
                    "wall_s"])
        for d in (64, 128, 256):
            for seed in SEEDS:
                if d == 128:
                    rows = [r for r in sweep
                            if r["tau"] == str(best["tau"])
                            and r["masked"] in ("True", "true", "1")
                            and int(r["seed"]) == seed]
                    if not rows:
                        sys.exit("sweep missing best-config d=128 rows")
                    w.writerow([d, best["masked"], best["tau"], seed,
                                rows[0]["val_cR@10"], rows[0]["frozen_val"],
                                4 * d, 2 * d, d, rows[0]["wall_s"]])
                    continue
                head, _ = train_head(best["tau"], best["masked"], seed, d_out=d)
                head.eval()
                with torch.no_grad():
                    proj = [head(torch.from_numpy(e).to(DEV)).cpu().numpy()
                            for e in va_embs]
                v10 = eval_embeddings(proj, va_rows, f"val_d{d}_s{seed}")
                w.writerow([d, best["masked"], best["tau"], seed,
                            round(v10, 4), round(frozen_val, 4),
                            4 * d, 2 * d, d, round(time.time() - t0, 1)])
                f.flush()
                print(f"d{d} seed{seed}: val cR@10 {v10:.4f}", flush=True)
    log_run("T2.5", {"dims": [64, 128, 256], "config": best,
                     "split": "scope_val", "device": DEV},
            {"out": str(dims_out)}, seed=0, wall_s=time.time() - t0)
    return 0


def select_best() -> dict:
    rows = list(csv.DictReader(OUT.open()))
    for r in rows:
        r["tau"] = float(r["tau"])
        r["val_cR@10"] = float(r["val_cR@10"])
        r["seed"] = int(r["seed"])
        r["masked"] = r["masked"] in ("True", "true", "1")
    best, best_mean = None, -1.0
    for tau in TAUS:
        for masked in (True, False):
            vs = [r["val_cR@10"] for r in rows
                  if r["tau"] == tau and r["masked"] == masked]
            if vs and np.mean(vs) > best_mean:
                best_mean = float(np.mean(vs))
                best = {"tau": tau, "masked": masked, "best_val_mean": round(best_mean, 4)}
    if best is None:
        sys.exit("no sweep rows; run --sweep first")
    return best


if __name__ == "__main__":
    sys.exit(main())
