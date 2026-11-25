#!/usr/bin/env python3
"""Head-only EC contrastive training on cached ESM-C 960-D rows.

This is the first, cheap stage from the EC plan.  It deliberately mirrors the
M2 recipe: frozen ESM-C rows, 960 -> 128 projection, MaxSim, symmetric InfoNCE
with same-EC4 false-negative masking.  Hard negatives share EC3 but not EC4.
A catalytic-decoy triplet term is optional.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esmc_li.train import Projection, false_negative_mask, maxsim, symmetric_nce

ROOT = Path(__file__).resolve().parents[1]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline=""), delimiter="\t"))


def ec4(r: dict) -> str:
    return (r.get("ec4") or "").split(";")[0]


def load_cache(ids: list[str], cache: Path) -> dict[str, np.ndarray]:
    out = {}
    for i in ids:
        p = cache / f"{i}.npy"
        if p.exists():
            out[i] = np.load(p)
    return out


def build_maps(rows: list[dict]):
    by_ec: dict[str, list[dict]] = defaultdict(list)
    by_ec3: dict[str, list[dict]] = defaultdict(list)
    by_ec1: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        e = ec4(r)
        if not e:
            continue
        by_ec[e].append(r)
        by_ec3[".".join(e.split(".")[:3])].append(r)
        by_ec1[e.split(".")[0]].append(r)
    return by_ec, by_ec3, by_ec1


def pad_batch(pairs, emb: dict[str, np.ndarray]):
    q, d, qm, dm, labels = [], [], [], [], []
    for a, b in pairs:
        ea, eb = emb.get(a["id"]), emb.get(b["id"])
        if ea is None or eb is None:
            continue
        q.append(ea)
        d.append(eb)
        labels.append(ec4(a))
    if not q:
        return None
    la, lb = max(len(x) for x in q), max(len(x) for x in d)
    qb = np.zeros((len(q), la, 960), dtype=np.float32)
    db = np.zeros((len(d), lb, 960), dtype=np.float32)
    qmask = np.zeros((len(q), la), dtype=bool)
    dmask = np.zeros((len(d), lb), dtype=bool)
    for i, (x, y) in enumerate(zip(q, d)):
        qb[i, :len(x)] = x
        db[i, :len(y)] = y
        qmask[i, :len(x)] = True
        dmask[i, :len(y)] = True
    return (torch.from_numpy(qb), torch.from_numpy(db),
            torch.from_numpy(qmask), torch.from_numpy(dmask), labels)


def make_batch(rows, by_ec, by_ec3, by_ec1, emb, rng, batch):
    pairs = []
    eligible = [r for r in rows if ec4(r) and len(by_ec[ec4(r)]) >= 2]
    rng.shuffle(eligible)
    for a in eligible:
        if len(pairs) >= batch:
            break
        e = ec4(a)
        pos = [r for r in by_ec[e] if r["id"] != a["id"]]
        if not pos:
            continue
        b = rng.choice(pos)
        pairs.append((a, b))
    # Hard negatives enter through the in-batch denominator; do not add them as
    # extra query rows (their diagonal would not be a positive).
    return pad_batch(pairs, emb)


def quick_top1(head, ref_rows, qry_rows, ref_emb, qry_emb, k=1):
    """Small in-memory MaxSim validation; ref/query are already sampled."""
    ids = [r["id"] for r in ref_rows]
    e4 = [ec4(r) for r in ref_rows]
    vecs = []
    with torch.no_grad():
        for r in ref_rows:
            x = torch.from_numpy(ref_emb[r["id"]][None]).to(DEV)
            vecs.append(head(x).squeeze(0).cpu())
    ref = torch.cat(vecs).numpy().astype(np.float16)
    ref_t = torch.from_numpy(ref.astype(np.float32)).to(DEV)
    hit = 0
    with torch.no_grad():
        for r in qry_rows:
            q = head(torch.from_numpy(qry_emb[r["id"]][None]).to(DEV)).squeeze(0)
            s = q @ ref_t.T
            # segment max over reference docs
            starts = np.cumsum([0] + [len(ref_emb[i]) for i in ids])[:-1]
            # ref rows are per-protein; use per-doc max via reduceat
            m = np.maximum.reduceat(s.cpu().numpy(), starts, axis=1)
            order = np.argsort(-m.sum(axis=0), kind="stable")[:k]
            hit += int(ec4(r) in {e4[j] for j in order})
    return hit / max(1, len(qry_rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--val-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decoy-meta", type=Path, default=None)
    ap.add_argument("--decoy-cache", type=Path, default=None)
    ap.add_argument("--decoy-weight", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--clusters", type=Path, default=None)
    ap.add_argument("--val-ref", type=int, default=1000)
    ap.add_argument("--val-qry", type=int, default=500)
    ap.add_argument("--skip-val", action="store_true",
                    help="skip the in-memory quick_top1 check; use ec_eval.py")
    args = ap.parse_args()
    clusters = {}
    if args.clusters:
        for line in args.clusters.open():
            rep, mem = line.split()
            clusters[mem] = rep
    train_rows = load_rows(args.train)
    val_rows = load_rows(args.val)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    by_ec, by_ec3, by_ec1 = build_maps(train_rows)
    emb = load_cache([r["id"] for r in train_rows], args.cache)
    ref_rows = rng.sample(train_rows, min(args.val_ref, len(train_rows)))
    qry_rows = rng.sample(val_rows, min(args.val_qry, len(val_rows)))
    ref_emb = load_cache([r["id"] for r in ref_rows], args.cache)
    qry_emb = load_cache([r["id"] for r in qry_rows],
                         args.val_cache or args.cache)
    ref_rows = [r for r in ref_rows if r["id"] in ref_emb]
    qry_rows = [r for r in qry_rows if r["id"] in qry_emb]

    decoy_by_parent = {}
    decoy_emb = {}
    if args.decoy_meta and args.decoy_cache:
        with args.decoy_meta.open(newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                decoy_by_parent[r["parent"]] = r
        decoy_emb = load_cache([r["decoy_id"] for r in decoy_by_parent.values()],
                               args.decoy_cache)

    head = Projection(960, 128).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    steps = max(1, len(train_rows) // args.batch)
    t0 = time.time()
    losses = []
    for epoch in range(args.epochs):
        for step in range(steps):
            batch = make_batch(train_rows, by_ec, by_ec3, by_ec1, emb, rng,
                               args.batch)
            if batch is None:
                continue
            q, d, qm, dm, labels = batch
            q, d, qm, dm = (t.to(DEV) for t in (q, d, qm, dm))
            scores = maxsim(head(q), qm, head(d), dm)
            loss = symmetric_nce(scores, args.tau,
                                 false_negative_mask(labels).to(DEV))
            if args.decoy_weight and decoy_by_parent:
                # One parent/decoy/positive triplet per step, cheap stabilizer.
                candidates = [r for r in train_rows
                              if r["id"] in decoy_by_parent
                              and r["id"] in emb
                              and decoy_by_parent[r["id"]]["decoy_id"] in decoy_emb]
                if not candidates:
                    continue
                parent = rng.choice(candidates)
                if parent["id"] in emb:
                    e = ec4(parent)
                    pos = [r for r in by_ec[e] if r["id"] != parent["id"]]
                    if clusters:
                        pos = [r for r in pos if clusters.get(r["id"])
                               != clusters.get(parent["id"])]
                    if pos:
                        positive = rng.choice(pos)
                        if positive["id"] in emb:
                            dd = decoy_by_parent[parent["id"]]["decoy_id"]
                            pa = torch.from_numpy(emb[parent["id"]][None]).to(DEV)
                            da = torch.from_numpy(decoy_emb[dd][None]).to(DEV)
                            po = torch.from_numpy(emb[positive["id"]][None]).to(DEV)
                            s = maxsim(head(torch.cat([pa, da])),
                                       torch.ones(2, pa.shape[1], dtype=torch.bool, device=DEV),
                                       head(torch.cat([po, po])),
                                       torch.ones(2, po.shape[1], dtype=torch.bool, device=DEV))
                            loss = loss + args.decoy_weight * F.relu(
                                args.margin - (s[0, 0] - s[1, 0]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(f"epoch {epoch+1} step {step}/{steps} loss {losses[-1]:.4f}",
                      flush=True)
            if DEV == "mps" and step % 50 == 0:
                torch.mps.empty_cache()
    head.eval()
    if DEV == "mps":
        torch.mps.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    val_top1 = None
    if not args.skip_val:
        val_top1 = quick_top1(head, ref_rows, qry_rows, ref_emb, qry_emb)
    torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()},
                "d": 128, "tau": args.tau, "seed": args.seed,
                "val_top1": val_top1}, args.out)
    np.save(args.out.with_suffix(".npy"),
            head.state_dict()["proj.weight"].detach().cpu().numpy())
    summary = {"seed": args.seed, "epochs": args.epochs, "batch": args.batch,
               "tau": args.tau, "val_top1": val_top1,
               "train_rows": len(train_rows), "cache_rows": len(emb),
               "final_loss": losses[-1] if losses else None,
               "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
