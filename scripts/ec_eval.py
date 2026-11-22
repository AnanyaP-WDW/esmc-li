#!/usr/bin/env python3
"""EC retrieval evaluation.

Methods:
  pooled   mean of residue rows, cosine
  maxsim   frozen 960-D MaxSim
  head     MaxSim after a 960->128 projection head

The reference is a labelled protein set; queries are scored leave-one-out when
they come from the same split.  Outputs per-query predictions and aggregate
metrics as CSV + JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from esmc_li.search import maxsim_topk_numpy  # noqa: E402


def rows(path: Path) -> list[dict]:
    out = list(csv.DictReader(path.open(newline=""), delimiter="\t"))
    for r in out:
        r["id"] = r.get("id") or r.get("decoy_id")
    return out


def ec4(r: dict) -> str:
    return (r.get("ec4") or "").split(";")[0]


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_head(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["head"]["proj.weight"].numpy().astype(np.float32)


def encode(ids, cache, method, head, limit=0):
    if limit:
        ids = ids[:limit]
    vals, got = [], []
    for i in ids:
        p = cache / f"{i}.npy"
        if not p.exists():
            continue
        x = np.load(p).astype(np.float32)
        if method == "pooled":
            v = l2(x.mean(axis=0))[None]
        elif method == "head":
            v = l2(x @ head.T)
        else:
            v = l2(x)
        vals.append(v)
        got.append(i)
    return got, vals


def ref_arrays(got, vals, method):
    if method == "pooled":
        return np.stack([v[0] for v in vals]).astype(np.float16), None
    lens = [len(v) for v in vals]
    off = np.cumsum([0] + lens).astype(np.uint64)
    db = np.concatenate(vals).astype(np.float16)
    return db, off


def predict(q_vals, db, off, method, k):
    if method == "pooled":
        q = np.stack([v[0] for v in q_vals]).astype(np.float32)
        s = q @ db.astype(np.float32).T
        order = np.argsort(-s, axis=1, kind="stable")[:, :k]
        scores = np.take_along_axis(s, order, axis=1)
        return scores, order
    out = [maxsim_topk_numpy(v, db, off, k) for v in q_vals]
    return (np.stack([o[0] for o in out]),
            np.stack([o[1] for o in out]))


def prefix(ec: str, n: int) -> str:
    return ".".join(ec.split(".")[:n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--ref-cache", type=Path, required=True)
    ap.add_argument("--query-cache", type=Path, required=True)
    ap.add_argument("--method", choices=["pooled", "maxsim", "head"],
                    default="head")
    ap.add_argument("--head", type=Path, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ref-limit", type=int, default=0)
    ap.add_argument("--query-limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--self-exclude", action="store_true")
    ap.add_argument("--decoy", action="store_true",
                    help="query rows have parent_ec4; report decoy FPR")
    args = ap.parse_args()
    ref_rows, qry_rows = rows(args.ref), rows(args.query)
    if args.ref_limit:
        ref_rows = ref_rows[:args.ref_limit]
    if args.query_limit:
        qry_rows = qry_rows[:args.query_limit]
    head = load_head(args.head) if args.method == "head" and args.head else None
    if args.method == "head" and head is None:
        raise SystemExit("--head is required for method=head")
    rids, rvals = encode([r["id"] for r in ref_rows], args.ref_cache,
                         args.method, head)
    qids, qvals = encode([r["id"] for r in qry_rows], args.query_cache,
                         args.method, head)
    by_id = {r["id"]: r for r in ref_rows}
    db, off = ref_arrays(rids, rvals, args.method)
    scores, order = predict(qvals, db, off, args.method, args.k)

    # Remove self from rankings if the query is also in the reference.
    if args.self_exclude:
        rank = {q: i for i, q in enumerate(rids)}
        for qi, q in enumerate(qids):
            j = rank.get(q)
            if j is not None:
                order[qi][order[qi] == j] = -1
                scores[qi][order[qi] == -1] = -np.inf

    qmap = {q: r for q, r in zip(qids, qry_rows)}
    hits1 = hitsk = 0
    mrr = 0.0
    multi_p, multi_r, multi_f = [], [], []
    levels = {n: {"tp": 0, "fp": 0, "fn": 0} for n in (1, 2, 3, 4)}
    preds = []
    for qi, q in enumerate(qids):
        qr = qmap[q]
        if args.decoy:
            true = [e for e in (qr.get("parent_ec4") or qr.get("ec4") or "").split(";") if e]
        else:
            true = [e for e in (qr.get("ec4") or "").split(";") if e]
        if not true:
            continue
        neigh = [by_id[rids[j]] for j in order[qi] if j >= 0]
        necs = [ec4(r) for r in neigh]
        top1 = necs[0] if necs else ""
        hits1 += int(top1 in true)
        hitsk += int(any(e in true for e in necs))
        if top1 in true:
            mrr += 1.0
        else:
            pos = next((i + 1 for i, e in enumerate(necs) if e in true), None)
            if pos:
                mrr += 1.0 / pos
        pred_set = set(e for e in necs if e)
        true_set = set(true)
        inter = len(pred_set & true_set)
        multi_p.append(inter / max(1, len(pred_set)))
        multi_r.append(inter / max(1, len(true_set)))
        multi_f.append(2 * multi_p[-1] * multi_r[-1] /
                       max(1e-9, multi_p[-1] + multi_r[-1]))
        for n in (1, 2, 3, 4):
            pt = {prefix(e, n) for e in pred_set}
            tt = {prefix(e, n) for e in true_set}
            levels[n]["tp"] += len(pt & tt)
            levels[n]["fp"] += len(pt - tt)
            levels[n]["fn"] += len(tt - pt)
        preds.append({"query": q, "true": ";".join(true), "top1": top1,
                      "top": ";".join(necs), "score": float(scores[qi, 0])})
    n = max(1, len(preds))
    hn = len([1 for _ in preds])  # all rows counted above
    out = {
        "method": args.method, "ref": len(rids), "query": len(qids),
        "k": args.k, "decoy": args.decoy,
        "top1_acc": hits1 / n, "topk_acc": hitsk / n, "mrr": mrr / n,
        "multi_p": float(np.mean(multi_p)) if multi_p else 0.0,
        "multi_r": float(np.mean(multi_r)) if multi_r else 0.0,
        "multi_f1": float(np.mean(multi_f)) if multi_f else 0.0,
    }
    for lv, c in levels.items():
        p = c["tp"] / max(1, c["tp"] + c["fp"])
        r = c["tp"] / max(1, c["tp"] + c["fn"])
        out[f"f1_ec{lv}"] = 2 * p * r / max(1e-9, p + r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0].keys()) if preds else
                          ["query", "true", "top1", "top", "score"])
        w.writeheader()
        w.writerows(preds)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
