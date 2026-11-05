#!/usr/bin/env python3
"""M7 T7.2 + T7.3 measurement per subset size.

  --size N   open cache/m7/db{N}.esmci; queries = queries.tsv ids mapped
             into the build order (leave-one-out, shared across subsets)

Measures: exhaustive fused-metal search latency (M5 stack), pooled-docs
build, coarse candidate recall vs exhaustive for C in the sweep (T7.2 exit:
>= 0.99 at the chosen C), and rerank latency. Appends
results/m7_scaling.csv; logs T7.3 (measure) + T7.2 (per size).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

M7 = ROOT / "cache" / "m7"
OUT = ROOT / "results" / "m7_scaling.csv"
FIELDS = ["size", "n_residues", "esmci_bytes", "build_s", "open_s",
          "exhaustive_50q_s", "exhaustive_s_per_q", "C_chosen",
          "candidate_recall", "rerank_50q_s", "coarse_s_per_q",
          "speedup_vs_exhaustive"]
C_SWEEP = (300, 1000, 3000, 10_000, 30_000, 100_000)
K = 10
NQ = 50


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, required=True)
    args = ap.parse_args()
    from esmc_li.coarse import (candidate_recall, pool_docs,
                                       pool_query, rerank, top_c)
    from esmc_li.index import Esmci
    from esmc_li.metal_search import maxsim_topk_metal
    from esmc_li.runlog import log_run
    from esmc_li.search import maxsim_topk

    N = args.size
    t = time.perf_counter()
    idx = Esmci(M7 / f"db{N}.esmci")
    open_s = time.perf_counter() - t
    qids = [r["id"] for r in csv.DictReader(
        (ROOT / "data" / "raw" / "uniref50" / "queries.tsv").open(
            newline=""), delimiter="\t")][:NQ]
    id2i = {rid: i for i, rid in enumerate(idx.ids)}
    self_ids = np.asarray([id2i[q] for q in qids], dtype=np.int64)
    q = [np.asarray(idx.vecs[idx.offsets[i] : idx.offsets[i + 1]])
         for i in self_ids]
    print(f"size={N} rows={idx.n_residues} open={open_s*1000:.1f}ms",
          flush=True)

    # exhaustive (fused metal, M5 stack) — 1 warmup query then timed 50.
    # Query i owns doc self_ids[i], so request K+1 and drop self for LOO.
    maxsim_topk_metal(q[:1], idx.vecs, idx.offsets, K + 1, normalize=False)
    t = time.perf_counter()
    s_ex_all, i_ex_all = maxsim_topk_metal(q, idx.vecs, idx.offsets, K + 1,
                                           normalize=False)
    ex_s = time.perf_counter() - t
    s_ex = np.empty((NQ, K), dtype=np.float32)
    i_ex = np.empty((NQ, K), dtype=np.int64)
    for qi in range(NQ):
        keep = i_ex_all[qi] != self_ids[qi]
        s_ex[qi] = s_ex_all[qi][keep][:K]
        i_ex[qi] = i_ex_all[qi][keep][:K]
    print(f"  exhaustive 50q: {ex_s:.2f}s ({ex_s/NQ:.3f}s/q)", flush=True)

    pool_path = M7 / f"db{N}_pool.npy"
    t = time.perf_counter()
    if pool_path.exists():
        d_pool = np.load(pool_path)
    else:
        d_pool = pool_docs(idx.vecs, idx.offsets)
        np.save(pool_path, d_pool)
    pool_s = time.perf_counter() - t

    qp = np.stack([pool_query(x) for x in q])
    recalls, chosen = {}, None
    for C in C_SWEEP:
        if C > N:
            continue
        cand = top_c(qp, d_pool, C)
        r = candidate_recall(cand, i_ex, self_ids=self_ids)
        recalls[C] = r
        if chosen is None and r >= 0.99:
            chosen = C
        print(f"  C={C:>6}: candidate recall {r:.4f}", flush=True)
    if chosen is None:
        chosen = max(recalls)

    cand = top_c(qp, d_pool, chosen)
    t = time.perf_counter()
    s_cr, i_cr = rerank(q, idx.vecs, idx.offsets, cand, K, device="metal",
                        self_ids=self_ids)
    rerank_s = time.perf_counter() - t
    agree = float(np.mean([np.array_equal(np.asarray(i_cr[j]),
                                          np.asarray(i_ex[j]))
                           for j in range(NQ)]))
    # coarse query cost = rerank + pooled top-C (pool build is index-time)
    t = time.perf_counter()
    for _ in range(10):
        top_c(qp, d_pool, chosen)
    topc_s = (time.perf_counter() - t) / 10
    coarse_q = (rerank_s / NQ) + topc_s
    speedup = (ex_s / NQ) / coarse_q
    rec = recalls[chosen]

    build_s = 0.0
    for line in (ROOT / "results" / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("task") == "T7.3" and r["config"].get("size") == N \
                and r["config"].get("stage") == "build":
            build_s = r["metrics"]["build_s"]
    row = {"size": N, "n_residues": idx.n_residues,
           "esmci_bytes": idx.nbytes(), "build_s": round(build_s, 1),
           "open_s": round(open_s, 4),
           "exhaustive_50q_s": round(ex_s, 2),
           "exhaustive_s_per_q": round(ex_s / NQ, 4),
           "C_chosen": chosen, "candidate_recall": round(rec, 4),
           "rerank_50q_s": round(rerank_s, 2),
           "coarse_s_per_q": round(coarse_q, 4),
           "speedup_vs_exhaustive": round(speedup, 1)}
    OUT.parent.mkdir(exist_ok=True)
    keep = ([r for r in csv.DictReader(OUT.open())
             if r["size"] != str(N)] if OUT.exists() else [])
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(keep)
        w.writerow(row)
    log_run("T7.3", {"size": N, "stage": "measure",
                     "queries": NQ, "k": K}, row, seed=0)
    log_run("T7.2", {"size": N, "C_sweep": list(C_SWEEP), "pool": "D=128",
                     "rerank": "frozen MaxSim"}, {"C_chosen": chosen,
                     "candidate_recall": rec, "rerank_agreement": agree,
                     "coarse_s_per_q": row["coarse_s_per_q"],
                     "exhaustive_s_per_q": row["exhaustive_s_per_q"]},
            seed=0)
    print(json.dumps(row, indent=1))
    ok = rec >= 0.99
    print(f"{'PASS' if ok else 'FAIL'}: recall@C={rec:.4f} >= 0.99, "
          f"rerank agreement {agree:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
