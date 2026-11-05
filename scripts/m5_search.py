#!/usr/bin/env python
"""M5 T5.2: exhaustive search throughput vs measured GEMM peak -> results/m5_search.csv.

DB = the first N docs of cache/m4/db_tr128.npy — the frozen default scorer
(D=128, prefix rows incl. per-window CLS/EOS, exactly what trained_m3 scored
in M4). Queries = M4 leave-one-out query rows at qpos (same rows, same L2 +
raw MaxSim). Efficiency = achieved FLOP/s over the full search wall (read +
normalize + transfer + GEMM + segment-max + top-k) / raw GEMM peak measured
with the same shapes on the same device.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "m4"
OUT = ROOT / "results" / "m5_search.csv"
FIELDNAMES = ["device", "n_db", "n_queries", "n_rows", "wall_s", "qps",
              "tflops_achieved", "tflops_gemm_peak", "efficiency_pct"]
K = 10
NQ = {"cpu": {10_000: 50, 100_000: 10}, "mps": {10_000: 50, 100_000: 50}}
GEMM_ROWS = 100_000  # tile shape raw GEMM is measured at


def gemm_peak(dev: str) -> float:
    """TFLOP/s for [340,128] @ [128,GEMM_ROWS] on `dev` (best of 5)."""
    import torch

    dt = torch.float16 if dev == "mps" else torch.float32
    A = torch.randn(340, 128, dtype=dt, device=dev)
    B = torch.randn(128, GEMM_ROWS, dtype=dt, device=dev)
    best = float("inf")
    for _ in range(5):
        if dev == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        C = A @ B
        del C
        if dev == "mps":
            torch.mps.synchronize()
        best = min(best, time.perf_counter() - t)
    return 2 * 340 * 128 * GEMM_ROWS / best / 1e12


def main() -> int:
    import torch

    from esmc_li.runlog import log_run
    from esmc_li.search import maxsim_topk

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    meta = json.loads((CACHE / "db26.meta.json").read_text())  # db_tr128 layout
    off = np.asarray(meta["offsets"], dtype=np.int64)
    id2i = {r: i for i, r in enumerate(meta["ids"])}
    qids = [r["id"] for r in csv.DictReader(
        (ROOT / "data" / "fulllength" / "queries.tsv").open(newline=""),
        delimiter="\t")]
    qpos = [id2i[q] for q in qids]
    A = np.load(CACHE / "db_tr128.npy", mmap_mode="r")

    OUT.parent.mkdir(exist_ok=True)
    rows_out = []
    for dev in devices:
        peak = gemm_peak(dev)
        print(f"{dev}: GEMM peak {peak:.2f} TFLOP/s "
              f"([340,128]@[128,{GEMM_ROWS}] f{16 if dev == 'mps' else 32})")
        for n in sorted(NQ[dev]):
            nq = NQ[dev][n]
            sub_off = off[: n + 1]
            sub_rows = int(sub_off[-1])
            q = [np.asarray(A[off[p] : off[p + 1]]) for p in qpos[:nq]]
            # warm run first: first search in a process pays MPS kernel and
            # page-in costs (observed ~2x); only the second run is reported
            maxsim_topk(q, A, sub_off, K, device=dev, normalize=False)
            t = time.perf_counter()
            # rows are pre-normalized (head F.normalize at build; f16 ~3e-4)
            s, ids = maxsim_topk(q, A, sub_off, K, device=dev, normalize=False)
            wall = time.perf_counter() - t
            flops = 2 * sum(len(qi) for qi in q) * sub_rows * A.shape[1]
            ach = flops / wall / 1e12
            row = {"device": dev, "n_db": n, "n_queries": nq,
                   "n_rows": sub_rows, "wall_s": round(wall, 2),
                   "qps": round(nq / wall, 3),
                   "tflops_achieved": round(ach, 3),
                   "tflops_gemm_peak": round(peak, 3),
                   "efficiency_pct": round(100 * ach / peak, 2)}
            print(f"  {dev} n={n:>6}: {wall:7.1f}s  {row['qps']:6.3f} q/s  "
                  f"{ach:6.2f} TFLOP/s = {row['efficiency_pct']}% of peak")
            rows_out.append(row)
            log_run("T5.2", {"device": dev, "n_db": n, "k": K,
                             "db": "db_tr128 first-n docs",
                             "queries": "m4 leave-one-out first-nq"},
                    row, seed=0, wall_s=wall)
            del s, ids
            if dev == "mps":
                torch.mps.empty_cache()
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows_out)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
