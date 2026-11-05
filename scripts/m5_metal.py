#!/usr/bin/env python
"""M5 T5.3: fused-metal search vs T5.1 mps at 10K/100K docs -> gate.

T5.2 showed mps efficiency 43.7% (<50%), so the plan's T5.3 gate fires:
custom Metal kernel (tiled f16 GEMM + segment-max, esmc_li/metal_search.py).
Exit = same top-k ids as T5.1 (tie-tolerant) AND >= 2x T5.1 wall at 100K.

Mps and metal run in separate subprocesses: torch's MPS allocator and live
Metal buffers in one process get the run SIGKILLed (observed exit 137 at the
second 100K pass; each phase alone is clean).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "m4"
TMP = ROOT / "results" / "_m5_metal"
K = 10
NQ = 50
SIZES = (10_000, 100_000)


def _queries(n_db: int):
    meta = json.loads((CACHE / "db26.meta.json").read_text())
    off = np.asarray(meta["offsets"], dtype=np.int64)
    id2i = {r: i for i, r in enumerate(meta["ids"])}
    qids = [r["id"] for r in csv.DictReader(
        (ROOT / "data" / "fulllength" / "queries.tsv").open(newline=""),
        delimiter="\t")]
    qpos = [id2i[q] for q in qids]
    A = np.load(CACHE / "db_tr128.npy", mmap_mode="r")
    q = [np.asarray(A[off[p] : off[p + 1]]) for p in qpos[:NQ]]
    return q, A, off[: n_db + 1]


def phase(device: str) -> int:
    """Warm + timed run per size; save top-k arrays + walls to an npz."""
    TMP.mkdir(exist_ok=True)
    walls, ss, iss = [], [], []
    for n in SIZES:
        q, A, sub = _queries(n)
        if device == "mps":
            from esmc_li.search import maxsim_topk
            run = lambda: maxsim_topk(q, A, sub, K, device="mps",
                                      normalize=False)
        else:
            from esmc_li.metal_search import maxsim_topk_metal
            run = lambda: maxsim_topk_metal(q, A, sub, K, normalize=False,
                                            rows_budget=6_000)
        run()  # warm: kernel compile / page-in
        t = time.perf_counter()
        s, i = run()
        walls.append(time.perf_counter() - t)
        ss.append(s)
        iss.append(i)
        print(f"  [{device}] n={n}: {walls[-1]:.1f}s", flush=True)
    np.savez(TMP / f"{device}.npz", walls=np.array(walls),
             s=np.stack(ss), i=np.stack(iss))
    return 0


def tie_ok(s_a, i_a, s_b, i_b) -> tuple[bool, float]:
    """(sets_ok and per-id score agree +-0.1, max |dscore|) — same rule as search."""
    sets_ok = all(set(i_a[qi].tolist()) == set(i_b[qi].tolist())
                  for qi in range(i_a.shape[0]))
    ok, mx = sets_ok, float(np.abs(s_a - s_b).max())
    if sets_ok:
        for qi in range(i_a.shape[0]):
            for j in range(i_a.shape[1]):
                nb = int(np.where(i_b[qi] == i_a[qi, j])[0][0])
                if abs(float(s_a[qi, j]) - float(s_b[qi, nb])) > 0.1:
                    ok = False
    return ok, mx


def main() -> int:
    if len(sys.argv) > 1:  # child
        return phase(sys.argv[1])
    from esmc_li.runlog import log_run

    for dev in ("mps", "metal"):
        r = subprocess.run([sys.executable, __file__, dev], cwd=ROOT)
        if r.returncode != 0:
            print(f"phase {dev} failed ({r.returncode})")
            return 1
    mps, met = (np.load(TMP / f"{d}.npz") for d in ("mps", "metal"))
    results, gate_ok = [], True
    for j, n in enumerate(SIZES):
        ok, mx = tie_ok(met["s"][j], met["i"][j], mps["s"][j], mps["i"][j])
        w_mps, w_met = float(mps["walls"][j]), float(met["walls"][j])
        row = {"n_db": n, "n_queries": NQ, "k": K,
               "mps_wall_s": round(w_mps, 2), "metal_wall_s": round(w_met, 2),
               "mps_qps": round(NQ / w_mps, 3),
               "metal_qps": round(NQ / w_met, 3),
               "speedup": round(w_mps / w_met, 2),
               "topk_ok": bool(ok), "max_dscores": round(mx, 6),
               "identical": bool((mps["i"][j] == met["i"][j]).all())}
        print(f"n={n:>6}: mps {w_mps:7.1f}s  metal {w_met:7.1f}s  "
              f"speedup {row['speedup']}x  topk_ok={ok} max|dscore|={mx:.2e}")
        results.append(row)
        log_run("T5.3", {"device": "metal", "n_db": n, "k": K, "NQ": NQ,
                         "db": "db_tr128 first-n docs",
                         "baseline": "T5.1 maxsim_topk mps (separate process)"},
                row, seed=0, wall_s=w_met)
        gate_ok = gate_ok and ok
    for f in TMP.glob("*.npz"):
        f.unlink()
    r100 = results[-1]
    gate = r100["topk_ok"] and r100["speedup"] >= 2.0
    print(f"T5.3 gate @100K: topk_ok={r100['topk_ok']} "
          f"speedup={r100['speedup']}x (need >=2.0) -> "
          f"{'PASS' if gate else 'FAIL'}")
    return 0 if (gate_ok and gate) else 1


if __name__ == "__main__":
    sys.exit(main())
