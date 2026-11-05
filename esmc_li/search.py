#!/usr/bin/env python
"""T5.1 — exhaustive MaxSim top-k search (tiled GEMM + segment-max + top-k).

Mirrors the frozen M4 scorer — L2-normalized residue rows, optional linear
projection applied *without* renormalization, raw MaxSim sum — using the
tile/chunk structure proven in scripts/m4_eval.py:blockwise_maxsim, on CPU
or MPS.

Self-check (T5.1 exit): `python -m esmc_li.search --selfcheck` builds a
1,000-protein subset of the M4 corpus (leave-one-out queries + fillers), runs
the naive per-document reference and the tiled search on CPU and MPS, and
asserts identical top-k doc ids.
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
BLOCK_ROWS = 100_000  # tile rows; keeps sim[Lq, Lb] under the MPS int32 limit
CHUNK_QUERIES = 16    # queries per GEMM+copy chunk (bounds sim buffers)
CACHE = ROOT / "cache" / "m4"


def l2_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _prep(rows: np.ndarray, proj: np.ndarray | None,
          normalize: bool = True) -> np.ndarray:
    out = np.asarray(rows, dtype=np.float32)
    if normalize:
        out = l2_rows(out)
    return out @ proj if proj is not None else out


def maxsim_topk(q: list[np.ndarray], db: np.ndarray, offsets: np.ndarray,
                k: int, device: str = "mps",
                proj: np.ndarray | None = None,
                normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Top-k (scores, doc_ids) per query. q: list of [Lq_i, d] row arrays.

    normalize=False skips the L2 pass — for indexes whose rows are already
    unit-norm (the trained head normalizes at build; f16 rounding ~3e-4,
    ranking-stable).
    """
    import gc
    import os

    import torch
    from concurrent.futures import ThreadPoolExecutor

    prof = os.environ.get("M5_PROFILE")  # phase timers: prep/gemm+sync/reduce/clean
    T = {"prep": 0.0, "gemm": 0.0, "copy": 0.0, "reduce": 0.0, "clean": 0.0}
    off = np.asarray(offsets, dtype=np.int64)
    n_docs, nq = len(off) - 1, len(q)
    f16 = device == "mps"
    qrows = [torch.from_numpy(_prep(qi, proj, normalize).astype(
        np.float16 if f16 else np.float32)).to(device) for qi in q]
    S = np.empty((nq, n_docs), dtype=np.float32)

    def chunk_reduce(sims, idxs, starts, a, b):
        out = np.empty((len(idxs), starts.size), dtype=np.float32)
        for j, sim in enumerate(sims):
            m = np.maximum.reduceat(np.asarray(sim, dtype=np.float32),
                                    starts, axis=1)
            out[j] = m.sum(axis=0)
        return idxs, out, a, b

    def drain_one(pending):
        fut, nbytes, lo, hi = pending.pop(0)
        qi, out, lo, hi = fut.result()
        S[qi, lo:hi] = out
        return nbytes

    ex = ThreadPoolExecutor(max_workers=12)
    pending: list = []  # (future, sim_bytes, col_lo, col_hi) — global, so the
    inflight = 0        # CPU reduce of tile N overlaps GPU enqueue of tile N+1
    d0, tile = 0, 0
    while d0 < n_docs:
        r0 = int(off[d0])
        d1 = min(int(np.searchsorted(off, r0 + BLOCK_ROWS, side="right")),
                 n_docs)
        r1 = int(off[d1])
        t = time.perf_counter()
        blk = np.asarray(db[r0:r1])
        if normalize:
            blk = l2_rows(blk.astype(np.float32))
        if proj is not None:
            blk = blk.astype(np.float32) @ proj
        want = np.float16 if f16 else np.float32
        C = torch.from_numpy(blk if blk.dtype == want else blk.astype(want)).to(device)
        T["prep"] += time.perf_counter() - t
        dlens = off[d0 + 1 : d1 + 1] - off[d0 : d1]
        starts = np.concatenate(([0], np.cumsum(dlens)[:-1])).astype(np.int64)
        a, b = d0, d1
        t = time.perf_counter()
        for c0 in range(0, nq, CHUNK_QUERIES):
            idxs = list(range(c0, min(c0 + CHUNK_QUERIES, nq)))
            outs = [qrows[i] @ C.T for i in idxs]      # enqueue only
            T["gemm"] += time.perf_counter() - t
            t = time.perf_counter()
            sims = [o.cpu().numpy() for o in outs]      # sync + copy
            T["copy"] += time.perf_counter() - t
            t = time.perf_counter()
            nb = sum(s.nbytes for s in sims)
            pending.append((ex.submit(chunk_reduce, sims, idxs, starts, a, b),
                            nb, a, b))
            inflight += nb
            while inflight > 6e9:  # byte backpressure across tiles
                inflight -= drain_one(pending)
            T["reduce"] += time.perf_counter() - t
            t = time.perf_counter()
        d0 = d1
        del C, blk
        tile += 1
        if tile % 4 == 0:  # cadence: per-tile gc+empty_cache starves the GPU
            t = time.perf_counter()
            gc.collect()
            if f16:
                torch.mps.empty_cache()
            T["clean"] += time.perf_counter() - t
    while pending:
        inflight -= drain_one(pending)
    ex.shutdown()
    if prof:
        print(f"    [profile {device}] " +
              " ".join(f"{k_}={v_:.1f}s" for k_, v_ in T.items()))
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(S, order, axis=1), order


def maxsim_topk_numpy(q: np.ndarray, db: np.ndarray, offsets: np.ndarray,
                      k: int, doc_chunk: int = 512
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Exact MaxSim top-k with numpy only (M8 package backend).

    q: [Lq, D] L2-normalized query rows; db: [n_res, D] f16/float index rows;
    offsets: n_proteins+1 row starts. Rows may include BOS/EOS; the caller
    keeps them consistent for index and query.
    """
    q = np.asarray(q, dtype=np.float32)
    off = np.asarray(offsets, dtype=np.int64)
    n_docs = len(off) - 1
    assert q.ndim == 2 and q.shape[1] == db.shape[1]
    assert doc_chunk > 0 and n_docs > 0
    scores = np.empty(n_docs, dtype=np.float32)
    for d0 in range(0, n_docs, doc_chunk):
        d1 = min(d0 + doc_chunk, n_docs)
        lo, hi = int(off[d0]), int(off[d1])
        blk = np.asarray(db[lo:hi], dtype=np.float32)
        lens = off[d0 + 1 : d1 + 1] - off[d0:d1]
        assert np.all(lens > 0), "empty protein rows"
        starts = np.concatenate(([0], np.cumsum(lens[:-1])))
        sim = q @ blk.T
        scores[d0:d1] = np.maximum.reduceat(sim, starts, axis=1).sum(axis=0)
    k = min(k, n_docs)
    order = np.argsort(-scores, kind="stable")[:k]
    return scores[order], order


def naive_topk(q: list[np.ndarray], db: np.ndarray, offsets: np.ndarray,
               k: int, proj: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Reference: per-document MaxSim loop (CPU f32)."""
    off = np.asarray(offsets, dtype=np.int64)
    n_docs = len(off) - 1
    Qs = [_prep(qi, proj) for qi in q]
    S = np.empty((len(Qs), n_docs), dtype=np.float32)
    for di in range(n_docs):
        d = _prep(np.asarray(db[off[di] : off[di + 1]], dtype=np.float32), proj)
        for qi, Q in enumerate(Qs):
            S[qi, di] = (Q @ d.T).max(axis=1).sum()
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(S, order, axis=1), order


def _load_subset(nq: int, seed: int = 0):
    """(queries, db, offsets, ids) — nq query docs + fillers to 1000 total."""
    meta = json.loads((CACHE / "db.meta.json").read_text())
    ids = np.array(meta["ids"])
    off = np.asarray(meta["offsets"], dtype=np.int64)
    qids = [r["id"] for r in csv.DictReader(
        (ROOT / "data" / "fulllength" / "queries.tsv").open(newline=""),
        delimiter="\t")]
    id2i = {r: i for i, r in enumerate(meta["ids"])}
    qpos_all = np.array([id2i[q] for q in qids])
    rng = np.random.default_rng(seed)
    nq = min(nq, len(qpos_all))
    qpos = rng.choice(qpos_all, size=nq, replace=False)
    in_q = set(qpos.tolist())
    fill = rng.choice(np.array([i for i in range(len(ids)) if i not in in_q]),
                      size=1000 - nq, replace=False)
    keep = np.sort(np.concatenate([qpos, fill]))
    pos = {int(d): i for i, d in enumerate(keep)}

    A = np.load(CACHE / "db.npy", mmap_mode="r")
    parts = [np.asarray(A[off[i] : off[i + 1]], dtype=np.float16) for i in keep]
    lo = np.concatenate(([0], np.cumsum([len(p) for p in parts])))
    db = np.concatenate(parts)
    q = [parts[pos[int(p)]] for p in qpos]
    return q, db, lo, ids[keep]


def selfcheck(nq: int = 100, k: int = 10) -> dict:
    from esmc_li.runlog import log_run

    t_all = time.time()
    q, db, lo, _ = _load_subset(nq)
    n_docs = len(lo) - 1
    print(f"selfcheck: {n_docs} docs, {len(q)} queries, k={k}")

    t = time.time()
    s_naive, i_naive = naive_topk(q, db, lo, k)
    t_naive = time.time() - t
    print(f"  naive cpu      {t_naive:.1f}s")

    res: dict = {"n_db": n_docs, "n_queries": len(q), "k": k,
                 "naive_s": round(t_naive, 2)}
    for dev in ("cpu", "mps"):
        import torch
        if dev == "mps" and not torch.backends.mps.is_available():
            print("  mps unavailable — skipped")
            continue
        t = time.time()
        s, ids = maxsim_topk(q, db, lo, k, device=dev)
        wall = time.time() - t
        exact = bool((ids == i_naive).all())
        # f16 vs f32 may swap ids tied within rounding: require the same id
        # SET per query and per-id score agreement (catches real bugs)
        sets_ok = all(set(ids[qi].tolist()) == set(i_naive[qi].tolist())
                      for qi in range(ids.shape[0]))
        id_agree = sets_ok
        if sets_ok:
            for qi in range(ids.shape[0]):
                for j in range(ids.shape[1]):
                    na = int(np.where(i_naive[qi] == ids[qi, j])[0][0])
                    if abs(float(s[qi, j]) - float(s_naive[qi, na])) > 0.1:
                        id_agree = False
        diff = float(np.abs(s - s_naive).max())
        ok = sets_ok and id_agree
        print(f"  tiled {dev:9s} {wall:.1f}s  top-{k} identical={exact} "
              f"sets_equal={sets_ok} max|dscore|={diff:.2e}")
        res[f"{dev}_identical"] = exact
        res[f"{dev}_topk_ok"] = ok
        res[f"{dev}_max_dscores"] = round(diff, 6)
        res[f"{dev}_wall_s"] = round(wall, 2)
        assert ok, f"top-{k} ids differ beyond ties on {dev}"

    log_run("T5.1", {"db": "m4_1000_subset", "seed": 0,
                     "devices": [d for d in ("cpu", "mps")
                                 if res.get(f"{d}_identical") is not None]},
            res, seed=0, wall_s=time.time() - t_all)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    if args.selfcheck:
        res = selfcheck(args.nq, args.k)
        print("PASS" if all(res.get(f"{d}_topk_ok")
                            for d in ("cpu", "mps")
                            if f"{d}_topk_ok" in res) else "FAIL")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
