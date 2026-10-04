"""T5.3 — fused Metal MaxSim pipeline (PyObjC, no MPS/torch on device).

Per tile: custom tiled f16 GEMM (sim stays GPU-side) -> segment-max kernel
(scores reduced to [nq, nseg] f32) -> tiny CPU readout. The score block is
never materialized host-side. Exits same top-k as esmc_retrieval.search.
"""
from __future__ import annotations

import numpy as np

MSL = r"""
#include <metal_stdlib>
using namespace metal;

// sim[M, Npad] = A[M, K] x B[N, K]^T  (f16 in, f32 accumulate).
// TG = 128 threads, output tile 64x64, thread owns an 8x4 block.
kernel void gemm_tiled(
    device const half* A [[buffer(0)]],   // [M, K]
    device const half* B [[buffer(1)]],   // [N, K]
    device half*       sim [[buffer(2)]], // [M, Npad]
    constant int& M  [[buffer(3)]],
    constant int& N  [[buffer(4)]],
    constant int& K  [[buffer(5)]],
    constant int& Npad [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tp [[thread_position_in_threadgroup]])
{
    uint tid = tp.x;
    constexpr uint TM = 64, TN = 64, KT = 32;
    threadgroup half Asm[64][32];
    threadgroup half Bsm[64][32];
    uint m0 = gid.x * TM, n0 = gid.y * TN;
    uint bi = tid >> 4, bj = tid & 15u;   // row-block 8, col-block 4
    float acc[8][4];
    for (uint a = 0; a < 8; ++a) for (uint b = 0; b < 4; ++b) acc[a][b] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += (int)KT) {
        for (uint e = tid; e < TM * KT; e += 128u) {
            uint rr = e >> 5, kk = e & 31u;
            uint m = m0 + rr;
            Asm[rr][kk] = (m < (uint)M) ? A[(long)m * K + k0 + kk] : half(0);
            uint n = n0 + rr;
            Bsm[rr][kk] = (n < (uint)N) ? B[(long)n * K + k0 + kk] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < KT; ++kk) {
            half av[8], bv[4];
            for (uint a = 0; a < 8; ++a) av[a] = Asm[bi * 8 + a][kk];
            for (uint b = 0; b < 4; ++b) bv[b] = Bsm[bj * 4 + b][kk];
            for (uint a = 0; a < 8; ++a)
                for (uint b = 0; b < 4; ++b)
                    acc[a][b] = fma(float(av[a]), float(bv[b]), acc[a][b]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint a = 0; a < 8; ++a) {
        uint m = m0 + bi * 8 + a;
        if (m >= (uint)M) break;
        for (uint b = 0; b < 4; ++b) {
            uint n = n0 + bj * 4 + b;
            if (n >= (uint)N) continue;
            sim[(long)m * Npad + n] = half(acc[a][b]);
        }
    }
}

// out[qid, d] += max over doc d's columns of sim[r, :]; warp per (r, d).
kernel void segmax_sim(
    device const half* sim [[buffer(0)]],   // [R, Npad]
    device const int*  starts [[buffer(1)]],
    device const int*  dlens  [[buffer(2)]],
    device const int*  rowqid [[buffer(3)]],
    device atomic_float* out [[buffer(4)]],
    constant int& Npad  [[buffer(5)]],
    constant int& nseg  [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],   // x = r * nseg + d
    uint lid [[thread_index_in_threadgroup]])
{
    int d = (int)gid.x % nseg;
    int r = (int)gid.x / nseg;
    int s = starts[d], n = dlens[d];
    float best = -INFINITY;
    for (int c = s + (int)lid; c < s + n; c += 32)
        best = max(best, float(sim[(long)r * Npad + c]));
    for (uint w = 16; w > 0; w >>= 1)
        best = max(best, simd_shuffle_down(best, w));
    if (lid == 0)
        atomic_fetch_add_explicit(out + (long)rowqid[r] * nseg + d,
                                  best, memory_order_relaxed);
}
"""

_dev = None
_queue = None
_lib = None
_pso = {}


def _init():
    global _dev, _queue, _lib
    if _dev is not None:
        return
    from Metal import MTLCreateSystemDefaultDevice
    _dev = MTLCreateSystemDefaultDevice()
    _queue = _dev.newCommandQueue()
    _lib, err = _dev.newLibraryWithSource_options_error_(MSL, None, None)
    if _lib is None:
        raise RuntimeError(f"MSL compile failed: {err}")


def _pipeline(name: str):
    if name not in _pso:
        fn = _lib.newFunctionWithName_(name)
        if fn is None:
            raise RuntimeError(f"kernel {name} not found")
        _pso[name] = _dev.newComputePipelineStateWithFunction_error_(fn, None)[0]
    return _pso[name]


def _rel(*bufs) -> None:
    """pyobjc Metal buffers are NOT freed on Python dealloc (measured: an
    unreleased 25MB buffer leaks per tile_scores call, ~18GB over one 100K
    search) — explicit release() is required and safe (no double-free)."""
    for b in bufs:
        try:
            b.release()
        except Exception:
            pass


def _buf(arr) -> object:
    a = np.ascontiguousarray(arr)
    return _dev.newBufferWithBytes_length_options_(a.tobytes(),
                                                   max(a.nbytes, 4), 0)


def tile_scores(A: np.ndarray, C: np.ndarray, starts: np.ndarray,
                dlens: np.ndarray, rowqid: np.ndarray, nq: int,
                sim_buf, M: int, N: int, K: int) -> np.ndarray:
    """GEMM + segment-max for one tile; returns out[nq, nseg] f32 (copy)."""
    from Metal import MTLSize
    _init()
    nseg = starts.size
    Npad = (N + 63) & ~63
    ps_g, ps_s = _pipeline("gemm_tiled"), _pipeline("segmax_sim")
    bA = A if not isinstance(A, np.ndarray) else _buf(A)
    bC = _buf(C)
    bS = _buf(np.ascontiguousarray(starts, np.int32))
    bL = _buf(np.ascontiguousarray(dlens, np.int32))
    bQ = _buf(np.ascontiguousarray(rowqid, np.int32))
    out_nbytes = max(nq * nseg * 4, 4)
    bO = _dev.newBufferWithBytes_length_options_(b"\0" * out_nbytes,
                                                 out_nbytes, 0)
    cb = _queue.commandBuffer()
    # GEMM
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(ps_g)
    enc.setBuffer_offset_atIndex_(bA, 0, 0)
    enc.setBuffer_offset_atIndex_(bC, 0, 1)
    enc.setBuffer_offset_atIndex_(sim_buf, 0, 2)
    for i, v in enumerate([M, N, K, Npad], start=3):
        enc.setBytes_length_atIndex_(np.int32(v).tobytes(), 4, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        MTLSize((M + 63) // 64, (N + 63) // 64, 1), MTLSize(128, 1, 1))
    enc.endEncoding()
    # segment-max (separate encoder in same cb = ordered)
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(ps_s)
    for i, b in enumerate([sim_buf, bS, bL, bQ, bO]):
        enc.setBuffer_offset_atIndex_(b, 0, i)
    for i, v in enumerate([Npad, nseg], start=5):
        enc.setBytes_length_atIndex_(np.int32(v).tobytes(), 4, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        MTLSize(M * nseg, 1, 1), MTLSize(32, 1, 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if cb.error() is not None:
        raise RuntimeError(f"metal cb error: {cb.error()}")
    got = np.frombuffer(bO.contents().as_buffer(out_nbytes),
                        dtype=np.float32, count=nq * nseg).reshape(
                            nq, nseg).copy()
    _rel(bC, bS, bL, bQ, bO)
    if isinstance(A, np.ndarray):
        _rel(bA)
    return got


def selfcheck_tile() -> float:
    """Correctness + throughput vs numpy at a realistic tile shape."""
    import time
    rng = np.random.default_rng(0)
    R, K, N, nq = 4096, 128, 20000, 8
    dlens = np.repeat([340, 512, 128, 1024], N // (4 * 340) + 1)
    dlens = dlens[dlens.cumsum() <= N].astype(np.int32)
    starts = np.concatenate(([0], np.cumsum(dlens)[:-1])).astype(np.int32)
    A = rng.standard_normal((R, K)).astype(np.float16)
    C = rng.standard_normal((N, K)).astype(np.float16)
    rowqid = np.repeat(np.arange(nq), R // nq).astype(np.int32)
    Npad = (N + 63) & ~63
    _init()
    sim_buf = _dev.newBufferWithLength_options_(R * Npad * 2, 0)
    # reference: numpy GEMM + reduceat
    t0 = time.perf_counter()
    # match kernel's f16 output rounding
    sim = (A.astype(np.float32) @ C.astype(np.float32).T).astype(np.float16).astype(np.float32)
    ref = np.zeros((nq, starts.size), np.float32)
    for r in range(R):
        m = np.maximum.reduceat(sim[r], starts.astype(np.int64)).copy()
        m[-1] = sim[r, starts[-1]:starts[-1] + dlens[-1]].max()  # clip: reduceat's last seg runs to end of row
        ref[rowqid[r]] += m
    t_np = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = tile_scores(A, C, starts, dlens, rowqid, nq, sim_buf, R, N, K)
    t_m = time.perf_counter() - t0
    err = float(np.abs(got - ref).max())
    flops = 2 * R * K * N
    print(f"  metal {t_m*1000:7.1f}ms  {flops/t_m/1e12:.2f} TFLOP/s  "
          f"(numpy ref {t_np:.1f}s)  maxerr={err:.3e}")
    assert err < 0.1, f"mismatch {err}"
    _rel(sim_buf)
    return t_m


def maxsim_topk_metal(q: list[np.ndarray], db: np.ndarray, offsets: np.ndarray,
                      k: int, proj: np.ndarray | None = None,
                      normalize: bool = True,
                      rows_budget: int = 15_000) -> tuple[np.ndarray, np.ndarray]:
    """Top-k (scores, doc_ids) per query — all GPU math in this file.

    Mirrors search.maxsim_topk's prep/tiling/top-k; the score block never
    materializes host-side (GEMM -> segment-max on device, tiny readout).
    """
    import os
    import time

    from esmc_retrieval.search import BLOCK_ROWS, _prep

    prof = os.environ.get("M5_PROFILE")
    T = {"prep": 0.0, "gemm": 0.0, "reduce": 0.0}
    off = np.asarray(offsets, dtype=np.int64)
    n_docs, nq = len(off) - 1, len(q)
    _init()
    prep_q = [_prep(qi, proj, normalize).astype(np.float16) for qi in q]
    K = prep_q[0].shape[1]
    lens = [len(p) for p in prep_q]
    A_all = np.concatenate(prep_q)

    # query chunks bounded by stacked rows (sizes sim buffer)
    chunks: list[list[int]] = []
    cur, rows = [], 0
    for i, L in enumerate(lens):
        if cur and rows + L > rows_budget:
            chunks.append(cur)
            cur, rows = [], 0
        cur.append(i)
        rows += L
    if cur:
        chunks.append(cur)
    chunk_bounds = []  # (a, b) row range in A_all per chunk
    a = 0
    for c in chunks:
        b = a + sum(lens[i] for i in c)
        chunk_bounds.append((a, b))
        a = b

    # precompute doc tiles (same rule as search.maxsim_topk)
    tiles: list[tuple[int, int, int, int]] = []
    d0 = 0
    while d0 < n_docs:
        r0 = int(off[d0])
        d1 = min(int(np.searchsorted(off, r0 + BLOCK_ROWS, side="right")),
                 n_docs)
        tiles.append((d0, d1, r0, int(off[d1])))
        d0 = d1
    maxR = max(b - a for a, b in chunk_bounds)
    maxN = max(r1 - r0 for _, _, r0, r1 in tiles)
    sim_buf = _dev.newBufferWithLength_options_(maxR * ((maxN + 63) & ~63) * 2, 0)

    chunk_meta = []  # (qidx, local rowqid, A buffer) per chunk
    for c, (ca, cb) in zip(chunks, chunk_bounds):
        local = np.repeat(np.arange(len(c), dtype=np.int32),
                          [lens[i] for i in c])  # per A-row -> chunk qid
        chunk_meta.append((c, local, _buf(A_all[ca:cb])))
    S = np.empty((nq, n_docs), dtype=np.float32)
    for d0, d1, r0, r1 in tiles:
        t = time.perf_counter()
        blk = np.asarray(db[r0:r1])
        if normalize:
            out = blk.astype(np.float32)
            out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
            blk = out
        if proj is not None:
            blk = blk.astype(np.float32) @ proj
        C = np.ascontiguousarray(blk, dtype=np.float16)
        dlens = (off[d0 + 1 : d1 + 1] - off[d0 : d1]).astype(np.int32)
        starts = np.concatenate(([0], np.cumsum(dlens)[:-1])).astype(np.int32)
        T["prep"] += time.perf_counter() - t
        N = r1 - r0
        for c, local, bA in chunk_meta:
            t = time.perf_counter()
            out = tile_scores(bA, C, starts, dlens, local, len(c),
                              sim_buf, local.size, N, K)
            T["gemm"] += time.perf_counter() - t
            t = time.perf_counter()
            S[np.asarray(c), d0:d1] = out
            T["reduce"] += time.perf_counter() - t
    if prof:
        print("    [profile metal] " +
              " ".join(f"{k_}={v_:.1f}s" for k_, v_ in T.items()))
    _rel(sim_buf, *(b for _, _, b in chunk_meta))
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(S, order, axis=1), order


def selfcheck_metal(nq: int = 20, k: int = 10) -> dict:
    """T5.3 exit check: metal top-k == naive reference (tie-tolerant)."""
    import time

    from esmc_retrieval.runlog import log_run
    from esmc_retrieval.search import _load_subset, naive_topk

    t_all = time.time()
    q, db, lo, _ = _load_subset(nq)
    n_docs = len(lo) - 1
    print(f"metal selfcheck: {n_docs} docs, {nq} queries, k={k}")
    t = time.time()
    s_naive, i_naive = naive_topk(q, db, lo, k)
    print(f"  naive cpu {time.time()-t:.1f}s")
    t = time.time()
    maxsim_topk_metal(q, db, lo, k)  # warm (compile already done, fill cache)
    t1 = time.time()
    s, ids = maxsim_topk_metal(q, db, lo, k)
    wall = time.time() - t1
    exact = bool((ids == i_naive).all())
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
    print(f"  metal          {wall:.1f}s  top-{k} identical={exact} "
          f"sets_equal={sets_ok} max|dscore|={diff:.2e}")
    res = {"n_db": n_docs, "n_queries": nq, "k": k,
           "metal_identical": exact, "metal_topk_ok": ok,
           "metal_max_dscores": round(diff, 6), "metal_wall_s": round(wall, 2),
           "naive_s": round(time.time() - t_all - wall, 2)}
    assert ok, "top-k ids differ beyond ties on metal"
    log_run("T5.3", {"db": "m4_1000_subset", "seed": 0, "device": "metal",
                     "check": "selfcheck"}, res, seed=0,
            wall_s=time.time() - t_all)
    return res


def main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--nq", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    if args.tile:
        selfcheck_tile()
        print("warm:", end="")
        selfcheck_tile()
        print("OK")
        return 0
    if args.selfcheck:
        res = selfcheck_metal(args.nq, args.k)
        print("PASS" if res["metal_topk_ok"] else "FAIL")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
