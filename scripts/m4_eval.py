#!/usr/bin/env python3
"""M4 T4.3: full-length retrieval benchmark (results/m4_fulllength.csv).

Protocol: queries = the 500 test-clan chains (data/fulllength/queries.tsv),
DB = 100K single-clan Swiss-Prot chains; queries are rows of the corpus and
rankings exclude self, so the FROZEN evaluate() harness applies unchanged
(leave-one-out; cR@k over clan groups).

Scorers:
  random / minhash5mer / mmseqs2_s7.5     (M0 baseline protocol, cross)
  pooled_esmc   mean-pooled cosine, 960D raw residues
  maxsim960     MaxSim on L2-normalized residue rows (blockwise over db.npy)
  maxsim128     same after the frozen gaussian 960->128 projection
  trained_m3    M3 k=3 seed0 suffix+head (D=128), MaxSim over db26 prefix rows

Normalizations (vector scorers): raw, div_lenA (query), div_lenB (candidate),
div_len_geomean — sym_* variants need a square matrix and do not survive the
cross protocol. rho(candidate length, score) over non-homolog pairs via
m1_run.spearman_len_score. Prep artifacts are cached in cache/m4/.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from esmc_li.eval import evaluate
from esmc_li.runlog import log_run
from esmc_li.scoring import random_projection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m1_run import spearman_len_score  # noqa: E402

FL = ROOT / "data" / "fulllength"
CACHE = ROOT / "cache" / "m4"
OUT = ROOT / "results" / "m4_fulllength.csv"

FIELDNAMES = ["method", "normalization", "rho_len_score_nonhomolog",
              "n_db", "n_queries", "wall_s", "cR@1", "cR@5", "cR@10",
              "cR@100", "mrr", "sens_family", "sens_superfamily"]
KMER_K, NUM_PERM = 5, 256
BLOCK_ROWS = 100_000  # keeps sim[Lq, Lb] element count under MPS int32 limit
CHUNK_QUERIES = 4  # GEMM chunk: bounds GPU sim buffers before CPU scatter

_db_open: dict[str, np.ndarray] = {}


def arr(p: Path) -> np.ndarray:
    if str(p) not in _db_open:
        _db_open[str(p)] = np.load(p, mmap_mode="r")
    return _db_open[str(p)]


def get_matrix(name: str, fn) -> np.ndarray:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"S_{name}.npy"
    if p.exists():
        return np.load(p)
    t = time.time()
    M = fn()
    np.save(p, M)
    print(f"{name}: computed in {time.time() - t:.1f}s -> {p}")
    return M


def cross_rankings(S: np.ndarray, qids: list[str], ids: list[str],
                   qpos: list[int]) -> dict[str, list[str]]:
    out = {}
    for i, q in enumerate(qids):
        order = np.argsort(-S[i], kind="stable")
        out[q] = [ids[j] for j in order if j != qpos[i]]
    return out


# -- prep stages ---------------------------------------------------------------

def doc_means(db: Path, meta: dict) -> np.ndarray:
    p = CACHE / "doc_means.npy"
    if p.exists():
        return np.load(p)
    off, lens = meta["offsets"], meta["lens"]
    A = arr(db)
    means = np.empty((len(lens), 960), dtype=np.float32)
    for i, (lo, ln) in enumerate(zip(off[:-1], lens)):
        means[i] = A[lo : lo + ln].mean(axis=0)
    np.save(p, means)
    return means


def blockwise_maxsim(db: Path, meta: dict, qpos: list[int],
                     proj: np.ndarray | None) -> np.ndarray:
    """S[q, d] = sum_qrows max_drows cos(qrow, drow), blockwise over rows.

    GEMM on GPU; segment-max on CPU threads. MPS scatter_reduce on broadcast
    views is slow (and heats the GPU into thermal creep); reduceat over
    contiguous per-doc segments is bandwidth-bound and overlaps the GEMM.
    """
    import gc
    import torch
    from concurrent.futures import ThreadPoolExecutor

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    off, dlens = meta["offsets"], meta["lens"]
    n_docs, total = len(dlens), off[-1]
    A = arr(db)

    def rows_of(doc: int) -> np.ndarray:
        q = np.asarray(A[off[doc] : off[doc] + dlens[doc]], dtype=np.float32)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        return q @ proj if proj is not None else q

    qrows = [torch.from_numpy(rows_of(i).astype(np.float16)).to(dev)
             for i in qpos]
    S = np.empty((len(qpos), n_docs), dtype=np.float32)

    def chunk_rows(sims, idxs, starts):
        out = np.empty((len(idxs), starts.size), dtype=np.float32)
        for j, sim in enumerate(sims):
            m = np.maximum.reduceat(sim.astype(np.float32), starts, axis=1)
            out[j] = m.sum(axis=0)
        return idxs, out

    ex = ThreadPoolExecutor(max_workers=8)
    t0, d0, r0 = time.time(), 0, 0
    while r0 < total:
        d1 = min(int(np.searchsorted(off, r0 + BLOCK_ROWS, side="right")),
                 n_docs)
        r1 = int(off[d1])
        blk = np.asarray(A[r0:r1], dtype=np.float32)
        blk /= np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-12)
        if proj is not None:
            blk = blk @ proj
        C = torch.from_numpy(blk.astype(np.float16)).to(dev)
        nb = d1 - d0
        csum = np.cumsum(dlens[d0:d1])
        starts = np.concatenate(([0], csum[:-1])).astype(np.int64)
        pending = []
        for c0 in range(0, len(qrows), CHUNK_QUERIES):
            idxs = range(c0, min(c0 + CHUNK_QUERIES, len(qrows)))
            sims = [qrows[i] @ C.T for i in idxs]
            sims = [s.cpu().numpy() for s in sims]  # drains this chunk's GEMMs
            pending.append(ex.submit(chunk_rows, sims, list(idxs), starts))
            if len(pending) > 2:
                qi, out = pending.pop(0).result()
                S[qi, d0:d1] = out
        while pending:
            qi, out = pending.pop(0).result()
            S[qi, d0:d1] = out
        print(f"  block docs {d0}-{d1}/{n_docs} ({time.time() - t0:.0f}s)")
        d0, r0 = d1, r1
        del C, blk
        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()
        if dev == "mps":
            torch.mps.empty_cache()
    ex.shutdown()
    return S


def build_trained(meta26: dict, rows: list[dict], ckpt: Path,
                  db26: Path | None = None, out: Path | None = None
                  ) -> None:
    """suffix+head over prefix rows -> f16 D=128 rows (default M4 paths)."""
    import torch
    from esm.models.esmc.tokenizer import EsmcTokenizer

    from esmc_li.embed import chunk_windows
    from esmc_li.train import Projection, Suffix, _esmc

    db26 = db26 or CACHE / "db26.npy"
    out = out or CACHE / "db_tr128.npy"
    progress = out.with_suffix(".progress.json")
    off, lens = meta26["offsets"], meta26["lens"]
    n = len(rows)
    done = 0
    if out.exists() and progress.exists():
        done = json.loads(progress.read_text())["done"]
        if done >= n:
            print(f"[skip] trained rows: {done}/{n}")
            return
        mm = np.lib.format.open_memmap(out, mode="r+")
        assert mm.shape == (off[-1], 128)
    else:
        CACHE.mkdir(exist_ok=True)
        mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                       shape=(off[-1], 128))

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    esmc, _ = _esmc()
    suffix = Suffix(esmc, ck["cut"]).to(dev).eval()
    suffix.load_state_dict(ck["suffix"])
    head = Projection(960, 128).to(dev).eval()
    head.load_state_dict(ck["head"])
    tok = EsmcTokenizer()

    def span_rows(seq: str) -> list[tuple[int, int]]:
        """(row_offset_in_prefix, n_rows) per window of seq."""
        out_, lo = [], 0
        for s, e in chunk_windows(seq):
            out_.append((lo, e - s + 2))
            lo += e - s + 2
        return out_

    A26 = arr(db26)
    t0 = time.time()
    with torch.no_grad():
        i = done
        while i < n:
            items: list[tuple[int, int, int]] = []  # (row, row_lo, n_rows)
            budget, ri = 0, i
            while ri < n:
                cost = lens[ri]
                if items and budget + cost > 8192:
                    break
                row_lo = span_rows(rows[ri]["sequence"])
                for lo, cnt in row_lo:
                    items.append((ri, off[ri] + lo, cnt))
                budget += cost
                ri += 1
            L = max(c for _, _, c in items)
            x = np.zeros((len(items), L, 960), dtype=np.float32)
            m = np.zeros((len(items), L), dtype=bool)
            for j, (_, lo, cnt) in enumerate(items):
                x[j, :cnt] = A26[lo : lo + cnt]
                m[j, :cnt] = True
            mt = torch.from_numpy(m).to(dev)
            h = suffix(torch.from_numpy(x).to(dev), mt)
            h = head(h).cpu().numpy()  # [B, L, 128], normalized rows
            for j, (_, lo, cnt) in enumerate(items):
                mm[lo : lo + cnt] = h[j, :cnt].astype(np.float16)
            i = ri
            progress.write_text(json.dumps({"done": i}))
            print(f"  trained {i}/{n} rows ({time.time() - t0:.0f}s)")
            del x, h
            if dev == "mps":
                torch.mps.empty_cache()
    mm.flush()
    del mm


def minhash_scores(qpos: list[int], rows: list[dict]) -> np.ndarray:
    from datasketch import MinHash

    p = CACHE / "minhash_sigs.npy"
    if p.exists():
        sigs = np.load(p)
    else:
        sigs = np.empty((len(rows), NUM_PERM), dtype=np.uint64)
        for i, r in enumerate(rows):
            m = MinHash(num_perm=NUM_PERM)
            seq = r["sequence"]
            for j in range(len(seq) - KMER_K + 1):
                m.update(seq[j : j + KMER_K].encode())
            sigs[i] = m.hashvalues
        np.save(p, sigs)
    s64 = sigs.astype(np.int64)
    S = np.empty((len(qpos), len(rows)), dtype=np.float32)
    for a in range(0, len(qpos), 50):
        chunk = s64[qpos[a : a + 50]]
        eq = (chunk[:, None, :] == s64[None, :, :]).sum(axis=2)
        S[a : a + 50] = eq / NUM_PERM
    return S


def mmseqs_rankings(qids: list[str], rows: list[dict]) -> dict[str, list[str]]:
    if not shutil.which("mmseqs"):
        raise RuntimeError("mmseqs not found (brew install mmseqs2)")
    hits_tsv = CACHE / "mmseqs_hits.tsv"
    if not hits_tsv.exists():
        with tempfile.TemporaryDirectory(prefix="m4mm_") as td:
            td = Path(td)
            qf, df = td / "q.fasta", td / "db.fasta"
            by_id = {r["id"]: r for r in rows}
            with qf.open("w") as fq, df.open("w") as fd:
                for r in rows:
                    fd.write(f">{r['id']}\n{r['sequence']}\n")
                for q in qids:
                    fq.write(f">{q}\n{by_id[q]['sequence']}\n")
            subprocess.run(
                ["mmseqs", "easy-search", str(qf), str(df), str(hits_tsv),
                 str(td / "tmp"), "-s", "7.5", "--max-seqs", "1000",
                 "-e", "10000000000", "--format-output",
                 "query,target,evalue", "-v", "1"],
                check=True, capture_output=True)
    from collections import defaultdict

    hits: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for line in hits_tsv.read_text().splitlines():
        p = line.split("\t")
        if len(p) < 3 or p[0] == p[1]:
            continue
        hits[p[0]].append((p[1], float(p[2])))
    return {q: [t for t, _ in sorted(hits.get(q, []), key=lambda x: x[1])]
            for q in qids}


def random_rankings(qids: list[str], ids: list[str], qpos: list[int]):
    rng = random.Random(0)
    out = {}
    for i, q in enumerate(qids):
        cand = ids[: qpos[i]] + ids[qpos[i] + 1 :]
        cand = list(cand)
        rng.shuffle(cand)
        out[q] = cand
    return out


# -- output ---------------------------------------------------------------------

def append_row(row: dict) -> None:
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        f.flush()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["cpu", "gpu", "all"], default="all",
                    help="cpu: pooled + baselines (no MPS); gpu: maxsim + "
                         "trained prep; all: everything")
    args = ap.parse_args()
    t_all = time.time()
    rows = list(csv.DictReader((FL / "db.tsv").open(newline=""), delimiter="\t"))
    qids = [r["id"] for r in
            csv.DictReader((FL / "queries.tsv").open(newline=""), delimiter="\t")]
    meta = json.loads((CACHE / "db.meta.json").read_text())
    ids = meta["ids"]
    by_id = {r["id"]: r for r in rows}
    rows = [by_id[i] for i in ids]  # rows must be in meta (embed) order
    id2i = {r: i for i, r in enumerate(ids)}
    qpos = [id2i[q] for q in qids]
    lens = np.array(meta["lens"], dtype=np.int64)
    clans = np.array([r["superfamily"] for r in rows])
    same_sf = clans[qpos][:, None] == clans[None, :]
    n, nq = len(rows), len(qids)
    print(f"db={n} queries={nq} (clans={len(set(clans[qpos]))}) stage={args.stage}")

    done_pairs = set()
    if OUT.exists():
        done_pairs = {(r["method"], r["normalization"])
                      for r in csv.DictReader(OUT.open())}

    def eval_S(S) -> dict:
        return evaluate(cross_rankings(S, qids, ids, qpos), rows)

    def emit(method: str, norm: str, S_scores: np.ndarray | None,
             rankings: dict | None, wall: float, r: float | None) -> None:
        if (method, norm) in done_pairs:
            print(f"[skip] {method} {norm}")
            return
        m = (evaluate(rankings, rows) if rankings is not None
             else eval_S(S_scores))
        row = {"method": method, "normalization": norm,
               "rho_len_score_nonhomolog": ("" if r is None else round(r, 4)),
               "n_db": n, "n_queries": nq, "wall_s": round(wall, 2),
               **{k: round(v, 4) for k, v in m.items()
                  if k in FIELDNAMES}}
        append_row(row)
        done_pairs.add((method, norm))
        print(row)
        log_run("T4.3", {"method": method, "normalization": norm,
                         "db": "swissprot_fulllength", "protocol": "loo",
                         "rho_len_score": row["rho_len_score_nonhomolog"]},
                m, seed=0, wall_s=wall)

    def emit_vector(name: str, S0: np.ndarray) -> None:
        S0 = np.asarray(S0, dtype=np.float32)
        r = spearman_len_score(S0, lens, same_sf)
        q_len = lens[qpos][:, None].astype(np.float64)
        d_len = lens[None, :].astype(np.float64)
        variants = {
            "raw": S0,
            "div_lenA": (S0 / q_len).astype(np.float32),
            "div_lenB": (S0 / d_len).astype(np.float32),
            "div_len_geomean": (S0 / np.sqrt(q_len * d_len)).astype(np.float32),
        }
        for vname, Sv in variants.items():
            t = time.time()
            emit(name, vname, Sv, None, time.time() - t, r)

    if args.stage in ("cpu", "all"):
        # non-vector baselines (raw only)
        t = time.time()
        emit("random", "raw", None, random_rankings(qids, ids, qpos),
             time.time() - t, None)
        t = time.time()
        emit("minhash5mer", "raw",
             get_matrix("minhash5mer", lambda: minhash_scores(qpos, rows)),
             None, time.time() - t, None)
        t = time.time()
        emit("mmseqs2_s7.5", "raw", None, mmseqs_rankings(qids, rows),
             time.time() - t, None)
        # pooled (CPU BLAS)
        S_pool = get_matrix("pooled", lambda: (
            lambda dm: (dm[qpos]
                        / np.linalg.norm(dm[qpos], axis=1, keepdims=True))
            @ (dm / np.linalg.norm(dm, axis=1, keepdims=True)).T
        )(doc_means(CACHE / "db.npy", meta)).astype(np.float32))
        emit_vector("pooled_esmc", S_pool)

    if args.stage in ("gpu", "all"):
        meta26 = json.loads((CACHE / "db26.meta.json").read_text())
        assert meta26["ids"] == ids, "db26 meta must match db meta"
        build_trained(meta26, rows, ROOT / "results" / "m3_models" / "k3_seed0.pt")
        emit_vector("maxsim960",
                    get_matrix("maxsim960", lambda: blockwise_maxsim(
                        CACHE / "db.npy", meta, qpos, None)))
        emit_vector("maxsim128",
                    get_matrix("maxsim128", lambda: blockwise_maxsim(
                        CACHE / "db.npy", meta, qpos,
                        random_projection(seed=0))))
        emit_vector("trained_m3",
                    get_matrix("trained_m3", lambda: blockwise_maxsim(
                        CACHE / "db_tr128.npy", meta26, qpos, None)))

    print(f"total {time.time() - t_all:.1f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
