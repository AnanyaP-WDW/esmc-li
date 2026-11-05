#!/usr/bin/env python
"""M5 T5.4: engineering baselines -> results/m5_baselines.csv.

Rows:
- T5.1/T5.2 search (cpu/mps) copied from results/m5_search.csv
- T5.3 fused-metal search copied from the latest T5.3 runs.jsonl entries
- FAISS IndexFlatIP on mean-pooled index rows (the latency floor), 10K+100K
- PyLate (PLAID late-interaction) on a 1K-doc subset; full-DB encode is
  not feasible on a laptop, so its encode rate is measured and extrapolated

Queries are the same leave-one-out corpus rows used by T5.1-3 (pooled and
MaxSim query vectors come from identical residues).
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
OUT = ROOT / "results" / "m5_baselines.csv"
K = 10
NQ = 50
N_PYLATE = 1_000
FIELDS = ["method", "n_db", "n_queries", "build_s", "index_mb",
          "search_s", "qps", "notes"]


def load_rows(nq: int = NQ):
    """(query doc rows f16, all docs mmap, offsets[:n+1 per call], query ids)"""
    meta = json.loads((CACHE / "db26.meta.json").read_text())
    off = np.asarray(meta["offsets"], dtype=np.int64)
    id2i = {r: i for i, r in enumerate(meta["ids"])}
    qids = [r["id"] for r in csv.DictReader(
        (ROOT / "data" / "fulllength" / "queries.tsv").open(newline=""),
        delimiter="\t")]
    qpos = [id2i[q] for q in qids][:nq]
    A = np.load(CACHE / "db_tr128.npy", mmap_mode="r")
    return A, off, qpos


def faiss_rows(rows_out: list) -> None:
    import faiss

    A, off, qpos = load_rows()
    # pooled queries: mean-pool the same query doc rows (leave-one-out)
    qvec = []
    for p in qpos:
        x = np.asarray(A[off[p] : off[p + 1]], dtype=np.float32)
        x = x.mean(axis=0)
        qvec.append(x / max(float(np.linalg.norm(x)), 1e-12))
    qvec = np.ascontiguousarray(np.vstack(qvec))
    for n in (10_000, 100_000):
        t = time.perf_counter()
        vecs = np.empty((n, A.shape[1]), dtype=np.float32)
        for d in range(n):
            x = np.asarray(A[off[d] : off[d + 1]], dtype=np.float32)
            vecs[d] = x.mean(axis=0)
        norms = np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        vecs /= norms
        index = faiss.IndexFlatIP(A.shape[1])
        index.add(vecs)
        build = time.perf_counter() - t
        index.search(qvec[:1], K)  # warm
        t = time.perf_counter()
        _, I = index.search(qvec, K)
        search = time.perf_counter() - t
        row = {"method": "faiss_flat_pooled", "n_db": n, "n_queries": NQ,
               "build_s": round(build, 2),
               "index_mb": round(index.ntotal * index.d * 4 / 1e6, 1),
               "search_s": round(search, 3),
               "qps": round(NQ / search, 1),
               "notes": "IndexFlatIP on mean-pooled f32 rows"}
        print(f"  {row}")
        rows_out.append(row)
        del vecs, index


def copy_engine_rows(rows_out: list) -> None:
    with (ROOT / "results" / "m5_search.csv").open() as f:
        for r in csv.DictReader(f):
            rows_out.append({
                "method": f"maxsim_{r['device']}", "n_db": int(r["n_db"]),
                "n_queries": int(r["n_queries"]), "build_s": 0.0,
                "index_mb": round(int(r["n_rows"]) * 128 * 2 / 1e6, 1),
                "search_s": r["wall_s"], "qps": r["qps"],
                "notes": f"T5.1/T5.2, efficiency {r['efficiency_pct']}%"})
    metal = {}
    for line in (ROOT / "results" / "runs.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("task") == "T5.3" and rec["config"].get("n_db"):
            metal[int(rec["config"]["n_db"])] = rec
    for n, rec in sorted(metal.items()):
        m = rec["metrics"]
        rows_out.append({
            "method": "maxsim_metal", "n_db": n, "n_queries": m["n_queries"],
            "build_s": 0.0, "index_mb": m.get("index_mb", round(
                rec["metrics"].get("n_rows", 34054457 if n == 100000
                                   else 8875335) * 128 * 2 / 1e6, 1)),
            "search_s": m["metal_wall_s"], "qps": m["metal_qps"],
            "notes": "T5.3 fused kernel; same-run mps %.1fs"
                     % m["mps_wall_s"]})


ROW_TMP = ROOT / "results" / "_m5_pylate_row.json"


def pylate_main() -> int:
    """Child process: faiss and torch each bundle libomp — together they
    deadlock at an OpenMP fork barrier (0% CPU, all threads parked), so
    PyLate runs here alone and drops its row to ROW_TMP."""
    import os
    rows_out: list = []
    try:
        from pylate import indexes as pl_indexes
        from pylate.models import ColBERT
        from pylate.retrieve import ColBERT as Retriever
    except Exception as e:  # pragma: no cover
        ROW_TMP.write_text(json.dumps(
            {"method": "pylate_plaid", "n_db": 0, "n_queries": 0,
             "build_s": 0.0, "index_mb": 0.0, "search_s": 0.0, "qps": 0.0,
             "notes": f"import failed: {e}"}))
        return 0
    seqs, doc_ids = {}, []
    with (ROOT / "data" / "fulllength" / "db.tsv").open(newline="") as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            if i >= N_PYLATE:
                break
            seqs[r["id"]] = r["sequence"]
            doc_ids.append(r["id"])
    meta = json.loads((CACHE / "db26.meta.json").read_text())
    id2i = {x: i for i, x in enumerate(meta["ids"])}
    qids = [r["id"] for r in csv.DictReader(
        (ROOT / "data" / "fulllength" / "queries.tsv").open(newline=""),
        delimiter="\t")][:NQ]
    qseqs = []
    qdb = np.load(CACHE / "db_tr128.npy", mmap_mode="r")
    qoff = np.asarray(meta["offsets"], dtype=np.int64)
    for q in qids:
        p = id2i[q]
        # sequence for query rows: pull from db.tsv index position via ids
        qseqs.append(q)  # placeholder replaced below
    # sequences keyed by the same ids exist in db.tsv only for db rows;
    # queries.tsv has its own sequences:
    with (ROOT / "data" / "fulllength" / "queries.tsv").open(newline="") as f:
        qseqs = [r["sequence"] for r in csv.DictReader(f, delimiter="\t")][:NQ]
    del qdb, qoff
    docs = list(seqs.values())
    print(f"  pylate: encoding {len(docs)} docs + {len(qseqs)} queries ...",
          flush=True)
    t = time.perf_counter()
    model = ColBERT("colbert-ir/colbertv2.0")
    enc_doc = time.perf_counter() - t
    t = time.perf_counter()
    docs_emb = model.encode(docs, batch_size=16, show_progress_bar=True)
    enc_docs = time.perf_counter() - t
    t = time.perf_counter()
    qs_emb = model.encode(qseqs, batch_size=16, show_progress_bar=False)
    enc_q = time.perf_counter() - t
    t = time.perf_counter()
    idx = pl_indexes.PLAID(index_folder="/tmp/m5_plaid", index_name="m5",
                           override=True, show_progress=True)
    idx.add_documents(doc_ids, docs_emb)
    build = time.perf_counter() - t
    rtr = Retriever(idx)
    rtr.retrieve(qs_emb[:1], k=K)  # warm
    t = time.perf_counter()
    hits = rtr.retrieve(qs_emb, k=K)
    search = time.perf_counter() - t
    rows_out.append({
        "method": "pylate_plaid", "n_db": len(docs), "n_queries": NQ,
        "build_s": round(build, 2),
        "index_mb": round(sum(len(e) for e in docs_emb)
                          * 128 * 4 / 1e6, 1),
        "search_s": round(search, 3), "qps": round(NQ / search, 1),
        "notes": (f"colbertv2 encode {enc_docs:.0f}s docs "
                  f"({len(docs)/enc_docs:.1f} docs/s) + {enc_q:.1f}s queries; "
                  f"1K subset, PLAID build incl.")})
    print(f"  {rows_out[-1]}", flush=True)
    ROW_TMP.write_text(json.dumps(rows_out[-1]))
    return 0


def main() -> int:
    from esmc_li.runlog import log_run

    if len(sys.argv) > 1 and sys.argv[1] == "--pylate":
        return pylate_main()
    rows: list = []
    copy_engine_rows(rows)
    faiss_rows(rows)
    if "--no-pylate" not in sys.argv:
        import subprocess
        r = subprocess.run([sys.executable, __file__, "--pylate"], cwd=ROOT)
        if r.returncode == 0 and ROW_TMP.exists():
            rows.append(json.loads(ROW_TMP.read_text()))
        else:
            rows.append({"method": "pylate_plaid", "n_db": 0, "n_queries": 0,
                         "build_s": 0.0, "index_mb": 0.0, "search_s": 0.0,
                         "qps": 0.0, "notes": f"failed rc={r.returncode}"})
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    log_run("T5.4", {"k": K, "NQ": NQ, "note": "engineering baselines"},
            {"rows": len(rows), "file": str(OUT.name)}, seed=0)
    print(f"-> {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
