#!/usr/bin/env python3
"""M8 T8.1 smoke: build a 1K SCOPe index and search 10 held-out homologs.

The DB is the first N rows of scope_test; queries are rows outside the DB that
have a same-family member in the DB. Exit 0 iff every query's top-1 hit is in
the query's superfamily (the headline homolog level; family is stricter).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from esmc_li import ProteinIndex  # noqa: E402
from esmc_li.runlog import log_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits/scope_test.tsv")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--output", default="cache/m8/smoke1k.esmci")
    ap.add_argument("--model",
                    default="/Users/ananyapathak/Desktop/oss/esmc.cpp/"
                            "models/esmc-300m-ft-f16.gguf")
    ap.add_argument("--embed-bin",
                    default="/Users/ananyapathak/Desktop/oss/esmc.cpp/"
                            "build/esmc-embed")
    args = ap.parse_args()
    t0 = time.time()

    rows = list(csv.DictReader(
        (ROOT / args.split).open(newline=""), delimiter="\t"))
    if len(rows) <= args.n:
        raise SystemExit("split is too small for DB + queries")
    db_rows = rows[: args.n]
    db_families = {r["family"] for r in db_rows}
    queries = [r for r in rows[args.n:] if r["family"] in db_families]
    if len(queries) < args.queries:
        raise SystemExit(f"only {len(queries)} held-out same-family queries")
    queries = queries[: args.queries]
    for q in queries:
        q["positive"] = next(r["id"] for r in db_rows
                             if r["family"] == q["family"])

    idx = ProteinIndex.build(
        db_rows, ROOT / args.output, model=args.model,
        embed_bin=args.embed_bin,
    )
    hits = idx.search_batch([q["sequence"] for q in queries], k=1)
    out = []
    ok = True
    for q, h in zip(queries, hits):
        top = h[0]
        same = top["metadata"].get("superfamily") == q["superfamily"]
        ok &= same
        out.append({"query": q["id"], "family": q["family"],
                    "superfamily": q["superfamily"],
                    "positive": q["positive"], "top1": top["id"],
                    "top1_family": top["metadata"].get("family"),
                    "top1_superfamily": top["metadata"].get("superfamily"),
                    "score": round(top["score"], 4),
                    "same_superfamily": same})
    log_run("T8.1", {"split": args.split, "n_db": args.n,
                     "n_queries": len(queries), "engine": "numpy",
                     "keep_special": True},
            {"n": len(idx), "index_bytes": idx.nbytes,
             "top1_superfamily": sum(r["same_superfamily"] for r in out),
             "top1_family": sum(r["top1_family"] == r["family"]
                                for r in out)},
            model_hash=idx._idx.meta.get("model_sha256"),
            wall_s=time.time() - t0)
    print(json.dumps(out, indent=2))
    print(f"PASS {sum(r['same_superfamily'] for r in out)}/{len(out)}"
          if ok else f"FAIL {sum(r['same_superfamily'] for r in out)}/{len(out)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
