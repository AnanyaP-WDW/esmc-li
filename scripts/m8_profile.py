#!/usr/bin/env python3
"""M8 profile: split query encode vs numpy MaxSim before any Rust work."""
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
    ap.add_argument("--index", required=True)
    ap.add_argument("--query-tsv", required=True)
    ap.add_argument("--nq", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", required=True)
    ap.add_argument("--embed-bin", required=True)
    args = ap.parse_args()

    idx = ProteinIndex.load(args.index, model=args.model,
                            embed_bin=args.embed_bin)
    q = list(csv.DictReader((ROOT / args.query_tsv).open(newline=""),
                            delimiter="\t"))[: args.nq]
    seqs = [r["sequence"] for r in q]
    t0 = time.perf_counter()
    qs = idx._encode(seqs)
    encode_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    idx._search(qs, args.k)
    search_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    idx.search(seqs[0], k=args.k)
    single_s = time.perf_counter() - t0

    out = {
        "index": args.index, "n": len(idx), "nq": len(seqs),
        "encode_s": round(encode_s, 4),
        "encode_ms_per_q": round(1000 * encode_s / len(seqs), 3),
        "search_s": round(search_s, 4),
        "search_ms_per_q": round(1000 * search_s / len(seqs), 3),
        "qps": round(len(seqs) / (encode_s + search_s), 3),
        "single_s": round(single_s, 4),
        "device": idx.device,
        "model_hash": idx._idx.meta.get("model_sha256"),
    }
    log_run("T8.profile", {"index": args.index, "nq": len(seqs),
                           "k": args.k, "device": idx.device},
            {k: out[k] for k in ("n", "encode_ms_per_q", "search_ms_per_q",
                                 "qps", "single_s")},
            model_hash=out["model_hash"], wall_s=encode_s + search_s)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
