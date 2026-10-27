#!/usr/bin/env python3
"""M7 T7.1 — compressed-index quality (results/m7_compression.csv).

Schemes quantize cache/m4/db_tr128.npy; decoded f16 rows go through the
frozen pathway (m4_eval.blockwise_maxsim -> cross_rankings -> evaluate()).
Baselines: `frozen` = S_trained_m3.npy (M4 headline, cR@10 0.8974) and
`fp16` = blockwise on the raw rows (validates the pathway before attributing
any drop to quantization).

Exit: cR@10 drop <= 0.01 at >= 4x size reduction vs the f16 payload,
else a documented negative result.
"""
from __future__ import annotations

import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from esmc_retrieval.compress import (decode_int8, decode_res_int8,  # noqa: E402
                                     decode_res_pq, encode_int8,
                                     encode_res_int8, encode_res_pq,
                                     scheme_bytes)
from esmc_retrieval.eval import evaluate  # noqa: E402
from esmc_retrieval.runlog import log_run  # noqa: E402

CACHE = ROOT / "cache" / "m4"
SRC = CACHE / "db_tr128.npy"
OUT = ROOT / "results" / "m7_compression.csv"
DEFAULT_SCHEMES = ["fp16", "int8", "residual_int8", "residual_pq_m8",
                   "residual_pq_m16"]
SCHEMES = sys.argv[1:] or DEFAULT_SCHEMES  # extra: pass scheme names
FIELDS = ["scheme", "bytes", "ratio_vs_f16", "cR@1", "cR@5", "cR@10",
          "cR@100", "mrr", "drop_cR@10", "pass", "encode_s", "decode_s",
          "search_s"]
CH = 1_000_000


def decode_chunked(decode_fn, n: int, d: int) -> np.ndarray:
    out = np.empty((n, d), dtype=np.float16)
    for s in range(0, n, CH):
        out[s : s + CH] = decode_fn(s, min(s + CH, n)).astype(np.float16)
    return out


def main() -> int:
    import m4_eval

    rows = list(csv.DictReader(
        (ROOT / "data" / "fulllength" / "db.tsv").open(newline=""),
        delimiter="\t"))
    qids = [r["id"] for r in csv.DictReader(
        (ROOT / "data" / "fulllength" / "queries.tsv").open(newline=""),
        delimiter="\t")]
    meta = json.loads((CACHE / "db26.meta.json").read_text())
    ids = meta["ids"]
    by_id = {r["id"]: r for r in rows}
    rows = [by_id[i] for i in ids]
    id2i = {r: i for i, r in enumerate(ids)}
    qpos = [id2i[q] for q in qids]
    src = np.load(SRC, mmap_mode="r")
    n, d = src.shape
    base_bytes = n * d * 2

    def eval_S(S) -> dict:
        return evaluate(m4_eval.cross_rankings(S, qids, ids, qpos), rows)

    ref = eval_S(np.load(CACHE / "S_trained_m3.npy"))
    print(f"frozen ref: cR@10={ref['cR@10']:.4f} mrr={ref['mrr']:.4f}",
          flush=True)

    out_rows = []
    for scheme in SCHEMES:
        print(f"== {scheme}", flush=True)
        t0 = time.time()
        codes = aux = None
        encode_s = decode_s = 0.0
        if scheme == "fp16":
            arr = src
            nbytes = base_bytes
        else:
            t = time.time()
            if scheme == "int8":
                codes, scale = encode_int8(src)
                nbytes = scheme_bytes(scheme, n, d)
                decode_fn = lambda a, b: decode_int8(  # noqa: E731
                    codes[a:b], scale)
            elif scheme == "residual_int8":
                codes, mu, scale = encode_res_int8(src)
                nbytes = scheme_bytes(scheme, n, d,
                                      aux=(d * 4 + d * 4))
                decode_fn = lambda a, b: decode_res_int8(  # noqa: E731
                    codes[a:b], mu, scale)
            else:
                m = int(scheme.rsplit("m", 1)[1])
                q = encode_res_pq(src, m=m, seed=0)
                nbytes = scheme_bytes(scheme, n, d)
                decode_fn = lambda a, b: decode_res_pq({**q,  # noqa: E731
                    "coarse": q["coarse"][a:b], "sub": q["sub"][a:b]})
            encode_s = time.time() - t
            t = time.time()
            arr = decode_chunked(decode_fn, n, d)
            decode_s = time.time() - t
            del codes
            if scheme.startswith("residual_pq"):
                del q
        print(f"  encode {encode_s:.0f}s decode {decode_s:.0f}s "
              f"bytes={nbytes:,}", flush=True)

        path = (SRC if scheme == "fp16"
                else CACHE / f"quant_{scheme}.npy")
        m4_eval._db_open[str(path)] = arr  # blockwise reads from here
        t = time.time()
        S = m4_eval.blockwise_maxsim(path, meta, qpos, None)
        search_s = time.time() - t
        m = eval_S(S)
        nbytes = base_bytes if scheme == "fp16" else nbytes
        ratio = base_bytes / nbytes
        drop = ref["cR@10"] - m["cR@10"]
        ok = drop <= 0.01 and ratio >= 4
        row = {"scheme": scheme, "bytes": nbytes,
               "ratio_vs_f16": round(ratio, 2),
               **{k: round(m[k], 4) for k in
                  ("cR@1", "cR@5", "cR@10", "cR@100", "mrr")},
               "drop_cR@10": round(drop, 4),
               "pass": ok, "encode_s": round(encode_s, 1),
               "decode_s": round(decode_s, 1),
               "search_s": round(search_s, 1)}
        print(f"  {row}", flush=True)
        out_rows.append(row)
        m4_eval._db_open.pop(str(path), None)  # blockwise's array cache
        del S, arr
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    OUT.parent.mkdir(exist_ok=True)
    old = ([r for r in csv.DictReader(OUT.open())
            if r["scheme"] not in SCHEMES] if OUT.exists() else [])
    rows_out = old + out_rows
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows_out)
    metrics = {f"{r['scheme']}_{k}": r[k] for r in rows_out
               for k in ("cR@10", "drop_cR@10", "ratio_vs_f16")}
    metrics["any_pass"] = any(r["pass"] for r in rows_out
                              if r["scheme"] != "fp16")
    metrics["frozen_cR@10"] = ref["cR@10"]
    log_run("T7.1", {"db": "db_tr128", "schemes": [r["scheme"]
                     for r in rows_out],
                     "harness": "frozen evaluate()"}, metrics, seed=0)
    print(f"-> {OUT}")
    for r in rows_out:
        tag = "PASS" if r["pass"] else ("ref" if r["scheme"] == "fp16"
                                        else "fail")
        print(f"  {r['scheme']:18s} {r['ratio_vs_f16']:>6.1f}x  "
              f"cR@10 {r['cR@10']:.4f}  drop {r['drop_cR@10']:+.4f}  {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
