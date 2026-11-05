#!/usr/bin/env python3
"""M6 T6.2/T6.3 — build the real .esmci and verify its exits.

T6.2: round-trip (tests/test_esmci.py) + 100K index opens in <1 s +
search results identical to the in-memory (.npy) path.
T6.3: file size == header + u64 offsets + n_res x D x 2 B (±1%),
i.e. built with the projected D=128 rows, not 960.

Source: cache/m4/db_tr128.npy + db26.meta.json (projected at build by
scripts/m4_eval.py:build_trained). -> cache/m4/db_tr128.esmci (+ .json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esmc_li.index import Esmci, expected_bytes, write  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cache" / "m4" / "db_tr128.npy"
META = ROOT / "cache" / "m4" / "db26.meta.json"
OUT = ROOT / "cache" / "m4" / "db_tr128.esmci"
K = 10
NQ = 50


def main() -> int:
    from esmc_li.runlog import log_run
    from esmc_li.search import maxsim_topk

    meta = json.loads(META.read_text())
    src = np.load(SRC, mmap_mode="r")
    off_u64 = np.asarray(meta["offsets"], dtype=np.uint64)
    assert int(off_u64[-1]) == src.shape[0]

    if not OUT.exists():
        t0 = time.perf_counter()
        stage = OUT.with_name(OUT.stem + ".stage")
        tmp = np.lib.format.open_memmap(stage, mode="w+", dtype=np.float16,
                                        shape=src.shape)
        CH = 2_000_000
        for i in range(0, src.shape[0], CH):
            tmp[i : i + CH] = src[i : i + CH]
        tmp.flush()
        del tmp
        write(OUT, np.load(stage, mmap_mode="r"), off_u64,
              {"ids": meta["ids"], "source": "db_tr128.npy", "d": 128,
               "dtype": "float16", "built": time.strftime("%Y-%m-%d")})
        stage.unlink()
        print(f"built {OUT.name} in {time.perf_counter() - t0:.1f}s")

    t = time.perf_counter()
    idx = Esmci(OUT)
    open_s = time.perf_counter() - t
    actual = OUT.stat().st_size
    expect = expected_bytes(idx.n_proteins, idx.n_residues, idx.d)
    dev_pct = 100.0 * (actual - expect) / expect
    print(f"open: {open_s * 1000:.1f} ms  n={idx.n_proteins} "
          f"rows={idx.n_residues} d={idx.d} size={actual}B "
          f"(expected {expect}B, {dev_pct:+.3f}%)")

    # T6.2 search identity: .npy in-memory vs .esmci mmap, same query
    id2i = {r: i for i, r in enumerate(meta["ids"])}
    import csv
    qids = [r["id"] for r in csv.DictReader(
        (ROOT / "data" / "fulllength" / "queries.tsv").open(newline=""),
        delimiter="\t")]
    qpos = [id2i[q] for q in qids][:NQ]
    q = [np.asarray(src[off_u64[p] : off_u64[p + 1]]) for p in qpos]
    t = time.perf_counter()
    s_a, i_a = maxsim_topk(q, src, off_u64, K, device="cpu",
                           normalize=False)
    s_b, i_b = maxsim_topk(q, idx.vecs, idx.offsets, K, device="cpu",
                           normalize=False)
    same = (np.array_equal(i_a, i_b) and np.array_equal(s_a, s_b))
    print(f"search identity (cpu, {NQ}q, k={K}): "
          f"{'IDENTICAL' if same else 'MISMATCH'} "
          f"({time.perf_counter() - t:.1f}s)")

    m = {"open_s": round(open_s, 4), "n_proteins": idx.n_proteins,
         "n_residues": idx.n_residues, "d": idx.d, "bytes": actual,
         "expected_bytes": expect, "size_dev_pct": round(dev_pct, 4),
         "search_identical": bool(same), "n_queries": NQ, "k": K}
    log_run("T6.2", {"file": str(OUT.relative_to(ROOT)),
                     "check": "open + search identity vs npy"}, m)
    log_run("T6.3", {"file": str(OUT.relative_to(ROOT)),
                     "projected_d": 128, "source_d": 960},
            {"bytes": actual, "expected_bytes": expect,
             "size_dev_pct": round(dev_pct, 4)})
    ok = (open_s < 1.0) and same and abs(dev_pct) <= 1.0
    print(f"{'PASS' if ok else 'FAIL'}: T6.2 open<1s={open_s < 1.0} "
          f"identical={same}; T6.3 size±1%={abs(dev_pct) <= 1.0}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
