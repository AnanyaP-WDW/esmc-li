#!/usr/bin/env python3
"""M7 T7.3 build — fused default index for UniRef50 subsets.

  --validate   run the fused builder on M4's first 100 chains and compare
               against cache/m4/db_tr128.npy (must match before any long run)
  --size N     data/raw/uniref50/db_N.tsv -> prefix+suffix+head fused ->
               cache/m7/db{N}.npy (+meta) -> db{N}.esmci; logs T7.3 build
               stages (walls) per size. Rows are length-sorted for batching;
               output ids follow that order.
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
sys.path.insert(0, str(ROOT))

from esmc_retrieval.build import build_default_index  # noqa: E402
from esmc_retrieval.index import write  # noqa: E402

M7 = ROOT / "cache" / "m7"
CKPT = ROOT / "results" / "m3_models" / "k3_seed0.pt"


def load_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline=""), delimiter="\t"))


def to_esmci(npy: Path, meta_path: Path, esmci_path: Path) -> float:
    t = time.time()
    meta = json.loads(meta_path.read_text())
    src = np.load(npy, mmap_mode="r")
    off = np.asarray(meta["offsets"], dtype=np.uint64)
    assert int(off[-1]) == src.shape[0]
    write(esmci_path, src, off,
          {"ids": meta["ids"], "d": 128, "dtype": "float16",
           "source": npy.name})
    return time.time() - t


def validate() -> int:
    rows = list(csv.DictReader(
        (ROOT / "data" / "fulllength" / "db.tsv").open(newline=""),
        delimiter="\t"))
    rows.sort(key=lambda r: (-len(r["sequence"]), r["id"]))
    rows = rows[:100]
    out_npy, meta = M7 / "validate.npy", M7 / "validate.meta.json"
    # build_default_index itself resumes/skips (progress json) — no outer skip
    build_default_index(rows, out_npy, meta, CKPT, token_budget=4096)
    ref = np.load(ROOT / "cache" / "m4" / "db_tr128.npy", mmap_mode="r")
    ref_off = json.loads(
        (ROOT / "cache" / "m4" / "db26.meta.json").read_text())["offsets"]
    got = np.load(out_npy, mmap_mode="r")
    hi = ref_off[100]
    assert got.shape[0] == hi, f"rows {got.shape[0]} != M4 prefix {hi}"
    g = np.asarray(got[:hi], dtype=np.float32)
    r = np.asarray(ref[:hi], dtype=np.float32)
    diff = np.abs(g - r)
    rel = diff.max() / max(float(np.abs(r).max()), 1e-9)
    identical = np.array_equal(g, r)
    nrm = np.linalg.norm(g, axis=1)
    ok = rel < 0.02 and np.all(np.abs(nrm - 1) < 0.01)
    print(f"validate: max_abs={diff.max():.5f} rel={rel:.5f} "
          f"bitwise={identical} rownorm in [1±0.01]: "
          f"{bool(np.all(np.abs(nrm - 1) < 0.01))} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, choices=[10_000, 100_000, 1_000_000])
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        return validate()
    if not args.size:
        ap.error("--size or --validate")
    from esmc_retrieval.runlog import log_run

    N = args.size
    rows = load_rows(ROOT / "data" / "raw" / "uniref50" / f"db_{N}.tsv")
    assert len(rows) == N
    # Length-sorted processing minimizes padding waste; the index order is
    # the sorted order and queries are mapped by id in m7_measure.py.
    rows.sort(key=lambda r: (-len(r["sequence"]), r["id"]))
    out_npy, meta = M7 / f"db{N}.npy", M7 / f"db{N}.meta.json"
    M7.mkdir(exist_ok=True)

    t0 = time.time()
    build_default_index(rows, out_npy, meta, CKPT)
    embed_s = time.time() - t0
    t = time.time()
    esmci_s = to_esmci(out_npy, meta, M7 / f"db{N}.esmci")
    size = (M7 / f"db{N}.esmci").stat().st_size
    residues = json.loads(meta.read_text())["offsets"][-1]
    m = {"n_proteins": N, "residues": residues,
         "embed_project_s": round(embed_s, 1),
         "esmci_s": round(esmci_s, 1), "build_s": round(embed_s + esmci_s, 1),
         "esmci_bytes": size,
         "prot_s": round(N / (embed_s + esmci_s), 3),
         "res_s": round(residues / (embed_s + esmci_s), 1)}
    log_run("T7.3", {"size": N, "stage": "build",
                     "corpus": "uniref50_seed0", "path": f"cache/m7/db{N}"},
            m, seed=0, wall_s=embed_s + esmci_s)
    print(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
