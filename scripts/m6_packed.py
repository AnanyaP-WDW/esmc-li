#!/usr/bin/env python3
"""M6 T6.1: packed single-file embedding exit check.

Per-file path (exit criterion): esmc-embed -> one .npy per window ->
assemble -> f16 -> `cache/m6/t6_10k.npy` (10K proteins, one file).
Packed reference: `cache/m4/db.npy` (embed_dataset, also one file, 100K
proteins) sliced to the same ids.

Exit: 10K proteins -> one file; values equal the per-file .npy path after
f16 cast (byte-exact, chunked compare against the mmap).

T6.1's plan artifact (esmc.cpp packed-output CLI flag) is deliberately not
built: embed_dataset already packs on the Python side, into one file, and
esmc-embed's per-file npys only ever live in a per-batch temp dir.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esmc_retrieval.embed import D_MODEL, assemble, chunk_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = 10_000
OUT = ROOT / "cache" / "m6" / "t6_10k.npy"


def main() -> int:
    from esmc_retrieval.runlog import log_run

    t0 = time.time()
    man = json.loads((ROOT / "data" / "manifest.json").read_text())
    binary = Path(man["esmc_cpp"]["esmc_embed_binary"])
    gguf = Path(man["esmc_cpp"]["gguf"]["path"])
    rows = list(csv.DictReader(
        (ROOT / "data" / "fulllength" / "db.tsv").open(newline=""),
        delimiter="\t"))
    rows.sort(key=lambda r: (-len(r["sequence"]), r["id"]))
    rows = rows[:N]
    lens = [len(r["sequence"]) for r in rows]
    total = sum(lens)
    print(f"{N} proteins, {total} residues, "
          f"{sum(1 for r in lens if r > 2046)} chains >2046", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    have = (OUT.exists()
            and np.load(OUT, mmap_mode="r").shape == (total, D_MODEL))
    embed_s = 0.0
    if have:
        print(f"[skip] {OUT.name} already packed", flush=True)
    if not have:
      mm = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float16,
                                   shape=(total, D_MODEL))
      with tempfile.TemporaryDirectory(prefix="m6p_") as td:
        td = Path(td)
        fa = td / "batch.fa"
        with fa.open("w") as f:
            for r in rows:
                seq = r["sequence"]
                for s, e in chunk_windows(seq):
                    if s == 0 and e == len(seq):
                        f.write(f">{r['id']}\n{seq}\n")
                    else:
                        f.write(f">{r['id']}|{s}\n{seq[s:e]}\n")
        npy_dir = td / "npy"
        npy_dir.mkdir()
        t = time.perf_counter()
        subprocess.run([str(binary), "-m", str(gguf), "--fasta", str(fa),
                        "--output-dir", str(npy_dir)], check=True,
                       capture_output=True)
        embed_s = time.perf_counter() - t
        lo = 0
        for r in rows:
            seq = r["sequence"]
            spans = chunk_windows(seq)
            if len(spans) == 1:
                part = np.load(npy_dir / f"{r['id']}.npy")
            else:
                parts = [np.load(npy_dir / f"{r['id']}|{s}.npy")
                         for s, _ in spans]
                part = assemble(spans, parts)
            assert part.shape == (len(seq), D_MODEL)
            mm[lo : lo + len(seq)] = part.astype(np.float16)
            lo += len(seq)
        assert lo == total
        mm.flush()
        del mm

    ref = np.load(ROOT / "cache" / "m4" / "db.npy", mmap_mode="r")
    got = np.load(OUT, mmap_mode="r")
    assert got.shape == (total, D_MODEL)
    CH = 500_000
    exact, max_abs = True, 0.0
    for i in range(0, total, CH):
        hi = min(i + CH, total)
        a = np.asarray(got[i:hi])
        b = np.asarray(ref[i:hi])
        if not np.array_equal(a, b):
            exact = False
            max_abs = max(max_abs, float(np.abs(
                a.astype(np.float32) - b.astype(np.float32)).max()))
    one_file = sum(1 for p in OUT.parent.glob("*.npy"))
    size = OUT.stat().st_size
    wall = time.time() - t0
    metrics = {"proteins": N, "residues": total, "bytes": size,
               "expected_bytes": total * D_MODEL * 2 + 128,
               "embed_s": round(embed_s, 1), "npy_files_in_outdir": one_file,
               "exact_vs_packed_db": exact,
               "max_abs_diff": max_abs,
               "shape": list(got.shape)}
    log_run("T6.1", {"n": N, "out": str(OUT.relative_to(ROOT)),
                     "reference": "cache/m4/db.npy (embed_dataset)"},
            metrics, wall_s=wall)
    ok = exact and one_file == 1
    print(f"{'PASS' if ok else 'FAIL'}: one_file={one_file} "
          f"exact={exact} max_abs_diff={max_abs} "
          f"size={size}B (header+off={size - total * D_MODEL * 2}B) "
          f"embed={embed_s:.1f}s total={wall:.1f}s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
