#!/usr/bin/env python3
"""Prototype EC annotation CLI for an .esmci index + 960->128 head.

Query sequences are embedded with esmc-embed (base ESM-C), projected by the
head, and searched with exact MaxSim.  The output is top-k neighbours plus a
score-weighted EC4 consensus and confidence; this is a retrieval/evidence tool,
not a calibrated classifier.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from esmc_li.index import Esmci  # noqa: E402
from esmc_li.search import maxsim_topk_numpy  # noqa: E402

DEFAULT_MODEL = Path("/Users/ananyapathak/Desktop/oss/esmc.cpp/models/esmc-300m-f16.gguf")
DEFAULT_BIN = Path("/Users/ananyapathak/Desktop/oss/esmc.cpp/build/esmc-embed")


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def read_fasta(path: Path) -> list[dict]:
    rows, rid, chunks = [], None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if rid is not None:
                rows.append({"id": rid, "sequence": "".join(chunks).upper()})
            rid, chunks = line[1:].split()[0], []
        else:
            chunks.append(line)
    if rid is not None:
        rows.append({"id": rid, "sequence": "".join(chunks).upper()})
    return rows


def embed(rows, model, binary):
    with tempfile.TemporaryDirectory(prefix="ecann_") as td:
        td = Path(td)
        fa = td / "q.fa"
        with fa.open("w") as f:
            for r in rows:
                f.write(f">{r['id']}\n{r['sequence']}\n")
        out = td / "npy"
        out.mkdir()
        subprocess.run([str(binary), "-m", str(model), "--fasta", str(fa),
                        "--output-dir", str(out)], check=True,
                       capture_output=True)
        return [np.load(out / f"{r['id']}.npy") for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--query", type=Path, default=None)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--embed-bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if bool(args.sequence) == bool(args.query):
        raise SystemExit("pass exactly one of --sequence or --query")
    rows = ([{"id": "q0", "sequence": args.sequence}] if args.sequence
            else read_fasta(args.query))
    idx = Esmci(args.index)
    head = np.load(args.head).astype(np.float32)
    embs = embed(rows, args.model, args.embed_bin)
    out = []
    for r, x in zip(rows, embs):
        q = l2(x.astype(np.float32) @ head.T)
        s, ids = maxsim_topk_numpy(q, idx.vecs, idx.offsets, args.k)
        ec4s = idx.meta.get("ec4") or [m.get("ec4", "")
                                       for m in idx.meta.get("metadata", [])]
        vote = defaultdict(float)
        hits = []
        for score, j in zip(s, ids):
            j = int(j)
            ec = (ec4s[j] or "").split(";")[0]
            hits.append({"id": idx.ids[j], "ec4": ec, "score": float(score)})
            if ec:
                vote[ec] += float(score)
        total = sum(vote.values()) or 1.0
        ranked = sorted(vote.items(), key=lambda kv: -kv[1])
        out.append({
            "query": r["id"],
            "ec4": ranked[0][0] if ranked else "",
            "confidence": ranked[0][1] / total if ranked else 0.0,
            "top": hits,
        })
    text = json.dumps(out, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
