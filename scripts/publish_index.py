#!/usr/bin/env python3
"""Sanitize an .esmci + head for public HF dataset upload.

Removes build-machine paths, adds model/head hashes and public metadata, and
writes a self-contained upload folder:

  <name>.esmci
  <name>.esmci.json
  <name>.esmci.head.npy
  SHA256SUMS
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from esmc_retrieval.index import Esmci  # noqa: E402
from esmc_retrieval.protein_index import DEFAULT_HEAD  # noqa: E402

MODEL_NAME = "esmc-300m-homology-maxsim-v1-f16.gguf"
MODEL_SHA = "e87b42f1a6344e57b7a0d8a6624cc9d7e2e15f34d4a6927447b4cb3c90209bab"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_metadata(path: Path) -> dict[str, dict]:
    out = {}
    for r in csv.DictReader(path.open(newline=""), delimiter="\t"):
        out[r["id"]] = {k: r.get(k, "") for k in
                        ("family", "superfamily", "fold") if r.get(k)}
        if r.get("description"):
            out[r["id"]]["description"] = r["description"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--metadata-tsv", default=None)
    ap.add_argument("--head", default=str(DEFAULT_HEAD))
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--model-sha", default=MODEL_SHA)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    idx = Esmci(args.index)
    meta = dict(idx.meta)
    ids = idx.ids
    if not ids:
        raise SystemExit("index has no ids sidecar")
    if args.metadata_tsv:
        by_id = read_metadata(Path(args.metadata_tsv))
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} ids missing from metadata tsv")
        metadata = [by_id[i] for i in ids]
    else:
        metadata = meta.get("metadata")
        if not isinstance(metadata, list) or len(metadata) != len(ids):
            metadata = [{} for _ in ids]

    head = Path(args.head)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_out = out / f"{args.name}.esmci"
    json_out = out / f"{args.name}.esmci.json"
    head_out = out / f"{args.name}.esmci.head.npy"
    shutil.copyfile(args.index, index_out)
    shutil.copyfile(head, head_out)

    meta.update({
        "ids": ids,
        "metadata": metadata,
        "d": idx.d,
        "dtype": "float16",
        "win": 2046,
        "overlap": 128,
        "head_file": head_out.name,
        "head_sha256": sha256(head_out),
        "model": args.model_name,
        "model_sha256": args.model_sha,
        "source": args.source,
        "published": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    meta.pop("embed_bin", None)
    if args.note:
        meta["note"] = args.note
    json_out.write_text(json.dumps(meta, indent=2) + "\n")

    files = [index_out, json_out, head_out]
    sums = "".join(f"{sha256(p)}  {p.name}\n" for p in files)
    (out / "SHA256SUMS").write_text(sums)
    print(json.dumps({
        "name": args.name, "out": str(out), "n": len(idx),
        "index_bytes": index_out.stat().st_size,
        "files": [p.name for p in files],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
