"""Fused default-index build (M7 T7.3; M8's production build path).

prefix (frozen `cut` blocks) -> frozen M3 suffix -> head -> f16 rows
[n_residues, 128], in one pass. The M4 route materialized the f32 960-d
prefix cache first (130 GB at 100K, ~1.3 TB at 1M) and then projected it;
here every window's rows go straight to the final memmap — no intermediate,
so the 1M build fits on disk. Resumable per batch via progress json; meta
sidecar follows embed._pack_meta (ids / offsets over n_prefix_rows / lens).

Numeric contract: same math as prefix_dataset + m4_eval.build_trained
composition (same modules, same masks); validated against M4's
cache/m4/db_tr128.npy by scripts/m7_build.py --validate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from esmc_retrieval.embed import _pack_meta, chunk_windows, n_prefix_rows


def build_default_index(rows: list[dict], out_npy: Path, meta_path: Path,
                        ckpt: Path, token_budget: int = 8192) -> None:
    import torch
    from esm.models.esmc.tokenizer import EsmcTokenizer

    from esmc_retrieval.train import Projection, Suffix, _esmc

    n = len(rows)
    progress = meta_path.with_suffix(".progress.json")
    lens = [n_prefix_rows(r["sequence"]) for r in rows]
    offsets, done = _pack_meta(rows, meta_path, progress, lens, "float16",
                               d=128)
    total = offsets[-1]

    if done == 0:
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float16,
                                       shape=(total, 128))
    else:
        mm = np.lib.format.open_memmap(out_npy, mode="r+")
        assert mm.shape == (total, 128)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  build device: {dev}", flush=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    cut = ck["cut"]
    esmc, tok = _esmc()
    esmc = esmc.to(dev).eval()
    suffix = Suffix(esmc, cut).to(dev).eval()
    suffix.load_state_dict(ck["suffix"])
    head = Projection(960, 128).to(dev).eval()
    head.load_state_dict(ck["head"])

    def span_rows(seq: str) -> list[tuple[int, int]]:
        out_, lo = [], 0
        for s, e in chunk_windows(seq):
            out_.append((lo, e - s + 2))
            lo += e - s + 2
        return out_

    t0 = time.time()
    with torch.no_grad():
        i = done
        while i < n:
            batch_rows: list[dict] = []
            budget = 0
            while i + len(batch_rows) < n:
                cost = lens[i + len(batch_rows)]
                if batch_rows and budget + cost > token_budget:
                    break
                batch_rows.append(rows[i + len(batch_rows)])
                budget += cost
            # entries = (row_in_batch, span, subseq); one per window
            entries: list[tuple[int, int, tuple[int, int], str]] = []
            for bi, r in enumerate(batch_rows):
                seq = r["sequence"]
                for w, span in enumerate(chunk_windows(seq)):
                    entries.append((bi, w, span,
                                    seq[span[0] : span[1]]))
            enc = tok([e[3] for e in entries], padding=True,
                      return_tensors="pt")
            enc = {kk: vv.to(dev) for kk, vv in enc.items()}
            mask = enc["attention_mask"].bool()
            seq_id = None if bool(mask.all()) else mask[:, None, None, :]
            x = esmc.embed(enc["input_ids"])
            for block in list(esmc.transformer.blocks)[:cut]:
                x, _ = block(x, seq_id)
            h = head(suffix(x, mask)).cpu().numpy()
            for ei, (bi, w, (s, e), _) in enumerate(entries):
                cnt = e - s + 2
                assert int(mask[ei].sum()) == cnt
                row_lo = offsets[i + bi] + span_rows(
                    batch_rows[bi]["sequence"])[w][0]
                mm[row_lo : row_lo + cnt] = h[ei, :cnt].astype(np.float16)
            i += len(batch_rows)
            progress.write_text(json.dumps({"done": i}))
            print(f"  index {i}/{n} rows ({time.time() - t0:.0f}s)",
                  flush=True)
            del x, h, enc
            if dev == "mps":
                torch.mps.empty_cache()
    mm.flush()
    del mm
