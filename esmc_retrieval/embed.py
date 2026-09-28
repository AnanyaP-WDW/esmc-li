"""Chunk-aware embedding (M4 T4.2).

esmc.cpp's context is 2048 tokens (2 special + 2046 residues), so chains
longer than WIN are split into overlapping windows; each window's residue
rows are reassembled so the final row count equals the sequence length
exactly (the shared overlap rows are emitted once, by the earlier window).
esmc-embed strips BOS/EOS itself, so no special-token rows ever appear.

embed_dataset() packs all chains into one .npy memmap [total_residues, 960]
(f16) + a meta json (ids/offsets/lens), batched so a crash loses at most one
batch — the "packed output" T6.1 defers, done on the Python side.

prefix_dataset() is the same packing but for the RAW block-`cut` hidden state
(pre final-norm), computed with the HF model exactly like
train.build_prefix_cache — input to the M3 suffix. esmc-embed cannot emit
this: its graph applies output_norm unconditionally.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

WIN = 2046
OVERLAP = 128
D_MODEL = 960


def chunk_windows(seq: str, win: int = WIN, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    """(start, end) spans covering [0, len(seq)) exactly; windows <= win wide."""
    n = len(seq)
    if n <= win:
        return [(0, n)]
    stride = win - overlap
    spans, s = [], 0
    while s + win < n:
        spans.append((s, s + win))
        s += stride
    spans.append((s, n))
    return spans


def assemble(spans: list[tuple[int, int]], parts: list[np.ndarray]) -> np.ndarray:
    """Row-exact reassembly: window i contributes rows [e_{i-1}, e_i)."""
    out, prev = [], 0
    for (s, e), p in zip(spans, parts):
        assert p.shape[0] == e - s, f"rows {p.shape[0]} != span {e - s}"
        assert prev >= s, "overlap must cover the previous window's tail"
        out.append(p[prev - s : e - s])
        prev = e
    return np.concatenate(out, axis=0)


def embed_seq(seq: str, embed_fn) -> np.ndarray:
    spans = chunk_windows(seq)
    if len(spans) == 1:
        return embed_fn(seq)
    parts = [embed_fn(seq[s:e]) for s, e in spans]
    return assemble(spans, parts)


def _pack_meta(rows: list[dict], meta_path: Path, progress: Path,
               lens: list[int], dtype: str) -> tuple[list[int], int]:
    """Write/validate meta; return (offsets, done_index)."""
    n = len(rows)
    offsets = np.cumsum([0] + lens)
    ids = [r["id"] for r in rows]
    done = 0
    meta_ok = (meta_path.exists()
               and json.loads(meta_path.read_text())["ids"] == ids
               and json.loads(meta_path.read_text())["dtype"] == dtype)
    if meta_ok and progress.exists():
        done = json.loads(progress.read_text())["done"]
        if done >= n:
            print(f"[skip] packed dataset: all {n} rows done")
        elif done > 0:
            print(f"[resume] packed dataset: {done}/{n} rows done")
    if not (meta_ok and done > 0):
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(
            {"ids": ids, "offsets": offsets.tolist(),
             "lens": lens, "dtype": dtype, "d": D_MODEL,
             "win": WIN, "overlap": OVERLAP}))
    return [int(o) for o in offsets], done


def embed_dataset(rows: list[dict], out_npy: Path, meta_path: Path,
                  binary: Path, gguf: Path, batch: int = 10_000) -> None:
    """Embed rows (id/sequence) into a packed f16 memmap; resumable per batch."""
    n = len(rows)
    progress = meta_path.with_suffix(".progress.json")
    offsets, done = _pack_meta(rows, meta_path, progress,
                               [len(r["sequence"]) for r in rows], "float16")
    total = offsets[-1]

    if done == 0:
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float16,
                                       shape=(total, D_MODEL))
    else:
        mm = np.lib.format.open_memmap(out_npy, mode="r+")
        assert mm.shape == (total, D_MODEL)

    for start in range(done, n, batch):
        chunk = rows[start : start + batch]
        with tempfile.TemporaryDirectory(prefix="m4embed_") as td:
            td = Path(td)
            fa = td / "batch.fa"
            with fa.open("w") as f:
                for r in chunk:
                    seq = r["sequence"]
                    for s, e in chunk_windows(seq):
                        if s == 0 and e == len(seq):
                            f.write(f">{r['id']}\n{seq}\n")
                        else:
                            f.write(f">{r['id']}|{s}\n{seq[s:e]}\n")
            npy_dir = td / "npy"
            npy_dir.mkdir()
            subprocess.run([str(binary), "-m", str(gguf), "--fasta", str(fa),
                            "--output-dir", str(npy_dir)], check=True,
                           capture_output=True)
            for j, r in enumerate(chunk):
                seq = r["sequence"]
                spans = chunk_windows(seq)
                if len(spans) == 1:
                    part = np.load(npy_dir / f"{r['id']}.npy")
                else:
                    parts = [np.load(npy_dir / f"{r['id']}|{s}.npy")
                             for s, _ in spans]
                    part = assemble(spans, parts)
                assert part.shape[0] == len(seq) and part.shape[1] == D_MODEL
                lo, hi = offsets[start + j], offsets[start + j + 1]
                mm[lo:hi] = part.astype(np.float16)
        done = min(start + batch, n)
        progress.write_text(json.dumps({"done": done}))
        print(f"  embedded {done}/{n} rows ({offsets[done]} residues)")
    mm.flush()
    del mm


def n_prefix_rows(seq: str) -> int:
    """Rows prefix_dataset writes for seq: CLS/EOS kept per window (M3's
    prefix cache includes them; check_suffix drops [1:-1] only for comparison)."""
    spans = chunk_windows(seq)
    return sum(e - s for s, e in spans) + 2 * len(spans)


def prefix_dataset(rows: list[dict], out_npy: Path, meta_path: Path,
                   cut: int = 27, token_budget: int = 8192) -> None:
    """RAW block-`cut` hidden state (M3 suffix input), HF fp32 on MPS,
    packed f32 memmap. Token-budget batches (long chains -> more windows).
    Overlap residues appear once per owning window (no dedup: each window is
    its own M3-style sequence, specials included).
    """
    import torch
    from esm.models.esmc.model import EsmcForMaskedLM
    from esm.models.esmc.tokenizer import EsmcTokenizer

    n = len(rows)
    progress = meta_path.with_suffix(".progress.json")
    lens = [n_prefix_rows(r["sequence"]) for r in rows]
    offsets, done = _pack_meta(rows, meta_path, progress, lens, "float32")
    total = offsets[-1]

    if done == 0:
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float32,
                                       shape=(total, D_MODEL))
    else:
        mm = np.lib.format.open_memmap(out_npy, mode="r+")
        assert mm.shape == (total, D_MODEL)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = EsmcForMaskedLM.from_pretrained(
        "biohub/ESMC-300M", dtype=torch.float32).esmc.to(dev).eval()
    tok = EsmcTokenizer()

    i = done
    with torch.no_grad():
        while i < n:
            batch_rows: list[dict] = []
            budget = 0
            while i + len(batch_rows) < n:
                r = rows[i + len(batch_rows)]
                cost = sum(e - s for s, e in chunk_windows(r["sequence"]))
                if batch_rows and budget + cost > token_budget:
                    break
                batch_rows.append(r)
                budget += cost
            # entries = (row_offset_in_batch, span, subseq)
            entries: list[tuple[int, tuple[int, int], str]] = []
            for bi, r in enumerate(batch_rows):
                seq = r["sequence"]
                for span in chunk_windows(seq):
                    entries.append((bi, span, seq[span[0] : span[1]]))
            enc = tok([e[2] for e in entries], padding=True, return_tensors="pt")
            enc = {kk: vv.to(dev) for kk, vv in enc.items()}
            x = model.embed(enc["input_ids"])
            mask = enc["attention_mask"].bool()
            seq_id = None if bool(mask.all()) else mask[:, None, None, :]
            for block in list(model.transformer.blocks)[:cut]:
                x, _ = block(x, seq_id)
            x = x.cpu().numpy()
            parts: dict[int, list[np.ndarray]] = {}
            for ei, (bi, (s, e), _) in enumerate(entries):
                t = int(mask[ei].sum())
                parts.setdefault(bi, []).append(x[ei, :t])
                assert parts[bi][-1].shape[0] == e - s + 2
            for bi, r in enumerate(batch_rows):
                part = np.concatenate(parts[bi], axis=0)
                assert part.shape[0] == n_prefix_rows(r["sequence"])
                lo, hi = offsets[i + bi], offsets[i + bi + 1]
                mm[lo:hi] = part
            i += len(batch_rows)
            progress.write_text(json.dumps({"done": i}))
            print(f"  prefix {i}/{n} rows ({offsets[i]} residues)")
            del x, enc
            if dev == "mps":
                torch.mps.empty_cache()
    mm.flush()
    del mm
