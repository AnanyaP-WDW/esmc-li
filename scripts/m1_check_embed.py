#!/usr/bin/env python3
"""T1.1 exit checks: cache/*.npy count/shape vs data/scope.fa, wall time logged;
10 random domains vs PyTorch esm ESMC: min cosine >= 0.999.

Usage: scripts/m1_check_embed.py [--torch]   (--torch adds the esm cross-check)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from esmc_retrieval.runlog import log_run

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / "data" / "scope.fa"
CACHE = ROOT / "cache" / "scope"


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    sid, chunks = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if sid is not None:
                seqs[sid] = "".join(chunks)
            sid, chunks = line[1:].split()[0], []
        else:
            chunks.append(line.strip())
    if sid is not None:
        seqs[sid] = "".join(chunks)
    return seqs


def check_shapes(seqs: dict[str, str]) -> None:
    files = sorted(CACHE.glob("*.npy"))
    missing = [s for s in seqs if not (CACHE / f"{s}.npy").exists()]
    extra = [p.stem for p in files if p.stem not in seqs]
    bad_shape = []
    for p in files:
        a = np.load(p, mmap_mode="r")
        want = len(seqs[p.stem])
        if a.shape != (want, 960):
            bad_shape.append((p.stem, a.shape, want))
    assert not missing, f"{len(missing)} missing, e.g. {missing[:5]}"
    assert not extra, f"{len(extra)} extra, e.g. {extra[:5]}"
    assert not bad_shape, f"bad shapes: {bad_shape[:5]}"
    assert len(files) == len(seqs), f"{len(files)} != {len(seqs)}"
    print(f"OK: {len(files)} files, all [len, 960], match scope.fa")


def torch_crosscheck(seqs: dict[str, str], n: int = 10, seed: int = 0) -> float:
    import torch
    from esm.models.esmc.model import EsmcForMaskedLM
    from esm.models.esmc.tokenizer import EsmcTokenizer

    torch.manual_seed(seed)
    ids = sorted(seqs)
    pick = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))[:n]]

    print("loading PyTorch ESMC-300M (downloads on first run)...")
    model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-300M", dtype=torch.float32)
    model.eval()
    tok = EsmcTokenizer()

    min_cos = 1.0
    for sid in pick:
        seq = seqs[sid].upper()  # ASTRAL is lowercase; esmc-embed uppercases, HF's tokenizer does not
        enc = tok(seq, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=False, return_dict=True)
        ref = out.last_hidden_state[0, 1 : 1 + len(seq)].float().numpy()
        ours = np.load(CACHE / f"{sid}.npy").astype(np.float64)
        assert ref.shape == ours.shape, (sid, ref.shape, ours.shape)
        num = (ref * ours).sum()
        den = np.linalg.norm(ref) * np.linalg.norm(ours)
        cos = float(num / den)
        min_cos = min(min_cos, cos)
        print(f"  {sid} len={len(seq)} cos={cos:.6f}")
    assert min_cos >= 0.999, f"min cos {min_cos} < 0.999"
    print(f"OK: min cosine over {n} domains = {min_cos:.6f} >= 0.999")
    return min_cos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch", action="store_true", help="run esm cross-check")
    args = ap.parse_args()

    seqs = read_fasta(FA)
    print(f"scope.fa: {len(seqs)} domains")
    t0 = time.time()
    check_shapes(seqs)
    check_s = time.time() - t0

    extras: dict = {}
    if args.torch:
        extras["min_cos_vs_pytorch"] = torch_crosscheck(seqs)
        extras["torch_check_s"] = round(time.time() - t0 - check_s, 1)
    log_run("T1.1", {"fasta": str(FA.relative_to(ROOT)),
                     "cache": str(CACHE.relative_to(ROOT)),
                     "shape_check_s": round(check_s, 1),
                     **extras}, {"n_files": len(seqs)}, wall_s=round(check_s, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
