#!/usr/bin/env python3
"""M1 diagnostic: why is frozen ESM-C MaxSim (cR@10 ~0.63) below ProtoCol's
frozen variant (0.81, ESM-2 35M + random projection)?

Isolates four factors in one harness (same eval as everything else):
  A  ESM-C, strip CLS/EOS, full length        (= our T1.3 pipeline; cross-check)
  B  ESM-C, keep CLS/EOS, truncate 256 tokens  (ProtoCol encoding protocol)
  C  ESM-2 35M, keep CLS/EOS, truncate, random Linear(480->128) proj
     (faithful ProtoCol-frozen replica; must land near 0.8096)
  D  ESM-2 35M, strip CLS/EOS, full length     (backbone effect, our protocol)

Writes results/m1_protocol_gap.csv + cache/m1hf/*.npy; logs runs.jsonl.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from esmc_li.eval import evaluate, load_split
from esmc_li.runlog import log_run
from esmc_li.scoring import l2_normalize, maxsim_matrix, rankings_from_scores

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "splits" / "scope_test.tsv"
OUT = ROOT / "results" / "m1_protocol_gap.csv"
CACHE = ROOT / "cache" / "m1hf"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_LEN = 256  # ProtoCol MAX_LEN (tokens incl. CLS/EOS)


def embed_esmc(seqs: list[str]) -> list[np.ndarray]:
    from esm.models.esmc.model import EsmcForMaskedLM
    from esm.models.esmc.tokenizer import EsmcTokenizer

    model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-300M", dtype=torch.float32)
    model = model.to(DEV).eval()
    tok = EsmcTokenizer()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 16):
            enc = tok(seqs[i:i + 16], padding=True, return_tensors="pt")
            enc = {k: v.to(DEV) for k, v in enc.items()}
            res = model(**enc, return_dict=True).last_hidden_state
            mask = enc["attention_mask"].bool()
            for k in range(res.size(0)):
                t = int(mask[k].sum())
                out.append(res[k, :t].float().cpu().numpy())
            if i % 512 == 0:
                print(f"  esmc {i}/{len(seqs)}", flush=True)
    return out


def embed_esm2(seqs: list[str]) -> list[np.ndarray]:
    from transformers import AutoModel, AutoTokenizer

    name = "facebook/esm2_t12_35M_UR50D"
    model = AutoModel.from_pretrained(name).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(name)
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 32):
            enc = tok(seqs[i:i + 32], padding=True, return_tensors="pt")
            enc = {k: v.to(DEV) for k, v in enc.items()}
            res = model(**enc).last_hidden_state
            mask = enc["attention_mask"].bool()
            for k in range(res.size(0)):
                t = int(mask[k].sum())
                out.append(res[k, :t].float().cpu().numpy())
            if i % 512 == 0:
                print(f"  esm2 {i}/{len(seqs)}", flush=True)
    return out


def strip_special(embs: list[np.ndarray]) -> list[np.ndarray]:
    """Drop first (CLS) and last (EOS) row."""
    return [e[1:-1] for e in embs]


def truncate_tokens(embs: list[np.ndarray]) -> list[np.ndarray]:
    """Keep at most MAX_LEN token rows (CLS + first residues + EOS)."""
    return [e[:MAX_LEN] for e in embs]


def variant_scores(name: str, embs: list[np.ndarray], proj: torch.nn.Linear | None,
                   rows: list[dict]) -> tuple[dict, np.ndarray]:
    if proj is not None:
        with torch.no_grad():
            v = [F.normalize(proj(torch.from_numpy(e)), dim=-1).numpy().astype(np.float32)
                 for e in embs]
    else:
        v = [l2_normalize(e) for e in embs]
    p = CACHE / f"{name}.npy"
    if p.exists():
        S = np.load(p)
    else:
        S = maxsim_matrix(v, device=DEV)
        np.save(p, S)
    ids = [r["id"] for r in rows]
    m = evaluate(rankings_from_scores(S, ids), rows)
    return m, S


def main() -> int:
    t0 = time.time()
    CACHE.mkdir(exist_ok=True)
    rows = load_split(SPLIT)
    ids = [r["id"] for r in rows]
    seqs = [r["sequence"] for r in rows]
    print(f"n={len(rows)} device={DEV}")

    variants: list[tuple[str, dict]] = []

    t = time.time()
    e_c = embed_esmc(seqs)
    print(f"esmc embed {time.time()-t:.1f}s")

    m, _ = variant_scores("A_esmc_strip_full", strip_special(e_c), None, rows)
    variants.append(("A_esmc_strip_full", m))

    m, _ = variant_scores("B_esmc_keep_trunc", truncate_tokens(e_c), None, rows)
    variants.append(("B_esmc_keep_trunc", m))
    del e_c

    t = time.time()
    e_2 = embed_esm2(seqs)
    print(f"esm2 embed {time.time()-t:.1f}s")
    torch.manual_seed(0)
    proj = torch.nn.Linear(480, 128, bias=False)  # ProtoCol random init
    m, _ = variant_scores("C_esm2_keep_trunc_randproj", truncate_tokens(e_2), proj, rows)
    variants.append(("C_esm2_keep_trunc_randproj", m))

    m, _ = variant_scores("D_esm2_strip_full", strip_special(e_2), None, rows)
    variants.append(("D_esm2_strip_full", m))

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "cR@1", "cR@5", "cR@10", "cR@100", "mrr"])
        for name, m in variants:
            w.writerow([name, round(m["cR@1"], 4), round(m["cR@5"], 4),
                        round(m["cR@10"], 4), round(m["cR@100"], 4),
                        round(m["mrr"], 4)])
            print(name, {k: round(m[k], 4) for k in ("cR@1", "cR@5", "cR@10", "cR@100")})
            log_run("T1.diag", {"split": "scope_test", "variant": name,
                                "device": DEV,
                                "note": "protocol-gap diagnosis: special tokens / "
                                        "truncation / backbone"},
                    m, seed=0)
    print(f"total {time.time()-t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
