#!/usr/bin/env python3
"""M3 T3.3: export the best fine-tuned model to GGUF and validate it.

  best results/m3_models/*.pt (by val cR@10)
  -> inject suffix blocks+norm into HF ESMC-300M -> save_pretrained
  -> esmc.cpp tools/convert_esmc_to_gguf.py --dtype f16
  -> esmc.cpp tests/generate_reference.py on sequences_correctness.fasta
     (reference generated from OUR fine-tuned weights)
  -> esmc.cpp benchmarks/correctness.py --model f16=<ft gguf>

Exit: 100/100 sequences pass and min cosine over all sequences >= 0.999.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from esmc_retrieval.runlog import log_run

ROOT = Path(__file__).resolve().parents[1]
ESMC_CPP = Path("/Users/ananyapathak/Desktop/oss/esmc.cpp")
MODELS = ROOT / "results" / "m3_models"
HF_DIR = ROOT / "cache" / "m3" / "ft_hf"
GGUF = ESMC_CPP / "models" / "esmc-300m-ft-f16.gguf"
REF_NPZ = HF_DIR / "reference_ft_correctness.npz"
FASTA = ESMC_CPP / "benchmarks" / "sequences_correctness.fasta"
OUT_PREFIX = ESMC_CPP / "results" / "correctness_ft_300m"
VENV = ROOT / ".venv" / "bin" / "python"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def main() -> int:
    t0 = time.time()
    cands = sorted(MODELS.glob("k*_seed*.pt"))
    if not cands:
        sys.exit(f"no checkpoints in {MODELS}; run m3_train.py --final first")
    best_path = max(cands, key=lambda p: torch.load(p)["val_cR@10"])
    ck = torch.load(best_path, map_location="cpu")
    k, cut, seed = ck["k"], ck["cut"], ck["seed"]
    print(f"best: {best_path.name} val={ck['val_cR@10']} test={ck['test_cR@10']}")

    from esm.models.esmc.model import EsmcForMaskedLM

    model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-300M",
                                            dtype=torch.float32)
    esmc = model.esmc
    suffix_sd = ck["suffix"]
    for local in range(k):
        pfx = f"blocks.{local}."
        sub = {kk[len(pfx):]: vv for kk, vv in suffix_sd.items()
               if kk.startswith(pfx)}
        esmc.transformer.blocks[cut + local].load_state_dict(sub)
    esmc.transformer.norm.load_state_dict(
        {kk[len("norm."):]: vv for kk, vv in suffix_sd.items()
         if kk.startswith("norm.")})
    HF_DIR.mkdir(parents=True, exist_ok=True)
    # esmc.cpp's converter expects the NATIVE key layout (esmc.embed.weight,
    # esmc.transformer.blocks...). save_pretrained() would emit a
    # transformers-style layout (embed_tokens/layers) it cannot read, so write
    # the in-memory (native) state dict straight to safetensors.
    from safetensors.torch import save_file

    sd = {kk: vv.detach().cpu().contiguous() for kk, vv in
          model.state_dict().items()}
    save_file(sd, HF_DIR / "model.safetensors")
    shutil.copyfile(ESMC_CPP / "esmc-300m" / "config.json",
                    HF_DIR / "config.json")
    print(f"saved fine-tuned native checkpoint -> {HF_DIR}")

    run([VENV, ESMC_CPP / "tools" / "convert_esmc_to_gguf.py",
         HF_DIR, GGUF, "--dtype", "f16"])
    run([VENV, ESMC_CPP / "tests" / "generate_reference.py",
         "--weights", HF_DIR / "model.safetensors",
         "--fasta", FASTA, "--output", REF_NPZ])
    # correctness.py exits 1 on any threshold failure — we parse the CSV and
    # apply T3.3's own criterion, so tolerate its exit code.
    try:
        run([VENV, ESMC_CPP / "benchmarks" / "correctness.py",
             "--model", f"f16={GGUF}", "--reference", REF_NPZ,
             "--backend", "metal", "--output-prefix", OUT_PREFIX])
    except subprocess.CalledProcessError as e:
        print(f"correctness.py exited {e.returncode}; parsing CSV anyway")

    rows = list(csv.DictReader(open(f"{OUT_PREFIX}.csv")))
    n = len(rows)
    n_pass = sum(1 for r in rows if r["passed"] == "True")
    min_cos = min(float(r["min_cosine"]) for r in rows)
    ok = n == 100 and n_pass == n and min_cos >= 0.999
    print(f"correctness: {n_pass}/{n} pass, min cosine {min_cos:.6f} "
          f"-> {'OK' if ok else 'FAIL'}")
    log_run("T3.3", {"gguf": str(GGUF), "source_ckpt": best_path.name,
                     "k": k, "seed": seed, "backend": "metal",
                     "criterion": "100/100 pass, min cosine >= 0.999"},
            {"n_pass": n_pass, "n": n, "min_cos": round(min_cos, 6),
             "val_cR@10": ck["val_cR@10"], "test_cR@10": ck["test_cR@10"]},
            seed=seed, wall_s=time.time() - t0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
