#!/usr/bin/env python3
"""T0.7: reproduce ProtoCol (trained SCOPe cR@10 = 0.8947) using THEIR code.

Runs the vendored protocol_experiments.py unmodified except DEVICE
(cpu -> mps when available; their code only wires cuda). Writes
results/m0_protocol_repro.csv and a results/runs.jsonl entry.
"""
from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ProtoCol"))

import torch  # noqa: E402
import protocol_experiments as pe  # noqa: E402

from esmc_retrieval.runlog import log_run  # noqa: E402

# Their DataLoader uses num_workers=2 + a closure collate_fn; macOS (spawn
# start method) can't pickle the closure. num_workers=0 avoids the worker
# processes entirely -- no numerical effect, their file stays unmodified.
_orig_dataloader = pe.DataLoader
pe.DataLoader = lambda *a, **k: _orig_dataloader(*a, **{**k, "num_workers": 0})

PAPER = {"frozen": 0.8093, "trained": 0.8947}  # Table 1, cR@10
OUT = ROOT / "results" / "m0_protocol_repro.csv"


def main() -> int:
    t0 = time.time()
    if torch.backends.mps.is_available():
        pe.DEVICE = torch.device("mps")
    print(f"DEVICE={pe.DEVICE}")
    random.seed(pe.SEED)
    torch.manual_seed(pe.SEED)

    proteins = pe.load_scope()
    train, test = pe.split_by_superfamily(proteins, 0.15, 3)
    test = pe.subsample_test_set(test, pe.MAX_TEST_PROTEINS)
    print(f"train={len(train)} test={len(test)}")

    tok = pe.AutoTokenizer.from_pretrained(pe.MODEL_NAME)
    model = pe.ProteinColBERT(pe.MODEL_NAME, proj_dim=128,
                              n_trainable_layers=3).to(pe.DEVICE)

    results: dict[str, dict[int, float]] = {}

    emb, labels = pe.encode_all(model, test, tok, batch_size=32)
    results["protocol_frozen"] = pe.evaluate_retrieval(emb, labels)
    print("frozen:", {k: round(v, 4) for k, v in results["protocol_frozen"].items()})

    pe.train_colbert(model, pe.PairDataset(train), tok, epochs=3,
                     batch_size=16, lr=2e-5, temperature=1.0)

    emb, labels = pe.encode_all(model, test, tok, batch_size=32)
    results["protocol_trained"] = pe.evaluate_retrieval(emb, labels)
    print("trained:", {k: round(v, 4) for k, v in results["protocol_trained"].items()})

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "cR@1", "cR@5", "cR@10", "cR@100",
                    "paper_cR@10", "delta_vs_paper"])
        for name, r in results.items():
            key = "frozen" if "frozen" in name else "trained"
            d = r[10] - PAPER[key]
            w.writerow([name, round(r[1], 4), round(r[5], 4), round(r[10], 4),
                        round(r[100], 4), PAPER[key], round(d, 4)])
            print(name, "cR@10 delta vs paper:", round(d, 4))

    log_run("T0.7", {"code": "vendor/ProtoCol (unmodified except DEVICE=mps)",
                     "model": pe.MODEL_NAME, "epochs": 3, "lr": 2e-5,
                     "batch": 16, "temperature": 1.0, "seed": pe.SEED},
            {name: r for name, r in results.items()}, seed=pe.SEED,
            wall_s=time.time() - t0)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
