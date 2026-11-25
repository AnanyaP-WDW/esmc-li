# EC suffix fine-tune on the 20K reference

**Date:** 2026-11-25
**Branch:** `exp/ec-local-objective`
**Status:** running. Results table to fill after the 3-seed sweep.

## Why this run

The full 153K reference is disk-blocked: the 20K reference cache is 28GB and
only ~119GB is free after deleting the regenerable 960-D train/val/ref caches;
100K (~140GB) and 153K (~214GB) do not fit. This run tests the remaining
planned signal variable, the suffix fine-tune, on the existing 20K reference.

## Setup

- Backbone: ESM-C 300M; fine-tune the last k=3 blocks + final norm (`cut=27`).
- Input: M3 prefix cache `[L+2, 960]` (CLS/EOS included), built with the HF
  ESM-C fp32 model. All EC sequences are <= 1023 residues, so one window.
- Head: 960→128 linear, L2 rows, MaxSim.
- Train: `train_5k.tsv` (5,000 proteins), 3 epochs, batch 16, τ=1.0.
- Optimizer: AdamW, suffix lr 2e-5, head lr 1e-3, weight decay 0.01.
- Seeds: 0, 1, 2.
- Eval: CARE ood30/promiscuous on the 20K reference; EnzymARC catalytic and
  5 Å decoys on the homolog-free reference (same 500 decoys / 30% identity,
  80% coverage filter as `DECOY_GLOBAL.md`).

The 960-D train/val/ref/decoys caches were deleted to free disk; the base
results are in `SUMMARY.md`, `DIAGNOSTIC.md`, and `DECOY_GLOBAL.md`. They can be
rebuilt with `scripts/ec_embed.py` if needed.

## Results

| Seed | CARE ood30 top1 | CARE ood30 top10 | CARE prom top1 | Catalytic FPR | 5 Å FPR |
|---|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — |
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| mean | — | — | — | — | — |

Base head (head-only, no suffix; `DECOY_GLOBAL.md`): CARE ood30 0.4907,
promiscuous 0.8421, catalytic FPR 0.346, 5 Å FPR 0.346.

## Decision rule

- If the suffix improves the homolog-free FPR toward ≤17% and does not regress
  CARE by >1–2 points, it is worth the full 100K/153K run on a larger machine.
- If the FPR stays ~34%, the EC active-site hypothesis is closed under this
  architecture and the 100K `annotate` product is the path.

## Reproduce

```bash
# prefix caches (one window per sequence; prefix = input to block 27)
.venv/bin/python scripts/ec_prefix.py data/ec/splits/train_5k.tsv cache/ec/prefix/k3/train_5k --k 3
.venv/bin/python scripts/ec_prefix.py data/ec/splits/val_5k.tsv cache/ec/prefix/k3/val_5k --k 3
.venv/bin/python scripts/ec_prefix.py data/ec/splits/ref_20k.tsv cache/ec/prefix/k3/ref20k --k 3
.venv/bin/python scripts/ec_prefix.py data/ec/splits/care_ood30.tsv cache/ec/prefix/k3/care_ood30 --k 3
.venv/bin/python scripts/ec_prefix.py data/ec/splits/care_promiscuous.tsv cache/ec/prefix/k3/care_promiscuous --k 3
.venv/bin/python scripts/ec_prefix.py data/ec/splits/enzymarc_catalytic.tsv cache/ec/prefix/k3/enzymarc_catalytic --k 3 --limit 500
.venv/bin/python scripts/ec_prefix.py data/ec/splits/enzymarc_5A.tsv cache/ec/prefix/k3/enzymarc_5A --k 3 --limit 500

# train suffix+head, seed 0
.venv/bin/python scripts/ec_suffix_train.py \
  --prefix cache/ec/prefix/k3/train_5k \
  --out-dir results/ec_suffix_models --seed 0

# suffix+head rows for the reference and queries
.venv/bin/python scripts/ec_suffix_embed.py data/ec/splits/ref_20k.tsv \
  cache/ec/prefix/k3/ref20k results/ec_suffix_models/k3_seed0.pt \
  cache/ec/suffix/k3/seed0/ref20k
.venv/bin/python scripts/ec_suffix_embed.py data/ec/splits/care_ood30.tsv \
  cache/ec/prefix/k3/care_ood30 results/ec_suffix_models/k3_seed0.pt \
  cache/ec/suffix/k3/seed0/care_ood30

# eval
.venv/bin/python scripts/ec_eval.py \
  --ref data/ec/splits/ref_20k.tsv --query data/ec/splits/care_ood30.tsv \
  --ref-cache cache/ec/suffix/k3/seed0/ref20k \
  --query-cache cache/ec/suffix/k3/seed0/care_ood30 \
  --method maxsim --k 10 --out results/ec/care_ood30_suffix_k3_seed0.csv
```
