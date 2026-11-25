# EC suffix fine-tune on the 20K reference

**Date:** 2026-11-26
**Branch:** `exp/ec-local-objective`
**Status:** **negative for the EC active-site claim; positive retrieval.**
The last-3-block suffix fine-tune improves CARE ood30 by +2.1 points but the
homolog-free EnzymARC FPR is 0.361 catalytic / 0.363 5 Å, slightly worse than the
0.346 / 0.346 head-only baseline and far from the ≤17% target.

## Why this run

The full 153K reference is disk-blocked: the 20K reference cache is 28GB and
only ~64GB is free after the suffix rows; 100K (~140GB) and 153K (~214GB) do not
fit. This run tests the remaining planned signal variable, the suffix
fine-tune, on the existing 20K reference.

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

| Seed | CARE ood30 top1 | CARE ood30 top10 | CARE prom top1 | Catalytic FPR | Catalytic FPR10 | 5 Å FPR | 5 Å FPR10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5069 | 0.5995 | 0.8517 | 0.366 | 0.428 | 0.366 | 0.434 |
| 1 | 0.5278 | 0.6019 | 0.8421 | 0.358 | 0.424 | 0.364 | 0.428 |
| 2 | 0.5000 | 0.5741 | 0.8325 | 0.358 | 0.428 | 0.360 | 0.440 |
| **mean** | **0.5116** | **0.5918** | **0.8421** | **0.3607** | **0.4267** | **0.3633** | **0.4340** |

Base head (head-only, no suffix; `DECOY_GLOBAL.md`): CARE ood30 0.4907,
promiscuous 0.8421, catalytic FPR 0.346 (top10 0.416), 5 Å FPR 0.346
(top10 0.412).

## Findings

1. **Retrieval improves.** CARE ood30 top1 is 0.5116 (+2.1 points over the
   head-only 0.4907); promiscuous is tied at 0.8421. The suffix adapts the
   backbone to EC retrieval.

2. **The active-site FPR does not improve.** Catalytic 0.3607 vs 0.346; 5 Å
   0.3633 vs 0.346; top-10 0.4267/0.4340 vs 0.416/0.412. The FPR is consistent
   across seeds (0.358–0.366), so the ~1.5-point regression is not seed noise.
   The ≤17% target is not met.

3. **The EC active-site hypothesis is closed under this architecture.** Three
   training-signal changes failed the meaningful test: the local patch loss
   overfit 32 triples without transferring (`OVERFIT32.md`); the global decoy
   triplet at scale left the FPR at 0.346 (`DECOY_GLOBAL.md`); the suffix
   fine-tune leaves it at 0.361/0.363. The only robust gain is retrieval, not
   active-site sensitivity.

4. **The product path is the 20K `annotate` retrieval/evidence index.** The
   100K/153K reference is disk-blocked; the 20K head index and
   `scripts/ec_annotate.py` work. The suffix model's CARE gain is not
   productized here because the CLI query encoder is the `esmc.cpp` GGUF, not
   the HF suffix; exporting the suffix to GGUF or adding a PyTorch query path
   is a separate task.

## Decision

Stop the EC active-site research. The falsifiable prediction (homolog-free FPR
≤17% at matched coverage) is not met at ~36%. Ship the 20K `annotate` product;
the 100K reference is not buildable on this machine.

## Caveats

- 20K reference, 3 seeds, k=3, one window per sequence. The full 100K/153K
  reference and suffix were not run.
- The FPR is top-1/top-10, not matched-coverage. The diagnostic removes
  reference homologs at ≥30% identity / ≥80% coverage, so the comparison is
  directionally fair but not the final matched-coverage number.
- The suffix FPR is slightly worse than the head-only baseline; the CARE gain
  is on the 20K reference and is a retrieval gain, not evidence for active-site
  sensitivity.
- The 960-D train/val/ref caches were deleted; the base results are recorded
  and the caches can be rebuilt with `scripts/ec_embed.py`.

## Artifacts

- `results/ec/suffix_k3_summary.csv`
- `results/ec/care_ood30_suffix_k3_seed{0,1,2}.csv` / `.json`
- `results/ec/care_promiscuous_suffix_k3_seed{0,1,2}.csv` / `.json`
- `results/ec/enzymarc_catalytic_suffix_k3_seed{0,1,2}.csv` / `.json`
- `results/ec/enzymarc_5A_suffix_k3_seed{0,1,2}.csv` / `.json`
- `results/ec_suffix_models/k3_seed{0,1,2}.pt` (checkpoints; gitignored)
- `cache/ec/prefix/k3/`, `cache/ec/suffix/k3/` (gitignored)

## Reproduce

```bash
# prefix caches (one window per sequence; prefix = input to block 27)
.venv/bin/python scripts/ec_prefix.py data/ec/splits/train_5k.tsv cache/ec/prefix/k3/train_5k --k 3
.venv/bin/python scripts/ec_prefix.py data/ec/splits/ref_20k.tsv cache/ec/prefix/k3/ref20k --k 3
.venv/bin/python scripts/ec_prefix.py data/ec/splits/care_ood30.tsv cache/ec/prefix/k3/care_ood30 --k 3
.venv/bin/python scripts/ec_prefix.py data/ec/splits/care_promiscuous.tsv cache/ec/prefix/k3/care_promiscuous --k 3
.venv/bin/python scripts/ec_prefix.py data/ec/splits/enzymarc_catalytic.tsv cache/ec/prefix/k3/enzymarc_catalytic --k 3 --limit 500
.venv/bin/python scripts/ec_prefix.py data/ec/splits/enzymarc_5A.tsv cache/ec/prefix/k3/enzymarc_5A --k 3 --limit 500

# 3-seed suffix+head training
for s in 0 1 2; do
  .venv/bin/python scripts/ec_suffix_train.py \
    --prefix cache/ec/prefix/k3/train_5k \
    --out-dir results/ec_suffix_models --seed "$s"
done

# suffix rows + CARE/EnzymARC evals + summary
.venv/bin/python scripts/ec_suffix_eval.py
```
