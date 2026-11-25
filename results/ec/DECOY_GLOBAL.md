# Global decoy triplet at full scale

**Date:** 2026-11-25
**Branch:** `exp/ec-local-objective`
**Status:** **negative.** The global decoy triplet with cluster-disjoint positives
and a stronger weight does not lower the homolog-free EnzymARC FPR.

## What was tested

The 32-overfit showed the global decoy triplet is the only variant that reaches
32/32 parent-vs-decoy. This run scales that signal to the full 5K training set:

- `L = L_global + 3.0 · relu(2.0 - (s(parent, positive) - s(decoy, positive)))`
- Positive is same EC4 and a different MMseqs cluster (30% identity, 80%
  coverage), matching the 32-overfit.
- Decoys are the 1,907 catalytic Ala/Gly train pairs.
- Head-only 960→128, 3 epochs, τ = 1.0, seed 0.
- Evaluation: CARE ood30/promiscuous on the 20K reference; EnzymARC catalytic
  and 5 Å decoys on the homolog-free reference (all reference proteins at ≥30%
  identity / ≥80% coverage to any decoy parent removed).

The first pass used decoy weight 0.3, margin 0.3, same-EC4 positives without
cluster filtering. This run is the fair version of the 32-overfit's winning
variant.

## Results

| Model | CARE ood30 top1 | CARE promiscuous top1 | EnzymARC catalytic filtered top1 | EnzymARC 5 Å filtered top1 |
|---|---:|---:|---:|---:|
| no decoy, τ=1.0 (first pass) | 0.4769 | 0.8325 | 0.344 | 0.340 |
| decoy 0.3, no cluster filter (first pass) | — | — | 0.340 | 0.334 |
| **decoy 3.0, cluster-disjoint (this run)** | **0.4907** | **0.8421** | **0.346** | **0.346** |

Top-10 FPR (filtered): catalytic 0.416 (first pass decoy-aware 0.410, plain
0.416? — plain top10 0.416); 5 Å 0.412 (first pass decoy-aware 0.408). The new
head is within noise and slightly worse on top-1 for both decoy sets.

CARE metrics:

- ood30: top1 0.4907, top10 0.5579, MRR 0.5145, multi-F1 0.2542.
- promiscuous: top1 0.8421, top10 0.9043, MRR 0.8622, multi-F1 0.4051.

The CARE numbers do not regress; ood30 improves by +1.4 points over the no-decoy
τ=1.0 head. That is a retrieval gain, not an active-site gain.

## Findings

1. **The global decoy triplet does not lower the homolog-free FPR.** Catalytic
   0.346 vs 0.340 decoy-aware / 0.344 plain; 5 Å 0.346 vs 0.334 / 0.340. The
   effect is null/negative, not the planned ≤17%.

2. **The 32-overfit did not predict the full-scale decoy result.** The same
   triplet reaches 32/32 on 32 cluster-disjoint triples but is tied on the
   500-decoy EnzymARC diagnostic. A 32-triple fit is not a sufficient signal for
   the full-scale FPR.

3. **CARE retrieval improves slightly.** ood30 0.4907 and promiscuous 0.8421 are
   +1.4 and +1.0 points over the no-decoy τ=1.0 head. The decoy term is not
   hurting retrieval, but it is not the active-site sensor the EC hypothesis
   needs.

4. **The local-objective and global-triplet results point the same way.** The
   local patch loss overfits the patch without transferring; the global triplet
   fits 32 triples without lowering the FPR at 500 decoys. Under the fixed
   architecture, neither training-signal change makes late interaction
   active-site sensitive.

## Decision

**Stop the EC training-signal experiment.** The falsifiable prediction
(homolog-free FPR ≤17% at matched coverage) is not met; the FPR is ~34.6%,
essentially the no-decoy baseline. The two candidate signals (local patch,
global decoy triplet) both fail the meaningful test. The remaining planned
variables (full 153K reference, suffix fine-tune, 3 seeds) are a 3–5 day bet
against a signal that has now failed twice. For the one-month user-value goal,
the 100K `annotate` retrieval/evidence product is the better path.

## Caveats

- 5K train / 20K reference, one seed, head-only. The full 153K reference and
  suffix fine-tune were not run.
- The FPR is top-1/top-10, not matched-coverage. The diagnostic removes
  reference homologs at ≥30% identity / ≥80% coverage, so the comparison is
  directionally fair but not the final matched-coverage number.
- The EnzymARC decoys are external test only; no EnzymARC data was trained on.
- The CARE gains are retrieval gains on a 20K reference; they are not evidence
  for active-site sensitivity.

## Artifacts

- `results/ec_models/head_ec_decoy_cluster_w3_seed0.pt`
- `results/ec/care_ood30_decoy_cluster_w3.csv` / `.json`
- `results/ec/care_promiscuous_decoy_cluster_w3.csv` / `.json`
- `results/ec/diag_w3_catalytic/catalytic_nohom.csv` / `.json`
- `results/ec/diag_w3_5A/5A_nohom.csv` / `.json`
- `results/ec/diag_w3_catalytic/catalytic_parent_identity.csv`
- `results/ec/diag_w3_5A/5A_parent_identity.csv`

## Reproduce

```bash
.venv/bin/python scripts/ec_train.py \
  --train data/ec/splits/train_5k.tsv \
  --val data/ec/splits/val_5k.tsv \
  --cache cache/ec/esmc960/train \
  --val-cache cache/ec/esmc960/val \
  --out results/ec_models/head_ec_decoy_cluster_w3_seed0.pt \
  --epochs 3 --tau 1.0 \
  --decoy-meta data/ec/decoys/train_decoys.tsv \
  --decoy-cache cache/ec/esmc960/decoys_train \
  --decoy-weight 3.0 --margin 2.0 \
  --clusters data/ec/clusters/train5k_cluster_cluster.tsv \
  --skip-val

.venv/bin/python scripts/ec_eval.py \
  --ref data/ec/splits/ref_20k.tsv \
  --query data/ec/splits/care_ood30.tsv \
  --ref-cache cache/ec/esmc960/ref20k \
  --query-cache cache/ec/esmc960/care_ood30 \
  --method head \
  --head results/ec_models/head_ec_decoy_cluster_w3_seed0.pt \
  --k 10 --out results/ec/care_ood30_decoy_cluster_w3.csv

.venv/bin/python scripts/ec_decoy_diagnostic.py \
  --category catalytic --limit 500 \
  --ref data/ec/splits/ref20k_for_catalytic.tsv \
  --train data/ec/splits/enzymarc_parents_500.tsv \
  --decoy data/ec/splits/enzymarc_catalytic.tsv \
  --ref-cache cache/ec/esmc960/ref20k \
  --query-cache cache/ec/esmc960/enzymarc_catalytic \
  --head results/ec_models/head_ec_decoy_cluster_w3_seed0.pt \
  --out-dir results/ec/diag_w3_catalytic
```
