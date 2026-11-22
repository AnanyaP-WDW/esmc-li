# Homolog-free decoy diagnostic

**Date:** 2026-11-22
**Question:** is the ~67% EnzymARC FPR a model failure or a close-homolog
label-transfer artifact?

## Setup

- 500 catalytic and 500 5 Å EnzymARC decoys (first rows of each file).
- Reference: `ref20k` with exact parents removed (19,788 proteins).
- Parent sequences fetched from UniProt 2026_04; all 514 unique parents found.
- MMseqs2 `easy-search` parents vs reference: `--min-seq-id 0.3 -c 0.8`.
- Removed every reference protein with **≥30% identity and ≥80% coverage** to
  any sampled decoy parent.
- Re-ran the same head-only models on the filtered reference.

## Reference reduction

| Category | Ref before | Ref after | Parents with ≥30% homolog |
|---|---:|---:|---:|
| catalytic | 19,788 | 16,740 | **378 / 500** |
| 5 Å | 19,788 | 16,685 | **384 / 500** |

Most decoy parents had at least one close homolog in the 20K reference. This is
why the unfiltered FPR was so high.

## FPR before vs after homolog removal

Top-1 prediction of the parent EC (i.e. false positive):

| Category | Model | Unfiltered FPR | Homolog-free FPR |
|---|---|---:|---:|
| catalytic | head, decoy loss 0.3 | 0.672 | **0.340** |
| catalytic | head, no decoy loss | 0.676 | **0.344** |
| 5 Å | head, decoy loss 0.3 | 0.674 | **0.334** |
| 5 Å | head, no decoy loss | 0.674 | **0.340** |

Removing 30% homologs cuts the FPR roughly in half. The remaining one-third is
not explained by close-homolog retrieval.

## FPR by parent → reference identity

Catalytic decoys, decoy-aware head:

| Max identity to reference | FPR |
|---|---:|
| <30% | 0.320 |
| 30–50% | 0.300 |
| 50–70% | 0.374 |
| 70–90% | 0.291 |
| ≥90% | 0.464 |

5 Å decoys, decoy-aware head:

| Max identity to reference | FPR |
|---|---:|
| <30% | 0.302 |
| 30–50% | 0.292 |
| 50–70% | 0.403 |
| 70–90% | 0.253 |
| ≥90% | 0.451 |

There is no identity bin where the FPR approaches zero. Even with no ≥30%
homolog in the reference, the model assigns the parent EC about one-third of the
time.

## Decoy loss effect

Decoy-aware training changes the FPR by at most ~1 point in every bin; it is
within noise. The triplet term did not teach active-site sensitivity.

## Conclusion

- The original 67% FPR was **roughly half homolog transfer**; the diagnostic was
  worth running.
- The residual 32–34% FPR on homolog-free references means the model still
  retrieves the parent EC from distant proteins that share global fold/function
  but not close sequence identity.
- Late interaction at 128-D with this training recipe is **not** an active-site
  sensor. The EC method claim remains **no-go**.
- The remaining signal is useful for retrieval/evidence, not for a competitive
  EC classifier.

## Artifacts

- `results/ec/nohom_catalytic_decoy2/` — filtered ref, hits, parent identities,
  per-query predictions.
- `results/ec/nohom_catalytic_plain2/`
- `results/ec/nohom_5A_decoy2/`
- `results/ec/nohom_5A_plain2/`
- `data/ec/splits/enzymarc_parents_500.tsv`

## What would change the conclusion

A full 100K–153K reference and suffix fine-tune could shift the absolute FPR,
but the diagnostic shows the current model is not locally sensitive even when
close homologs are removed. Scaling the same recipe is unlikely to fix the
failure mode; if continuing, the next experiment should change the training
signal (active-site decoys with cluster-filtered negatives, or a structure-aware
local objective), not just the reference size.
