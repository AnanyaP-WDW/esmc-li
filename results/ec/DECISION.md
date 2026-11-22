# EC experiment decision

**Date:** 2026-11-22
**Branch:** `exp/ec-active-site-late-interaction`
**Decision:** **No-go for an EC method/SOTA claim in this run.** Keep the
retrieval/evidence prototype, document the negative decoy result, and do not
publish CARE/EnzymARC numbers as a competitive EC method.

## Gate results

| Gate | Criterion | Result | Status |
|---|---|---|---|
| G0 data integrity | counts, zero accession overlap, licenses | counts recorded; train∩val/S3/CARE = 0; EnzymARC parent overlap 53,795; full 30% cluster scan not run; EnzymARC license unclear | **partial** |
| G1 eval harness | baselines run; deterministic; random near chance | `ec_eval.py` + tests; pooled/frozen/head runs deterministic; random baseline near chance | **pass** |
| G2 tiny overfit | 32/32 top-1; loss <0.1; no NaN | 32/32 top-1; loss reaches 0; no NaN | **pass** |
| G3 head-only | val EC4 top-1 beats frozen/pooled by ≥3 | val 5K: 0.57 vs 0.28 pooled; CARE ood30 0.477 vs 0.093 pooled | **pass (local), but below published** |
| G4 suffix | val ≥ head-only +1, or decoy gain | suffix fine-tune not run | **not run** |
| G5 decoy | parent-vs-decoy AUC / FPR improves | internal FPR 0.085 vs 0.090; EnzymARC 0.672 vs 0.676 catalytic; no gain | **fail** |
| G6 external | beat FuncSeek/CLEAN or ≥50% FPR reduction or multi-domain win | CARE ood30 0.477 vs FuncSeek 0.646 / CLEAN 0.551; no decoy reduction | **fail** |
| G7 product | 100K index; CLI; latency | 20K head-only index + annotate prototype works; 100K not built | **partial** |

## Why stop

The only real hypothesis was **locality**: residue-level late interaction should
be more sensitive to active-site disruption than global/single-vector methods.
The first pass does not show that:

- The decoy-aware head is statistically tied with the no-decoy head on internal
  catalytic decoys (8.5% vs 9.0% FPR).
- On EnzymARC external decoys, the FPR is ~67% at top-1 for both heads.
- Exact-parent filtering leaves close homologs in the reference; the model
  retrieves the parent EC from those homologs, which is the failure mode the
  EnzymARC benchmark was designed to expose.
- CARE ood30 top-1 is 47.7% with a 20K reference; even allowing for the smaller
  reference, this is below CLEAN 55.1% and FuncSeek 64.6%. No suffix training or
  full reference was run, so this is not a final comparison, but it is not a
  signal to keep scaling blindly.

## What is still worth keeping

- The data pipeline, leakage checks, splits, tests, and evaluation harness.
- The head-only EC retriever: 0.57 val top-1 vs 0.28 pooled, 0.477 CARE ood30
  with a 20K reference. It is a useful **retrieval/evidence** prototype even if
  it is not a SOTA EC classifier.
- `scripts/ec_annotate.py` returns top-k EC neighbours + consensus + confidence.
  It is the honest product fallback from the plan.

## What would change the decision

A future run needs all three:

1. **Full reference** (100K–153K) and **suffix fine-tune** with 3 seeds.
2. **30% cluster-disjoint** train/test and decoy parents removed by cluster,
   not just accession.
3. A decoy objective that actually lowers EnzymARC FPR at matched coverage
   (not just a triplet term on 2K sequence-only decoys).

If those still fail, the negative result is the paper: late interaction is not
the active-site sensor we hoped for. If they pass, revisit the EC claim.

## Next actions

- Do **not** update the model card or README with EC SOTA claims.
- Keep `results/ec/` and the data card for reproducibility.
- If continuing, run one focused experiment: full 100K reference + suffix
  fine-tune + 3 seeds, with cluster-filtered EnzymARC evaluation. Do not add
  new architecture.
