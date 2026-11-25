# EC local-objective 32-triple overfit

**Date:** 2026-11-25
**Branch:** `exp/ec-local-objective`
**Status:** **negative.** The local patch objective overfits 32 triples but does
not improve full-sequence parent-vs-decoy discrimination. The global decoy
triplet is the only variant that reaches 32/32.

## What was tested

The approved intervention was `L = L_global + λ · L_local`, with λ = 1.0 and a
local patch objective on active-site windows. The first deliverable was a
32-triple overfit plus a random-window control. This run tests whether the local
loss can fit 32 triples before any 100K/153K scale-up.

## Triples

32 triples, one per EC4, all cluster-disjoint:

- **parent:** `train_5k` protein with experimental active sites and a catalytic
  Ala/Gly decoy.
- **decoy:** parent with active-site residues mutated to Ala/Gly.
- **positive:** same EC4 as the parent, in a different MMseqs cluster
  (30% identity, 80% coverage).
- 32 unique EC4s; 0 parent/positive pairs in the same cluster; 85 active sites
  total.

The positives are cluster-disjoint from the parents, so the 32-overfit is not a
homology test. It is a within-triple discrimination test: can the model put the
parent closer to a same-EC4 positive than its own active-site-disrupted decoy?

## Losses

- **Global:** symmetric InfoNCE over full-sequence MaxSim, τ = 0.1.
- **Global decoy:** full-sequence margin triplet
  `relu(0.3 - (s(parent, positive) - s(decoy, positive)))`.
- **Local patch (patch-vs-patch):** 2-way contrastive on active-site windows
  (±8 residues), parent patch vs positive patch vs decoy patch.
- **Local patch (local query, full document):** parent patch query against the
  full positive/decoy documents.
- **Local patch (full query, local document):** full parent query against the
  positive/decoy active-site patches.
- **Random-window control:** same local losses, but windows are sampled so their
  full span avoids active-site windows.

All variants use head-only 960→128 projection, AdamW lr 1e-3, batch 8, 300
steps, seed 0. The local loss weight is 1.0.

## Results

Full accuracy is the parent-vs-decoy paired accuracy on the full sequence
(fraction of the 32 triples where `s(parent, positive) > s(decoy, positive)`).
Active/random accuracy is the same paired accuracy on the active-site or
random-window patch. Full/active/random AUC is the global AUC over all 32
positives and 32 negatives; it is secondary because scores are not comparable
across queries. Local loss is the final local component.

| Variant | Full acc | Full AUC | Active acc | Active AUC | Random acc | Random AUC | Local loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `global` | 0.906 | 0.536 | 0.969 | 0.579 | 0.812 | 0.541 | — |
| `global+global_decoy` | **1.000** | 0.544 | 0.969 | 0.587 | 0.969 | 0.547 | — |
| `global+local_active` | 0.781 | 0.533 | **1.000** | 0.687 | 0.594 | 0.545 | 0.085 |
| `global+local_random` | 0.688 | 0.526 | 0.812 | 0.584 | 0.906 | 0.636 | 0.484 |
| `global+local_active_full` | 0.781 | 0.541 | 0.875 | 0.683 | 0.625 | 0.541 | 0.149 |
| `global+local_random_full` | 0.750 | 0.532 | 0.750 | 0.573 | 0.719 | 0.571 | 0.535 |
| `global+local_active_doc` | 0.906 | 0.576 | 0.969 | 0.698 | 0.938 | 0.587 | 0.021 |
| `global+local_random_doc` | 0.938 | 0.572 | 0.938 | 0.615 | 0.938 | 0.588 | 0.023 |
| `local_active` | 0.156 | 0.483 | 0.281 | 0.466 | 0.156 | 0.456 | 0.032 |
| `local_random` | 0.281 | 0.491 | 0.406 | 0.470 | 0.344 | 0.482 | 0.200 |

## Findings

1. **The local loss overfits the patch.** `global+local_active` reaches active
   patch accuracy 1.000 and local loss 0.085. `global+local_active_doc` reaches
   local loss 0.021. The local objective can fit its own 32 triples.

2. **The local loss does not transfer to the full sequence.** The same variants
   reach full accuracy 0.781 and 0.906, both below the global decoy triplet's
   1.000. The patch fit is not the bottleneck; the full-sequence aggregation is.
   The head is a per-residue projection and cannot reweight the MaxSim sum, so
   making active-site residues more discriminative does not make them dominate
   the full-sequence score.

3. **The global decoy triplet is the only variant that reaches 32/32.** It is a
   full-sequence margin triplet, not a local patch loss. The first pass already
   tested a global decoy triplet on 2K sequence-only decoys and saw no EnzymARC
   FPR gain. The 32-overfit does not revive that result; it only shows the
   global triplet can fit 32 cluster-disjoint triples.

4. **The active-window local loss is better than the random-window control, but
   both are worse than the global baseline.** On patch-vs-patch, active 0.781
   vs random 0.688 full accuracy; on the active patch, active 0.687 vs random
   0.584 AUC. The locality effect is real but small. On the full-query/local-
   document formulation, active and random are statistically tied (0.906 vs
   0.938 full accuracy). The active-site locality is not a robust advantage.

5. **Local-only variants collapse.** `local_active` and `local_random` reach
   full accuracy 0.156 and 0.281. A local loss without the global loss is not a
   useful training signal.

## Gate

The approved gate was: `global+local_active` full accuracy 32/32 and local loss
<0.1. Result: **25/32 (0.781) full accuracy, 0.085 local loss.** The local loss
passes; the full-sequence accuracy fails. The local objective is not a viable
training signal under the fixed architecture.

## Recommendation

- **Do not scale to 100K/153K with the local patch objective.** The 32-overfit
  gate fails on the metric that matters. More data will not fix a loss that
  cannot move the full-sequence score on 32 triples.
- **If continuing EC research, the global decoy triplet is the only variant
  that passes the 32-overfit.** But the first pass already tested a global
  decoy triplet on 2K sequence-only decoys and saw no EnzymARC FPR gain. A
  scale-up would need a new reason to expect the global triplet to generalize,
  not just a 32-triple fit.
- **For user value within a month, switch to the 100K `annotate` product.**
  The 20K retrieval/evidence prototype works; the 100K index is the next step.
  The negative local-objective result is the honest research outcome.

## Caveats

- 32 triples, one seed, head-only, 300 steps. This is a debugging test, not a
  full evaluation. No EnzymARC FPR is measured here.
- The random-window control is not a perfect "identical patch" control: ESM-C
  embeddings are contextual, so mutating active sites changes embeddings at all
  positions. The parent/decoy random patches differ slightly (max abs diff
  ~0.33 on one triple). The control still tests whether the head can use the
  contextual effect; the active-window control is better, so the locality effect
  is real but small.
- The positives are cluster-disjoint from the parents, so the 32-overfit is not
  a homology test. The decoy is the parent's mutation. The full accuracy is a
  within-triple discrimination, not a reference FPR.
- The full/active/random AUC is a global AUC over 32 positives and 32 negatives.
  Scores are not comparable across queries, so the paired accuracy is the
  primary metric. The global AUC is reported for completeness.

## Reproduce

```bash
.venv/bin/python scripts/ec_overfit32.py --out results/ec/overfit32_final
```

Defaults: `--n 32 --window 8 --steps 300 --batch 8 --lr 1e-3 --tau 0.1
--lambda 1.0 --seed 0`.

## Artifacts

- `results/ec/overfit32_final/overfit32_results.json`
- `results/ec/overfit32_final/overfit32_triples.tsv`
- `results/ec/overfit32_final/overfit32_scores.csv`
- `scripts/ec_overfit32.py`
- `tests/test_ec_overfit32.py`
