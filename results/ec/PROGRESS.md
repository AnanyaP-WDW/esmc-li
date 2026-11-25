# EC experiment progress

Branch: `exp/ec-local-objective` (from `exp/ec-active-site-late-interaction`)
Status: first pass completed; EC method gate failed; local-objective 32-overfit
failed; retrieval/evidence prototype kept.

| Phase | Exit criterion | Status |
|---|---|---|
| 0 Preflight | clean branch; pytest; deps | pass |
| 1 Data card/splits | counts, exact accession overlap, licenses | partial: exact overlap pass; full 30% MMseqs2 scan not run; EnzymARC license unclear |
| 2 Decoys | 10–20K pairs; no test parents | partial: 2,119 catalytic pairs (Ala/Gly); no 5/10/15 Å training decoys |
| 3 Eval/baselines | deterministic; baselines; tests | pass |
| 4 Tiny overfit | 32/32; loss <0.1; no NaN | pass (loss reaches 0; 32/32) |
| 5 Head-only | val EC4 top-1 > frozen/pooled | pass locally: 0.57 vs 0.28 (5K ref); CARE ood30 0.477 vs 0.093 pooled |
| 6 Suffix | 3 seeds; val gain | not run (compute/timebox) |
| 7 Decoy-aware | decoy FPR/AUC improves | fail: internal 8.5% vs 9.0%; EnzymARC 67.2% vs 67.6% |
| 8 External | beat FuncSeek/CLEAN or ≥50% FPR cut | fail: 47.7% CARE ood30 vs 64.6/55.1; no FPR cut |
| 9 Decision | explicit go/no-go | **no-go for EC method claim**; homolog-free diagnostic halved FPR to ~34% but did not eliminate it; see DECISION.md and DIAGNOSTIC.md |
| 10 Product | 100K index; CLI | partial: 20K head index + `ec_annotate.py` works; 100K not built |
| 11 Write-up | reproducible claims | summary + data card + decision written; no SOTA claim |
| 12 Local-objective 32-overfit | 32/32 full accuracy; local loss <0.1 | fail: local loss 0.085 but full accuracy 25/32; global decoy triplet 32/32; see OVERFIT32.md |

## Blockers / deviations

- `S1` has 153,888 rows but 153,295 unique accessions present in UniProt 2026_04; 16 missing.
- `S2` 18,224/18,293 unique accessions.
- `S3` 6,893/8,031 sequences available; 1,138 obsolete/missing in current UniProt.
- EnzymARC parses to 76,480 catalytic / 70,747 / 70,783 / 70,783 decoys with complete EC4 and valid sequence; source headers/records differ from README counts.
- `price_protein_test.csv` is 149 lines incl header; parsed 148 rows.
- Full MMseqs2 30% cluster scan not completed; exact accession overlap only.
- Homology-model zero-shot 20K index build was killed (memory); no zero-shot fine-tuned-GGUF baseline.
- Frozen 960-D ref20K eval was killed (memory); frozen 5K result is reported instead.
- Single-domain validation eval killed after 31 min; multi-domain validation result is reported, but it is validation, not an independent test.
- No EC-Bench Track B run; no suffix fine-tune; no 3 seeds.
- Decoy training is sequence-only catalytic Ala/Gly, not structure-based shells.

## Artifacts

See `results/ec/SUMMARY.md`, `results/ec/DATA_CARD.md`, and
`results/ec/OVERFIT32.md`. Run log: `results/ec/runs.jsonl`.
