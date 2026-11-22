# EC active-site late-interaction — results (first pass)

**Status:** preliminary; head-only, 5K training subset, 20K retrieval reference.
This is **not** the final planned 20K-training / 100K-reference / suffix-finetuned run.
The negative decoy result is real; the absolute EC numbers are not comparable to
CLEAN/FuncSeek because the reference is smaller and the backbone is not suffix-tuned.

## Setup

- Backbone: base ESM-C 300M (`esmc-300m-f16.gguf`), 960-D residue rows.
- Head: 960 → 128 linear, L2 rows, MaxSim.
- Train: `data/ec/splits/train_5k.tsv` (5,000 proteins, S1/CARE universe).
- Validation: `data/ec/splits/val_5k.tsv` (5,000).
- Retrieval reference: `data/ec/splits/ref_20k.tsv` (20,000 S1 proteins).
- Methods: pooled mean vector, frozen 960-D MaxSim, head-only EC fine-tune.
- Seeds: 0 only (head-only; no suffix/3 seeds yet).

## CARE retrieval

| Test | Method | Ref | EC4 top-1 | top-10 | MRR | multi-label F1 |
|---|---|---:|---:|---:|---:|---:|
| CARE ood30 | pooled | 20K | 0.0926 | 0.1551 | 0.1101 | 0.0375 |
| CARE ood30 | frozen MaxSim 960D | 5K | 0.2755 | 0.3588 | 0.3026 | 0.0793 |
| CARE ood30 | head, tau=0.1 | 20K | 0.4745 | 0.5532 | 0.5020 | 0.2553 |
| CARE ood30 | head, tau=1.0 | 20K | **0.4769** | **0.5509** | **0.5016** | **0.2499** |
| CARE 30–50 | head, tau=1.0 | 20K | 0.7424 | 0.8140 | 0.7679 | 0.4614 |
| CARE promiscuous | head, tau=1.0 | 20K | 0.8325 | 0.8947 | 0.8532 | 0.3946 (multi) |
| CARE Price-149 | head, tau=1.0 | 20K | 0.1554 | 0.4189 | 0.2198 | 0.1361 |

Published reference points (FuncSeek README; different training/reference):
CARE ood30 top-1 EC4: FuncSeek **0.646**, CLEAN **0.551**, BLASTp **0.514**.
CARE promiscuous: FuncSeek **0.937**, CLEAN **0.694**, BLASTp **0.937**.

Our 20K reference is ~13% of the S1 training universe; the published methods use
their full training reference sets. Do not compare the absolute numbers.

## Active-site decoy results

### Internal catalytic decoys (Ala/Gly scan, 1,907 train / 212 val)

| Reference | Model | Top-1 parent EC | Top-10 parent EC |
|---|---|---:|---:|
| 1K train (parents removed) | head, no decoy loss | 0.090 | 0.100 |
| 1K train (parents removed) | head, decoy loss 0.3 | 0.085 | 0.105 |

No meaningful improvement; the decoy-aware head is essentially tied.

### EnzymARC external decoys (first 500 rows; parents exact-filtered)

| Category | Reference | Model | Top-1 parent EC (FPR) | Top-10 |
|---|---|---|---:|---:|
| catalytic | 20K ref20k minus parents | head, decoy loss 0.3 | **0.672** | 0.738 |
| catalytic | 20K ref20k minus parents | head, no decoy loss | 0.676 | 0.736 |
| 5 Å | 20K ref20k minus parents | head, decoy loss 0.3 | **0.674** | 0.742 |
| 5 Å | 20K ref20k minus parents | head, no decoy loss | 0.674 | 0.742 |

The decoy loss does not reduce the external FPR; exact-parent filtering leaves
close homologs in the reference, and the model still retrieves the parent EC.
This is the same failure mode reported for CLEAN/DIAMOND in the EnzymARC paper.

### Multi-domain validation (not an independent test)

On the `val_5k` subset with ≥2 Pfam domains (2,190 proteins), the head-only
model gets EC4 top-1 **0.936** and top-10 **0.963** against the 20K reference.
This is validation data, not a held-out test, and the reference shares the S1
universe; treat it as a smoke signal only. The single-domain comparison did not
finish within the timebox.

## Product prototype

`scripts/ec_annotate.py` works on the 20K EC index:

```bash
esmc-li-like annotate data/ec/reference/ec_head20k.esmci \
  --sequence "<protein>" --head results/ec_models/head_ec_tau1_seed0.npy --k 5
```

Example: a query from CARE ood30 returned top-5 neighbors all EC `1.1.1.35`,
consensus `1.1.1.35`, confidence 1.0.

## Artifacts

- `data/ec/splits/` — splits and reference sets
- `data/ec/manifest.json` — raw counts/hashes
- `results/ec_models/head_ec_seed0.pt` (tau=0.1), `.npy`
- `results/ec_models/head_ec_tau1_seed0.pt` (tau=1.0), `.npy`
- `results/ec_models/head_ec_decoy_seed0.pt` (tau=1.0 + decoy loss 0.3), `.npy`
- `data/ec/reference/ec_head20k.esmci` + `.json` (head-only, 20K)
- `results/ec/*.csv` — per-query predictions and metrics
- `results/ec/*.json` — aggregate metrics
- `results/ec/valdecoy_*.csv`, `results/ec/enzymarc_*.csv`

## Caveats

- Only 5K training proteins and seed 0; no suffix fine-tune, no 3 seeds.
- Reference is 20K, not 100K/153K; absolute CARE numbers are pessimistic.
- No full MMseqs2 30% cluster scan; exact accession overlap only.
- No EC-Bench Track B run yet.
- No structure-based training decoys (5/10/15 Å); only EnzymARC external.
- Price source file: 149 lines incl header; parsed 148 rows.
- S3 TrEMBL: 6,893/8,031 sequences available; 1,138 obsolete/missing.
- EnzymARC parsed 76,480/79,505 catalytic (complete EC4/valid sequence filter).
