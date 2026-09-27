# M1 failure analysis (T1.5)

Scorer: `maxsim960` (L2-normalized ESM-C residue vectors, raw MaxSim sum), split `scope_test` (n=2314).
Rank-1 false positives (top-1 not same superfamily): **306/2314 = 13.2%**.

## 20 highest-scoring rank-1 false positives

| # | query | qlen | q fold | top-1 | clen | t fold | score | hypothesis |
|---|-------|------|--------|-------|------|--------|-------|------------|
| 1 | d2a65a1 | 509 | f.54 | d1rh5a_ | 422 | f.41 | 363.8 | same SCOP class f but different folds — surface/shape convergence at 960D MaxSim, no alignment to distinguish them |
| 2 | d6ddba_ | 467 | c.108 | d3qowa_ | 327 | c.66 | 337.6 | same SCOP class c but different folds — surface/shape convergence at 960D MaxSim, no alignment to distinguish them |
| 3 | d2cfua2 | 505 | d.157 | d1d8wa_ | 416 | c.1 | 334.0 | cross-class pair (SCOP d vs c) — MaxSim rewards any structurally similar local fragments regardless of class |
| 4 | d2ez2a_ | 456 | c.67 | d5nh6a_ | 436 | c.1 | 319.1 | same SCOP class c but different folds — surface/shape convergence at 960D MaxSim, no alignment to distinguish them |
| 5 | d2gi3a1 | 475 | c.117 | d1v9na_ | 360 | c.122 | 314.2 | same SCOP class c but different folds — surface/shape convergence at 960D MaxSim, no alignment to distinguish them |
| 6 | d1vpra1 | 351 | b.60 | d1iu4a_ | 331 | d.3 | 307.9 | cross-class pair (SCOP b vs d) — MaxSim rewards any structurally similar local fragments regardless of class |
| 7 | d1u2za_ | 406 | c.66 | d1x3za_ | 320 | d.3 | 302.0 | cross-class pair (SCOP c vs d) — MaxSim rewards any structurally similar local fragments regardless of class |
| 8 | d1iu4a_ | 331 | d.3 | d7b1sc_ | 265 | d.58 | 299.6 | same SCOP class d but different folds — surface/shape convergence at 960D MaxSim, no alignment to distinguish them |
| 9 | d2hoxa_ | 425 | c.67 | d4ohfa1 | 455 | c.108 | 296.3 | same SCOP class c but different folds — surface/shape convergence at 960D MaxSim, no alignment to distinguish them |
| 10 | d2i7va_ | 453 | d.157 | d4ohfa1 | 455 | c.108 | 295.4 | cross-class pair (SCOP d vs c) — MaxSim rewards any structurally similar local fragments regardless of class |
| 11 | d3qrya_ | 426 | a.102 | d5nh6a_ | 436 | c.1 | 294.2 | cross-class pair (SCOP a vs c) — MaxSim rewards any structurally similar local fragments regardless of class |
| 12 | d3tg7a2 | 310 | b.121 | d1dlca2 | 210 | b.77 | 286.0 | same SCOP class b but different folds — surface/shape convergence at 960D MaxSim, no alignment to distinguish them |
| 13 | d3ieka_ | 431 | d.157 | d1wyua_ | 437 | c.67 | 280.9 | cross-class pair (SCOP d vs c) — MaxSim rewards any structurally similar local fragments regardless of class |
| 14 | d2z0da1 | 350 | d.3 | d6ddba_ | 467 | c.108 | 254.0 | cross-class pair (SCOP d vs c) — MaxSim rewards any structurally similar local fragments regardless of class |
| 15 | d6xxja_ | 356 | c.31 | d6ddba_ | 467 | c.108 | 252.2 | same SCOP class c but different folds — surface/shape convergence at 960D MaxSim, no alignment to distinguish them |
| 16 | d1wiya_ | 385 | a.104 | d3ieka_ | 431 | d.157 | 251.2 | cross-class pair (SCOP a vs d) — MaxSim rewards any structurally similar local fragments regardless of class |
| 17 | d2q40a_ | 360 | e.50 | d3u7qa_ | 477 | c.92 | 243.8 | cross-class pair (SCOP e vs c) — MaxSim rewards any structurally similar local fragments regardless of class |
| 18 | d5a0ya1 | 268 | d.58 | d1t3ta4 | 209 | d.79 | 243.7 | same SCOP class d but different folds — surface/shape convergence at 960D MaxSim, no alignment to distinguish them |
| 19 | d7b1sc_ | 265 | d.58 | d1t3ta4 | 209 | d.79 | 243.7 | same SCOP class d but different folds — surface/shape convergence at 960D MaxSim, no alignment to distinguish them |
| 20 | d1dkia_ | 335 | d.3 | d1v5da_ | 386 | a.102 | 242.4 | cross-class pair (SCOP d vs a) — MaxSim rewards any structurally similar local fragments regardless of class |

## Aggregate: are FPs length-driven?

- rank-1 TP pairs: mean |len ratio| 1.11 (n=2008)
- rank-1 FP pairs: mean |len ratio| 2.27 (n=306)
- fraction of FPs crossing SCOP class boundary: 60.8%

## Residue similarity maps

Heatmaps of the [Lq x Lc] residue cosine matrix actually fed to MaxSim (`maxsim960`); the score is the sum over query rows of each row maximum.

- `docs/plots/m1_map_tp_homolog_d1rq5a1_d1clca1.png` — tp_homolog: row-max mean 0.799, off-diagonal background 0.423
- `docs/plots/m1_map_fp_cross_class_d2cfua2_d1d8wa_.png` — fp_cross_class: row-max mean 0.661, off-diagonal background 0.363
- `docs/plots/m1_map_fp_length_mismatch_d1z1ba1_d4ohfa1.png` — fp_length_mismatch: row-max mean 0.612, off-diagonal background 0.359
