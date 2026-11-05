**Table 2. SCOPe ASTRAL40 retrieval**

| Method | cR@1 | cR@5 | cR@10 | cR@100 | MRR | sens family | sens superfamily | sens fold |
|---|---|---|---|---|---|---|---|---|
| Random | 0.0389 | 0.0445 | 0.0454 | 0.0615 | 0.1192 | 0.0008 | 0.0007 | 0.0005 |
| MinHash 5-mer | 0.3254 | 0.2242 | 0.1956 | 0.1624 | 0.4333 | 0.0356 | 0.0182 | 0.0155 |
| MMseqs2 | 0.8578 | 0.7100 | 0.6179 | 0.3424 | 0.8695 | 0.1753 | 0.2402 | 0.2166 |
| Pooled ESM-C | 0.6089 | 0.4640 | 0.4045 | 0.3313 | 0.6848 | 0.0834 | 0.0756 | 0.0671 |
| Frozen MaxSim 960D | 0.8678 | 0.7182 | 0.6304 | 0.4817 | 0.8937 | 0.1770 | 0.2264 | 0.2029 |
| Frozen MaxSim 128D | 0.8297 | 0.6603 | 0.5717 | 0.4391 | 0.8620 | 0.1636 | 0.1903 | 0.1695 |
| ProtoCol frozen (reproduced) | 0.8829 | 0.8335 | 0.8096 | 0.7421 | — | — | — | — |
| ProtoCol trained (reproduced) | 0.9447 | 0.9064 | 0.8933 | 0.8771 | — | — | — | — |
| ESM-C + trained head (M2) | 0.9549 ± 0.0014 | 0.8965 ± 0.0012 | 0.8574 ± 0.0023 | 0.7502 ± 0.0072 | 0.9654 ± 0.0007 | 0.2305 ± 0.0011 | 0.4369 ± 0.0012 | 0.4051 ± 0.0022 |
| ESM-C fine-tuned k=3 (M3) | 0.9705 ± 0.0010 | 0.9305 ± 0.0020 | 0.9065 ± 0.0014 | 0.8329 ± 0.0034 | 0.9765 ± 0.0006 | 0.2426 ± 0.0019 | 0.5566 ± 0.0058 | 0.5275 ± 0.0069 |

Leave-one-out; cR@k is ProtoCol's capped recall. M2/M3 are mean ± population std over the seeds; all other rows are single runs. ProtoCol's released evaluator reports only capped recall.
