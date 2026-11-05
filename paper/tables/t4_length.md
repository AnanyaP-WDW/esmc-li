**Table 4. Length-normalization ablation**

| Method | Normalization | cR@10 | sens superfamily | rho(len, score) |
|---|---|---|---|---|
| ESM-C fine-tuned k=3 | raw | 0.8974 | 0.6048 | 0.172 |
| ESM-C fine-tuned k=3 | ÷ len_A | 0.8974 | 0.6048 | 0.172 |
| ESM-C fine-tuned k=3 | ÷ len_B | 0.0825 | 0.0054 | 0.172 |
| ESM-C fine-tuned k=3 | ÷ sqrt(len_A len_B) | 0.6926 | 0.3010 | 0.172 |
| Frozen MaxSim 960D | raw | 0.8024 | 0.3914 | 0.069 |
| Frozen MaxSim 960D | ÷ len_A | 0.8024 | 0.3914 | 0.069 |
| Frozen MaxSim 960D | ÷ len_B | 0.0090 | 0.0000 | 0.069 |
| Frozen MaxSim 960D | ÷ sqrt(len_A len_B) | 0.0570 | 0.0101 | 0.069 |

Full-length 100K corpus. div_lenA is rank-invariant within a query; the candidate-length variants change the ranking.
