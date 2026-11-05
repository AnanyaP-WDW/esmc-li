**Table 3. Full-length Swiss-Prot retrieval**

| Method | cR@1 | cR@5 | cR@10 | cR@100 | MRR | sens family | sens superfamily |
|---|---|---|---|---|---|---|---|
| Random | 0.0020 | 0.0048 | 0.0040 | 0.0048 | 0.0183 | 0.0000 | 0.0000 |
| MinHash 5-mer | 0.8180 | 0.6756 | 0.5957 | 0.4092 | 0.8273 | 0.3299 | 0.2067 |
| MMseqs2 | 0.9540 | 0.8779 | 0.8304 | 0.6612 | 0.9542 | 0.7065 | 0.4765 |
| Pooled ESM-C | 0.8880 | 0.7544 | 0.6870 | 0.4643 | 0.8990 | 0.3213 | 0.2118 |
| Frozen MaxSim 960D | 0.9500 | 0.8528 | 0.8024 | 0.6355 | 0.9548 | 0.5854 | 0.3914 |
| Frozen MaxSim 128D | 0.9100 | 0.7985 | 0.7374 | 0.5530 | 0.9190 | 0.4862 | 0.3091 |
| ESM-C fine-tuned k=3 | 0.9780 | 0.9252 | 0.8974 | 0.8172 | 0.9792 | 0.7557 | 0.6048 |

100,000 single-clan chains; 500 test-clan queries; leave-one-out; raw MaxSim for the vector scorers.
