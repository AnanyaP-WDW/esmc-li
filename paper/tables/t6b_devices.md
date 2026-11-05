**Table 6b. Search backends at 100K (PyLate at 1K)**

| Backend | n_db | q/s | s/query | index MB | GEMM eff % |
|---|---|---|---|---|---|
| numpy CPU | 100,000 | 0.394 | 2.536 | 8717.9 | 91.7 |
| MPS | 100,000 | 0.564 | 1.774 | 8717.9 | 43.7 |
| Metal (fused) | 100,000 | 1.001 | 0.999 | 8717.9 | — |
| FAISS flat (pooled) | 100,000 | 7,559 | 0.00014 | 51.2 | — |
| PyLate/PLAID | 1,000 | 23.4 | 0.0427 | 16.4 | — |

CPU/MPS/Metal are exact MaxSim. FAISS is an approximate latency floor over pooled vectors, not a quality-comparable retriever. PyLate/PLAID is 1K only because full-DB encoding was infeasible here.
