**Table 6a. Exact search and coarse stage, 10K–1M**

| n_db | residues | index GB | build h | open ms | exhaustive s/q | q/s | C | candidate recall | coarse s/q | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 3,151,934 | 0.81 | 0.17 | 6.5 | 0.112 | 8.905 | 3,000 | 0.994 | 0.106 | 1.1 |
| 100,000 | 31,787,100 | 8.14 | 1.79 | 7.8 | 1.150 | 0.869 | 10,000 | 0.996 | 0.450 | 2.6 |
| 1,000,000 | 317,031,935 | 81.17 | 17.49 | 6.0 | 11.134 | 0.0898 | 30,000 | 0.994 | 10.261 | 1.1 |

M4 Max, fused Metal, 50 queries. Build/index from UniRef50 subsets; coarse stage reranks C candidates with exact MaxSim.
