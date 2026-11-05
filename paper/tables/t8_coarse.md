**Table 8. Coarse candidate stage vs exhaustive**

| n_db | C | candidate recall | coarse s/q | exhaustive s/q | rerank agreement | speedup |
|---|---|---|---|---|---|---|
| 10,000 | 3,000 | 0.994 | 0.106 | 0.112 | 0.94 | 1.1 |
| 100,000 | 10,000 | 0.996 | 0.450 | 1.150 | 0.98 | 2.6 |
| 1,000,000 | 30,000 | 0.994 | 10.261 | 11.134 | 0.94 | 1.1 |

Candidate recall is recall of the exhaustive top-10 candidate set. At 1M, random candidate gather makes rerank almost exhaustive.
