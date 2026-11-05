**Table 9. Package profile (appendix)**

| Index / backend | encode ms/q | search ms/q | search s/q | q/s |
|---|---|---|---|---|
| 1K numpy | 122.7 | 16.3 | 0.02 | 7.192 |
| 1K Metal | 61.5 | 19.0 | 0.02 | 12.4 |
| 100K numpy | 113.7 | 9057.8 | 9.06 | 0.109 |
| 100K Metal | 112.5 | 1294.0 | 1.29 | 0.711 |
| 1M numpy | 308.5 | 67051.5 | 67.05 | 0.015 |

M4 Max. Single-query encode+search; model/context stays loaded. The largest row is numpy (no Metal backend at that size in M8).
