**Table 7. Compression trade-off (100K full-length)**

| Scheme | ratio vs f16 | cR@10 | drop cR@10 | pass |
|---|---|---|---|---|
| fp16 | 1.00 | 0.8974 | 0.0000 | no |
| int8 | 2.00 | 0.8972 | 0.0002 | no |
| residual int8 | 2.00 | 0.8972 | 0.0002 | no |
| residual PQ m8 | 28.43 | 0.8513 | 0.0461 | no |
| residual PQ m16 | 15.06 | 0.8816 | 0.0158 | no |
| residual PQ m32 | 7.76 | 0.8921 | 0.0052 | yes |

Pass = >=4x smaller and cR@10 drop <=0.01. Only PQ m32 passes; it was not shipped because decode+MaxSim was slower than f16.
