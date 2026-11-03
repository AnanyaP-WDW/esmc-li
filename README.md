# esmc-retrieval

Fast local protein retrieval using ESM-C residue embeddings and late
interaction (MaxSim). Exact search runs on numpy, with an optional fused Metal
backend; ESM-C inference runs through
[`esmc.cpp`](https://github.com/AnanyaP-WDW/esmc.cpp).

Model: [`AnanyaPathak/esmc-300m-homology-maxsim-v1`](https://huggingface.co/AnanyaPathak/esmc-300m-homology-maxsim-v1)
Indexes: [`AnanyaPathak/esmc-retrieval-indexes`](https://huggingface.co/datasets/AnanyaPathak/esmc-retrieval-indexes)

## Requirements

- Python 3.10+ and numpy
- `esmc-embed` built from esmc.cpp (commit `218cc69` or later, which adds
  `--keep-special`)
- The fine-tuned retrieval GGUF
  `esmc-300m-homology-maxsim-v1-f16.gguf` (query encoder)
- The packaged projection head (`esmc_retrieval/data/head_k3_seed0.npy`)

Set the runtime paths once:

```bash
export ESMC_EMBED=/path/to/esmc.cpp/build/esmc-embed
export ESMC_RETRIEVAL_GGUF=/path/to/esmc-300m-homology-maxsim-v1-f16.gguf
```

## Install

```bash
pip install .
# optional, macOS: fused Metal search
pip install ".[metal]"
```

## Build an index

```bash
esmc-retrieval build proteins.fasta \
    --output proteins.esmci
```

The build writes `proteins.esmci` plus `proteins.esmci.json` and
`proteins.esmci.head.npy`. It embeds prefix-token rows (BOS/EOS kept),
projects them 960 -> 128, L2-normalizes, and stores f16 rows.

## Search

```bash
esmc-retrieval search proteins.esmci \
    --sequence "MKTIIALSYIFCLVFADYKDDDDK" \
    --top-k 10
```

Batch FASTA queries:

```bash
esmc-retrieval search proteins.esmci \
    --query queries.fasta \
    --output results.tsv
```

```python
from esmc_retrieval import ProteinIndex

index = ProteinIndex.load("proteins.esmci")
hits = index.search("MKTIIALSYIFCLVFADYKDDDDK", k=10)
```

## Notes

- Search is exact MaxSim over projected residue rows. `device="auto"` uses
  fused Metal when PyObjC is available on macOS, otherwise numpy.
- `search_batch` uses one `esmc-embed` process for all queries.
- Single-query `search()` pays one model load; use `search_batch` for throughput.
- The index is not self-contained: keep the matching head and FT GGUF with it.
- `--keep-special` is required because the frozen scorer was trained on rows
  including BOS/EOS.
- ESM-C weights and derivatives are governed by the EvolutionaryScale
  Cambrian Open License Agreement.

## M8 checks

```bash
python scripts/m8_smoke.py     # 1K SCOPe, 10 held-out queries -> 10/10
python scripts/m8_profile.py   # encode/search split
```

Profile on M4 Max: 1K ~19 ms search/query; 100K ~1.3 s/query with Metal
(numpy fallback is much slower).
