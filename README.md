# esmc-retrieval

Fast local protein retrieval using ESM-C residue embeddings and late
interaction (MaxSim). Exact search runs on numpy; ESM-C inference runs through
[`esmc.cpp`](https://github.com/AnanyaP-WDW/esmc.cpp).

## Requirements

- Python 3.10+ and numpy
- `esmc-embed` built from esmc.cpp (commit `28a3913` or later, which adds
  `--keep-special`)
- The fine-tuned retrieval GGUF `esmc-300m-ft-f16.gguf` (the query encoder)
- The packaged projection head (`esmc_retrieval/data/head_k3_seed0.npy`)

Set the runtime paths once:

```bash
export ESMC_EMBED=/path/to/esmc.cpp/build/esmc-embed
export ESMC_RETRIEVAL_GGUF=/path/to/esmc-300m-ft-f16.gguf
```

## Install

```bash
pip install .
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

- Search is exact MaxSim over projected residue rows, implemented in numpy.
- `search_batch` uses one `esmc-embed` process for all queries.
- The index is not self-contained: keep the matching head and FT GGUF with it.
- `--keep-special` is required because the frozen scorer was trained on rows
  including BOS/EOS.
- ESM-C weights and derivatives are governed by the EvolutionaryScale
  Cambrian Open License Agreement.

## M8 smoke

```bash
python scripts/m8_smoke.py
```

Builds a 1K SCOPe index and checks top-1 superfamily recovery on 10 held-out
queries. Measured result: 10/10.
