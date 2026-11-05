# ESMC-LI

Fast local protein retrieval using ESM-C residue embeddings and late
interaction (MaxSim). Exact search runs on numpy, with an optional fused Metal
backend; ESM-C inference runs through
[`esmc.cpp`](https://github.com/AnanyaP-WDW/esmc.cpp).

Model: [`AnanyaPathak/esmc-li-300m-v1`](https://huggingface.co/AnanyaPathak/esmc-li-300m-v1)
Indexes: [`AnanyaPathak/esmc-li-indexes`](https://huggingface.co/datasets/AnanyaPathak/esmc-li-indexes)

The Hugging Face model/index repositories are now ESMC-LI; their file names
keep the historical `esmc-300m-homology-maxsim-v1` prefix.

Paper: [`paper/main.md`](paper/main.md) · API: [`docs/api.md`](docs/api.md)

Code is MIT-licensed (`LICENSE`); the ESM-C weights and GGUF derivatives are
governed by the Cambrian Open License (`NOTICE`).

## Install

Prebuilt macOS arm64 / Linux x86_64 wheels include `libesmc`; no compiler or
separate esmc.cpp install is needed.

```bash
# from a release wheel (PyPI once published)
pip install dist/esmc_li-*.whl
# optional, macOS: fused Metal search
pip install pyobjc-framework-Metal
```

The fine-tuned GGUF is downloaded from Hugging Face on first use and cached
under `~/.cache/esmc-retrieval` (the historical cache path is kept so
existing downloads still work). The packaged projection head ships in the wheel. No model paths are required.

For a source install, `libesmc` is not bundled; set `ESMC_LIB` or use the
legacy CLI fallback (`ESMC_EMBED`). Release wheels are the supported path.

Optional overrides:

```bash
export ESMC_LIB=/path/to/libesmc.dylib   # source/dev only
export ESMC_RETRIEVAL_GGUF=/path/to/local.gguf
export ESMC_RETRIEVAL_CACHE=/custom/cache
```

## Prebuilt indexes

```bash
esmc-li index list
esmc-li index pull scope-1k
esmc-li search scope-1k --sequence "MKT..." --top-k 10
```

## Build an index

```bash
esmc-li build proteins.fasta --output proteins.esmci
```

The build writes `proteins.esmci` plus `proteins.esmci.json` and
`proteins.esmci.head.npy`. It embeds prefix-token rows (BOS/EOS kept),
projects them 960 -> 128, L2-normalizes, and stores f16 rows.

## Search

```bash
esmc-li search proteins.esmci \
    --sequence "MKTIIALSYIFCLVFADYKDDDDK" \
    --top-k 10
```

Batch FASTA queries:

```bash
esmc-li search proteins.esmci \
    --query queries.fasta \
    --output results.tsv
```

```python
from esmc_li import ProteinIndex

index = ProteinIndex.load("proteins.esmci")
hits = index.search("MKTIIALSYIFCLVFADYKDDDDK", k=10)
```

## Notes

- The in-process `libesmc` context stays loaded; single-query `search()` does
  not reload the model. Source/legacy CLI fallback does.
- Search is exact MaxSim over projected residue rows. `device="auto"` uses
  fused Metal when PyObjC is available on macOS, otherwise numpy.
- The index is not self-contained: keep the matching head and FT GGUF with it.
- The C API keeps BOS/EOS rows; no CLI `--keep-special` is needed on the
  runtime path.
- v1 macOS runtime binaries are unsigned; the package prints a warning. Verify
  the release checksum or build esmc.cpp locally if that matters.
- ESM-C weights and derivatives are governed by the EvolutionaryScale
  Cambrian Open License Agreement.

## Paper and reproducibility

The M9 paper is [`paper/main.md`](paper/main.md). All data tables and figures
are generated from a pinned results lock; no model or embedding cache is needed:

```bash
make tables     # regenerate 15 tables, 3 figures, and paper/main.md
make audit      # claims + citations + lock + tests + zero-diff
```

`make audit` currently reports 82/82 numeric claims mapped and 26/26 citations
verified. See [`paper/README.md`](paper/README.md) and
[`docs/claims.md`](docs/claims.md).

## M8 checks

```bash
python scripts/m8_smoke.py     # 1K SCOPe, 10 held-out queries -> 10/10
python scripts/m8_profile.py   # encode/search split
```

Profile on M4 Max: 1K ~19 ms search/query; 100K ~1.3 s/query with Metal
(numpy fallback is much slower).
