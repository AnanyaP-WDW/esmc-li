---
license: other
license_name: cambrian-open-license-agreement
license_link: https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement
library_name: gguf
pipeline_tag: feature-extraction
base_model: EvolutionaryScale/esmc-300m-2024-12
pretty_name: ESMC-LI 300M Homology v1
tags:
  - protein-language-model
  - esmc
  - protein-retrieval
  - homology
  - late-interaction
  - maxsim
  - gguf
  - esmc.cpp
  - esmc-li
---

# ESMC-LI 300M Homology v1

Fine-tuned **ESM-C 300M** for local protein homology and remote-homology
retrieval with residue-level late interaction (MaxSim).

This repository contains the **GGUF encoder** and the matching **128×960
projection head**. The first-party package downloads both automatically, so
you normally do not need to download files by hand.

**Links:** [GitHub](https://github.com/AnanyaP-WDW/esmc-li) ·
[PyPI](https://pypi.org/project/esmc-li/) ·
[Paper](https://github.com/AnanyaP-WDW/esmc-li/blob/main/paper/main.md) ·
[Indexes](https://huggingface.co/datasets/AnanyaPathak/esmc-li-indexes)

## Quickstart

```bash
pip install esmc-li

esmc-li index pull scope-1k
esmc-li search scope-1k \
  --sequence "GERRRSQLDRDQCAYCKEKGHWAKDCPKKPRGPRGPRPQT" \
  --top-k 5
```

```python
from esmc_li import ProteinIndex

index = ProteinIndex.load("scope-1k")
hits = index.search("GERRRSQLDRDQCAYCKEKGHWAKDCPKKPRGPRGPRPQT", k=5)
print(hits[0])
```

Build an index from your own FASTA; the package resolves this model from the
Hugging Face cache automatically:

```bash
esmc-li build proteins.fasta --output proteins.esmci
esmc-li search proteins.esmci --sequence "MKT..." --top-k 10
```

On macOS, `pip install "esmc-li[metal]"` enables the optional fused Metal
search backend.

### If you want the raw files

```python
from huggingface_hub import hf_hub_download

gguf = hf_hub_download(
    "AnanyaPathak/esmc-li-300m-homology-v1",
    "esmc-300m-homology-maxsim-v1-f16.gguf",
)
head = hf_hub_download(
    "AnanyaPathak/esmc-li-300m-homology-v1",
    "esmc-300m-homology-maxsim-v1-head.npy",
)
```

For a source/advanced install, pass these paths explicitly and set
`ESMC_LIB` or `ESMC_EMBED` for the native runtime. The package wheel is the
supported path.

## Files

| File | Size | SHA256 |
|---|---:|---|
| `esmc-300m-homology-maxsim-v1-f16.gguf` | 634 MiB | `e87b42f1a6344e57b7a0d8a6624cc9d7e2e15f34d4a6927447b4cb3c90209bab` |
| `esmc-300m-homology-maxsim-v1-head.npy` | 480 KiB | `b86b04e28a658ea9a3a209e476a6e130a9f3c83e6fc0ab7f7cd9a48694a09bc3` |

The GGUF is the fine-tuned encoder; the `.npy` is the frozen 128×960
projection head. Both are required. The file names retain the historical
`esmc-300m-homology-maxsim-v1` prefix.

## What it is

- **Backbone:** ESM-C 300M, 960-dimensional residue embeddings
- **Training data:** SCOPe 2.08 ASTRAL 40% domains, superfamily-positive pairs
- **Method:** fine-tune the last 3 transformer blocks + final LayerNorm,
  then a 960 → 128 linear projection with L2-normalized rows
- **Scoring:** raw MaxSim sum over query and candidate residue rows
- **Context:** 2048 tokens; long chains are chunked at 2046 residues with
  128-residue overlap
- **Seed:** 0

## Results

| Benchmark | Result |
|---|---|
| SCOPe ASTRAL40, cR@10 | **0.9065** (three-seed mean) |
| Full-length Swiss-Prot 100K, cR@10 | **0.8974** (MMseqs2 0.8304) |
| Full-length 100K, sensitivity at superfamily | **0.6048** (MMseqs2 0.4765) |
| 1K SCOPe smoke, top-1 superfamily | **10/10** |

These are retrieval results on the project’s frozen evaluation split. See the
[paper](https://github.com/AnanyaP-WDW/esmc-li/blob/main/paper/main.md) for
the protocol, baselines, and limitations.

## Intended use

Local protein homology / remote-homology retrieval, motif and domain search,
and ranking candidates for downstream analysis.

## Not intended for

- functional annotation without independent validation
- structure prediction
- clinical or diagnostic use
- claims of official EvolutionaryScale affiliation

## Limitations

- Exact search is interactive at 100K proteins and batch-oriented at 1M.
- The package supports macOS arm64 and Linux x86_64; no Windows or Intel
  macOS wheel in v1.
- macOS runtime binaries are unsigned.
- The model is downloaded on first use (~664 MB).
- This model was trained on SCOPe domains; full-length behavior is evaluated
  but not guaranteed for every protein family.

## Citation

```bibtex
@software{pathak2026esmcli,
  author  = {Pathak, Ananya},
  title   = {ESMC-LI: Fast Local Protein Retrieval with ESM-C Residue Embeddings and Late Interaction},
  version = {1.0.1},
  year    = {2026},
  url     = {https://github.com/AnanyaP-WDW/esmc-li}
}
```

## License

The ESM-C 300M model weights and these fine-tuned derivatives are governed by
the [EvolutionaryScale Cambrian Open License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)
and the [Acceptable Use Policy](https://www.evolutionaryscale.ai/policies/acceptable-use-policy).
A copy of the agreement is included in the project repository at
[CAMBRIAN-OPEN-LICENSE.md](https://github.com/AnanyaP-WDW/esmc-li/blob/main/CAMBRIAN-OPEN-LICENSE.md).

Built with ESM. `esmc.cpp` and the ESMC-LI Python package are separate
open-source components; see their repositories.
