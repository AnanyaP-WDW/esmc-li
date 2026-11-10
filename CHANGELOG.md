# Changelog

## 1.0.1 — 2026-11-10

- Polished README with white-background charts and retrieval, ProtoCol, and
  scaling graphs.
- Added `CONTRIBUTING.md` and `make readme-assets`.
- ABI3 marker extension compiled with the limited API.
- No public API changes.

## 1.0.0 — 2026-11-07

First PyPI release.

- `pip install esmc-li`; Python import `esmc_li`; CLI `esmc-li`.
- macOS arm64 and Linux x86_64 wheels bundling `libesmc`, ggml, and the
  projection head.
- Exact MaxSim search, memory-mapped `.esmci` indexes, and offline operation.
- Fine-tuned ESM-C 300M model and prebuilt indexes downloaded from Hugging
  Face on first use.
- Python 3.10+ via an ABI3 marker extension.
- No Windows or Intel-macOS wheel in this release.
- macOS runtime binaries are unsigned.
- Model weights and GGUF derivatives are governed by the EvolutionaryScale
  Cambrian Open License; see `NOTICE`.
