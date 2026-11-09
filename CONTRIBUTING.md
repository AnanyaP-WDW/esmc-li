# Contributing

ESMC-LI is a small Python package around a native ESM-C runtime. The
user-facing surface is intentionally narrow; the paper and benchmark harness
live in the same repository.

## Development setup

```bash
git clone https://github.com/AnanyaP-WDW/esmc-li.git
cd esmc-li
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The package is importable without the native library. For a source runtime,
build `esmc.cpp` separately and set one of:

```bash
export ESMC_LIB=/path/to/libesmc.dylib
# or use the CLI fallback
export ESMC_EMBED=/path/to/esmc-embed
```

## Tests

```bash
pytest -q
make audit
```

`make audit` runs the package tests plus the paper lock, claims, citations, and
zero-diff checks.

## Development checks

```bash
python scripts/m8_smoke.py     # 1K SCOPe smoke: 10 held-out queries
python scripts/m8_profile.py   # encode/search profile
```

## Paper

The manuscript is `paper/main.md`. Its tables and figures are generated from
`paper/results.lock.json`:

```bash
make tables     # regenerate tables, figures, and paper/main.md
make claims     # regenerate docs/claims.md
```

Do not edit generated files by hand.

## Release

See [docs/release.md](https://github.com/AnanyaP-WDW/esmc-li/blob/main/docs/release.md).
The release workflow is `.github/workflows/release.yml` and uses TestPyPI/PyPI
Trusted Publishing through the `testpypi` and `pypi` GitHub environments.

## License

Code contributions are under the MIT License. Model weights and GGUF
derivatives remain under the Cambrian Open License; see `NOTICE` and
`CAMBRIAN-OPEN-LICENSE.md`.
