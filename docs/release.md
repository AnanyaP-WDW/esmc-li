# Release checklist (v1.0.0)

The API is frozen in `docs/api.md`. The wheel bundles `libesmc` and its ggml
dylibs; it is not a pure-Python wheel.

## Artifacts

Target one wheel per platform:

```text
esmc_li-1.0.0-cp310-abi3-macosx_11_0_arm64.whl
esmc_li-1.0.0-cp310-abi3-manylinux_2_35_x86_64.whl
```

- macOS arm64 only; no Intel macOS or Windows wheel in v1.
- Linux glibc 2.35+ (Ubuntu 22.04 class). Build with `GGML_NATIVE=OFF` and
  AVX2 enabled.
- No sdist in v1. Source users install from GitHub and supply `ESMC_LIB` or
  `ESMC_EMBED`.

## Build

Use a dedicated release environment; the dev `.venv` is not a release
environment.

```bash
python3 -m venv /tmp/esmc-release
source /tmp/esmc-release/bin/activate
python -m pip install --upgrade pip build twine
```

```bash
rm -rf build dist *.egg-info
ESMC_CPP=/path/to/esmc.cpp make wheel PYTHON=python3
twine check dist/*.whl
```

The macOS build must produce a `macosx_11_0_arm64` tag. The Linux build is
repaired with:

```bash
python -m pip install auditwheel
auditwheel repair --plat manylinux_2_35_x86_64 -w dist-repaired dist/*.whl
rm dist/*.whl
mv dist-repaired/*.whl dist/
```

Inspect contents:

```bash
python -m zipfile -l dist/*.whl | grep -E 'esmc_li/_runtime|data/head|LICENSE|NOTICE'
```

The wheel must not contain `esmc_retrieval`, `cache/`, `results/`, `paper/`,
`build/`, or `__pycache__`.

## TestPyPI

```bash
twine upload \
  --repository-url https://test.pypi.org/legacy/ \
  dist/*.whl
```

Install in a clean venv:

```bash
python -m venv /tmp/esmc-testpypi
source /tmp/esmc-testpypi/bin/activate
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  esmc-li==1.0.0
esmc-li --help
```

Then run a real search on macOS arm64 and Linux x86_64:

```bash
esmc-li index pull scope-1k
esmc-li search scope-1k \
  --sequence "GERRRSQLDRDQCAYCKEKGHWAKDCPKKPRGPRGPRPQT" \
  --top-k 5
```

Do not promote until both platforms pass.

## PyPI

Preferred: Trusted Publishing from `.github/workflows/release.yml`.

Manual fallback:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
twine upload dist/*.whl
```

Never reuse a version after a failed production upload. Yank and bump.

## Publish settings

- TestPyPI trusted publisher: repository `AnanyaP-WDW/esmc-li`, workflow
  `release.yml`, environment `testpypi`.
- PyPI trusted publisher: same repository/workflow, environment `pypi`.
- Add required reviewers to the `pypi` GitHub environment.
- Tag production with `v1.0.0`; publish the exact wheels tested on TestPyPI.

## Post-release

```bash
git tag -a v1.0.0 -m "ESMC-LI v1.0.0"
git push origin v1.0.0
```

Create a GitHub release with wheel SHA256 checksums, supported platforms, and
model/index links. Update `README.md` from local-wheel instructions to
`pip install esmc-li`.

## Rollback

PyPI files are immutable. If a wheel is broken:

1. Yank `1.0.0` in the PyPI UI.
2. Fix and publish `1.0.1`.
3. Never delete/re-upload the same version.

TestPyPI can be cleaned; PyPI cannot.
