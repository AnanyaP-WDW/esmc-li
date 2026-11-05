# Release checklist (v1.0.0)

The API is frozen in `docs/api.md`. The wheel bundles `libesmc` and its ggml
dylibs; it is not a pure-Python wheel.

1. Build the platform wheel against the pinned esmc.cpp checkout:

   ```bash
   ESMC_CPP=/path/to/esmc.cpp make wheel PYTHON=python3
   shasum -a 256 dist/esmc_li-1.0.0-*.whl
   ```

2. Check the wheel in a clean venv, offline after the model/index cache is
   warm:

   ```bash
   python3 -m venv /tmp/esmc-release
   /tmp/esmc-release/bin/pip install dist/esmc_li-1.0.0-*.whl
   /tmp/esmc-release/bin/python scripts/m8_smoke.py
   ```

   Expected: `10/10` top-1 superfamily on the 1K SCOPe smoke.

3. Verify the published artifacts:

   - model: `AnanyaPathak/esmc-li-300m-v1`
   - indexes: `AnanyaPathak/esmc-li-indexes`
   - SHA256 values in `data/manifest.json` and the model card.

4. Tag and upload (requires credentials; not run by the paper gate):

   ```bash
   git tag -a v1.0.0 -m "ESMC-LI v1.0.0"
   twine upload dist/esmc_li-1.0.0-*.whl
   ```

   If `twine` is unavailable, `python -m pip install --upgrade twine` first.
