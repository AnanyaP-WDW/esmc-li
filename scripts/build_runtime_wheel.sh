#!/bin/bash
# Build a platform wheel with libesmc + its ggml dylibs bundled.
#
# macOS arm64 and Linux x86_64 are supported. Set ESMC_CPP to the esmc.cpp
# checkout; the commit should be >= 218cc69. This script is for CI/release
# builds, not for `pip install` from source.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ESMC_CPP="${ESMC_CPP:-/Users/ananyapathak/Desktop/oss/esmc.cpp}"
PYTHON="${PYTHON:-python3}"
RT="$ROOT/esmc_li/_runtime"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"

CMAKE_EXTRA=""
if [ "$(uname -s)" = "Linux" ]; then
  # A wheel built on a CI runner must not depend on its CPU features.
  CMAKE_EXTRA="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON"
fi
cmake -S "$ESMC_CPP" -B "$ESMC_CPP/build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" ${CMAKE_EXTRA}
cmake --build "$ESMC_CPP/build" --target esmc_shared -j "${JOBS:-8}"

rm -f "$RT"/libesmc.* "$RT"/libggml*
mkdir -p "$RT"

case "$(uname -s)" in
  Darwin)
    export _PYTHON_HOST_PLATFORM="macosx-${MACOSX_DEPLOYMENT_TARGET}-$(uname -m)"
    cp "$ESMC_CPP/build/libesmc.dylib" "$RT/"
    for f in "$ESMC_CPP"/build/bin/libggml*.dylib; do
      cp -L "$f" "$RT/$(basename "$f")"
    done
    for f in "$RT"/lib*.dylib; do
      install_name_tool -add_rpath "@loader_path" "$f" 2>/dev/null || true
    done
    ;;
  Linux)
    cp "$ESMC_CPP/build/libesmc.so" "$RT/"
    for f in "$ESMC_CPP"/build/bin/libggml*.so; do
      cp -L "$f" "$RT/$(basename "$f")"
    done
    if command -v patchelf >/dev/null; then
      for f in "$RT"/lib*.so*; do
        patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null || true
      done
    fi
    ;;
  *)
    echo "unsupported OS: $(uname -s)" >&2
    exit 1
    ;;
esac

rm -rf "$ROOT/dist" "$ROOT/build" "$ROOT"/*.egg-info
"$PYTHON" -m pip wheel "$ROOT" --no-deps -w "$ROOT/dist"
ls -lh "$ROOT/dist"
