"""ctypes binding to the esmc.cpp C API (`libesmc`).

The model/context stay loaded for the lifetime of the encoder. This is the
in-process query/build path; the CLI remains a fallback for platforms without
a bundled runtime library.
"""
from __future__ import annotations

import ctypes
import os
import sys
import warnings
from pathlib import Path

import numpy as np


class ModelParams(ctypes.Structure):
    _fields_ = [
        ("use_mmap", ctypes.c_bool),
        ("use_metal", ctypes.c_bool),
        ("require_metal", ctypes.c_bool),
        ("n_threads", ctypes.c_int),
    ]


def find_library() -> Path | None:
    env = os.environ.get("ESMC_LIB")
    candidates = [Path(env)] if env else []
    name = "libesmc.dylib" if sys.platform == "darwin" else "libesmc.so"
    candidates.append(Path(__file__).with_name("_runtime") / name)
    candidates.append(Path.cwd() / name)
    for p in candidates:
        if p.is_file():
            return p
    return None


class EsmcRuntime:
    """One loaded esmc_model + esmc_context; returns full-token f32 rows."""

    def __init__(self, lib_path: str | Path, model_path: str | Path,
                 use_metal: bool | None = None, n_threads: int = 0):
        if sys.platform == "darwin" and "_runtime" in str(lib_path):
            warnings.warn(
                "bundled libesmc is unsigned in v1; verify the release "
                "checksum or build esmc.cpp locally", RuntimeWarning,
                stacklevel=2)
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        params = self.lib.esmc_default_model_params()
        if use_metal is None:
            use_metal = sys.platform == "darwin"
        params.use_metal = bool(use_metal)
        params.require_metal = False
        params.n_threads = int(n_threads)
        self.model = self.lib.esmc_load_model(str(model_path).encode(), params)
        if not self.model:
            raise RuntimeError(f"esmc_load_model failed: {model_path}")
        self.ctx = self.lib.esmc_new_context(self.model)
        if not self.ctx:
            self.lib.esmc_free_model(self.model)
            self.model = None
            raise RuntimeError("esmc_new_context failed")
        self.n_embd = int(self.lib.esmc_n_embd(self.model))
        self.n_ctx = int(self.lib.esmc_n_ctx(self.model))

    def _bind(self) -> None:
        L = self.lib
        L.esmc_default_model_params.restype = ModelParams
        L.esmc_load_model.argtypes = [ctypes.c_char_p, ModelParams]
        L.esmc_load_model.restype = ctypes.c_void_p
        L.esmc_new_context.argtypes = [ctypes.c_void_p]
        L.esmc_new_context.restype = ctypes.c_void_p
        L.esmc_free_context.argtypes = [ctypes.c_void_p]
        L.esmc_free_context.restype = None
        L.esmc_free_model.argtypes = [ctypes.c_void_p]
        L.esmc_free_model.restype = None
        L.esmc_n_embd.argtypes = [ctypes.c_void_p]
        L.esmc_n_embd.restype = ctypes.c_int
        L.esmc_n_ctx.argtypes = [ctypes.c_void_p]
        L.esmc_n_ctx.restype = ctypes.c_int
        L.esmc_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_int32),
                                    ctypes.c_int32]
        L.esmc_tokenize.restype = ctypes.c_int
        L.esmc_embed.argtypes = [ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_int32),
                                 ctypes.c_int32,
                                 ctypes.POINTER(ctypes.c_float)]
        L.esmc_embed.restype = ctypes.c_int

    def embed(self, sequence: str) -> np.ndarray:
        raw = sequence.encode("ascii")
        tokens = (ctypes.c_int32 * (len(raw) + 2))()
        n = self.lib.esmc_tokenize(self.model, raw, tokens, len(tokens))
        if n < 0:
            raise RuntimeError("esmc_tokenize failed")
        out = (ctypes.c_float * (n * self.n_embd))()
        if self.lib.esmc_embed(self.ctx, tokens, n, out) != 0:
            raise RuntimeError("esmc_embed failed")
        return np.ctypeslib.as_array(out).reshape(n, self.n_embd).copy()

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.esmc_free_context(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.lib.esmc_free_model(self.model)
            self.model = None

    def __enter__(self) -> "EsmcRuntime":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
