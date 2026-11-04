"""M8 `ProteinIndex`: build, load, and search a local `.esmci` index.

The query encoder is esmc.cpp's fine-tuned GGUF with `--keep-special`, followed
by the frozen 960->128 projection head. Index rows and query rows both retain
BOS/EOS, matching the M3/M4/M7 scorer. Search is exact MaxSim with numpy so the
package needs only numpy at query time; esmc.cpp remains the model runtime.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

from esmc_retrieval.embed import chunk_windows
from esmc_retrieval.index import Esmci, write
from esmc_retrieval.models import DEFAULT_MODEL, resolve_model
from esmc_retrieval.runtime import EsmcRuntime, find_library
from esmc_retrieval.search import maxsim_topk_numpy

D_MODEL = 960
D_PROJ = 128
WIN = 2046
OVERLAP = 128
DEFAULT_HEAD = Path(__file__).with_name("data") / "head_k3_seed0.npy"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _clean(seq: str) -> str:
    return "".join(seq.split()).upper()


def _read_fasta(path: Path) -> list[dict]:
    rows: list[dict] = []
    rid, chunks, desc = None, [], ""
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if rid is not None:
                rows.append({"id": rid, "sequence": _clean("".join(chunks)),
                             "description": desc})
            head = line[1:].split(None, 1)
            rid = head[0] if head else ""
            desc = head[1] if len(head) > 1 else ""
            chunks = []
        else:
            chunks.append(line)
    if rid is not None:
        rows.append({"id": rid, "sequence": _clean("".join(chunks)),
                     "description": desc})
    if not rows:
        raise ValueError(f"no sequences in {path}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate FASTA ids")
    if any(not r["sequence"] for r in rows):
        raise ValueError("empty sequence in FASTA")
    return rows


def _window_names(seq: str, rid: str) -> list[tuple[str, tuple[int, int]]]:
    spans = chunk_windows(seq)
    if len(spans) == 1:
        return [(rid, spans[0])]
    return [(f"{rid}|{s}", (s, e)) for s, e in spans]


def _run_esmc(rows: list[dict], embed_bin: Path, model: Path) -> list[np.ndarray]:
    """Embed full-token rows (BOS/EOS kept) for each row; one CLI call."""
    if "Q4" in model.name.upper():
        warnings.warn(
            "Q4 GGUF is not the frozen f16 scorer; retrieval quality may differ",
            RuntimeWarning, stacklevel=2,
        )
    with tempfile.TemporaryDirectory(prefix="m8embed_") as td:
        td = Path(td)
        fa = td / "batch.fa"
        with fa.open("w") as f:
            for r in rows:
                for name, (s, e) in _window_names(r["sequence"], r["id"]):
                    f.write(f">{name}\n{r['sequence'][s:e]}\n")
        out = td / "npy"
        out.mkdir()
        cmd = [str(embed_bin), "-m", str(model), "--fasta", str(fa),
               "--output-dir", str(out), "--keep-special"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            tail = (p.stderr or p.stdout)[-2000:]
            raise RuntimeError(f"esmc-embed failed ({p.returncode}):\n{tail}")

        parts: list[np.ndarray] = []
        for r in rows:
            spans = _window_names(r["sequence"], r["id"])
            arrs = [np.load(out / f"{name}.npy") for name, _ in spans]
            for (name, (s, e)), a in zip(spans, arrs):
                want = e - s + 2
                if a.shape != (want, D_MODEL):
                    raise RuntimeError(
                        f"{name}: {a.shape} != ({want}, {D_MODEL}); "
                        "esmc-embed without --keep-special?")
            parts.append(np.concatenate(arrs, axis=0))
        return parts


def _project(rows: np.ndarray, head: np.ndarray) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float32) @ np.asarray(
        head, dtype=np.float32).T
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def _resolve(path: str | Path | None, meta: dict, key: str,
             env: str, default: str | Path | None = None) -> Path | None:
    value = path if path is not None else meta.get(key) or os.environ.get(env)
    if value is None and default is not None:
        value = default
    return Path(value) if value else None


def _make_runtime(model: Path | None) -> EsmcRuntime | None:
    if model is None or not model.exists():
        return None
    lib = find_library()
    return EsmcRuntime(lib, model) if lib is not None else None


class ProteinIndex:
    """Local exact MaxSim index over projected ESM-C residue rows."""

    def __init__(self, path: str | Path, head: np.ndarray, model: Path | None,
                 embed_bin: Path | None, metadata: list[dict] | None = None,
                 device: str = "auto", runtime: EsmcRuntime | None = None):
        self.path = Path(path)
        self._idx = Esmci(self.path)
        self._head = np.asarray(head, dtype=np.float32)
        self.model = model
        self.embed_bin = embed_bin
        self.metadata = metadata or [{} for _ in range(len(self._idx))]
        self.device = device
        self._runtime = runtime

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def __enter__(self) -> "ProteinIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self._idx)

    @property
    def ids(self) -> list[str]:
        return self._idx.ids

    @property
    def nbytes(self) -> int:
        return self._idx.nbytes()

    @classmethod
    def load(cls, path: str | Path, model: str | Path | None = None,
             embed_bin: str | Path | None = None,
             head: str | Path | None = None,
             device: str = "auto") -> "ProteinIndex":
        if not Path(path).exists():
            from esmc_retrieval.indexes import resolve_index
            path = resolve_index(path)
        idx = Esmci(path)
        meta = idx.meta
        companion = Path(str(path) + ".head.npy")
        if head is None:
            head = companion if companion.exists() else DEFAULT_HEAD
        head = Path(head)
        h = np.load(head)
        if h.shape != (D_PROJ, D_MODEL):
            raise ValueError(f"head shape {h.shape} != ({D_PROJ},{D_MODEL})")
        if meta.get("head_sha256") and _sha256(head) != meta["head_sha256"]:
            raise ValueError("projection head hash does not match index")
        raw_model = (model if model is not None
                     else meta.get("model") or os.environ.get(
                         "ESMC_RETRIEVAL_GGUF"))
        mdl = resolve_model(raw_model) if raw_model else None
        emb = _resolve(embed_bin, meta, "embed_bin", "ESMC_EMBED",
                       shutil.which("esmc-embed"))
        if mdl is not None and meta.get("model_sha256"):
            if _sha256(mdl) != meta["model_sha256"]:
                raise ValueError("model hash does not match index")
        md = meta.get("metadata")
        if not isinstance(md, list) or len(md) != len(idx):
            md = [{} for _ in range(len(idx))]
        runtime = _make_runtime(mdl)
        return cls(path, h, mdl, emb, md, device=device, runtime=runtime)

    @classmethod
    def build(cls, rows: list[dict] | str | Path, output: str | Path,
              model: str | Path = DEFAULT_MODEL,
              embed_bin: str | Path | None = None,
              head: str | Path | None = None,
              metadata: list[dict] | None = None,
              device: str = "auto") -> "ProteinIndex":
        if isinstance(rows, (str, Path)):
            rows = _read_fasta(Path(rows))
        if not rows:
            raise ValueError("no rows to index")
        order = sorted(range(len(rows)),
                       key=lambda i: (-len(rows[i]["sequence"]),
                                      rows[i]["id"]))
        rows = [rows[i] for i in order]
        if metadata is not None:
            if len(metadata) != len(rows):
                raise ValueError("metadata length must match rows")
            metadata = [metadata[i] for i in order]
        model = resolve_model(model)
        if model is None or not model.exists():
            raise FileNotFoundError(model)
        runtime = _make_runtime(model)
        if runtime is None:
            raw_bin = (embed_bin or os.environ.get("ESMC_EMBED")
                       or shutil.which("esmc-embed"))
            if not raw_bin:
                raise FileNotFoundError(
                    "libesmc/esmc-embed not found; pass embed_bin= or set "
                    "ESMC_EMBED/ESMC_LIB")
            embed_bin = Path(raw_bin)
            if not embed_bin.exists():
                raise FileNotFoundError(embed_bin)
        head_path = Path(head) if head is not None else DEFAULT_HEAD
        head_arr = np.load(head_path)
        if head_arr.shape != (D_PROJ, D_MODEL):
            raise ValueError(f"head shape {head_arr.shape}")

        rows = [dict(r, sequence=_clean(r["sequence"])) for r in rows]
        try:
            embedded = ([runtime.embed(r["sequence"]) for r in rows]
                        if runtime is not None
                        else _run_esmc(rows, embed_bin, model))
        finally:
            if runtime is not None:
                runtime.close()
        projected = [_project(x, head_arr) for x in embedded]
        lens = [len(x) for x in projected]
        offsets = np.cumsum([0] + lens).astype(np.uint64)
        packed = np.concatenate(projected, axis=0).astype(np.float16)

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        companion = Path(str(output) + ".head.npy")
        shutil.copyfile(head_path, companion)
        if metadata is None:
            metadata = [{k: v for k, v in r.items()
                         if k not in ("id", "sequence")} for r in rows]
        if len(metadata) != len(rows):
            raise ValueError("metadata length must match rows")
        meta = {
            "ids": [r["id"] for r in rows],
            "metadata": metadata,
            "d": D_PROJ,
            "dtype": "float16",
            "win": WIN,
            "overlap": OVERLAP,
            "head_file": companion.name,
            "head_sha256": _sha256(companion),
            "model": str(model),
            "model_sha256": _sha256(model),
            "embed_bin": str(embed_bin) if embed_bin else "",
            "source": str(output),
            "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write(output, packed, offsets, meta)
        return cls.load(output, model=model, embed_bin=embed_bin,
                        head=companion, device=device)

    @classmethod
    def from_fasta(cls, fasta: str | Path, output: str | Path,
                   model: str | Path = DEFAULT_MODEL, **kwargs
                   ) -> "ProteinIndex":
        return cls.build(_read_fasta(Path(fasta)), output, model, **kwargs)

    def _encode(self, sequences: list[str]) -> list[np.ndarray]:
        if self.model is None or not self.model.exists():
            raise FileNotFoundError(
                "query GGUF not found; pass model= to load() or set "
                "ESMC_RETRIEVAL_GGUF")
        rows = [{"id": f"q{i}", "sequence": _clean(s)}
                for i, s in enumerate(sequences)]
        if any(not r["sequence"] for r in rows):
            raise ValueError("empty query sequence")
        if self._runtime is not None:
            embedded = [self._runtime.embed(r["sequence"]) for r in rows]
        else:
            if self.embed_bin is None or not self.embed_bin.exists():
                raise FileNotFoundError(
                    "libesmc/esmc-embed not found; pass embed_bin= to load() "
                    "or set ESMC_EMBED/ESMC_LIB")
            embedded = _run_esmc(rows, self.embed_bin, self.model)
        return [_project(x, self._head) for x in embedded]

    def _search(self, qs: list[np.ndarray], k: int
                ) -> tuple[np.ndarray, np.ndarray]:
        if self.device in ("auto", "metal"):
            try:
                from esmc_retrieval.metal_search import maxsim_topk_metal
                return maxsim_topk_metal(qs, self._idx.vecs, self._idx.offsets,
                                         k, normalize=False)
            except ImportError:
                if self.device == "metal":
                    raise
                self.device = "numpy"
        outs = [maxsim_topk_numpy(q, self._idx.vecs, self._idx.offsets, k)
                for q in qs]
        return (np.stack([o[0] for o in outs]),
                np.stack([o[1] for o in outs]))

    def _hits(self, scores: np.ndarray, rows: np.ndarray) -> list[dict]:
        out = []
        for s, i in zip(scores, rows):
            i = int(i)
            out.append({"id": self.ids[i], "score": float(s),
                        "metadata": self.metadata[i] if i < len(self.metadata)
                        else {}})
        return out

    def search(self, sequence: str, k: int = 10) -> list[dict]:
        s, i = self._search(self._encode([sequence]), k)
        return self._hits(s[0], i[0])

    def search_batch(self, sequences: list[str], k: int = 10
                     ) -> list[list[dict]]:
        s, i = self._search(self._encode(sequences), k)
        return [self._hits(s[j], i[j]) for j in range(len(sequences))]

    def benchmark(self, nq: int = 10, k: int = 10) -> dict:
        n = min(nq, len(self))
        if n == 0:
            raise ValueError("empty index")
        qs = [self._idx.protein(i) for i in range(n)]
        t = time.perf_counter()
        self._search(qs, k)
        dt = time.perf_counter() - t
        return {"n": len(self), "nq": n, "k": k, "wall_s": dt,
                "qps": n / dt if dt > 0 else float("inf")}
