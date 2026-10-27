"""`.esmci` single-file index (M6 T6.2/T6.3).

Layout (little-endian):
  - 64-byte header: magic(8s) version(u32) dtype_code(u32) D(u32)
    reserved(u32) n_proteins(u64) n_residues(u64) payload_offset(u64)
  - offsets: u64[n_proteins + 1]
  - payload: projected embeddings, dtype_code, shape (n_residues, D)

Metadata (ids, provenance) rides a JSON sidecar `<name>.esmci.json` — never
in the binary. Load is a pair of mmaps (opens 100K in well under a second);
`vecs`/`offsets` are ndarrays and drop straight into `search.maxsim_topk`.
Build-time projection (T6.3): write the D-dimensional rows, not the 960-d
base — the file is exactly header + offsets + n_res x D x 2 bytes.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"ESMCI\x00\x00\x00"
VERSION = 1
HEADER_SIZE = 64
_HDR = struct.Struct("<8sIIIIQQQ")  # 48 bytes, padded to HEADER_SIZE
_DT = {2: np.float16}
_DT_CODE = {np.dtype(np.float16): 2}


def _sidecar(path: Path) -> Path:
    return path.parent / (path.name + ".json")


def write(path: Path | str, vectors: np.ndarray, offsets: np.ndarray,
          meta: dict | None = None, chunk_rows: int = 1_000_000) -> Path:
    """Write `.esmci` + JSON sidecar. vectors: (n_res, D) any float dtype
    (stored f16); offsets: n_proteins+1 row starts.

    The payload is streamed in `chunk_rows` blocks so a memory-mapped 1M
    index (tens of GiB) never materializes a second full copy in RAM.
    """
    path = Path(path)
    off = np.ascontiguousarray(offsets, dtype=np.uint64)
    n_res, d = vectors.shape
    n_prot = len(off) - 1
    assert chunk_rows > 0
    assert off[0] == 0 and off[-1] == n_res, "offsets must span the payload"
    payload_off = HEADER_SIZE + (n_prot + 1) * 8
    hdr = _HDR.pack(MAGIC, VERSION, _DT_CODE[np.dtype(np.float16)], int(d), 0,
                    n_prot, int(n_res), payload_off)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(hdr.ljust(HEADER_SIZE, b"\0"))
        f.write(off.tobytes())
        for s in range(0, n_res, chunk_rows):
            blk = np.ascontiguousarray(vectors[s : s + chunk_rows],
                                       dtype=np.float16)
            f.write(blk.tobytes())
    if meta is not None:
        _sidecar(path).write_text(json.dumps(meta))
    return path


class Esmci:
    """mmap-backed index: `.vecs` [n_res, D] f16, `.offsets` u64[n+1]."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with self.path.open("rb") as f:
            hdr = f.read(_HDR.size)
        magic, self.version, code, self.d, _, self.n_proteins, self.n_residues, \
            payload_off = _HDR.unpack(hdr)
        if magic != MAGIC:
            raise ValueError(f"not an .esmci file: {self.path}")
        if self.version != VERSION:
            raise ValueError(f"unsupported .esmci version {self.version}")
        self.dtype = _DT[code]
        off_bytes = (self.n_proteins + 1) * 8
        self.offsets = np.memmap(self.path, dtype=np.uint64, mode="r",
                                 offset=HEADER_SIZE, shape=(self.n_proteins + 1,))
        self.vecs = np.memmap(self.path, dtype=self.dtype, mode="r",
                              offset=HEADER_SIZE + off_bytes,
                              shape=(self.n_residues, self.d))
        if int(self.offsets[-1]) != self.n_residues:
            raise ValueError("offsets[-1] != n_residues")

    @property
    def meta(self) -> dict:
        p = _sidecar(self.path)
        return json.loads(p.read_text()) if p.exists() else {}

    @property
    def ids(self) -> list[str]:
        return self.meta.get("ids", [])

    def protein(self, i: int) -> np.ndarray:
        """Copy of doc i's residue rows [L_i, D]."""
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        return np.array(self.vecs[lo:hi])

    def __len__(self) -> int:
        return self.n_proteins

    def nbytes(self) -> int:
        return self.path.stat().st_size


def expected_bytes(n_proteins: int, n_residues: int, d: int) -> int:
    return HEADER_SIZE + (n_proteins + 1) * 8 + n_residues * d * 2
