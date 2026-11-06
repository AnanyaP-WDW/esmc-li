"""T6.2 exit: .esmci round-trip."""
import numpy as np
import pytest

from esmc_li import ProteinIndex
from esmc_li.index import Esmci, expected_bytes, write


def make(n=7, d=5, seed=0):
    rng = np.random.default_rng(seed)
    lens = rng.integers(1, 12, size=n)
    off = np.concatenate(([0], np.cumsum(lens)))
    vecs = rng.standard_normal((int(off[-1]), d)).astype(np.float32)
    ids = [f"p{i}" for i in range(n)]
    return vecs, off, {"ids": ids, "note": "round-trip"}


def test_roundtrip(tmp_path):
    vecs, off, meta = make()
    p = write(tmp_path / "t.esmci", vecs, off, meta)
    idx = Esmci(p)
    assert len(idx) == 7
    assert idx.d == 5 and idx.n_residues == vecs.shape[0]
    np.testing.assert_array_equal(idx.offsets, off.astype(np.uint64))
    np.testing.assert_array_equal(np.asarray(idx.vecs),
                                  vecs.astype(np.float16))
    assert idx.ids == meta["ids"]
    assert idx.meta["note"] == "round-trip"
    np.testing.assert_array_equal(idx.protein(3),
                                  vecs[off[3] : off[4]].astype(np.float16))


def test_size_formula_exact(tmp_path):
    vecs, off, meta = make(n=10, d=128, seed=1)
    p = write(tmp_path / "s.esmci", vecs, off, meta)
    assert p.stat().st_size == expected_bytes(10, vecs.shape[0], 128)


def test_protein_index_requires_sidecar_ids(tmp_path):
    vecs, off, _ = make()
    p = write(tmp_path / "no-meta.esmci", vecs, off, meta=None)
    with pytest.raises(ValueError, match="sidecar ids invalid"):
        ProteinIndex.load(p, device="numpy")


def test_rejects_non_esmci(tmp_path):
    p = tmp_path / "bad.esmci"
    p.write_bytes(b"not an index" * 8)
    with pytest.raises(ValueError, match="not an .esmci"):
        Esmci(p)


def test_float64_input_stored_f16(tmp_path):
    vecs, off, meta = make(n=3, d=4, seed=2)
    p = write(tmp_path / "f.esmci", vecs.astype(np.float64), off, meta)
    idx = Esmci(p)
    assert idx.dtype == np.float16
    np.testing.assert_array_equal(np.asarray(idx.vecs),
                                  vecs.astype(np.float16))


def test_streaming_write_matches_single_chunk(tmp_path):
    vecs, off, meta = make(n=5, d=3, seed=3)
    one = write(tmp_path / "one.esmci", vecs, off, meta)
    many = write(tmp_path / "many.esmci", vecs, off, meta, chunk_rows=2)
    assert one.read_bytes() == many.read_bytes()
    assert many.stat().st_size == expected_bytes(5, vecs.shape[0], 3)
    np.testing.assert_array_equal(np.asarray(Esmci(many).vecs),
                                  vecs.astype(np.float16))
