"""T4.2 exit: chunked embedding outputs exactly len(seq) residue rows."""
import numpy as np

from esmc_li.embed import OVERLAP as OVERLAP_LEN
from esmc_li.embed import WIN, assemble, chunk_windows, embed_seq


def local_rows(n: int) -> np.ndarray:
    """Fake embedder: row i = window-local indices, 4 fake dims."""
    idx = np.arange(n, dtype=np.float32)
    return np.stack([idx, idx + 1, idx + 2, idx + 3], axis=1)


def global_rows(s: int, e: int) -> np.ndarray:
    """Fake window covering global positions [s, e)."""
    idx = np.arange(s, e, dtype=np.float32)
    return np.stack([idx, idx + 1, idx + 2, idx + 3], axis=1)


def test_short_seq_single_window():
    seq = "A" * 100
    assert chunk_windows(seq) == [(0, 100)]
    assert embed_seq(seq, lambda s: local_rows(len(s))).shape == (100, 4)


def test_windows_cover_exactly_and_fit_context():
    seq = "M" * 5000
    spans = chunk_windows(seq)
    assert spans[0][0] == 0 and spans[-1][1] == len(seq)
    for s, e in spans:
        assert e - s <= WIN
    starts = [s for s, _ in spans]
    assert starts == sorted(starts)
    for (_, e), (s2, _) in zip(spans, spans[1:]):
        assert s2 < e


def test_embed_seq_local_shape():
    for n in (1, WIN, WIN + 1, 3 * WIN + 7):
        seq = "ACDE" * (n // 4) + "A" * (n % 4)
        rows = embed_seq(seq, lambda s: local_rows(len(s)))
        assert rows.shape == (n, 4), "row count must equal sequence length"


def test_row_count_and_identity():
    for n in (1, WIN, WIN + 1, WIN + OVERLAP_LEN, 3 * WIN, 14507):
        spans = chunk_windows("X" * n)
        parts = [global_rows(s, e) for s, e in spans]
        out = assemble(spans, parts)
        assert out.shape[0] == n, "row count must equal sequence length"
        np.testing.assert_array_equal(out[:, 0], np.arange(n, dtype=np.float32))


def test_overlap_rows_emitted_once():
    spans = chunk_windows("G" * 3000)
    parts = [global_rows(s, e) for s, e in spans]
    out = assemble(spans, parts)
    vals = out[:, 0].astype(np.int64)
    assert np.array_equal(vals, np.arange(3000))
