"""Catalytic-decoy generation tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ec_decoys import mutate  # noqa: E402


def test_mutate_changes_positions():
    seq = "ACDEFGHIKL"
    out, changed = mutate(seq, [1, 2, 99])
    assert len(out) == len(seq)
    assert changed == [1, 2]
    assert out[0] != seq[0] and out[1] != seq[1]


def test_mutate_does_not_touch_out_of_range():
    seq = "ACDE"
    out, changed = mutate(seq, [0, 10])
    assert out == seq
    assert changed == []
