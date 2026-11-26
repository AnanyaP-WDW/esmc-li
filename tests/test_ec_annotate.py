"""Consensus/confidence test for the EC annotation CLI."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ec_annotate  # noqa: E402


class FakeIndex:
    vecs = np.zeros((4, 8), dtype=np.float16)
    offsets = np.array([0, 2, 4], dtype=np.uint64)
    ids = ["a", "b"]
    meta = {"ec4": ["1.1.1.1", "2.2.2.2"]}


def test_annotate_consensus(monkeypatch, capsys, tmp_path):
    head = tmp_path / "head.npy"
    np.save(head, np.eye(8, dtype=np.float32))
    monkeypatch.setattr(ec_annotate, "Esmci", lambda p: FakeIndex())
    monkeypatch.setattr(
        ec_annotate, "embed",
        lambda rows, model, binary: [np.ones((2, 8), dtype=np.float32)])
    monkeypatch.setattr(
        ec_annotate, "maxsim_topk_numpy",
        lambda q, vecs, off, k: (np.array([1.0, 0.5]), np.array([0, 1])))
    monkeypatch.setattr(sys, "argv", ["ec_annotate.py", "x",
                                      "--sequence", "AA",
                                      "--head", str(head), "--k", "2"])
    assert ec_annotate.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["ec4"] == "1.1.1.1"
    assert out[0]["confidence"] == 1.0 / 1.5
    assert [h["id"] for h in out[0]["top"]] == ["a", "b"]
