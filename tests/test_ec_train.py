"""Batch construction tests for EC head training."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ec_train import make_batch  # noqa: E402


def test_make_batch_with_duplicate_ec():
    rows = [
        {"id": "a", "ec4": "1.1.1.1", "ec1": "1"},
        {"id": "b", "ec4": "1.1.1.1", "ec1": "1"},
        {"id": "c", "ec4": "1.1.1.2", "ec1": "1"},
    ]
    emb = {r["id"]: np.ones((4, 960), dtype=np.float32) for r in rows}
    by_ec = {"1.1.1.1": rows[:2], "1.1.1.2": rows[2:]}
    by_ec3 = {"1.1.1": rows}
    by_ec1 = {"1": rows}
    batch = make_batch(rows, by_ec, by_ec3, by_ec1, emb,
                       __import__("random").Random(0), 2)
    assert batch is not None
    q, d, qm, dm, labels = batch
    assert q.shape[0] == d.shape[0] == len(labels)
    assert q.shape[2] == 960
    assert qm.any() and dm.any()
