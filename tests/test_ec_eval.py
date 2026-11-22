"""Metric/primitive tests for EC evaluation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ec_eval  # noqa: E402


def test_l2_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    y = ec_eval.l2(x)
    assert np.allclose(np.linalg.norm(y[0]), 1.0)
    assert np.allclose(y[1], 0.0)


def test_prefix():
    assert ec_eval.prefix("1.2.3.4", 1) == "1"
    assert ec_eval.prefix("1.2.3.4", 3) == "1.2.3"


def test_pooled_predict_shape():
    rng = np.random.default_rng(0)
    db = rng.standard_normal((5, 8)).astype(np.float16)
    q = [rng.standard_normal((1, 8)).astype(np.float32) for _ in range(3)]
    s, o = ec_eval.predict(q, db, None, "pooled", 2)
    assert s.shape == (3, 2)
    assert o.shape == (3, 2)
