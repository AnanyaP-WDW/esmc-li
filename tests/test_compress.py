"""T7.1 exits: compression round-trips."""
import numpy as np

from esmc_li.compress import (decode_int8, decode_res_int8,
                                     decode_res_pq, encode_int8,
                                     encode_res_int8, encode_res_pq,
                                     kmeans, scheme_bytes)


def toy(n=5_000, d=128, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_kmeans_shapes_and_determinism():
    x = toy(3_000, 16)
    a = kmeans(x, k=32, iters=5, seed=1)
    b = kmeans(x, k=32, iters=5, seed=1)
    assert a.shape == (32, 16)
    np.testing.assert_array_equal(a, b)


def test_int8_roundtrip_bound():
    x = toy()
    c, scale = encode_int8(x)
    y = decode_int8(c, scale)
    assert c.dtype == np.int8
    assert np.abs(x - y).max() <= (scale / 2 + 1e-6).max()


def test_res_int8_not_worse_than_int8():
    x = toy()
    c1, s1 = encode_int8(x)
    c2, mu, s2 = encode_res_int8(x)
    e1 = np.abs(x - decode_int8(c1, s1)).mean()
    e2 = np.abs(x - decode_res_int8(c2, mu, s2)).mean()
    assert e2 <= e1


def test_res_pq_beats_coarse_only():
    x = toy()
    q = encode_res_pq(x, m=16, seed=0)
    y = decode_res_pq(q)
    assert y.shape == x.shape
    e_coarse = np.abs(x - q["C"][q["coarse"]]).mean()
    e_pq = np.abs(x - y).mean()
    assert e_pq < e_coarse


def test_size_formula():
    n, d = 1_000, 128
    assert scheme_bytes("int8", n, d) == n * d
    # residual_pq m16: 1 coarse byte + 16 sub bytes per row + codebooks
    assert scheme_bytes("residual_pq_m16", n, d) == n * 17 + (
        256 * d * 2 + 16 * 256 * 8 * 2)
