"""Patch and AUC tests for the 32-triple local-objective overfit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ec_overfit32  # noqa: E402


def test_patch_indices_union():
    assert ec_overfit32.patch_indices(10, [1, 5], 1) == [0, 1, 3, 4, 5]


def test_random_patch_excludes_active_sites():
    import random
    rng = random.Random(0)
    idx = ec_overfit32.random_patch_indices(20, [5], 2, 1, rng)
    assert idx
    assert not ({4, 5, 6} & set(idx))


def test_auc_perfect_and_tied():
    assert ec_overfit32.auc_score([1.0, 2.0], [0.0, 0.5]) == 1.0
    assert ec_overfit32.auc_score([0.0, 0.5], [1.0, 2.0]) == 0.0
    assert ec_overfit32.auc_score([1.0], [1.0]) == 0.5


def test_build_triples_cluster_disjoint():
    train = [
        {"id": "p1", "ec4": "1.1.1.1", "active_sites": "2;4",
         "sequence": "AAAA"},
        {"id": "q1", "ec4": "1.1.1.1", "active_sites": "1;3",
         "sequence": "BBBB"},
        {"id": "p2", "ec4": "2.2.2.2", "active_sites": "1",
         "sequence": "CCCC"},
        {"id": "q2", "ec4": "2.2.2.2", "active_sites": "2",
         "sequence": "DDDD"},
    ]
    decoys = [{"parent": "p1", "decoy_id": "p1|catalytic"},
              {"parent": "p2", "decoy_id": "p2|catalytic"}]
    clusters = {"p1": "c1", "q1": "c2", "p2": "c3", "q2": "c4"}
    triples = ec_overfit32.build_triples(train, decoys, clusters, 2, 0)
    assert len(triples) == 2
    assert len({t["ec4"] for t in triples}) == 2
    for t in triples:
        assert clusters[t["parent"]["id"]] != clusters[t["positive"]["id"]]
        assert t["decoy"]["parent"] == t["parent"]["id"]
