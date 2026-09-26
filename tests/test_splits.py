import json
from pathlib import Path

import pytest

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


def load(name: str) -> list[dict]:
    lines = (SPLITS / name).read_text().splitlines()
    cols = lines[0].split("\t")
    return [dict(zip(cols, line.split("\t"))) for line in lines[1:]]


@pytest.mark.parametrize("dataset", ["scope", "pfam"])
def test_zero_group_overlap(dataset):
    parts = {s: load(f"{dataset}_{s}.tsv") for s in ("train", "val", "test")}
    groups = {s: {r["superfamily"] for r in rows} for s, rows in parts.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_no_id_overlap_across_splits():
    for dataset in ("scope", "pfam"):
        ids = [{r["id"] for r in load(f"{dataset}_{s}.tsv")}
               for s in ("train", "val", "test")]
        assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]


def test_protocol_test_equals_our_test():
    """Our split's test is ProtoCol's test (same procedure, seed 42)."""
    ours = {r["id"] for r in load("scope_test.tsv")}
    proto = {r["id"] for r in load("scope_test_protocol.tsv")}
    assert ours == proto


def test_protocol_train_disjoint_from_test():
    train = {r["id"] for r in load("scope_protocol_train.tsv")}
    test = {r["id"] for r in load("scope_test_protocol.tsv")}
    assert not train & test


def test_split_hashes_recorded():
    hashes = json.loads((SPLITS / "hashes.json").read_text())
    for key, h in hashes.items():
        path = SPLITS / f"{key}.tsv"
        assert path.exists(), key
        import hashlib
        assert hashlib.sha256(path.read_bytes()).hexdigest() == h, key
