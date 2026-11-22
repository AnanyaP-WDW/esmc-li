"""Data-contract tests for the EC experiment."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ec"


def rows(name):
    return list(csv.DictReader((DATA / "splits" / f"{name}.tsv").open(
        newline=""), delimiter="\t"))


def test_manifest_counts():
    man = json.loads((DATA / "manifest.json").read_text())
    c = man["counts"]
    assert c["train_s1_single"] == 153295
    assert c["val_s2"] == 18224
    assert c["brenda_trembl"] == 6893
    assert c["care"]["care_ood30"] == 432
    assert c["care"]["care_30_50"] == 559
    assert c["care"]["care_price"] == 148
    assert c["care"]["care_promiscuous"] == 209
    assert c["enzymarc"]["catalytic"] == 76480


def test_no_train_test_accession_overlap():
    train = {r["id"] for r in rows("train_s1_single")}
    for name in ("val_s2", "brenda_trembl", "care_ood30",
                 "care_30_50", "care_price", "care_promiscuous"):
        assert not (train & {r["id"] for r in rows(name)}), name


def test_ec4_shape():
    for name in ("train_5k", "val_5k", "care_ood30"):
        for r in rows(name):
            for ec in r["ec4"].split(";"):
                if ec:
                    assert len(ec.split(".")) == 4, (name, ec)


def test_sequences_nonempty():
    for name in ("train_5k", "val_5k", "care_ood30", "care_promiscuous"):
        assert all(r["sequence"] for r in rows(name))
