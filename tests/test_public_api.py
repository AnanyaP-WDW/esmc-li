"""Frozen public API surface for v1."""
from __future__ import annotations

import inspect

import esmc_retrieval
from esmc_retrieval import ProteinIndex


def params(fn):
    return list(inspect.signature(fn).parameters)


def test_exports():
    assert esmc_retrieval.__all__ == ["ProteinIndex"]
    assert esmc_retrieval.__version__ == "1.0.0"


def test_build_signature():
    assert params(ProteinIndex.build) == [
        "rows", "output", "model", "embed_bin", "head", "metadata", "device"]
    assert params(ProteinIndex.from_fasta) == ["fasta", "output", "model", "kwargs"]
    assert params(ProteinIndex.load) == [
        "path", "model", "embed_bin", "head", "device"]


def test_search_signature():
    assert params(ProteinIndex.search) == ["self", "sequence", "k"]
    assert params(ProteinIndex.search_batch) == ["self", "sequences", "k"]
    assert params(ProteinIndex.benchmark) == ["self", "nq", "k"]


def test_properties():
    assert isinstance(ProteinIndex.ids, property)
    assert isinstance(ProteinIndex.nbytes, property)
    assert hasattr(ProteinIndex, "__enter__")
    assert hasattr(ProteinIndex, "__exit__")
