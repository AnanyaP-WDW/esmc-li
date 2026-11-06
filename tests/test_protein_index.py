"""M8 end-to-end with a fake esmc-embed (no model download)."""
import sys

import numpy as np
import pytest

from esmc_li import ProteinIndex
from esmc_li.embed import chunk_windows
from esmc_li.protein_index import _embed_chunked


@pytest.fixture(autouse=True)
def _no_runtime(monkeypatch):
    monkeypatch.setattr("esmc_li.protein_index.find_library",
                        lambda: None)


def _fake_embed(tmp_path):
    p = tmp_path / "fake_embed"
    p.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "a = sys.argv[1:]\n"
        "fa = Path(a[a.index('--fasta') + 1])\n"
        "out = Path(a[a.index('--output-dir') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "name, seq = None, None\n"
        "for line in fa.read_text().splitlines():\n"
        "    if line.startswith('>'):\n"
        "        name, seq = line[1:], ''\n"
        "    elif seq is not None:\n"
        "        seq += line.strip()\n"
        "        if name is not None:\n"
        "            n = len(seq) + 2\n"
        "            h = float(sum(map(ord, seq)))\n"
        "            x = np.stack([np.arange(960, dtype=np.float32) + h + i\n"
        "                          for i in range(n)])\n"
        "            np.save(out / f'{name}.npy', x)\n"
        "            name = None\n"
    )
    p.chmod(0o755)
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake model")
    return p, model


def test_long_sequence_chunking(tmp_path):
    bin_path, model = _fake_embed(tmp_path)
    seq = "ACDE" * 525  # 2100 aa -> two windows with 128 overlap
    fasta = tmp_path / "long.fa"
    fasta.write_text(f">long\n{seq}\n")
    out = tmp_path / "long.esmci"
    idx = ProteinIndex.from_fasta(fasta, out, model=model, embed_bin=bin_path,
                              device="numpy")
    assert idx._idx.n_residues == 2048 + (2100 - 1918 + 2)
    assert idx.search(seq, k=1)[0]["id"] == "long"


def test_chunked_runtime_contract():
    seq = "ACDE" * 1300
    calls = []

    def fake_embed(part):
        calls.append(len(part))
        return np.zeros((len(part) + 2, 960), dtype=np.float32)

    out = _embed_chunked(seq, fake_embed)
    spans = chunk_windows(seq)
    assert calls == [e - s for s, e in spans]
    assert out.shape == (sum(e - s + 2 for s, e in spans), 960)


def test_build_load_search_roundtrip(tmp_path):
    bin_path, model = _fake_embed(tmp_path)
    fasta = tmp_path / "proteins.fa"
    fasta.write_text(">a alpha\nACDE\n>b beta\nMKT\n")
    out = tmp_path / "proteins.esmci"
    idx = ProteinIndex.from_fasta(fasta, out, model=model, embed_bin=bin_path,
                              device="numpy")
    assert len(idx) == 2
    assert idx.ids == ["a", "b"]  # longest first
    assert idx.metadata[idx.ids.index("a")]["description"] == "alpha"

    loaded = ProteinIndex.load(out, device="numpy")
    assert len(loaded) == 2
    hit = loaded.search("ACDE", k=1)[0]
    assert hit["id"] == "a"
    hits = loaded.search_batch(["ACDE", "MKT"], k=1)
    assert [h[0]["id"] for h in hits] == ["a", "b"]
    assert loaded.benchmark(nq=2)["n"] == 2
    assert loaded.search_batch([]) == []
