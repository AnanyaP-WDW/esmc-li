# Public API (frozen at v1.0.0)

**ESMC-LI** exposes one class and one CLI. The index format version is 1;
the public API below is frozen for the v1 line.

## Python

```python
from esmc_li import ProteinIndex

index = ProteinIndex.from_fasta("proteins.fasta", output="proteins.esmci")
index = ProteinIndex.load("proteins.esmci")
hits = index.search("MKTIIALSYIFCLVFADYKDDDDK", k=20)
batch = index.search_batch(["MKT...", "MKT..."], k=20)
```

| Member | Contract |
|---|---|
| `ProteinIndex.from_fasta(fasta, output, model=..., **kwargs)` | Build an index from FASTA and return it. |
| `ProteinIndex.build(rows, output, model=..., embed_bin=..., head=..., metadata=..., device="auto")` | Build from row dicts or a FASTA path. |
| `ProteinIndex.load(path, model=..., embed_bin=..., head=..., device="auto")` | Open an `.esmci` index; verifies head/model hashes when recorded. |
| `ProteinIndex.search(sequence, k=10)` | Return `[{"id": str, "score": float, "metadata": dict}, ...]`. |
| `ProteinIndex.search_batch(sequences, k=10)` | One hit list per sequence, same schema. |
| `ProteinIndex.benchmark(nq=10, k=10)` | Return encode/search timing metrics. |
| `len(index)`, `index.ids`, `index.nbytes`, `index.metadata` | Read-only index properties. |
| `with ProteinIndex.load(...) as index:` | Closes the underlying memory map / runtime. |

`device="auto"` uses the fused Metal backend when PyObjC is available on macOS
and falls back to numpy. `device="metal"` raises if Metal is unavailable.

## CLI

```bash
esmc-li build proteins.fasta --output proteins.esmci
esmc-li search proteins.esmci --sequence "MKT..." --top-k 20
esmc-li search proteins.esmci --query queries.fasta --output hits.tsv
esmc-li benchmark proteins.esmci
esmc-li index list
esmc-li index pull scope-1k
```

## Compatibility

- `.esmci` v1 layout is documented in `docs/esmci.md`.
- An index is not self-contained: keep its matching head (`.head.npy`) and the
  fine-tuned GGUF with it. The sidecar records hashes and `load` checks them.
- Model weights are governed by the Cambrian Open License; see `NOTICE`.
