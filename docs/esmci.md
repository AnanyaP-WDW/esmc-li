# `.esmci` — single-file ESMC retrieval index (M6)

Raw binary, little-endian. Written/read by `esmc_retrieval/index.py`.

## Layout

| region | bytes | contents |
|---|---:|---|
| header | 64 | `magic` 8 B `"ESMCI\0\0\0"`, `version` u32 (=1), `dtype_code` u32 (2 = f16), `D` u32, `reserved` u32, `n_proteins` u64, `n_residues` u64, `payload_offset` u64, zero pad |
| offsets | 8 × (n_proteins + 1) | `u64` row start of each protein in the payload; `offsets[0] = 0`, `offsets[-1] = n_residues` |
| payload | n_residues × D × 2 | residue-row embeddings, f16, shape `(n_residues, D)` |

Total size is exactly `64 + 8(n+1) + 2·n_res·D` bytes (verified ±0.000%
on the 100K index, T6.3).

## Sidecar

Metadata never enters the binary. A JSON file sits beside it:
`<name>.esmci.json` — `ids` (row-aligned protein ids) plus provenance
(`source`, `d`, `dtype`, `built`).

## Load

`Esmci(path)` reads the 48-byte header and mmaps the two regions; opening
the 100K / 34.05M-row index takes **7 ms** (T6.2). `.vecs` and `.offsets`
are ndarrays and drop straight into `search.maxsim_topk`.

## Build

Projection happens at build time (T6.3): store D=128 rows, not 960
(7.5× smaller). `scripts/m6_index.py` packs `db_tr128.npy` + offsets;
round-trip and size tests live in `tests/test_esmci.py`.

## M8 `ProteinIndex`

`ProteinIndex.build` uses `esmc-embed --keep-special` (BOS/EOS kept), applies
the packaged 960→128 head, and stores the same `.esmci` layout. The sidecar
adds `metadata`, `head_sha256`, `model_sha256`, and `embed_bin`; the head is
copied to `<index>.head.npy` so `ProteinIndex.load` can verify it. Query rows
are encoded with the same `--keep-special` path before exact numpy MaxSim.
