# Paper assets

`paper/main.md` is generated from `paper/main.template.md` and
`paper/results.lock.json`. Do not edit `main.md`, `tables/*.md`,
`figures/*.svg`, `numbers.json`, or `docs/claims.md` by hand.

```bash
make tables     # regenerate every data table and figure; rebuild main.md
make claims     # regenerate paper/claims.tsv and docs/claims.md
make audit      # lock check + claims + citations + pytest + zero-diff
```

The lock pins `results/runs.jsonl` at `results_commit` (currently 5a308ac).
`make verify-lock` re-reads that commit and checks every selected row's line
hash and metrics. `make tables` needs only the lock and Python (with
`matplotlib` for the figures): no model, no embedding cache, no network, no
`esmc.cpp`.

Claims are placeholders: `{{n:t2.m3.cR@10_pm}}` becomes the table cell value.
`scripts/check_claims.py` rejects any hardcoded number and any placeholder
without a `paper/claims.tsv` row. `scripts/check_citations.py` rejects an
unverified citation or an `UNVERIFIED` marker.

The paper is a Markdown preprint. If a PDF is needed, convert `main.md`; the
generated tables and figures are the source of truth.
