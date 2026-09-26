#!/usr/bin/env python3
"""Build group-disjoint train/val/test splits (T0.4).

SCOPe:  group = superfamily (sccs[:3]), from ASTRAL 40% 2.08.
Pfam:   group = clan (family -> clan from Pfam-A.clans.tsv), from Pfam-A.seed,
        max 50 seqs/family, only families in a clan (ProtoCol's loader).

Also materialises `test_protocol` = ProtoCol's exact split procedure
(seed 42, test_frac 0.15, min_members 3, subsample to 3000) so their
published numbers are comparable on an identical corpus.

Outputs data/splits/*.tsv + data/splits/hashes.json.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "splits"
SEED = 42
CANONICAL = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")
MAX_TEST = 3000


def ok_seq(s: str) -> bool:
    return bool(CANONICAL.fullmatch(s)) and 20 <= len(s) <= 512


def load_scope() -> list[dict]:
    fa = RAW / "scope" / "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
    rows, sid, sccs, chunks = [], None, None, []

    def flush():
        if sid is None:
            return
        seq = "".join(chunks).upper()
        if ok_seq(seq):
            sf = ".".join(sccs.split(".")[:3])
            rows.append({"id": sid, "sequence": seq, "family": sccs,
                         "superfamily": sf, "fold": ".".join(sccs.split(".")[:2])})

    for line in fa.read_text().splitlines():
        if line.startswith(">"):
            flush()
            parts = line[1:].split()
            sid, sccs, chunks = parts[0], (parts[1] if len(parts) > 1 else "x.x.x.x"), []
        else:
            chunks.append(line.strip())
    flush()
    return rows


def load_pfam(max_per_family: int = 50) -> list[dict]:
    fam2clan = {}
    with gzip.open(RAW / "pfam" / "Pfam-A.clans.tsv.gz", "rt", encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1]:
                fam2clan[p[0]] = p[1]

    rows = []
    fam, seqs = None, []
    with gzip.open(RAW / "pfam" / "Pfam-A.seed.gz", "rt", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("#=GF AC"):
                fam = line.split()[2].split(".")[0]
            elif line == "//":
                if fam and fam in fam2clan:
                    clan = fam2clan[fam]
                    for sid, gapped in seqs[:max_per_family]:
                        ungapped = re.sub(r"[^A-Za-z]", "", gapped).upper()
                        ungapped = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", ungapped)
                        if ok_seq(ungapped):
                            rows.append({"id": sid, "sequence": ungapped,
                                         "family": fam, "superfamily": clan,
                                         "fold": ""})
                fam, seqs = None, []
            elif line.startswith("#") or not line:
                continue
            else:
                p = line.split(None, 1)
                if len(p) == 2:
                    seqs.append((p[0], p[1]))
    return rows


def group_split(rows: list[dict], test_frac: float, min_members: int,
                val_frac: float = 0.10) -> dict[str, list[dict]]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["superfamily"]].append(r)
    groups = [g for g, m in by_group.items() if len(m) >= min_members]
    random.Random(SEED).shuffle(groups)
    n_test = max(1, int(len(groups) * test_frac))
    test_g = set(groups[:n_test])
    rest = groups[n_test:]
    n_val = max(1, int(len(rest) * val_frac))
    val_g = set(rest[:n_val])
    train_g = set(rest[n_val:])
    pick = lambda gs: [r for g in gs for r in by_group[g]]
    return {"train": pick(train_g), "val": pick(val_g), "test": pick(test_g)}


def protocol_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Exact ProtoCol split_by_superfamily + subsample_test_set (SEED=42)."""
    by_sf: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sf[r["superfamily"]].append(r)
    sfs = [s for s, m in by_sf.items() if len(m) >= 3]
    random.Random(SEED).shuffle(sfs)
    n_test = max(1, int(len(sfs) * 0.15))
    test_sfs = set(sfs[:n_test])
    train_sfs = set(sfs[n_test:])
    train = [r for s in train_sfs for r in by_sf[s]]
    test = [r for s in test_sfs for r in by_sf[s]]

    if len(test) > MAX_TEST:
        t_by = defaultdict(list)
        for r in test:
            t_by[r["superfamily"]].append(r)
        rng = random.Random(SEED)
        kept = []
        for members in t_by.values():
            rng.shuffle(members)
            kept.extend(members[:2])
        leftovers = [r for m in t_by.values() for r in m[2:]]
        rng.shuffle(leftovers)
        rem = MAX_TEST - len(kept)
        if rem > 0:
            kept.extend(leftovers[:rem])
        test = kept
    return train, test


def write_tsv(path: Path, rows: list[dict]) -> str:
    cols = ["id", "sequence", "family", "superfamily", "fold"]
    lines = ["\t".join(cols)]
    for r in sorted(rows, key=lambda x: x["id"]):
        lines.append("\t".join(r.get(c, "") for c in cols))
    path.write_text("\n".join(lines) + "\n")
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"  {path.name}: {len(rows)} rows sha256={h[:16]}…")
    return h


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    print("SCOPe:")
    scope = load_scope()
    sp = group_split(scope, test_frac=0.15, min_members=3)
    for name, rows in sp.items():
        hashes[f"scope_{name}"] = write_tsv(OUT / f"scope_{name}.tsv", rows)
    ptrain, ptest = protocol_split(scope)
    hashes["scope_protocol_train"] = write_tsv(OUT / "scope_protocol_train.tsv", ptrain)
    hashes["scope_test_protocol"] = write_tsv(OUT / "scope_test_protocol.tsv", ptest)

    print("Pfam:")
    pfam = load_pfam()
    pp = group_split(pfam, test_frac=0.15, min_members=3)
    for name, rows in pp.items():
        hashes[f"pfam_{name}"] = write_tsv(OUT / f"pfam_{name}.tsv", rows)

    (OUT / "hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    sys.path.insert(0, str(ROOT))
    from esmc_retrieval.runlog import log_run
    log_run("T0.4", {"seed": SEED, "test_frac": 0.15, "min_members": 3,
                     "hashes": hashes},
            {k: sum(1 for _ in open(OUT / f"{k}.tsv")) - 1 for k in hashes})
    print(f"split hashes -> {OUT / 'hashes.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
