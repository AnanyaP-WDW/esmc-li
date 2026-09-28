#!/usr/bin/env python3
"""M4 T4.1: full-length Swiss-Prot DB with Pfam clan labels (fulllength chains).

Downloads reviewed UniProtKB (sequence + Pfam xrefs), maps families -> clans
(Pfam-A.clans.tsv, same loader as make_splits.py), keeps single-clan chains
(the frozen evaluate() compares one clan string per row), takes ALL chains of
the Pfam test clans (query positives) and fills up to DB_SIZE with non-test-clan
chains (seed 0). Queries = a sample of the test-clan chains, leave-one-out
against the full DB via the frozen harness.

Writes data/fulllength/{db.tsv,queries.tsv,manifest.json}; logs T4.1.
"""
from __future__ import annotations

import csv
import gzip
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import fetch_data  # scripts/ dir is sys.path[0] when run as scripts/m4_fulllength.py

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "uniprot"
OUT = ROOT / "data" / "fulllength"
SPLITS = ROOT / "data" / "splits"
URL = ("https://rest.uniprot.org/uniprotkb/stream?format=tsv"
       "&fields=accession,sequence,xref_pfam&query=reviewed:true")
DB_SIZE = 100_000
QUERY_N = 500
SEED = 0
NONCANON = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")
HIST_EDGES = [50, 100, 200, 400, 800, 1600, 2046]


def download() -> Path:
    """Idempotent: skip iff the manifest sha matches (fetch_data pattern)."""
    dest = RAW / "sprot_pfam.tsv"
    RAW.mkdir(parents=True, exist_ok=True)
    if dest.exists() and fetch_data.MANIFEST.exists():
        man = json.loads(fetch_data.MANIFEST.read_text())
        rec = man.get("sources", {}).get("uniprot_sprot_pfam")
        if rec and fetch_data.sha256(dest) == rec["sha256"]:
            print(f"[skip] uniprot_sprot_pfam ({dest.stat().st_size} bytes)")
            return dest
    print(f"[get ] uniprot_sprot_pfam: {URL}")
    t0 = time.time()
    import requests

    with requests.get(URL, stream=True, timeout=(10, 600)) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".tsv.part")
        mb = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                mb += 1
                if mb % 256 == 0:
                    print(f"  ... {mb} MB")
        tmp.replace(dest)
    print(f"  done in {time.time() - t0:.1f}s -> {dest}")
    man = json.loads(fetch_data.MANIFEST.read_text()) if fetch_data.MANIFEST.exists() else {"sources": {}}
    man.setdefault("sources", {})["uniprot_sprot_pfam"] = {
        "url": URL, "path": str(dest.relative_to(ROOT)),
        "sha256": fetch_data.sha256(dest), "bytes": dest.stat().st_size,
    }
    fetch_data.MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    return dest


def load_fam2clan() -> dict[str, str]:
    fam2clan: dict[str, str] = {}
    p = ROOT / "data" / "raw" / "pfam" / "Pfam-A.clans.tsv.gz"
    with gzip.open(p, "rt", encoding="latin-1") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2 and c[1]:
                fam2clan[c[0]] = c[1]
    return fam2clan


def load_chains(tsv: Path, fam2clan: dict[str, str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats: Counter = Counter()
    with tsv.open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            stats["reviewed"] += 1
            fams = sorted({x for x in (r.get("Pfam") or "").split(";")
                           if x and x in fam2clan})
            if not fams:
                stats["no_clan_family"] += 1
                continue
            clans = {fam2clan[x] for x in fams}
            if len(clans) > 1:
                stats["multi_clan_dropped"] += 1
                continue
            stats["kept_single_clan"] += 1
            seq = (r["Sequence"] or "").upper()
            clean = NONCANON.sub("X", seq)
            if clean != seq:
                stats["noncanonical_replaced"] += 1
            rows.append({"id": r["Entry"], "sequence": clean,
                         "family": ";".join(fams),
                         "superfamily": next(iter(clans)), "fold": ""})
    return rows, dict(stats)


def clan_set(name: str) -> set[str]:
    out = set()
    with (SPLITS / name).open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("superfamily"):
                out.add(r["superfamily"])
    return out


def write_tsv(path: Path, rows: list[dict]) -> str:
    import hashlib

    cols = ["id", "sequence", "family", "superfamily", "fold"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histogram(lens: list[int]) -> dict[str, int]:
    hist: Counter = Counter()
    for L in lens:
        edge = next((e for e in HIST_EDGES if L <= e), None)
        hist[f"<={edge}" if edge else ">2046"] += 1
    order = {f"<={e}": i for i, e in enumerate(HIST_EDGES)}
    order[">2046"] = len(HIST_EDGES)
    return dict(sorted(hist.items(), key=lambda kv: order[kv[0]]))


def main() -> int:
    t0 = time.time()
    fam2clan = load_fam2clan()
    print(f"families with a clan: {len(fam2clan)}")

    tsv = download()
    rows, stats = load_chains(tsv, fam2clan)
    print({k: stats[k] for k in sorted(stats)})

    test_clans = clan_set("pfam_test.tsv")
    train_clans = clan_set("pfam_train.tsv") | clan_set("pfam_val.tsv")
    overlap = test_clans & train_clans
    if overlap:
        raise SystemExit(f"train/test clan overlap: {sorted(overlap)[:5]}")

    rows.sort(key=lambda r: r["id"])
    in_test = [r for r in rows if r["superfamily"] in test_clans]
    other = [r for r in rows if r["superfamily"] not in test_clans]
    if len(in_test) >= DB_SIZE:
        raise SystemExit(f"test-clan chains {len(in_test)} >= DB_SIZE")
    rng = random.Random(SEED)
    rng.shuffle(other)
    db = sorted(in_test + other[: DB_SIZE - len(in_test)], key=lambda r: r["id"])

    per_clan = Counter(r["superfamily"] for r in db)
    pool = [r for r in db if r["superfamily"] in test_clans
            and per_clan[r["superfamily"]] >= 2]
    # round-robin over clans so queries span every eligible test clan
    by_clan: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_clan[r["superfamily"]].append(r)
    qrng = random.Random(SEED + 1)
    clans = sorted(by_clan)
    qrng.shuffle(clans)
    for members in by_clan.values():
        qrng.shuffle(members)
    queries, rnd = [], 0
    while len(queries) < QUERY_N:
        added = False
        for cl in clans:
            if rnd < len(by_clan[cl]):
                queries.append(by_clan[cl][rnd])
                added = True
                if len(queries) >= QUERY_N:
                    break
        if not added:
            break
        rnd += 1
    queries.sort(key=lambda r: r["id"])
    if not queries:
        raise SystemExit("no queries: no test-clan chain has a DB positive")

    OUT.mkdir(parents=True, exist_ok=True)
    db_sha = write_tsv(OUT / "db.tsv", db)
    q_sha = write_tsv(OUT / "queries.tsv", queries)
    lens = [len(r["sequence"]) for r in db]
    gt = [L for L in lens if L > 2046]
    man = {
        "source": json.loads(fetch_data.MANIFEST.read_text())["sources"]["uniprot_sprot_pfam"],
        "db_size": len(db),
        "n_queries": len(queries),
        "query_clans": len({r["superfamily"] for r in queries}),
        "test_clans": len(test_clans),
        "train_val_clans": len(train_clans),
        "clan_overlap": len(overlap),
        "chains_from_test_clans": len(in_test),
        "counts": stats,
        "length_hist": histogram(lens),
        "n_gt_2046": len(gt),
        "max_length": max(lens),
        "sha256": {"db.tsv": db_sha, "queries.tsv": q_sha},
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")

    sys.path.insert(0, str(ROOT))
    from esmc_retrieval.runlog import log_run

    log_run("T4.1", {"db_size": len(db), "db_sha256": db_sha,
                     "query_sha256": q_sha, "source": "uniprotkb_stream_reviewed",
                     "single_clan_only": True, "query_protocol": "leave-one-out",
                     "length_hist": man["length_hist"]},
            {"n_queries": man["n_queries"], "n_gt_2046": man["n_gt_2046"],
             "max_length": man["max_length"], "clan_overlap": 0},
            wall_s=time.time() - t0)
    print(json.dumps({k: man[k] for k in ("db_size", "n_queries", "query_clans",
                                          "n_gt_2046", "max_length", "clan_overlap")}, indent=2))
    print(f"histogram: {man['length_hist']}")
    print(f"total {time.time() - t0:.1f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
