#!/usr/bin/env python3
"""Data pipeline for the EC active-site late-interaction experiment.

Stages:
  fetch     download raw FuncSeek/CARE/EnzymARC/UniProt files
  build     parse raw files into data/ec/splits/*.tsv and a manifest
  leakage   exact accession overlap + optional MMseqs2 identity scan
  all       fetch -> build -> leakage

The raw data is not committed.  The manifest records URLs, sizes, hashes,
counts, the UniProt release, and the split contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ec"
RAW = DATA / "raw"
SPLITS = DATA / "splits"
MANIFEST = DATA / "manifest.json"

FUNCSEEK = "https://raw.githubusercontent.com/leendertcloete/FuncSeek/main/supporting_information"
ENZYMARC = "https://raw.githubusercontent.com/jsartori12/EnzymARC/main/data"
UNIPROT = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_FIELDS = ("accession,sequence,ec,organism_id,lineage,"
                  "ft_act_site,ft_binding,xref_pdb,xref_pfam")

FUNCSEEK_FILES = {
    "s1_train.csv": "S1_Table_training_set.csv",
    "s2_val.csv": "S2_Table_validation_set.csv",
    "s3_brenda.csv": "S3_Table_brenda_trembl_set.csv",
    "care_ood30.csv": "CARE_test_splits/30_protein_test.csv",
    "care_30_50.csv": "CARE_test_splits/30-50_protein_test.csv",
    "care_price.csv": "CARE_test_splits/price_protein_test.csv",
    "care_promiscuous.csv": "CARE_test_splits/promiscuous_protein_test.csv",
}

ENZYMARC_FILES = {
    "enzymarc_catalytic.fasta": "decoys_cataliticals.fasta",
    "enzymarc_5A.fasta": "decoys_5A.fasta",
    "enzymarc_10A.fasta": "decoys_10A.fasta",
    "enzymarc_15A.fasta": "decoys_15A.fasta",
}

EC4_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
EC_RE = re.compile(r"\d+\.\d+\.\d+\.\d+")
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWXY]+$")


def clean_seq(seq: str) -> str:
    """Upper-case and map non-standard residues to X (keeps test rows)."""
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", (seq or "").upper())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url: str, dest: Path, *, params: dict | None = None) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url} -> {dest.relative_to(ROOT)}", flush=True)
    with requests.get(url, params=params, stream=True, timeout=(20, 600)) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    print(f"[done] {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)")


def fetch_all() -> None:
    for name, remote in FUNCSEEK_FILES.items():
        fetch(f"{FUNCSEEK}/{remote}", RAW / "funcseek" / name)
    for name, remote in ENZYMARC_FILES.items():
        fetch(f"{ENZYMARC}/{remote}", RAW / "enzymarc" / name)
    fetch(UNIPROT, RAW / "uniprot" / "sprot_2026_04.tsv",
          params={"format": "tsv", "fields": UNIPROT_FIELDS,
                  "query": "reviewed:true"})
    fetch_trembl_s3()


def fetch_trembl_s3() -> None:
    """Fetch sequences for the S3 TrEMBL accessions in batches."""
    dest = RAW / "uniprot" / "trembl_s3.tsv"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.relative_to(ROOT)}")
        return
    accs = list(read_ec_csv(RAW / "funcseek" / "s3_brenda.csv"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tsv.part")
    fields = "accession,sequence,ec"
    with tmp.open("w") as out:
        out.write("Entry\tSequence\tEC number\n")
        for i in range(0, len(accs), 50):
            batch = accs[i:i + 50]
            query = " OR ".join(f"accession:{a}" for a in batch)
            r = requests.get(UNIPROT, params={
                "query": query, "format": "tsv", "fields": fields,
            }, timeout=(20, 120))
            r.raise_for_status()
            lines = r.text.splitlines()
            for line in lines[1:]:
                out.write(line + "\n")
            print(f"[trembl] {min(i + 50, len(accs))}/{len(accs)}", flush=True)
    tmp.replace(dest)


def read_ec_csv(path: Path) -> dict[str, list[str]]:
    """accession -> ordered unique EC numbers."""
    out: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            acc = (r.get("uniprot_accession") or r.get("Entry") or "").strip()
            ecs = r.get("ec_number") or r.get("EC number") or r.get("EC All") or ""
            if not acc:
                continue
            for ec in EC_RE.findall(ecs):
                if ec not in out[acc]:
                    out[acc].append(ec)
    return dict(out)


def parse_active_sites(field: str) -> list[int]:
    """UniProt 'Active site' column contains positions; keep integers."""
    return [int(x) for x in re.findall(r"\b(\d+)\b", field or "")]


def parse_uniprot() -> dict[str, dict]:
    out: dict[str, dict] = {}
    paths = [RAW / "uniprot" / "sprot_2026_04.tsv"]
    trembl = RAW / "uniprot" / "trembl_s3.tsv"
    if trembl.exists():
        paths.append(trembl)
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                acc = r.get("Entry", "").strip()
                if not acc:
                    continue
                ecs = [e for e in EC_RE.findall(r.get("EC number", ""))]
                seq = clean_seq(r.get("Sequence", ""))
                out[acc] = {
                    "sequence": seq,
                    "ec_all": ecs,
                    "organism": r.get("Organism (ID)", ""),
                    "lineage": r.get("Taxonomic lineage", ""),
                    "active_sites": parse_active_sites(r.get("Active site", "")),
                    "binding_sites": parse_active_sites(r.get("Binding site", "")),
                    "pdb": r.get("PDB", ""),
                    "pfam": r.get("Pfam", ""),
                }
    return out


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t",
                           lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_rows(accessions: list[str], ecs: dict[str, list[str]],
              uni: dict[str, dict], source: str,
              single: bool = False) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for acc in accessions:
        rec = uni.get(acc)
        if rec is None:
            missing.append(acc)
            continue
        seq = rec["sequence"]
        if not seq or seq.count("X") > max(5, len(seq) // 20):
            missing.append(acc)
            continue
        all_ec = list(dict.fromkeys(ecs.get(acc, []) or rec["ec_all"]))
        complete = [e for e in all_ec if EC4_RE.match(e)]
        if single:
            # FuncSeek S1 label contract: first annotation only.
            complete = complete[:1]
        if not complete:
            missing.append(acc)
            continue
        rows.append({
            "id": acc,
            "sequence": seq,
            "ec_all": ";".join(complete),
            "ec4": ";".join(complete),
            "ec1": ";".join(sorted({e.split(".")[0] for e in complete})),
            "source": source,
            "organism": rec["organism"],
            "lineage": rec["lineage"],
            "pfam": rec["pfam"],
            "active_sites": ";".join(map(str, rec["active_sites"])),
            "pdb": rec["pdb"],
        })
    return rows, missing


def parse_care(path: Path, source: str) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            acc = (r.get("Entry") or "").strip()
            seq = clean_seq(r.get("Sequence", ""))
            if not acc or not seq:
                continue
            ecs = [e for e in EC_RE.findall(
                " ".join([r.get("EC number", ""), r.get("EC All", "")]))]
            ecs = list(dict.fromkeys(ecs))
            complete = [e for e in ecs if EC4_RE.match(e)]
            if not complete:
                continue
            rows.append({
                "id": acc,
                "sequence": seq,
                "ec_all": ";".join(complete),
                "ec4": ";".join(complete),
                "ec1": ";".join(sorted({e.split(".")[0] for e in complete})),
                "source": source,
                "organism": "",
                "lineage": "",
                "pfam": "",
                "active_sites": "",
                "pdb": "",
                "clusterRes30": r.get("clusterRes30", ""),
                "clusterRes50": r.get("clusterRes50", ""),
                "clusterRes70": r.get("clusterRes70", ""),
                "clusterRes90": r.get("clusterRes90", ""),
            })
    return rows


def parse_enzymarc(path: Path) -> list[dict]:
    rows, name, chunks = [], None, []
    def flush():
        if name is None:
            return
        parts = name.split("|")
        acc = parts[0]
        cat = parts[1] if len(parts) > 1 else ""
        ecs = [e for e in EC_RE.findall(name) if EC4_RE.match(e)]
        ec = ecs[0] if ecs else ""
        seq = clean_seq("".join(chunks))
        if seq and EC4_RE.match(ec):
            rows.append({"id": f"{acc}|{cat}",
                         "parent": acc,
                         "category": cat,
                         "ec4": ec,
                         "sequence": seq,
                         "source": "enzymarc"})
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                name, chunks = line[1:].strip(), []
            elif name is not None:
                chunks.append(line.strip())
    flush()
    return rows


def build() -> None:
    uni = parse_uniprot()
    s1 = RAW / "funcseek" / "s1_train.csv"
    s2 = RAW / "funcseek" / "s2_val.csv"
    s3 = RAW / "funcseek" / "s3_brenda.csv"
    ec1 = read_ec_csv(s1)
    ec2 = read_ec_csv(s2)
    ec3 = read_ec_csv(s3)

    train_single, miss1 = make_rows(list(ec1), ec1, uni, "funcseek_s1",
                                    single=True)
    train_multi, _ = make_rows(list(ec1), ec1, uni, "funcseek_s1",
                               single=False)
    val, miss2 = make_rows(list(ec2), ec2, uni, "funcseek_s2")
    transfer, miss3 = make_rows(list(ec3), ec3, uni, "funcseek_s3")

    fields = ["id", "sequence", "ec_all", "ec4", "ec1", "source",
              "organism", "lineage", "pfam", "active_sites", "pdb"]
    write_tsv(SPLITS / "train_s1_single.tsv", train_single, fields)
    write_tsv(SPLITS / "train_s1_multi.tsv", train_multi, fields)
    write_tsv(SPLITS / "val_s2.tsv", val, fields)
    write_tsv(SPLITS / "brenda_trembl.tsv", transfer, fields)

    care_counts = {}
    for name, source in (("care_ood30.csv", "care_ood30"),
                         ("care_30_50.csv", "care_30_50"),
                         ("care_price.csv", "care_price"),
                         ("care_promiscuous.csv", "care_promiscuous")):
        rows = parse_care(RAW / "funcseek" / name, source)
        care_counts[source] = len(rows)
        write_tsv(SPLITS / f"{source}.tsv", rows,
                  fields + ["clusterRes30", "clusterRes50",
                            "clusterRes70", "clusterRes90"])

    decoy_counts = {}
    for name, cat in (("enzymarc_catalytic.fasta", "catalytic"),
                      ("enzymarc_5A.fasta", "5A"),
                      ("enzymarc_10A.fasta", "10A"),
                      ("enzymarc_15A.fasta", "15A")):
        rows = parse_enzymarc(RAW / "enzymarc" / name)
        decoy_counts[cat] = len(rows)
        write_tsv(SPLITS / f"enzymarc_{cat}.tsv", rows,
                  ["id", "parent", "category", "ec4", "sequence", "source"])

    sources = {}
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and not p.name.endswith(".part"):
            sources[str(p.relative_to(ROOT))] = {
                "bytes": p.stat().st_size, "sha256": sha256(p)}
    man = {
        "release": "uniprot_2026_04",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": sources,
        "counts": {
            "train_s1_single": len(train_single),
            "train_s1_multi": len(train_multi),
            "val_s2": len(val),
            "brenda_trembl": len(transfer),
            "care": care_counts,
            "enzymarc": decoy_counts,
        },
        "missing": {"s1": len(miss1), "s2": len(miss2), "s3": len(miss3)},
    }
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(json.dumps(man["counts"], indent=2))
    print("missing:", man["missing"])


def leakage() -> None:
    def ids(path: Path) -> set[str]:
        with path.open(newline="") as f:
            return {r["id"] for r in csv.DictReader(f, delimiter="\t")}

    sets = {}
    for name in ("train_s1_single", "val_s2", "brenda_trembl",
                 "care_ood30", "care_30_50", "care_price",
                 "care_promiscuous"):
        sets[name] = ids(SPLITS / f"{name}.tsv")
    report = {"accession_overlap": {}}
    for a in ("train_s1_single",):
        for b in ("val_s2", "brenda_trembl", "care_ood30", "care_30_50",
                  "care_price", "care_promiscuous"):
            report["accession_overlap"][f"{a}__{b}"] = len(sets[a] & sets[b])
    for cat in ("catalytic", "5A", "10A", "15A"):
        d = ids(SPLITS / f"enzymarc_{cat}.tsv")
        parents = {r["parent"] for r in csv.DictReader(
            (SPLITS / f"enzymarc_{cat}.tsv").open(newline=""), delimiter="\t")}
        report["accession_overlap"][f"train__enzymarc_{cat}_parents"] = len(
            sets["train_s1_single"] & parents)
    (DATA / "leakage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["fetch", "build", "leakage", "all"])
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.stage in ("fetch", "all"):
        fetch_all()
    if args.stage in ("build", "all"):
        build()
    if args.stage in ("leakage", "all"):
        leakage()
    return 0


if __name__ == "__main__":
    sys.exit(main())
