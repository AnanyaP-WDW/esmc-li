#!/usr/bin/env python3
"""Fetch and pin all M0 datasets. Second run is a no-op (idempotent).

Writes data/manifest.json with url, sha256, bytes, and SCOPe/Pfam counts.
The esmc.cpp pin is sticky: rerunning keeps the manifest commit and records
the current checkout as `head`; edit the manifest to change the pin.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"

SOURCES = {
    "scope_dir_cla": "https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.08-stable.txt",
    "astral_40": "https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa",
    "pfam_clans": "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz",
    "pfam_version": "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam.version.gz",
    "pfam_seed": "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz",
}

ESMC_CPP = Path("/Users/ananyapathak/Desktop/oss/esmc.cpp")
ESMC_GGUF = ESMC_CPP / "models" / "esmc-300m-f16.gguf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        }
    )
    return s


def anubis_pass(s: requests.Session, url: str) -> requests.Response:
    """scop.berkeley.edu puts Anubis bot-protection on some paths; pass it."""
    r = s.get(url, timeout=60)
    if "anubis_challenge" not in r.text:
        return r
    m = re.search(r'id="anubis_challenge" type="application/json">(.*?)</script>', r.text, re.S)
    ch = json.loads(m.group(1))["challenge"]
    time.sleep(2.1)
    s.get(
        "https://scop.berkeley.edu/.within.website/x/cmd/anubis/api/pass-challenge",
        params={"challenge": ch["randomData"], "id": ch["id"], "redir": "/"},
        timeout=60,
    )
    return s.get(url, timeout=60)


def download(name: str, url: str, dest: Path) -> bool:
    """Fetch url -> dest unless dest already matches the recorded hash. True = downloaded."""
    if dest.exists() and MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
        rec = man.get("sources", {}).get(name)
        if rec and sha256(dest) == rec["sha256"]:
            print(f"[skip] {name} (hash matches manifest)")
            return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {name}: {url}")
    s = http_session()
    r = anubis_pass(s, url)
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return True


def scope_counts(dir_cla: Path, astral: Path) -> dict:
    folds, superfams, fams, domains = set(), set(), set(), 0
    astral_ids = set()
    for line in astral.read_text().splitlines():
        if line.startswith(">"):
            astral_ids.add(line[1:].split()[0])
    in_both = 0
    for line in dir_cla.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        did, sccs = cols[0], cols[3]
        domains += 1
        parts = sccs.split(".")
        fams.add(sccs)
        superfams.add(".".join(parts[:3]))
        folds.add(".".join(parts[:2]))
        if did in astral_ids:
            in_both += 1
    return {
        "scope_domains_dir_cla": domains,
        "scope_folds": len(folds),
        "scope_superfamilies": len(superfams),
        "scope_families": len(fams),
        "astral_40_domains": len(astral_ids),
        "astral_40_domains_in_dir_cla": in_both,
    }


def esmc_rows() -> dict:
    import subprocess

    head = subprocess.run(
        ["git", "-C", str(ESMC_CPP), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    prior = {}
    if MANIFEST.exists():
        prior = json.loads(MANIFEST.read_text()).get("esmc_cpp", {})
    commit = prior.get("commit", head)
    check = subprocess.run(
        ["git", "-C", str(ESMC_CPP), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
    )
    if check.returncode != 0:
        raise SystemExit(
            f"pinned esmc.cpp commit {commit} not found in {ESMC_CPP}; "
            "update data/manifest.json or fetch that commit"
        )
    rows = {
        "path": str(ESMC_CPP),
        "commit": commit,
        "head": head,
        "gguf": {
            "path": str(ESMC_GGUF),
            "sha256": sha256(ESMC_GGUF),
            "bytes": ESMC_GGUF.stat().st_size,
        },
        "esmc_embed_binary": str(ESMC_CPP / "build" / "esmc-embed"),
    }
    if prior.get("note"):
        rows["note"] = prior["note"]
    return rows


def main() -> int:
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"sources": {}}
    subdirs = {"scope_dir_cla": "scope", "astral_40": "scope"}
    for name, url in SOURCES.items():
        dest = RAW / subdirs.get(name, name.split("_")[0]) / Path(url).name
        download(name, url, dest)
        man["sources"][name] = {
            "url": url,
            "path": str(dest.relative_to(ROOT)),
            "sha256": sha256(dest),
            "bytes": dest.stat().st_size,
        }

    dir_cla = RAW / "scope" / Path(SOURCES["scope_dir_cla"]).name
    astral = RAW / "scope" / Path(SOURCES["astral_40"]).name
    man["counts"] = scope_counts(dir_cla, astral)

    with gzip.open(ROOT / man["sources"]["pfam_version"]["path"], "rt") as f:
        ver = f.read().strip()
    man["sources"]["pfam_version"]["text"] = ver

    man["esmc_cpp"] = esmc_rows()

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    sys.path.insert(0, str(ROOT))
    from esmc_retrieval.runlog import log_run
    log_run("T0.3", {"sources": sorted(man["sources"]),
                     "esmc_cpp_commit": man["esmc_cpp"]["commit"]},
            man["counts"], model_hash=man["esmc_cpp"]["gguf"]["sha256"])
    print(json.dumps(man["counts"], indent=2))
    print(f"manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
