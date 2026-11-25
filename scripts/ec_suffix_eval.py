#!/usr/bin/env python3
"""Run the suffix 3-seed evaluation.

For each seed: suffix+head rows for the 20K reference and the CARE/EnzymARC
queries, then MaxSim evals (CARE ood30/promiscuous, homolog-free EnzymARC
catalytic/5A).  Writes results/ec/suffix_k3_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPLITS = {
    "ref20k": ("data/ec/splits/ref_20k.tsv", "ref20k", 0),
    "care_ood30": ("data/ec/splits/care_ood30.tsv", "care_ood30", 0),
    "care_promiscuous": ("data/ec/splits/care_promiscuous.tsv",
                         "care_promiscuous", 0),
    "enzymarc_catalytic": ("data/ec/splits/enzymarc_catalytic.tsv",
                           "enzymarc_catalytic", 500),
    "enzymarc_5A": ("data/ec/splits/enzymarc_5A.tsv", "enzymarc_5A", 500),
}

FILTERED = {
    "enzymarc_catalytic": "results/ec/diag_w3_catalytic/ref_nohom_catalytic.tsv",
    "enzymarc_5A": "results/ec/diag_w3_5A/ref_nohom_5A.tsv",
}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path,
                    default=ROOT / "results/ec_suffix_models")
    ap.add_argument("--prefix", type=Path,
                    default=ROOT / "cache/ec/prefix/k3")
    ap.add_argument("--suffix", type=Path,
                    default=ROOT / "cache/ec/suffix/k3")
    ap.add_argument("--out", type=Path, default=ROOT / "results/ec")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--skip-embed", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        ckpt = args.models / f"k3_seed{seed}.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        for name, (split, prefix_name, limit) in SPLITS.items():
            out = args.suffix / f"seed{seed}" / name
            if not args.skip_embed:
                cmd = [sys.executable, str(ROOT / "scripts/ec_suffix_embed.py"),
                       str(ROOT / split), str(args.prefix / prefix_name),
                       str(ckpt), str(out)]
                if limit:
                    cmd += ["--limit", str(limit)]
                run(cmd)

        def ev(name: str, ref: str, query: str,
               ref_cache: Path, query_cache: Path) -> None:
            out = args.out / f"{name}_suffix_k3_seed{seed}.csv"
            cmd = [sys.executable, str(ROOT / "scripts/ec_eval.py"),
                   "--ref", ref, "--query", query,
                   "--ref-cache", str(ref_cache),
                   "--query-cache", str(query_cache),
                   "--method", "maxsim", "--k", "10", "--out", str(out)]
            if name.startswith("enzymarc"):
                cmd += ["--query-limit", "500", "--decoy"]
            run(cmd)

        suffix = args.suffix / f"seed{seed}"
        ev("care_ood30", str(ROOT / "data/ec/splits/ref_20k.tsv"),
           str(ROOT / "data/ec/splits/care_ood30.tsv"),
           suffix / "ref20k", suffix / "care_ood30")
        ev("care_promiscuous", str(ROOT / "data/ec/splits/ref_20k.tsv"),
           str(ROOT / "data/ec/splits/care_promiscuous.tsv"),
           suffix / "ref20k", suffix / "care_promiscuous")
        for name in ("enzymarc_catalytic", "enzymarc_5A"):
            filtered = ROOT / FILTERED[name]
            if not filtered.exists():
                raise SystemExit(f"missing filtered ref: {filtered}")
            ev(name, str(filtered),
               str(ROOT / "data/ec/splits" / f"{name}.tsv"),
               suffix / "ref20k", suffix / name)

    rows = []
    for seed in seeds:
        def js(name: str) -> dict:
            p = args.out / f"{name}_suffix_k3_seed{seed}.json"
            return json.loads(p.read_text())

        c30 = js("care_ood30")
        cp = js("care_promiscuous")
        cat = js("enzymarc_catalytic")
        a5 = js("enzymarc_5A")
        rows.append({
            "seed": seed,
            "care_ood30_top1": c30["top1_acc"],
            "care_ood30_top10": c30["topk_acc"],
            "care_prom_top1": cp["top1_acc"],
            "catalytic_fpr": cat["top1_acc"],
            "catalytic_fpr10": cat["topk_acc"],
            "5A_fpr": a5["top1_acc"],
            "5A_fpr10": a5["topk_acc"],
        })
    fields = list(rows[0])
    out_csv = args.out / "suffix_k3_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    means = {k: sum(r[k] for r in rows) / len(rows)
             for k in fields if k != "seed"}
    print(json.dumps({"seeds": seeds, "means": means, "out": str(out_csv)},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
