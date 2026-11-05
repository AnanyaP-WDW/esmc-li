#!/usr/bin/env python3
"""M9.0: pin the exact result rows the paper is allowed to use.

The lock is the only input to the paper asset generator. It records the
results commit, the sha256 of that commit's results/runs.jsonl, the selected
row's line hash, and a small ``facts`` block for dataset/hyperparameter
constants that are not run metrics.

Selection is explicit (task + timestamp + config), never "latest" or "max".
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs.jsonl"
OUT = ROOT / "paper" / "results.lock.json"

# result id -> (task, exact ts, extra config matcher)
SEL: dict[str, tuple[str, str, dict]] = {
    "t0.random": ("T0.6", "2026-09-27T01:41:38+0530", {"method": "random"}),
    "t0.minhash": ("T0.6", "2026-09-27T01:41:41+0530", {"method": "minhash5mer"}),
    "t0.mmseqs2": ("T0.6", "2026-09-27T01:41:44+0530", {"method": "mmseqs2_s7.5"}),
    "t0.protocol_frozen": ("T0.7", "2026-09-27T02:02:43+0530", {}),
    "t0.protocol_trained": ("T0.7", "2026-09-27T02:02:43+0530", {}),
    "t1.pooled": ("T1.2", "2026-09-27T15:45:15+0530", {}),
    "t1.maxsim960": ("T1.3", "2026-09-27T15:45:16+0530", {}),
    "t1.maxsim128": ("T1.3", "2026-09-27T15:45:17+0530", {}),
    "t2.d64.s0": ("T2.5", "2026-09-27T18:55:06+0530", {"d": 64, "seed": 0}),
    "t2.d64.s1": ("T2.5", "2026-09-27T18:55:06+0530", {"d": 64, "seed": 1}),
    "t2.d64.s2": ("T2.5", "2026-09-27T18:55:06+0530", {"d": 64, "seed": 2}),
    "t2.d128.s0": ("T2.5", "2026-09-27T18:55:06+0530", {"d": 128, "seed": 0}),
    "t2.d128.s1": ("T2.5", "2026-09-27T18:55:06+0530", {"d": 128, "seed": 1}),
    "t2.d128.s2": ("T2.5", "2026-09-27T18:55:06+0530", {"d": 128, "seed": 2}),
    "t2.d256.s0": ("T2.5", "2026-09-27T18:55:06+0530", {"d": 256, "seed": 0}),
    "t2.d256.s1": ("T2.5", "2026-09-27T18:55:06+0530", {"d": 256, "seed": 1}),
    "t2.d256.s2": ("T2.5", "2026-09-27T18:55:06+0530", {"d": 256, "seed": 2}),
    "t3.k2.s0": ("T3.2", "2026-09-27T21:37:25+0530", {"k": 2, "seed": 0}),
    "t3.k2.s1": ("T3.2", "2026-09-27T21:37:25+0530", {"k": 2, "seed": 1}),
    "t3.k2.s2": ("T3.2", "2026-09-27T21:37:25+0530", {"k": 2, "seed": 2}),
    "t3.k3.s0": ("T3.2", "2026-09-27T21:37:25+0530", {"k": 3, "seed": 0}),
    "t3.k3.s1": ("T3.2", "2026-09-27T21:37:25+0530", {"k": 3, "seed": 1}),
    "t3.k3.s2": ("T3.2", "2026-09-27T21:37:25+0530", {"k": 3, "seed": 2}),
    "t2.full.s0": ("T2.full", "2026-11-05T14:32:17+0530", {"seed": 0}),
    "t2.full.s1": ("T2.full", "2026-11-05T14:32:18+0530", {"seed": 1}),
    "t2.full.s2": ("T2.full", "2026-11-05T14:32:18+0530", {"seed": 2}),
    "t3.full.s0": ("T3.full", "2026-11-05T14:32:19+0530", {"seed": 0}),
    "t3.full.s1": ("T3.full", "2026-11-05T14:32:19+0530", {"seed": 1}),
    "t3.full.s2": ("T3.full", "2026-11-05T14:32:20+0530", {"seed": 2}),
}
# tau x masked x seed
for tau in (0.03, 0.1, 0.3, 1.0):
    for masked in (True, False):
        for seed in (0, 1, 2):
            SEL[f"t2.tau{tau}.m{int(masked)}.s{seed}"] = (
                "T2.4", "2026-09-27T17:36:00+0530",
                {"tau": tau, "masked": masked, "seed": seed})
# full-length rows: second (final) batch
FL = {
    "random": "2026-09-28T03:18:20+0530",
    "minhash": "2026-09-28T03:22:14+0530",
    "mmseqs2": "2026-09-28T03:22:14+0530",
    "pooled_raw": "2026-09-28T03:22:40+0530",
    "pooled_divA": "2026-09-28T03:22:50+0530",
    "pooled_divB": "2026-09-28T03:23:01+0530",
    "pooled_divGeo": "2026-09-28T03:23:11+0530",
    "maxsim960_raw": "2026-09-28T10:01:15+0530",
    "maxsim960_divA": "2026-09-28T10:01:22+0530",
    "maxsim960_divB": "2026-09-28T10:01:30+0530",
    "maxsim960_divGeo": "2026-09-28T10:01:37+0530",
    "maxsim128_raw": "2026-09-28T12:43:08+0530",
    "maxsim128_divA": "2026-09-28T12:43:16+0530",
    "maxsim128_divB": "2026-09-28T12:43:24+0530",
    "maxsim128_divGeo": "2026-09-28T12:43:31+0530",
    "trained_raw": "2026-09-28T14:57:05+0530",
    "trained_divA": "2026-09-28T14:57:12+0530",
    "trained_divB": "2026-09-28T14:57:19+0530",
    "trained_divGeo": "2026-09-28T14:57:26+0530",
}
for name, ts in FL.items():
    method = name.rsplit("_", 1)[0]
    if method in ("random", "minhash", "mmseqs2"):
        norm = "raw"
        method = {"minhash": "minhash5mer", "mmseqs2": "mmseqs2_s7.5"}.get(
            method, method)
    else:
        norm = name.rsplit("_", 1)[1]
        method = {"maxsim960": "maxsim960", "maxsim128": "maxsim128",
                  "pooled": "pooled_esmc", "trained": "trained_m3"}[method]
    SEL[f"fl.{name}"] = ("T4.3", ts,
                         {"method": method, "normalization": norm})
# systems
for n, ts in ((10000, "2026-09-28T16:52:28+0530"),
              (100000, "2026-09-28T16:53:23+0530")):
    SEL[f"m5.cpu.{n}"] = ("T5.2", ts, {"device": "cpu", "n_db": n})
for n, ts in ((10000, "2026-09-28T16:54:21+0530"),
              (100000, "2026-09-28T16:58:09+0530")):
    SEL[f"m5.mps.{n}"] = ("T5.2", ts, {"device": "mps", "n_db": n})
SEL["m5.metal.10000"] = ("T5.3", "2026-10-04T18:36:47+0530", {"n_db": 10000})
SEL["m5.metal.100000"] = ("T5.3", "2026-10-04T18:36:47+0530", {"n_db": 100000})
for m, n in (("maxsim_cpu", 10000), ("maxsim_cpu", 100000),
             ("maxsim_mps", 10000), ("maxsim_mps", 100000),
             ("maxsim_metal", 10000), ("maxsim_metal", 100000),
             ("faiss_flat_pooled", 10000), ("faiss_flat_pooled", 100000),
             ("pylate_plaid", 1000)):
    SEL[f"m5.base.{m}.{n}"] = ("T5.4.row", "2026-10-04T23:09:14+0530",
                              {"method": m, "n_db": n})
SEL["t6.buildrate"] = ("T6.4", "2026-10-05T00:50:09+0530", {})
SEL["t7.compress"] = ("T7.1", "2026-10-05T07:15:40+0530", {})
for n, ts in ((10000, "2026-10-27T11:42:38+0530"),
              (100000, "2026-10-27T11:44:13+0530"),
              (1000000, "2026-10-27T12:27:36+0530")):
    SEL[f"t7.coarse.{n}"] = ("T7.2", ts, {"size": n})
for n, ts in ((10000, "2026-10-26T15:58:05+0530"),
              (100000, "2026-10-26T17:46:03+0530"),
              (1000000, "2026-10-27T11:17:30+0530")):
    SEL[f"t7.build.{n}"] = ("T7.3", ts, {"size": n, "stage": "build"})
for n, ts in ((10000, "2026-10-27T11:42:38+0530"),
              (100000, "2026-10-27T11:44:13+0530"),
              (1000000, "2026-10-27T12:27:36+0530")):
    SEL[f"t7.measure.{n}"] = ("T7.3", ts, {"size": n, "stage": "measure"})
SEL["t8.smoke"] = ("T8.1", "2026-11-03T10:07:43+0530", {})
SEL["t8.profile.1k_numpy"] = ("T8.profile", "2026-11-03T21:52:28+0530", {})
SEL["t8.profile.1k_metal"] = ("T8.profile", "2026-11-03T22:01:47+0530", {})
SEL["t8.profile.100k_numpy"] = ("T8.profile", "2026-11-03T22:00:51+0530", {})
SEL["t8.profile.100k_metal"] = ("T8.profile", "2026-11-03T22:02:05+0530", {})
SEL["t8.profile.1m_numpy"] = ("T8.profile", "2026-11-03T22:08:00+0530", {})


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def match(row: dict, task: str, ts: str, extra: dict) -> bool:
    if row.get("task") != task or row.get("ts") != ts:
        return False
    cfg = row.get("config", {})
    for k, v in extra.items():
        if cfg.get(k) != v:
            return False
    return True


def fact_group(name: str, values: dict, sources: list[str]) -> dict:
    return {"values": values, "sources": sources}


def main() -> int:
    results_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    raw = subprocess.run(
        ["git", "-C", str(ROOT), "show",
         f"{results_commit}:results/runs.jsonl"],
        capture_output=True, check=True).stdout
    lines = raw.decode().splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    out: dict[str, dict] = {}
    for rid, (task, ts, extra) in SEL.items():
        found = [(i, r) for i, r in enumerate(rows)
                 if match(r, task, ts, extra)]
        if len(found) != 1:
            raise SystemExit(
                f"{rid}: expected 1 row, found {len(found)} for "
                f"{task} {ts} {extra}")
        i, r = found[0]
        out[rid] = {
            "line": i + 1,
            "line_sha256": sha256_bytes(lines[i].encode()),
            "task": r["task"],
            "ts": r["ts"],
            "seed": r.get("seed", 0),
            "config": r.get("config", {}),
            "metrics": r.get("metrics", {}),
            "git_sha": r.get("git_sha"),
            "derived": r.get("derived", False),
        }

    t04 = next(r for r in rows if r["task"] == "T0.4")
    t41 = next(r for r in rows if r["task"] == "T4.1")
    fl = json.loads((ROOT / "data" / "fulllength" /
                     "manifest.json").read_text())
    ident = (ROOT / "results" / "m0_split_identity.txt").read_text()
    m = re.search(r"max identity over >=80%-coverage alignments = ([\d.]+)%",
                  ident)
    if not m:
        raise SystemExit("could not parse identity file")
    facts = {
        "scope": fact_group("scope", {
            "train": t04["metrics"]["scope_train"],
            "val": t04["metrics"]["scope_val"],
            "test": t04["metrics"]["scope_test"],
            "superfamilies": t04["metrics"]["scope_superfamilies"],
            "families": t04["metrics"]["scope_families"],
            "folds": t04["metrics"]["scope_folds"],
            "max_train_test_identity_80cov": float(m.group(1)) / 100.0,
        }, ["results/runs.jsonl", "results/m0_split_identity.txt"]),
        "full_length": fact_group("full_length", {
            "n_db": fl["db_size"],
            "n_queries": fl["n_queries"],
            "test_clans": fl["test_clans"],
            "n_gt_2046": fl["n_gt_2046"],
            "max_length": fl["max_length"],
            "clan_overlap": fl["clan_overlap"],
        }, ["data/fulllength/manifest.json", "results/runs.jsonl"]),
        "hyperparams": fact_group("hyperparams", {
            "params_m": 300,
            "h": 960,
            "d": 128,
            "ctx": 2048,
            "win": 2046,
            "overlap": 128,
            "batch": 16,
            "epochs": 3,
            "seeds": 3,
            "tau": 1.0,
            "k_blocks": 3,
            "lr_head": 0.001,
            "lr_blocks": 0.00002,
        }, ["scripts/m2_train.py", "scripts/m3_train.py",
            "esmc_retrieval/protein_index.py", "data/manifest.json"]),
    }
    lock = {
        "schema": 1,
        "results_commit": results_commit,
        "runs_sha256": sha256_bytes(raw),
        "results": out,
        "facts": facts,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"pinned {len(out)} results from {results_commit[:12]} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
