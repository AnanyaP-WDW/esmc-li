#!/usr/bin/env python3
"""M6 T6.4 — end-to-end build rate on M4 Max (from measured M4/M6 runs).

Default-scorer index = prefix_dataset (HF block-27, T4.2) -> build_trained
projection 960->128 (mtime of cache/m4/db26.npy -> db_tr128.npy) -> .esmci
write (T6.2, 22.6 s for 34.05M rows). Effective rate is dominated by the
prefix embed. Logs T6.4; §15 build column consumes it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from esmc_retrieval.runlog import log_run

    runs = [json.loads(l) for l in (ROOT / "results" / "runs.jsonl")
            .read_text().splitlines()]
    t42 = [r for r in runs if r.get("task") == "T4.2"]
    embed = next(r for r in t42 if r["config"].get("mode") != "prefix26")
    prefix = next(r for r in t42 if r["config"].get("mode") == "prefix26")
    res = prefix["metrics"]["residues"]
    m4_dir = ROOT / "cache" / "m4"
    proj_s = (m4_dir / "db_tr128.npy").stat().st_mtime - \
        (m4_dir / "db26.npy").stat().st_mtime
    esmci_s = 22.6  # scripts/m6_index.py: build of cache/m4/db_tr128.esmci
    t6 = next(r for r in runs if r.get("task") == "T6.2")
    rows = t6["metrics"]["n_residues"]

    rates = {
        "embed_base_res_s": res / embed["wall_s"],
        "prefix_res_s": res / prefix["wall_s"],
        "project_res_s": res / proj_s,
        "esmci_res_s": rows / esmci_s,
    }
    # sequential end-to-end for the frozen default (prefix -> project -> esmci)
    eff = 1.0 / (1.0 / rates["prefix_res_s"] + 1.0 / rates["project_res_s"]
                 + 1.0 / rates["esmci_res_s"])
    m = {**{k: round(v, 1) for k, v in rates.items()},
         "effective_default_res_s": round(eff, 1),
         "effective_default_prot_s_400aa": round(eff / 400, 2),
         "embed_base_s_100k": round(embed["wall_s"], 1),
         "prefix_s_100k": round(prefix["wall_s"], 1),
         "project_s_100k": round(proj_s, 1),
         "esmci_s_100k": round(esmci_s, 1),
         "n_proteins": prefix["metrics"]["n"], "residues": res}
    log_run("T6.4", {"source": "measured M4/M6 runs: T4.2 (embed, prefix), "
                     "mtime db26->db_tr128 (project), T6.2 (esmci)",
                     "hardware": "M4 Max"}, m)
    print(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
