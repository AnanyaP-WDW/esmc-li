#!/usr/bin/env python3
"""M9/T9.1: generate every paper table and data figure from the pinned lock.

Inputs:  paper/results.lock.json only (plus paper/tables.tsv-independent code).
Outputs: paper/tables/*.md, paper/figures/*.svg, paper/numbers.json.

No model, cache, network, or esmc.cpp is touched. Formatting is fixed; run
twice and diff to prove determinism.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "paper" / "results.lock.json"
TABLE_DIR = ROOT / "paper" / "tables"
FIG_DIR = ROOT / "paper" / "figures"
NUMBERS = ROOT / "paper" / "numbers.json"

R: dict[str, dict] = {}
FACTS: dict[str, dict] = {}
NUM: dict[str, dict] = {}


def load() -> None:
    global R, FACTS
    lock = json.loads(LOCK.read_text())
    R = lock["results"]
    FACTS = {k: v["values"] for k, v in lock["facts"].items()}


def put(key: str, raw, display: str, table: str, row: str, col: str,
        sources: list[str]) -> str:
    NUM[key] = {"raw": raw, "display": display, "table": table,
                "row": row, "col": col, "sources": sources}
    return display


def num(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def ints(x: int) -> str:
    return f"{x:,}"


def sci(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.3g}"


def metric(rid: str, name: str):
    return R[rid]["metrics"][name]


def cfg(rid: str, name: str, default=None):
    return R[rid]["config"].get(name, default)


def mean_std(ids: list[str], name: str) -> tuple[float, float]:
    vals = [float(metric(i, name)) for i in ids]
    return statistics.fmean(vals), statistics.pstdev(vals)


def cell_pm(ids: list[str], name: str, key: str, table: str, row: str,
            col: str) -> str:
    m, s = mean_std(ids, name)
    return put(f"{key}_pm", m, f"{m:.4f} ± {s:.4f}", table, row, col, ids)


def cell(rid: str, name: str, key: str, table: str, row: str, col: str,
         nd: int = 4) -> str:
    v = float(metric(rid, name))
    return put(key, v, num(v, nd), table, row, col, [rid])


def proto(row: str, variant: str, key: str, k: int) -> str:
    m = R[row]["metrics"][variant]
    v = float(m[str(k)])
    return put(key, v, num(v), "t2_scope", "ProtoCol " + variant.split("_")[1],
               f"cR@{k}", [row])


def md_table(path: Path, title: str, header: list[str],
             rows: list[list[str]], note: str = "") -> None:
    lines = [f"**{title}**", "", "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    if note:
        lines += ["", note]
    path.write_text("\n".join(lines) + "\n")


def t1() -> None:
    s = FACTS["scope"]
    f = FACTS["full_length"]
    h = FACTS["hyperparams"]
    # Individual keys for prose; the table cells below carry the combined rows.
    for k, v, lab in [
        ("data.scope.train_n", s["train"], "SCOPe train"),
        ("data.scope.val_n", s["val"], "SCOPe val"),
        ("data.scope.test_n", s["test"], "SCOPe test"),
        ("data.scope.superfamilies_n", s["superfamilies"], "SCOPe superfamilies"),
        ("data.scope.families_n", s["families"], "SCOPe families"),
        ("data.scope.folds_n", s["folds"], "SCOPe folds"),
        ("data.fl.db_n", f["n_db"], "Full-length chains"),
        ("data.fl.queries_n", f["n_queries"], "Full-length queries"),
        ("data.fl.query_clans_n", f["query_clans"], "Full-length query clans"),
        ("data.fl.gt2046_n", f["n_gt_2046"], "Full-length >2046"),
        ("data.fl.maxlen_n", f["max_length"], "Full-length max length"),
    ]:
        put(k, v, ints(v), "t1a_datasets", lab, "value",
            ["T0.3" if k.startswith("data.scope") else "T4.1"])
    rows = [
        ["SCOPe ASTRAL40 domains",
         put("data.scope.domains", s["domains"], ints(s["domains"]),
             "t1a_datasets", "SCOPe ASTRAL40 domains", "value", ["T0.3"])],
        ["SCOPe train / val / test",
         put("data.scope.splits", s["test"],
             f'{ints(s["train"])} / {ints(s["val"])} / {ints(s["test"])}',
             "t1a_datasets", "SCOPe train / val / test", "value", ["T0.4"])],
        ["SCOPe superfamilies / families / folds",
         put("data.scope.groups", s["superfamilies"],
             f'{ints(s["superfamilies"])} / {ints(s["families"])} / {ints(s["folds"])}',
             "t1a_datasets", "SCOPe superfamilies / families / folds", "value",
             ["T0.3"])],
        ["Max train/test identity (>=80% coverage)",
         put("data.scope.identity", s["max_train_test_identity_80cov"],
             f'{s["max_train_test_identity_80cov"]:.1%}', "t1a_datasets",
             "Max train/test identity (>=80% coverage)", "value",
             ["m0_split_identity.txt"])],
        ["Full-length Swiss-Prot chains",
         put("data.fl.db", f["n_db"], ints(f["n_db"]), "t1a_datasets",
             "Full-length Swiss-Prot chains", "value", ["T4.1"])],
        ["Full-length queries / query clans",
         put("data.fl.queries", f["n_queries"],
             f'{ints(f["n_queries"])} / {ints(f["query_clans"])}',
             "t1a_datasets", "Full-length queries / query clans", "value",
             ["T4.1"])],
        ["Chains >2046 aa / max length",
         put("data.fl.long", f["n_gt_2046"],
             f'{ints(f["n_gt_2046"])} / {ints(f["max_length"])} aa',
             "t1a_datasets", "Chains >2046 aa / max length", "value",
             ["T4.1"])],
    ]
    md_table(TABLE_DIR / "t1a_datasets.md", "Table 1a. Data and splits",
             ["Item", "Value"], rows,
             "SCOPe: group-disjoint train/val/test. Full-length: single-clan "
             "Swiss-Prot chains, test-clan queries, leave-one-out.")

    for k, v, disp, lab in [
        ("model.params_m", h["params_m"], str(h["params_m"]), "Backbone"),
        ("model.h", h["h"], str(h["h"]), "Residue width H"),
        ("model.d", h["d"], str(h["d"]), "Projection D"),
        ("model.ctx_n", h["ctx"], str(h["ctx"]), "Context"),
        ("model.win_n", h["win"], str(h["win"]), "Window"),
        ("model.overlap_n", h["overlap"], str(h["overlap"]), "Overlap"),
        ("train.k_n", h["k_blocks"], str(h["k_blocks"]), "Fine-tuned blocks k"),
        ("train.batch_n", h["batch"], str(h["batch"]), "Batch"),
        ("train.epochs_n", h["epochs"], str(h["epochs"]), "Epochs"),
        ("train.seeds_n", h["seeds"], str(h["seeds"]), "Seeds"),
        ("train.tau_n", h["tau"], f'{h["tau"]:g}', "Temperature"),
        ("train.lr_blocks_n", h["lr_blocks"], f'{h["lr_blocks"]:g}', "Block lr"),
        ("train.lr_head_n", h["lr_head"], f'{h["lr_head"]:g}', "Head lr"),
    ]:
        put(k, v, disp, "t1b_hyperparams", lab, "value", ["data/manifest.json"])
    rows = [
        ["Backbone",
         put("model.params", h["params_m"], "ESM-C 300M", "t1b_hyperparams",
             "Backbone", "value", ["data/manifest.json"])],
        ["Residue width H",
         put("model.h", h["h"], str(h["h"]), "t1b_hyperparams", "Residue width H",
             "value", ["data/manifest.json"])],
        ["Projection D",
         put("model.d", h["d"], str(h["d"]), "t1b_hyperparams", "Projection D",
             "value", ["m2_dims.csv"])],
        ["Context / window / overlap",
         put("model.ctx", h["ctx"],
             f'{h["ctx"]} / {h["win"]} / {h["overlap"]}', "t1b_hyperparams",
             "Context / window / overlap", "value",
             ["esmc_retrieval/protein_index.py"])],
        ["Fine-tuned blocks k",
         put("train.k", h["k_blocks"], str(h["k_blocks"]), "t1b_hyperparams",
             "Fine-tuned blocks k", "value", ["m3_train.py"])],
        ["Batch / epochs / seeds",
         put("train.budget", h["batch"],
             f'{h["batch"]} / {h["epochs"]} / {h["seeds"]}', "t1b_hyperparams",
             "Batch / epochs / seeds", "value", ["m2_train.py"])],
        ["Temperature / block lr / head lr",
         put("train.opt", h["tau"],
             f'{h["tau"]} / {h["lr_blocks"]:g} / {h["lr_head"]:g}',
             "t1b_hyperparams", "Temperature / block lr / head lr", "value",
             ["m2_train.py", "m3_train.py"])],
    ]
    md_table(TABLE_DIR / "t1b_hyperparams.md",
             "Table 1b. Model and training", ["Item", "Value"], rows)


def t2() -> None:
    metrics = [("cR@1", "cR@1"), ("cR@5", "cR@5"), ("cR@10", "cR@10"),
               ("cR@100", "cR@100"), ("MRR", "mrr"),
               ("sens family", "sens_family"),
               ("sens superfamily", "sens_superfamily"),
               ("sens fold", "sens_fold")]
    rows: list[list[str]] = []

    def single(label: str, rid: str, prefix: str) -> None:
        vals = []
        for col, m in metrics:
            vals.append(cell(rid, m, f"t2.{prefix}.{m}", "t2_scope", label, col))
        rows.append([label] + vals)

    def agg(label: str, ids: list[str], prefix: str) -> None:
        vals = []
        for col, m in metrics:
            vals.append(cell_pm(ids, m, f"t2.{prefix}.{m}", "t2_scope", label, col))
        rows.append([label] + vals)

    single("Random", "t0.random", "random")
    single("MinHash 5-mer", "t0.minhash", "minhash")
    single("MMseqs2", "t0.mmseqs2", "mmseqs2")
    single("Pooled ESM-C", "t1.pooled", "pooled")
    single("Frozen MaxSim 960D", "t1.maxsim960", "maxsim960")
    single("Frozen MaxSim 128D", "t1.maxsim128", "maxsim128")
    # ProtoCol: only capped recall is available from their released eval.
    rows.append(["ProtoCol frozen (reproduced)",
                 proto("t0.protocol_frozen", "protocol_frozen",
                       "t2.protocol_frozen.cR@1", 1),
                 proto("t0.protocol_frozen", "protocol_frozen",
                       "t2.protocol_frozen.cR@5", 5),
                 proto("t0.protocol_frozen", "protocol_frozen",
                       "t2.protocol_frozen.cR@10", 10),
                 proto("t0.protocol_frozen", "protocol_frozen",
                       "t2.protocol_frozen.cR@100", 100),
                 "—", "—", "—", "—"])
    rows.append(["ProtoCol trained (reproduced)",
                 proto("t0.protocol_trained", "protocol_trained",
                       "t2.protocol_trained.cR@1", 1),
                 proto("t0.protocol_trained", "protocol_trained",
                       "t2.protocol_trained.cR@5", 5),
                 proto("t0.protocol_trained", "protocol_trained",
                       "t2.protocol_trained.cR@10", 10),
                 proto("t0.protocol_trained", "protocol_trained",
                       "t2.protocol_trained.cR@100", 100),
                 "—", "—", "—", "—"])
    agg("ESM-C + trained head (M2)",
        ["t2.full.s0", "t2.full.s1", "t2.full.s2"], "m2")
    agg("ESM-C fine-tuned k=3 (M3)",
        ["t3.full.s0", "t3.full.s1", "t3.full.s2"], "m3")
    md_table(TABLE_DIR / "t2_scope.md",
             "Table 2. SCOPe ASTRAL40 retrieval",
             ["Method"] + [m[0] for m in metrics], rows,
             "Leave-one-out; cR@k is ProtoCol's capped recall. M2/M3 are "
             "mean ± population std over the seeds; all other rows are single "
             "runs. ProtoCol's released evaluator reports only capped recall.")


def t2b() -> None:
    rows = []
    for label, rid in [("Frozen ESM-C 300M", "t1.maxsim960"),
                       ("Frozen ESM-2 35M (full)", "t1.diagD"),
                       ("Frozen ESM-2 35M (ProtoCol protocol)", "t1.diagC")]:
        rows.append([
            label,
            cell(rid, "cR@10", f"t2b.{rid}.cr10", "t2b_backbone", label, "cR@10"),
            cell(rid, "sens_superfamily", f"t2b.{rid}.sens", "t2b_backbone",
                 label, "sens superfamily"),
            cell(rid, "sens_fold", f"t2b.{rid}.fold", "t2b_backbone",
                 label, "sens fold"),
        ])
    md_table(TABLE_DIR / "t2b_backbone.md",
             "Table 2b. Frozen backbone control",
             ["Backbone", "cR@10", "sens superfamily", "sens fold"], rows,
             "Same harness and test split; no projection training. The ESM-2 "
             "35M rows are the M1 protocol-gap diagnostic.")


def t3() -> None:
    metrics = [("cR@1", "cR@1"), ("cR@5", "cR@5"), ("cR@10", "cR@10"),
               ("cR@100", "cR@100"), ("MRR", "mrr"),
               ("sens family", "sens_family"),
               ("sens superfamily", "sens_superfamily")]
    rows = []
    labels = [("Random", "random"), ("MinHash 5-mer", "minhash"),
              ("MMseqs2", "mmseqs2"), ("Pooled ESM-C", "pooled_raw"),
              ("Frozen MaxSim 960D", "maxsim960_raw"),
              ("Frozen MaxSim 128D", "maxsim128_raw"),
              ("ESM-C fine-tuned k=3", "trained_raw")]
    for label, name in labels:
        vals = []
        for col, m in metrics:
            key = f"t3.{name}.{m}"
            vals.append(cell(f"fl.{name}", m, key, "t3_fulllength", label, col))
        rows.append([label] + vals)
    md_table(TABLE_DIR / "t3_fulllength.md",
             "Table 3. Full-length Swiss-Prot retrieval",
             ["Method"] + [m[0] for m in metrics], rows,
             "Single-clan chains; test-clan queries; leave-one-out; "
             "raw MaxSim for the vector scorers.")


def t4() -> None:
    rows = []
    for label, method in [("ESM-C fine-tuned k=3", "trained"),
                          ("Frozen MaxSim 960D", "maxsim960")]:
        for norm, name in [("raw", "raw"), ("÷ len_A", "divA"),
                           ("÷ len_B", "divB"), ("÷ sqrt(len_A len_B)", "divGeo")]:
            rid = f"fl.{method}_{name}"
            c10 = cell(rid, "cR@10", f"t4.{method}.{name}.cr10",
                       "t4_length", f"{label} {norm}", "cR@10")
            sens = cell(rid, "sens_superfamily", f"t4.{method}.{name}.sens",
                        "t4_length", f"{label} {norm}", "sens superfamily")
            rho = cfg(rid, "rho_len_score")
            rho_disp = put(f"t4.{method}.{name}.rho", rho, num(float(rho), 3),
                           "t4_length", f"{label} {norm}", "rho", [rid])
            rows.append([label, norm, c10, sens, rho_disp])
    md_table(TABLE_DIR / "t4_length.md",
             "Table 4. Length-normalization ablation",
             ["Method", "Normalization", "cR@10", "sens superfamily",
              "rho(len, score)"], rows,
             "Full-length corpus. div_lenA is rank-invariant within a "
             "query; the candidate-length variants change the ranking.")


def t5a() -> None:
    rows = []
    for d in (64, 128, 256):
        ids = [f"t2.d{d}.s{s}" for s in (0, 1, 2)]
        label = f"D={d}"
        pm = cell_pm(ids, "val_cR@10", f"t5a.d{d}.cr10", "t5a_dims",
                     label, "val cR@10")
        put(f"t5a.d{d}.d", d, str(d), "t5a_dims", label, "D", [ids[0]])
        b = int(metric(f"t2.d{d}.s0", "B_per_res_fp16"))
        bd = put(f"t5a.d{d}.bytes", b, str(b), "t5a_dims", label,
                 "bytes/res (fp16)", [ids[0]])
        rows.append([label, pm, bd])
    md_table(TABLE_DIR / "t5a_dims.md",
             "Table 5a. Projection dimension (val)",
             ["D", "val cR@10", "bytes/residue (fp16)"], rows,
             "Mean ± population std over the seeds, selected on scope_val only.")


def t5b() -> None:
    rows = []
    for tau in (0.03, 0.1, 0.3, 1.0):
        for masked in (True, False):
            ids = [f"t2.tau{tau}.m{int(masked)}.s{s}" for s in (0, 1, 2)]
            label = f"tau={tau}, {'masked' if masked else 'unmasked'}"
            pm = cell_pm(ids, "val_cR@10",
                         f"t5b.tau{tau}.m{int(masked)}.cr10",
                         "t5b_tau", label, "val cR@10")
            rows.append([f"{tau:g}", "yes" if masked else "no", pm])
    md_table(TABLE_DIR / "t5b_tau.md",
             "Table 5b. Temperature and false-negative masking (val)",
             ["tau", "masked", "val cR@10"], rows,
             "Mean ± population std over the seeds. Masking removes in-batch "
             "pairs sharing the positive's superfamily from the denominator.")


def t5c() -> None:
    rows = []
    for k in (2, 3):
        ids = [f"t3.k{k}.s{s}" for s in (0, 1, 2)]
        val = cell_pm(ids, "val_cR@10", f"t5c.k{k}.val", "t5c_k",
                      f"k={k}", "val cR@10")
        if k == 3:
            test = cell_pm(["t3.full.s0", "t3.full.s1", "t3.full.s2"],
                           "cR@10", "t5c.k3.test", "t5c_k", "k=3",
                           "test cR@10")
        else:
            test = "—"
        rows.append([f"k={k}", val, test])
    md_table(TABLE_DIR / "t5c_k.md",
             "Table 5c. Fine-tuned blocks (val and test)",
             ["k", "val cR@10", "test cR@10"], rows,
             "Mean ± population std over the seeds. Test is evaluated once after "
             "val selection; k=2 test was not evaluated.")


def t6a() -> None:
    rows = []
    for n in (10000, 100000, 1000000):
        b = R[f"t7.build.{n}"]
        m = R[f"t7.measure.{n}"]
        label = ints(n)
        rows.append([
            put(f"t6.n.{n}", n, label, "t6a_scaling", label, "n_db",
                [f"t7.build.{n}", f"t7.measure.{n}"]),
            put(f"t6.res.{n}", b["metrics"]["residues"],
                ints(b["metrics"]["residues"]), "t6a_scaling", label,
                "residues", [f"t7.build.{n}"]),
            put(f"t6.gb.{n}", m["metrics"]["esmci_bytes"] / 1e9,
                f'{m["metrics"]["esmci_bytes"] / 1e9:.2f}', "t6a_scaling",
                label, "index GB", [f"t7.measure.{n}"]),
            put(f"t6.build_h.{n}", b["metrics"]["build_s"] / 3600,
                f'{b["metrics"]["build_s"] / 3600:.2f}', "t6a_scaling",
                label, "build h", [f"t7.build.{n}"]),
            put(f"t6.open_ms.{n}", m["metrics"]["open_s"] * 1000,
                f'{m["metrics"]["open_s"] * 1000:.1f}', "t6a_scaling",
                label, "open ms", [f"t7.measure.{n}"]),
            put(f"t6.sq.{n}", m["metrics"]["exhaustive_s_per_q"],
                f'{m["metrics"]["exhaustive_s_per_q"]:.3f}', "t6a_scaling",
                label, "exhaustive s/q", [f"t7.measure.{n}"]),
            put(f"t6.qps.{n}", 1.0 / m["metrics"]["exhaustive_s_per_q"],
                sci(1.0 / m["metrics"]["exhaustive_s_per_q"]), "t6a_scaling",
                label, "q/s", [f"t7.measure.{n}"]),
            put(f"t6.C.{n}", m["metrics"]["C_chosen"],
                ints(m["metrics"]["C_chosen"]), "t6a_scaling", label, "C",
                [f"t7.measure.{n}"]),
            put(f"t6.recall.{n}", m["metrics"]["candidate_recall"],
                f'{m["metrics"]["candidate_recall"]:.3f}', "t6a_scaling",
                label, "candidate recall", [f"t7.measure.{n}"]),
            put(f"t6.coarse.{n}", m["metrics"]["coarse_s_per_q"],
                f'{m["metrics"]["coarse_s_per_q"]:.3f}', "t6a_scaling",
                label, "coarse s/q", [f"t7.measure.{n}"]),
            put(f"t6.speedup.{n}", m["metrics"]["speedup_vs_exhaustive"],
                f'{m["metrics"]["speedup_vs_exhaustive"]:.1f}', "t6a_scaling",
                label, "speedup", [f"t7.measure.{n}"]),
        ])
    md_table(TABLE_DIR / "t6a_scaling.md",
             "Table 6a. Exact search and coarse stage across sizes",
             ["n_db", "residues", "index GB", "build h", "open ms",
              "exhaustive s/q", "q/s", "C", "candidate recall",
              "coarse s/q", "speedup"], rows,
             "M4 Max, fused Metal, a fixed query set. Build/index from UniRef50 "
             "subsets; coarse stage reranks C candidates with exact MaxSim.")


def t6b() -> None:
    rows = []
    # CPU/MPS efficiency + qps from T5.2; Metal from T5.3; baselines T5.4.
    for label, rid, idx in [
        ("numpy CPU", "m5.cpu.100000", "m5.base.maxsim_cpu.100000"),
        ("MPS", "m5.mps.100000", "m5.base.maxsim_mps.100000"),
        ("Metal (fused)", "m5.metal.100000", "m5.base.maxsim_metal.100000"),
        ("FAISS flat (pooled)", None, "m5.base.faiss_flat_pooled.100000"),
        ("PyLate/PLAID", None, "m5.base.pylate_plaid.1000"),
    ]:
        m = R[idx]["metrics"]
        n = int(m["n_db"]) if "n_db" in m else int(cfg(idx, "n_db"))
        qps = float(m["qps"]) if "qps" in m else float(m["metal_qps"])
        if rid:
            rm = R[rid]["metrics"]
            total = rm.get("wall_s", rm.get("metal_wall_s"))
            spq = float(total) / float(rm["n_queries"])
        else:
            spq = float(m["search_s"]) / float(cfg(idx, "n_queries"))
        mb = float(m["index_mb"]) if "index_mb" in m else float("nan")
        eff = (float(metric(rid, "efficiency_pct")) if rid and
               "efficiency_pct" in R[rid]["metrics"] else None)
        slug = (label.lower().replace(" ", "_").replace("(", "")
                .replace(")", "").replace("/", "_"))
        rows.append([
            label,
            put(f"t6b.n.{slug}", n, ints(n), "t6b_devices", label, "n_db", [idx]),
            put(f"t6b.qps.{slug}", qps, sci(qps), "t6b_devices", label, "q/s", [rid or idx]),
            put(f"t6b.sq.{slug}", spq, sci(spq), "t6b_devices", label, "s/query", [rid or idx]),
            put(f"t6b.mb.{slug}", mb, f"{mb:.1f}" if mb == mb else "—",
                "t6b_devices", label, "index MB", [idx]),
            put(f"t6b.eff.{slug}", eff if eff is not None else float("nan"),
                f"{eff:.1f}" if eff is not None else "—", "t6b_devices",
                label, "GEMM eff %", [rid] if rid else []),
        ])
    md_table(TABLE_DIR / "t6b_devices.md",
             "Table 6b. Search backends",
             ["Backend", "n_db", "q/s", "s/query", "index MB", "GEMM eff %"],
             rows,
             "CPU/MPS/Metal are exact MaxSim. FAISS is an approximate latency "
             "floor over pooled vectors, not a quality-comparable retriever. "
             "The PyLate/PLAID row is a small-index subset because full-DB "
             "encoding was infeasible here.")


def t7() -> None:
    m = R["t7.compress"]["metrics"]
    rows = []
    for label, key, ckey, dkey, rkey in [
        ("fp16", "fp16", "fp16_cR@10", "fp16_drop_cR@10", "fp16_ratio_vs_f16"),
        ("int8", "int8", "int8_cR@10", "int8_drop_cR@10", "int8_ratio_vs_f16"),
        ("residual int8", "residual_int8", "residual_int8_cR@10",
         "residual_int8_drop_cR@10", "residual_int8_ratio_vs_f16"),
        ("residual PQ m8", "pq_m8", "residual_pq_m8_cR@10",
         "residual_pq_m8_drop_cR@10", "residual_pq_m8_ratio_vs_f16"),
        ("residual PQ m16", "pq_m16", "residual_pq_m16_cR@10",
         "residual_pq_m16_drop_cR@10", "residual_pq_m16_ratio_vs_f16"),
        ("residual PQ m32", "pq_m32", "residual_pq_m32_cR@10",
         "residual_pq_m32_drop_cR@10", "residual_pq_m32_ratio_vs_f16"),
    ]:
        ratio = float(m[rkey])
        drop = float(m[dkey])
        ok = ratio >= 4.0 and drop <= 0.01
        rows.append([
            label,
            put(f"t7.{key}.ratio", ratio, f"{ratio:.2f}", "t7_compression",
                label, "ratio vs f16", ["t7.compress"]),
            put(f"t7.{key}.cr10", float(m[ckey]), f"{float(m[ckey]):.4f}",
                "t7_compression", label, "cR@10", ["t7.compress"]),
            put(f"t7.{key}.drop", drop, f"{drop:.4f}", "t7_compression",
                label, "drop", ["t7.compress"]),
            "yes" if ok else "no",
        ])
    md_table(TABLE_DIR / "t7_compression.md",
             "Table 7. Compression trade-off",
             ["Scheme", "ratio vs f16", "cR@10", "drop cR@10", "pass"],
             rows,
             "Pass = at least four times smaller and no more than one point of cR@10 drop. Only PQ m32 passes; "
             "it was not shipped because decode+MaxSim was slower than f16.")


def t8() -> None:
    rows = []
    for n in (10000, 100000, 1000000):
        m = R[f"t7.measure.{n}"]["metrics"]
        c = R[f"t7.coarse.{n}"]["metrics"]
        label = ints(n)
        rows.append([
            label,
            put(f"t8.C.{n}", m["C_chosen"], ints(m["C_chosen"]),
                "t8_coarse", label, "C", [f"t7.measure.{n}"]),
            put(f"t8.recall.{n}", m["candidate_recall"],
                f'{m["candidate_recall"]:.3f}', "t8_coarse", label,
                "candidate recall", [f"t7.measure.{n}"]),
            put(f"t8.coarse.{n}", m["coarse_s_per_q"],
                f'{m["coarse_s_per_q"]:.3f}', "t8_coarse", label,
                "coarse s/q", [f"t7.measure.{n}"]),
            put(f"t8.exhaustive.{n}", m["exhaustive_s_per_q"],
                f'{m["exhaustive_s_per_q"]:.3f}', "t8_coarse", label,
                "exhaustive s/q", [f"t7.measure.{n}"]),
            put(f"t8.agreement.{n}", c["rerank_agreement"],
                f'{c["rerank_agreement"]:.2f}', "t8_coarse", label,
                "rerank agreement", [f"t7.coarse.{n}"]),
            put(f"t8.speedup.{n}", m["speedup_vs_exhaustive"],
                f'{m["speedup_vs_exhaustive"]:.1f}', "t8_coarse", label,
                "speedup", [f"t7.measure.{n}"]),
        ])
    md_table(TABLE_DIR / "t8_coarse.md",
             "Table 8. Coarse candidate stage vs exhaustive",
             ["n_db", "C", "candidate recall", "coarse s/q",
              "exhaustive s/q", "rerank agreement", "speedup"], rows,
             "Candidate recall is recall of the exhaustive top-10 candidate set. "
             "At the largest size, random candidate gather makes rerank almost exhaustive.")


def t9() -> None:
    rows = []
    for label, rid, eng in [
        ("1K numpy", "t8.profile.1k_numpy", "numpy"),
        ("1K Metal", "t8.profile.1k_metal", "Metal"),
        ("100K numpy", "t8.profile.100k_numpy", "numpy"),
        ("100K Metal", "t8.profile.100k_metal", "Metal"),
        ("1M numpy", "t8.profile.1m_numpy", "numpy"),
    ]:
        m = R[rid]["metrics"]
        search_s = put(f"t9.{eng}.{m['n']}.search_s",
                       m["search_ms_per_q"] / 1000.0,
                       f'{m["search_ms_per_q"] / 1000.0:.2f}', "t9_profile",
                       label, "search s/q", [rid])
        rows.append([
            label,
            put(f"t9.{eng}.{m['n']}.enc", m["encode_ms_per_q"],
                f'{m["encode_ms_per_q"]:.1f}', "t9_profile", label,
                "encode ms/q", [rid]),
            put(f"t9.{eng}.{m['n']}.search", m["search_ms_per_q"],
                f'{m["search_ms_per_q"]:.1f}', "t9_profile", label,
                "search ms/q", [rid]),
            search_s,
            put(f"t9.{eng}.{m['n']}.qps", m["qps"], sci(m["qps"]),
                "t9_profile", label, "q/s", [rid]),
        ])
    smoke = R["t8.smoke"]["metrics"]
    put("t9.smoke.n", smoke["n"], ints(smoke["n"]), "t9b_smoke",
        "SCOPe 1K", "Queries", ["t8.smoke"])
    put("t9.smoke.superfamily", smoke["top1_superfamily"],
        str(smoke["top1_superfamily"]), "t9b_smoke", "SCOPe 1K",
        "top-1 superfamily", ["t8.smoke"])
    put("t9.smoke.family", smoke["top1_family"], str(smoke["top1_family"]),
        "t9b_smoke", "SCOPe 1K", "top-1 family", ["t8.smoke"])
    md_table(TABLE_DIR / "t9b_smoke.md",
             "Table 9b. Offline package smoke",
             ["Index", "Queries", "top-1 superfamily", "top-1 family"],
             [["SCOPe 1K", ints(smoke["n"]), str(smoke["top1_superfamily"]),
               str(smoke["top1_family"])]],
             "Fresh-install smoke from the M8 release check.")
    md_table(TABLE_DIR / "t9_profile.md",
             "Table 9. Package profile (appendix)",
             ["Index / backend", "encode ms/q", "search ms/q", "search s/q",
              "q/s"], rows,
             "M4 Max. Single-query encode+search; model/context stays loaded. "
             "The largest row is numpy (no Metal backend at that size in M8).")


def _svg(w: int, h: int, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" '
            f'height="{h}" viewBox="0 0 {w} {h}">' +
            "".join(body) + "</svg>\n")


def _line(x1, y1, x2, y2, color="#333", width=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
            f'y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>')


def _text(x, y, s, size=10, anchor="start", color="#222"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica,Arial,'
            f'sans-serif" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}">{s}</text>')


def _circle(x, y, color="#111"):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'


def _map(v, lo, hi, p0, p1):
    import math
    return p0 + (math.log10(v) - math.log10(lo)) / (
        math.log10(hi) - math.log10(lo)) * (p1 - p0)


def figures() -> None:
    import math
    n = [10000, 100000, 1000000]
    ex = [metric(f"t7.measure.{x}", "exhaustive_s_per_q") for x in n]
    co = [metric(f"t7.measure.{x}", "coarse_s_per_q") for x in n]
    gb = [metric(f"t7.measure.{x}", "esmci_bytes") / 1e9 for x in n]

    # Figure 1: two log-log panels, hand-drawn SVG for byte determinism.
    body = [_text(20, 18, "exact search", 11),
            _text(370, 18, "index size", 11)]
    for left, title in ((50, "seconds / query"), (400, "GB")):
        top, bot, right = 30, 220, left + 250
        body.append(_line(left, bot, right, bot))
        body.append(_line(left, bottom := bot, x2=left, y2=top))
        for gx in (10000, 100000, 1000000):
            x = _map(gx, 10000, 1000000, left, right)
            body.append(_line(x, bot, x, bot + 4))
            body.append(_text(x, bot + 16, f"10^{int(math.log10(gx))}",
                              9, "middle"))
        body.append(_text((left + right) / 2, bot + 32,
                          "database proteins", 9, "middle"))
        body.append(_text(left - 8, top + 4, title, 9, "end"))
    # left panel y ticks
    for yv in (0.1, 1, 10):
        y = _map(yv, 0.08, 20, 220, 30)
        body.append(_line(50, y, 54, y))
        body.append(_text(46, y + 3, f"{yv:g}", 9, "end"))
    # right panel y ticks
    for yv in (1, 10, 100):
        y = _map(yv, 0.7, 100, 220, 30)
        body.append(_line(400, y, 404, y))
        body.append(_text(396, y + 3, f"{yv:g}", 9, "end"))
    pts_ex = [(_map(x, 10000, 1000000, 50, 300),
               _map(y, 0.08, 20, 220, 30)) for x, y in zip(n, ex)]
    pts_co = [(_map(x, 10000, 1000000, 50, 300),
               _map(y, 0.08, 20, 220, 30)) for x, y in zip(n, co)]
    pts_gb = [(_map(x, 10000, 1000000, 400, 650),
               _map(y, 0.7, 100, 220, 30)) for x, y in zip(n, gb)]
    for pts, dash in ((pts_ex, ""), (pts_co, "5,4")):
        body.append(f'<polyline points="' +
                    " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
                    f'" fill="none" stroke="#111" stroke-width="1.5"'
                    + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        for x, y in pts:
            body.append(_circle(x, y))
    body.append(f'<polyline points="' +
                " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_gb) +
                '" fill="none" stroke="#111" stroke-width="1.5"/>')
    for x, y in pts_gb:
        body.append(_circle(x, y))
    body.append(_line(60, 245, 88, 245, "#111", 1.5))
    body.append(_text(94, 248, "exhaustive", 9))
    body.append(_line(170, 245, 198, 245, "#111", 1.5, "5,4"))
    body.append(_text(204, 248, "coarse + rerank", 9))
    (FIG_DIR / "f2_scaling.svg").write_text(_svg(680, 260, body))

    # Figure 2: compression ratio vs cR@10.
    m = R["t7.compress"]["metrics"]
    labels = ["fp16", "int8", "r-int8", "PQ m8", "PQ m16", "PQ m32"]
    ratios = [float(m["fp16_ratio_vs_f16"]), float(m["int8_ratio_vs_f16"]),
              float(m["residual_int8_ratio_vs_f16"]),
              float(m["residual_pq_m8_ratio_vs_f16"]),
              float(m["residual_pq_m16_ratio_vs_f16"]),
              float(m["residual_pq_m32_ratio_vs_f16"])]
    cr = [float(m["fp16_cR@10"]), float(m["int8_cR@10"]),
          float(m["residual_int8_cR@10"]), float(m["residual_pq_m8_cR@10"]),
          float(m["residual_pq_m16_cR@10"]), float(m["residual_pq_m32_cR@10"])]
    body = [_line(40, 220, 340, 220), _line(40, 220, 40, 20)]
    for xv in (1, 3, 10, 30):
        x = _map(xv, 1, 30, 40, 340)
        body.append(_line(x, 220, x, 224))
        body.append(_text(x, 236, f"{xv:g}", 9, "middle"))
    body.append(_text(190, 250, "compression ratio vs f16", 9, "middle"))
    for yv in (0.85, 0.87, 0.89):
        y = 220 - (yv - 0.84) / 0.06 * 200
        body.append(_line(36, y, 40, y))
        body.append(_text(32, y + 3, f"{yv:.2f}", 9, "end"))
    body.append(_text(14, 120, "cR@10", 9, "middle"))
    pts = [(_map(x, 1, 30, 40, 340), 220 - (y - 0.84) / 0.06 * 200)
           for x, y in zip(ratios, cr)]
    body.append(f'<polyline points="' +
                " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
                '" fill="none" stroke="#111" stroke-width="1.5"/>')
    for (x, y), t in zip(pts, labels):
        body.append(_circle(x, y))
        body.append(_text(x, y - 8, t, 8, "middle"))
    (FIG_DIR / "f3_compression.svg").write_text(_svg(360, 270, body))

    # Figure 3: grouped bars, raw vs normalized.
    body = [_line(50, 220, 400, 220), _line(50, 220, 50, 20)]
    norms = [("raw", "raw"), ("divA", "len_A"), ("divB", "len_B"),
             ("divGeo", "sqrt")]
    for i, (_, lab) in enumerate(norms):
        x = 80 + i * 80
        body.append(_text(x, 236, lab, 9, "middle"))
    for yv in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = 220 - yv * 200
        body.append(_line(46, y, 50, y))
        body.append(_text(42, y + 3, f"{yv:.2f}", 9, "end"))
    for j, (method, color) in enumerate((("trained", "#222"),
                                         ("maxsim960", "#999"))):
        for i, (norm, _) in enumerate(norms):
            v = metric(f"fl.{method}_{norm}", "cR@10")
            x = 65 + i * 80 + j * 22
            y = 220 - v * 200
            body.append(f'<rect x="{x}" y="{y:.1f}" width="20" '
                        f'height="{220 - y:.1f}" fill="{color}"/>')
    body.append(f'<rect x="260" y="12" width="10" height="10" fill="#222"/>')
    body.append(_text(274, 21, "fine-tuned k=3", 8))
    body.append(f'<rect x="260" y="26" width="10" height="10" fill="#999"/>')
    body.append(_text(274, 35, "frozen MaxSim 960D", 8))
    (FIG_DIR / "f1_length.svg").write_text(_svg(420, 260, body))


def main() -> int:
    load()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t1()
    t2()
    t2b()
    t3()
    t4()
    t5a()
    t5b()
    t5c()
    t6a()
    t6b()
    t7()
    t8()
    t9()
    figures()
    NUMBERS.write_text(json.dumps(NUM, indent=2, sort_keys=True) + "\n")
    print(f"tables: {len(list(TABLE_DIR.glob('*.md')))}, "
          f"figures: {len(list(FIG_DIR.glob('*.svg')))}, "
          f"numbers: {len(NUM)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
