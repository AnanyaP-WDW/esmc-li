#!/usr/bin/env python3
"""Generate the README charts from the committed result CSVs.

All charts are saved as opaque white RGB PNGs. Transparent PNGs disappear in
GitHub dark mode when their text/lines are dark.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def read(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as f:
        return list(csv.DictReader(f))


def row(rows: list[dict], key: str, value: str) -> dict:
    found = [r for r in rows if r.get(key) == value]
    if len(found) != 1:
        raise SystemExit(f"expected one {key}={value}, got {len(found)}")
    return found[0]


def metric(r: dict, key: str) -> float:
    return float(r[key])


def mean_metric(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(r[key]) for r in rows)


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
        metadata={"Software": "ESMC-LI README"},
    )
    plt.close(fig)
    # Flatten antialiased alpha onto white so dark-mode renderers never get
    # transparent black text.
    im = Image.open(path)
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.getchannel("A") if "A" in im.getbands() else None)
        bg.save(path)


def retrieval() -> None:
    scorers = read("m1_scorers.csv")
    baselines = read("m0_baselines.csv")
    protocol = read("m0_protocol_repro.csv")
    m3 = read("m3_full_metrics.csv")

    series = {
        "Pooled ESM-C": row(scorers, "method", "pooled_esmc"),
        "Frozen MaxSim 960D": row(scorers, "method", "maxsim960"),
        "MMseqs2": row(baselines, "method", "mmseqs2_s7.5"),
        "ProtoCol frozen": row(protocol, "method", "protocol_frozen"),
        "ProtoCol trained": row(protocol, "method", "protocol_trained"),
    }
    esmc_li = {
        key: mean_metric(m3, key) for key in ("cR@1", "cR@5", "cR@10", "cR@100")
    }
    colors = {
        "Pooled ESM-C": "#7f7f7f",
        "Frozen MaxSim 960D": "#8faadc",
        "MMseqs2": "#c55a11",
        "ProtoCol frozen": "#a6a6a6",
        "ProtoCol trained": "#2e75b6",
        "ESMC-LI fine-tuned": "#1f4e79",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [1, 5, 10, 100]
    for label, r in series.items():
        ax1.plot(xs, [metric(r, f"cR@{k}") for k in xs], marker="o",
                 linewidth=1.8, label=label, color=colors[label])
    ax1.plot(xs, [esmc_li[f"cR@{k}"] for k in xs], marker="o",
             linewidth=2.4, label="ESMC-LI fine-tuned",
             color=colors["ESMC-LI fine-tuned"])
    ax1.set_xscale("log")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["1", "5", "10", "100"])
    ax1.set_xlabel("recall cutoff k")
    ax1.set_ylabel("capped recall")
    ax1.set_ylim(0.15, 1.0)
    ax1.grid(True, alpha=0.25)
    ax1.set_title("SCOPe ASTRAL40 (n=2314)", fontsize=11)
    ax1.legend(fontsize=7, frameon=False, loc="lower right")

    fl = read("m4_fulllength.csv")
    raw = [r for r in fl if r["normalization"] == "raw"]
    methods = [
        ("pooled_esmc", "Pooled"),
        ("maxsim960", "MaxSim 960D"),
        ("mmseqs2_s7.5", "MMseqs2"),
        ("trained_m3", "ESMC-LI"),
    ]
    labels = [label for _, label in methods]
    c10 = [metric(row(raw, "method", method), "cR@10")
           for method, _ in methods]
    sens = [metric(row(raw, "method", method), "sens_superfamily")
            for method, _ in methods]
    x = list(range(len(labels)))
    w = 0.36
    ax2.bar([i - w / 2 for i in x], c10, w, label="cR@10", color="#2e75b6")
    ax2.bar([i + w / 2 for i in x], sens, w,
            label="sens superfamily", color="#8faadc")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("score")
    ax2.set_title("Full-length Swiss-Prot 100K (500 queries)", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "retrieval.png")


def protocol() -> None:
    repro = read("m0_protocol_repro.csv")
    gap = read("m1_protocol_gap.csv")
    m3 = read("m3_full_metrics.csv")
    bars = [
        ("ESMC-LI fine-tuned", mean_metric(m3, "cR@10"), "#1f4e79"),
        ("ProtoCol trained", metric(row(repro, "method", "protocol_trained"),
                                    "cR@10"), "#2e75b6"),
        ("ESM-2 35M, full", metric(row(gap, "variant", "D_esm2_strip_full"),
                                    "cR@10"), "#a6a6a6"),
        ("ESM-2 35M, ProtoCol", metric(row(gap, "variant",
                                          "C_esm2_keep_trunc_randproj"),
                                       "cR@10"), "#c9c9c9"),
        ("ProtoCol frozen", metric(row(repro, "method", "protocol_frozen"),
                                   "cR@10"), "#8faadc"),
        ("ESM-C 300M, ProtoCol", metric(row(gap, "variant",
                                           "B_esmc_keep_trunc"), "cR@10"),
         "#b4c6e7"),
        ("ESM-C 300M, full", metric(row(gap, "variant", "A_esmc_strip_full"),
                                    "cR@10"), "#d9e2f3"),
    ]
    labels = [b[0] for b in bars][::-1]
    vals = [b[1] for b in bars][::-1]
    colors = [b[2] for b in bars][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.barh(labels, vals, color=colors, edgecolor="#333333", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.55, 0.95)
    ax.set_xlabel("SCOPe cR@10")
    ax.set_title("ProtoCol reproduction and backbone control", fontsize=11)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "protocol.png")


def scaling() -> None:
    rows = read("m7_scaling.csv")
    n = [int(r["size"]) for r in rows]
    ex = [float(r["exhaustive_s_per_q"]) for r in rows]
    co = [float(r["coarse_s_per_q"]) for r in rows]
    gb = [float(r["esmci_bytes"]) / 1e9 for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.loglog(n, ex, "o-", color="#1f4e79", linewidth=1.8,
               label="exhaustive")
    ax1.loglog(n, co, "s--", color="#c55a11", linewidth=1.8,
               label="coarse + rerank")
    ax1.set_xlabel("database proteins")
    ax1.set_ylabel("seconds / query")
    ax1.grid(True, which="both", alpha=0.2)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("Search cost", fontsize=11)
    ax2.loglog(n, gb, "o-", color="#1f4e79", linewidth=1.8)
    ax2.set_xlabel("database proteins")
    ax2.set_ylabel("index size (GB)")
    ax2.grid(True, which="both", alpha=0.2)
    ax2.set_title("Index size", fontsize=11)
    fig.tight_layout()
    save(fig, "scaling.png")


def flatten_pipeline() -> None:
    path = OUT / "pipeline.png"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    im = Image.open(path)
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.getchannel("A") if "A" in im.getbands() else None)
        bg.save(path)


def main() -> int:
    retrieval()
    protocol()
    scaling()
    flatten_pipeline()
    for p in ("retrieval.png", "protocol.png", "scaling.png", "pipeline.png"):
        im = Image.open(OUT / p)
        if im.mode != "RGB" or im.getpixel((0, 0)) != (255, 255, 255):
            raise SystemExit(f"{p}: expected opaque white RGB image")
        print(f"{p}: {im.size[0]}x{im.size[1]} RGB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
