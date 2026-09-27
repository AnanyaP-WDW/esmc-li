#!/usr/bin/env python3
"""T1.5: look at the data. 20 rank-1 false positives for the best M1 scorer
(maxsim960) + 3 residue similarity maps -> docs/m1_failures.md + docs/plots/.
Every failure gets a one-line hypothesis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from esmc_retrieval.eval import load_split
from esmc_retrieval.runlog import log_run
from esmc_retrieval.scoring import load_embeddings, l2_normalize, rankings_from_scores

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "splits" / "scope_test.tsv"
OUT = ROOT / "docs" / "m1_failures.md"
PLOTS = ROOT / "docs" / "plots"
SCORER = "maxsim960"


def classify(fold: str) -> str:
    return fold.split(".")[0] if fold else "?"


def hypothesis(q: dict, c: dict, S: np.ndarray, i: int, j: int) -> str:
    ql, cl = len(q["sequence"]), len(c["sequence"])
    qlab, clab = classify(q["fold"]), classify(c["fold"])
    ratio = max(ql, cl) / min(ql, cl)
    if qlab == clab and q["fold"] != c["fold"]:
        return (f"same SCOP class {qlab} but different folds — surface/shape "
                f"convergence at 960D MaxSim, no alignment to distinguish them")
    if ratio >= 2.0:
        return (f"length mismatch ({ql} vs {cl} aa, {ratio:.1f}x) — "
                f"short protein's residues all find strong partners in the long one")
    if q["fold"] == c["fold"]:
        return (f"same SCOP fold {q['fold']} but different superfamily — "
                f"remote homology borderline; MaxSim sees the shared structural core")
    if qlab != clab:
        return (f"cross-class pair (SCOP {qlab} vs {clab}) — MaxSim rewards any "
                f"structurally similar local fragments regardless of class")
    return (f"same class {qlab}, unrelated folds — anisotropic residue space: "
            f"generic residues (helix/loop) dominate the max-sim sum")


def main() -> int:
    rows = load_split(SPLIT)
    ids = [r["id"] for r in rows]
    by_id = {r["id"]: r for r in rows}
    idx = {r["id"]: i for i, r in enumerate(rows)}
    S = np.load(ROOT / "cache" / "m1" / f"{SCORER}.npy")

    ranks = rankings_from_scores(S, ids)
    fps, tps = [], []
    for qi, q in enumerate(rows):
        top = ranks[q["id"]][0]
        j = idx[top]
        (fps if by_id[top]["superfamily"] != q["superfamily"] else tps).append(
            (qi, j, float(S[qi, j])))
    print(f"rank-1 FP: {len(fps)}/{len(rows)} ({len(fps)/len(rows):.1%}), TP: {len(tps)}")

    fps.sort(key=lambda x: -x[2])  # highest-scoring FPs first
    picks = fps[:20]

    PLOTS.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = [
        "# M1 failure analysis (T1.5)",
        "",
        f"Scorer: `{SCORER}` (L2-normalized ESM-C residue vectors, raw MaxSim sum), "
        f"split `scope_test` (n={len(rows)}).",
        f"Rank-1 false positives (top-1 not same superfamily): "
        f"**{len(fps)}/{len(rows)} = {len(fps)/len(rows):.1%}**.",
        "",
        "## 20 highest-scoring rank-1 false positives",
        "",
        "| # | query | qlen | q fold | top-1 | clen | t fold | score | hypothesis |",
        "|---|-------|------|--------|-------|------|--------|-------|------------|",
    ]
    for n, (qi, j, sc) in enumerate(picks, 1):
        q, c = rows[qi], rows[j]
        lines.append(
            f"| {n} | {q['id']} | {len(q['sequence'])} | {q['fold']} | "
            f"{c['id']} | {len(c['sequence'])} | {c['fold']} | {sc:.1f} | "
            f"{hypothesis(q, c, S, qi, j)} |")

    # length stats over ALL rank-1 FPs vs TPs
    fp_lens = [(len(rows[qi]["sequence"]), len(rows[j]["sequence"])) for qi, j, _ in fps]
    tp_lens = [(len(rows[qi]["sequence"]), len(rows[j]["sequence"])) for qi, j, _ in tps]
    lines += [
        "",
        "## Aggregate: are FPs length-driven?",
        "",
        f"- rank-1 TP pairs: mean |len ratio| "
        f"{_mean_ratio(tp_lens):.2f} (n={len(tp_lens)})",
        f"- rank-1 FP pairs: mean |len ratio| "
        f"{_mean_ratio(fp_lens):.2f} (n={len(fp_lens)})",
        f"- fraction of FPs crossing SCOP class boundary: "
        f"{_cross_class_frac(rows, fps):.1%}",
        "",
        "## Residue similarity maps",
        "",
        "Heatmaps of the [Lq x Lc] residue cosine matrix actually fed to MaxSim "
        f"(`{SCORER}`); the score is the sum over query rows of each row maximum.",
        "",
    ]

    # 3 maps: one correct homolog, two false positives (different class / length)
    map_picks: list[tuple[str, int, int]] = []
    if tps:
        best_tp = max(tps, key=lambda x: x[2])
        map_picks.append(("tp_homolog", best_tp[0], best_tp[1]))
    cross = [(qi, j) for qi, j, _ in fps
             if classify(rows[qi]["fold"]) != classify(rows[j]["fold"])]
    big_len = sorted(fps, key=lambda x: -max(len(rows[x[0]]["sequence"]),
                                             len(rows[x[1]]["sequence"])) /
                      min(len(rows[x[0]]["sequence"]), len(rows[x[1]]["sequence"])))
    if cross:
        map_picks.append(("fp_cross_class", cross[0][0], cross[0][1]))
    if big_len:
        map_picks.append(("fp_length_mismatch", big_len[0][0], big_len[0][1]))

    embs = {rid: l2_normalize(e) for rid, e in
            zip(ids, load_embeddings(ids)) if rid in
            {rows[a]["id"] for _, a, _ in map_picks} | {rows[b]["id"] for _, _, b in map_picks}}
    for tag, qi, j in map_picks:
        q, c = rows[qi], rows[j]
        A, B = embs[q["id"]], embs[c["id"]]
        M = A @ B.T
        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel(f"{c['id']} residues (len {len(c['sequence'])})")
        ax.set_ylabel(f"{q['id']} residues (len {len(q['sequence'])})")
        ax.set_title(f"{tag}: {q['id']} (sf {q['superfamily']}) vs "
                     f"{c['id']} (sf {c['superfamily']})\n"
                     f"MaxSim={S[qi, j]:.1f}  row-max mean={M.max(axis=1).mean():.3f}")
        fig.colorbar(im, ax=ax, label="cosine")
        fig.tight_layout()
        fname = PLOTS / f"m1_map_{tag}_{q['id']}_{c['id']}.png"
        fig.savefig(fname, dpi=110)
        plt.close(fig)
        lines.append(f"- `{fname.relative_to(ROOT)}` — {tag}: row-max mean "
                     f"{M.max(axis=1).mean():.3f}, off-diagonal background "
                     f"{np.median(M):.3f}")
        print("wrote", fname)

    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)
    log_run("T1.5", {"split": "scope_test", "scorer": SCORER,
                     "n_maps": len(map_picks)},
            {"rank1_fp_frac": round(len(fps) / len(rows), 4),
             "n_rank1_fp": len(fps), "n_rank1_tp": len(tps)},
            seed=0)
    return 0


def _mean_ratio(pairs) -> float:
    if not pairs:
        return 0.0
    return float(np.mean([max(a, b) / min(a, b) for a, b in pairs]))


def _cross_class_frac(rows, fps) -> float:
    if not fps:
        return 0.0
    n = sum(1 for qi, j, _ in fps
            if classify(rows[qi]["fold"]) != classify(rows[j]["fold"]))
    return n / len(fps)


if __name__ == "__main__":
    sys.exit(main())
