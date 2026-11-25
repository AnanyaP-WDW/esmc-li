#!/usr/bin/env python3
"""32-triple overfit for the EC local-patch objective.

Builds 32 cluster-disjoint (parent, decoy, positive) triples:
  parent   a train_5k protein with experimental active sites and a catalytic decoy
  decoy    parent with active-site residues mutated to Ala/Gly
  positive same EC4 as parent, in a different MMseqs cluster

The local loss is a 2-way contrastive on active-site patch MaxSim:
  positive = parent patch vs positive patch
  negative = parent patch vs decoy patch
The random-window control runs the same loss on random windows that exclude
active sites.  If the active-window loss overfits but the random-window loss
does not, the signal is active-site specific.

Gate: global+local_active must reach 32/32 paired parent-vs-decoy accuracy
and local loss <0.1 before any full training run.  The global AUC is also
reported, but it is not the gate: raw MaxSim scores are not comparable across
different queries, so the paired accuracy is the meaningful 32-triple metric.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from esmc_li.train import Projection, maxsim, symmetric_nce  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"

VARIANTS = {
    "global": {"global": 1.0},
    "global+global_decoy": {"global": 1.0, "global_decoy": 1.0},
    "global+local_active": {"global": 1.0, "local_active": 1.0},
    "global+local_random": {"global": 1.0, "local_random": 1.0},
    "global+local_active_full": {"global": 1.0, "local_active_full": 1.0},
    "global+local_random_full": {"global": 1.0, "local_random_full": 1.0},
    "global+local_active_doc": {"global": 1.0, "local_active_doc": 1.0},
    "global+local_random_doc": {"global": 1.0, "local_random_doc": 1.0},
    "local_active": {"local_active": 1.0},
    "local_random": {"local_random": 1.0},
}

EVAL_KEYS = {
    "full": ("parent_emb", "decoy_emb", "positive_emb"),
    "active": ("parent_patch_active", "decoy_patch_active",
               "positive_patch_active"),
    "random": ("parent_patch_random", "decoy_patch_random",
               "positive_patch_random"),
}


def load_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline=""), delimiter="\t"))


def load_clusters(path: Path) -> dict[str, str]:
    out = {}
    for line in path.open():
        rep, mem = line.split()
        out[mem] = rep
    return out


def active_positions(row: dict) -> list[int]:
    return [int(x) for x in (row.get("active_sites") or "").split(";") if x]


def build_triples(train: list[dict], decoys: list[dict],
                  clusters: dict[str, str], n: int,
                  seed: int) -> list[dict]:
    by_ec4: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        if r.get("active_sites"):
            by_ec4[r["ec4"].split(";")[0]].append(r)
    decoy_by_parent = {r["parent"]: r for r in decoys}

    candidates = []
    for ec4, rows in by_ec4.items():
        parents = [r for r in rows if r["id"] in decoy_by_parent]
        if not parents:
            continue
        for p in parents:
            for q in rows:
                if q["id"] == p["id"]:
                    continue
                if clusters.get(p["id"]) != clusters.get(q["id"]):
                    candidates.append((p, q, decoy_by_parent[p["id"]]))
                    break
            else:
                continue
            break

    rng = random.Random(seed)
    rng.shuffle(candidates)
    triples = []
    for p, q, d in candidates[:n]:
        triples.append({
            "ec4": p["ec4"].split(";")[0],
            "parent": p,
            "positive": q,
            "decoy": d,
        })
    return triples


def patch_indices(seq_len: int, positions: list[int], window: int) -> list[int]:
    idx: set[int] = set()
    for p in positions:
        i = p - 1
        for j in range(max(0, i - window), min(seq_len, i + window + 1)):
            idx.add(j)
    return sorted(idx)


def random_patch_indices(seq_len: int, positions: list[int], n_windows: int,
                         window: int, rng: random.Random) -> list[int]:
    forbidden: set[int] = set()
    for p in positions:
        i = p - 1
        for j in range(max(0, i - window), min(seq_len, i + window + 1)):
            forbidden.add(j)
    centers = [c for c in range(seq_len)
               if all(j not in forbidden
                      for j in range(max(0, c - window),
                                     min(seq_len, c + window + 1)))]
    if len(centers) < n_windows:
        centers = [c for c in range(seq_len) if c not in forbidden]
    chosen = rng.sample(centers, min(n_windows, len(centers)))
    idx: set[int] = set()
    for c in chosen:
        for j in range(max(0, c - window), min(seq_len, c + window + 1)):
            idx.add(j)
    return sorted(idx)


def load_emb(cache: Path, ids: list[str]) -> dict[str, np.ndarray]:
    missing = [i for i in ids if not (cache / f"{i}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} missing in {cache}: {missing[:5]}")
    return {i: np.load(cache / f"{i}.npy") for i in ids}


def collate(items: list[dict], key: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_n = max(x[key].shape[0] for x in items)
    out = np.zeros((len(items), max_n, 960), dtype=np.float32)
    mask = np.zeros((len(items), max_n), dtype=bool)
    for i, x in enumerate(items):
        n = x[key].shape[0]
        out[i, :n] = x[key]
        mask[i, :n] = True
    return torch.from_numpy(out).to(DEV), torch.from_numpy(mask).to(DEV)


def chunks(items: list[dict], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def global_loss(head, P, Pm, Q, Qm, tau):
    return symmetric_nce(maxsim(head(P), Pm, head(Q), Qm), tau)


def global_decoy_loss(head, P, Pm, D, Dm, Q, Qm, margin):
    s_pq = maxsim(head(P), Pm, head(Q), Qm).diagonal()
    s_dq = maxsim(head(D), Dm, head(Q), Qm).diagonal()
    return F.relu(margin - (s_pq - s_dq)).mean()


def local_loss(head, P, Pm, D, Dm, Q, Qm, tau):
    """2-way contrastive over MaxSim for an arbitrary query/document pair."""
    s_pq = maxsim(head(P), Pm, head(Q), Qm).diagonal()
    s_pd = maxsim(head(P), Pm, head(D), Dm).diagonal()
    n = Pm.sum(dim=1).clamp(min=1)
    s_pq = s_pq / n
    s_pd = s_pd / n
    logits = torch.stack([s_pq / tau, s_pd / tau], dim=1)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def auc_score(pos: list[float], neg: list[float]) -> float:
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    total = 0.0
    for p in pos:
        total += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(total / (len(pos) * len(neg)))


def evaluate(head, triples: list[dict], batch_size: int) -> dict:
    head.eval()
    out = {}
    with torch.no_grad():
        for name, (kp, kd, kq) in EVAL_KEYS.items():
            s_pos, s_neg = [], []
            for batch in chunks(triples, batch_size):
                P, Pm = collate(batch, kp)
                D, Dm = collate(batch, kd)
                Q, Qm = collate(batch, kq)
                s_pos.extend(maxsim(head(P), Pm, head(Q), Qm)
                             .diagonal().cpu().numpy().tolist())
                s_neg.extend(maxsim(head(D), Dm, head(Q), Qm)
                             .diagonal().cpu().numpy().tolist())
            s_pos_a = np.asarray(s_pos)
            s_neg_a = np.asarray(s_neg)
            out[name] = {
                "acc": float(np.mean(s_pos_a > s_neg_a)),
                "auc": auc_score(s_pos, s_neg),
                "margin": float(np.mean(s_pos_a - s_neg_a)),
                "s_pos": s_pos,
                "s_neg": s_neg,
            }
    head.train()
    return out


def train_variant(name: str, weights: dict[str, float], triples: list[dict],
                  args) -> dict:
    torch.manual_seed(args.seed)
    head = Projection(960, 128).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(args.seed)
    history = []
    components = []
    t0 = time.time()

    for step in range(args.steps):
        rng.shuffle(triples)
        total = 0.0
        step_components: dict[str, float] = defaultdict(float)
        count = 0
        for batch in chunks(triples, args.batch):
            loss = torch.zeros((), device=DEV)
            if "global" in weights:
                P, Pm = collate(batch, "parent_emb")
                Q, Qm = collate(batch, "positive_emb")
                l = global_loss(head, P, Pm, Q, Qm, args.tau)
                loss = loss + weights["global"] * l
                step_components["global"] += float(l.detach())
            if "global_decoy" in weights:
                P, Pm = collate(batch, "parent_emb")
                D, Dm = collate(batch, "decoy_emb")
                Q, Qm = collate(batch, "positive_emb")
                l = global_decoy_loss(head, P, Pm, D, Dm, Q, Qm,
                                      args.margin)
                loss = loss + weights["global_decoy"] * l
                step_components["global_decoy"] += float(l.detach())
            if "local_active" in weights:
                P, Pm = collate(batch, "parent_patch_active")
                D, Dm = collate(batch, "decoy_patch_active")
                Q, Qm = collate(batch, "positive_patch_active")
                l = local_loss(head, P, Pm, D, Dm, Q, Qm, args.tau_local)
                loss = loss + weights["local_active"] * l
                step_components["local_active"] += float(l.detach())
            if "local_random" in weights:
                P, Pm = collate(batch, "parent_patch_random")
                D, Dm = collate(batch, "decoy_patch_random")
                Q, Qm = collate(batch, "positive_patch_random")
                l = local_loss(head, P, Pm, D, Dm, Q, Qm, args.tau_local)
                loss = loss + weights["local_random"] * l
                step_components["local_random"] += float(l.detach())
            if "local_active_full" in weights:
                P, Pm = collate(batch, "parent_patch_active")
                D, Dm = collate(batch, "decoy_emb")
                Q, Qm = collate(batch, "positive_emb")
                l = local_loss(head, P, Pm, D, Dm, Q, Qm, args.tau_local)
                loss = loss + weights["local_active_full"] * l
                step_components["local_active_full"] += float(l.detach())
            if "local_random_full" in weights:
                P, Pm = collate(batch, "parent_patch_random")
                D, Dm = collate(batch, "decoy_emb")
                Q, Qm = collate(batch, "positive_emb")
                l = local_loss(head, P, Pm, D, Dm, Q, Qm, args.tau_local)
                loss = loss + weights["local_random_full"] * l
                step_components["local_random_full"] += float(l.detach())
            if "local_active_doc" in weights:
                P, Pm = collate(batch, "parent_emb")
                D, Dm = collate(batch, "decoy_patch_active")
                Q, Qm = collate(batch, "positive_patch_active")
                l = local_loss(head, P, Pm, D, Dm, Q, Qm, args.tau_local)
                loss = loss + weights["local_active_doc"] * l
                step_components["local_active_doc"] += float(l.detach())
            if "local_random_doc" in weights:
                P, Pm = collate(batch, "parent_emb")
                D, Dm = collate(batch, "decoy_patch_random")
                Q, Qm = collate(batch, "positive_patch_random")
                l = local_loss(head, P, Pm, D, Dm, Q, Qm, args.tau_local)
                loss = loss + weights["local_random_doc"] * l
                step_components["local_random_doc"] += float(l.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            count += 1
        history.append(total / max(1, count))
        components.append({k: v / max(1, count)
                           for k, v in step_components.items()})
        if step % 50 == 0 or step == args.steps - 1:
            ev = evaluate(head, triples, args.batch)
            print(f"{name} step {step} loss {history[-1]:.4f} "
                  f"full_auc {ev['full']['auc']:.3f} "
                  f"full_acc {ev['full']['acc']:.3f}", flush=True)
        if DEV == "mps" and step % 50 == 0:
            torch.mps.empty_cache()

    ev = evaluate(head, triples, args.batch)
    return {
        "weights": weights,
        "final_loss": history[-1] if history else None,
        "components": components[-1] if components else {},
        "history": history,
        "full": ev["full"],
        "active": ev["active"],
        "random": ev["random"],
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path,
                    default=ROOT / "data/ec/splits/train_5k.tsv")
    ap.add_argument("--decoys", type=Path,
                    default=ROOT / "data/ec/decoys/train_decoys.tsv")
    ap.add_argument("--clusters", type=Path,
                    default=ROOT / "data/ec/clusters/train5k_cluster_cluster.tsv")
    ap.add_argument("--train-cache", type=Path,
                    default=ROOT / "cache/ec/esmc960/train")
    ap.add_argument("--decoy-cache", type=Path,
                    default=ROOT / "cache/ec/esmc960/decoys_train")
    ap.add_argument("--out", type=Path, default=ROOT / "results/ec/overfit32")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--tau-local", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--lambda-local", type=float, default=1.0)
    ap.add_argument("--lambda-global-decoy", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")

    train = load_rows(args.train)
    decoys = load_rows(args.decoys)
    clusters = load_clusters(args.clusters)
    triples = build_triples(train, decoys, clusters, args.n, args.seed)
    if len(triples) < args.n:
        raise SystemExit(f"only {len(triples)} triples available, wanted {args.n}")

    train_emb = load_emb(args.train_cache,
                         [t["parent"]["id"] for t in triples]
                         + [t["positive"]["id"] for t in triples])
    decoy_emb = load_emb(args.decoy_cache,
                         [t["decoy"]["decoy_id"] for t in triples])

    rng = random.Random(args.seed)
    for t in triples:
        p = t["parent"]
        q = t["positive"]
        d = t["decoy"]
        p_pos = active_positions(p)
        q_pos = active_positions(q)
        p_len = len(p["sequence"])
        q_len = len(q["sequence"])
        t["parent_emb"] = train_emb[p["id"]]
        t["positive_emb"] = train_emb[q["id"]]
        t["decoy_emb"] = decoy_emb[d["decoy_id"]]
        t["parent_patch_active"] = t["parent_emb"][
            patch_indices(p_len, p_pos, args.window)]
        t["decoy_patch_active"] = t["decoy_emb"][
            patch_indices(p_len, p_pos, args.window)]
        t["positive_patch_active"] = t["positive_emb"][
            patch_indices(q_len, q_pos, args.window)]
        n_windows = len(set(p_pos))
        p_random = random_patch_indices(p_len, p_pos, n_windows, args.window, rng)
        t["parent_patch_random"] = t["parent_emb"][p_random]
        t["decoy_patch_random"] = t["decoy_emb"][p_random]
        t["positive_patch_random"] = t["positive_emb"][
            random_patch_indices(q_len, q_pos, n_windows, args.window, rng)]
        t["n_active"] = n_windows
        t["n_patch_active"] = len(t["parent_patch_active"])
        t["n_patch_random"] = len(t["parent_patch_random"])

    args.out.mkdir(parents=True, exist_ok=True)
    base = list(triples)
    results = {}
    for name in variants:
        weights = dict(VARIANTS[name])
        for key in ("local_active", "local_random",
                    "local_active_full", "local_random_full",
                    "local_active_doc", "local_random_doc"):
            if key in weights:
                weights[key] = args.lambda_local
        if "global_decoy" in weights:
            weights["global_decoy"] = args.lambda_global_decoy
        print(f"training {name} weights={weights}", flush=True)
        results[name] = train_variant(name, weights, list(base), args)

    gate = results.get("global+local_active")
    gate_summary = None
    if gate:
        gate_summary = {
            "variant": "global+local_active",
            "full_auc": gate["full"]["auc"],
            "full_acc": gate["full"]["acc"],
            "local_loss": gate["components"].get("local_active"),
            "passed": bool(gate["full"]["acc"] >= 0.99
                           and (gate["components"].get("local_active") or 0) < 0.1),
        }

    meta = {
        "config": {
            "n": args.n,
            "window": args.window,
            "steps": args.steps,
            "batch": args.batch,
            "lr": args.lr,
            "tau": args.tau,
            "tau_local": args.tau_local,
            "margin": args.margin,
            "lambda_local": args.lambda_local,
            "lambda_global_decoy": args.lambda_global_decoy,
            "seed": args.seed,
            "train": str(args.train),
            "decoys": str(args.decoys),
            "clusters": str(args.clusters),
        },
        "triples": [{
            "triple_id": i,
            "ec4": t["ec4"],
            "parent": t["parent"]["id"],
            "decoy": t["decoy"]["decoy_id"],
            "positive": t["positive"]["id"],
            "parent_cluster": clusters.get(t["parent"]["id"]),
            "positive_cluster": clusters.get(t["positive"]["id"]),
            "n_active": t["n_active"],
            "n_patch_active": t["n_patch_active"],
            "n_patch_random": t["n_patch_random"],
        } for i, t in enumerate(triples)],
        "variants": results,
        "gate": gate_summary,
    }
    (args.out / "overfit32_results.json").write_text(
        json.dumps(meta, indent=2) + "\n")

    with (args.out / "overfit32_triples.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["triple_id", "ec4", "parent", "decoy", "positive",
                    "parent_cluster", "positive_cluster", "n_active",
                    "n_patch_active", "n_patch_random"])
        for i, t in enumerate(triples):
            w.writerow([i, t["ec4"], t["parent"]["id"], t["decoy"]["decoy_id"],
                        t["positive"]["id"], clusters.get(t["parent"]["id"]),
                        clusters.get(t["positive"]["id"]), t["n_active"],
                        t["n_patch_active"], t["n_patch_random"]])

    with (args.out / "overfit32_scores.csv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["variant", "triple_id", "ec4", "parent", "decoy",
                    "positive", "s_pos_full", "s_neg_full",
                    "s_pos_active", "s_neg_active",
                    "s_pos_random", "s_neg_random"])
        for name, res in results.items():
            for i, t in enumerate(triples):
                w.writerow([name, i, t["ec4"], t["parent"]["id"],
                            t["decoy"]["decoy_id"], t["positive"]["id"],
                            res["full"]["s_pos"][i], res["full"]["s_neg"][i],
                            res["active"]["s_pos"][i], res["active"]["s_neg"][i],
                            res["random"]["s_pos"][i], res["random"]["s_neg"][i]])

    print(json.dumps({"gate": gate_summary,
                      "variants": {k: {
                          "full_auc": v["full"]["auc"],
                          "full_acc": v["full"]["acc"],
                          "active_auc": v["active"]["auc"],
                          "random_auc": v["random"]["auc"],
                          "final_loss": v["final_loss"],
                          "components": v["components"],
                      } for k, v in results.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
