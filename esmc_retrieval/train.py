"""M2: train a projection head on the frozen M1 cache (backbone never runs).

Symmetric InfoNCE over MaxSim (ProtoCol eq. 3). PyLate was checked first for
T2.1: its Contrastive loss calls `self.model(sentence_feature)` on tokenized
text through its own ColBERT wrapper — no precomputed-embedding path — so this
is the plan's ~100-line fallback.

One step runs on CPU and MPS: `python -m esmc_retrieval.train --demo`.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from esmc_retrieval.eval import load_split
from esmc_retrieval.runlog import log_run

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "scope"


class EmbeddingStore:
    """Lazy id -> float32 [L, 960] cache. ponytail: unbounded dict; trim to
    fp16 or an LRU if a full-epoch sweep's resident set hurts on 8GB."""

    def __init__(self, cache: Path = CACHE) -> None:
        self.cache = cache
        self._d: dict[str, np.ndarray] = {}

    def get(self, rid: str) -> np.ndarray:
        if rid not in self._d:
            self._d[rid] = np.load(self.cache / f"{rid}.npy")
        return self._d[rid]


class Projection(nn.Module):
    def __init__(self, d_in: int = 960, d_out: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(e), dim=-1)


def sample_pair_batch(rows: list[dict], store: EmbeddingStore, batch_size: int,
                      rng: random.Random) -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor, list[str]]:
    """Batch of (anchor, positive) pairs sharing a superfamily — mirrors
    ProtoCol's PairDataset — as padded tensors + masks + superfamily labels."""
    by_sf: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sf[r["superfamily"]].append(r)
    sfs = [sf for sf, rs in by_sf.items() if len(rs) >= 2]
    qs, ds, qm, dm, labels = [], [], [], [], []
    for _ in range(batch_size):
        sf = rng.choice(sfs)
        a, b = rng.sample(by_sf[sf], 2)
        ea, eb = store.get(a["id"]), store.get(b["id"])
        qs.append(ea)
        ds.append(eb)
        qm.append(np.arange(len(ea)) < len(ea))
        dm.append(np.arange(len(eb)) < len(eb))
        labels.append(sf)
    la = max(len(x) for x in qs)
    lb = max(len(x) for x in ds)
    qb = np.zeros((batch_size, la, 960), dtype=np.float32)
    db = np.zeros((batch_size, lb, 960), dtype=np.float32)
    qmask = np.zeros((batch_size, la), dtype=bool)
    dmask = np.zeros((batch_size, lb), dtype=bool)
    for i in range(batch_size):
        qb[i, :len(qs[i])] = qs[i]
        db[i, :len(ds[i])] = ds[i]
        qmask[i, :len(qs[i])] = True
        dmask[i, :len(ds[i])] = True
    return (torch.from_numpy(qb), torch.from_numpy(db),
            torch.from_numpy(qmask), torch.from_numpy(dmask), labels)


def maxsim(q: torch.Tensor, q_mask: torch.Tensor,
           d: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
    """Exact differentiable MaxSim: [B,La,D] x [B,Ld,D] -> [B, B] where
    scores[i, j] = sum over query residues of max over doc residues (ProtoCol's
    eq. 2). Padded doc residues are -inf before the max, padded query residues
    are zeroed after. ponytail: einsum materializes [B,B,La,Ld]; fine at
    batch<=16 up to 512x512 (~270MB fp32) — chunk the doc axis if it grows."""
    sim = torch.einsum("ial,jbl->ijab", q, d)
    sim = sim.masked_fill(~d_mask[None, :, None, :], float("-inf"))
    sim = sim.amax(dim=-1)                                   # [B, B, La]
    sim = sim.masked_fill(~q_mask[:, None, :], 0.0)
    return sim.sum(dim=-1)


def false_negative_mask(labels: list[str]) -> torch.Tensor:
    """[B, B] True where a pair shares a superfamily (excluded from the
    denominator). Diagonal never masked — it is the positive."""
    lab = torch.tensor([hash(l) for l in labels])
    same = lab[:, None] == lab[None, :]
    return same.fill_diagonal_(False)


def nce_logits(scores: torch.Tensor, tau: float,
               forbidden: torch.Tensor | None = None) -> torch.Tensor:
    """scores/tau with forbidden pairs set to -inf (never the diagonal)."""
    logits = scores / tau
    if forbidden is not None:
        logits = logits.masked_fill(forbidden, float("-inf"))
    return logits


def symmetric_nce(scores: torch.Tensor, tau: float,
                  forbidden: torch.Tensor | None = None) -> torch.Tensor:
    """0.5 * (CE(scores/tau, diag) + CE(scores.T/tau, diag)).
    `forbidden[i, j] = True` removes that symmetric pair from the denominator
    entirely (T2.3 false-negative masking; None = unmasked reference)."""
    logits = nce_logits(scores, tau, forbidden)
    labels = torch.arange(scores.size(0), device=scores.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.T, labels))


def train_step(head: nn.Module, opt: torch.optim.Optimizer,
               batch: tuple, tau: float) -> float:
    dev = next(head.parameters()).device
    q, d, q_mask, d_mask, _labels = (t.to(dev) if torch.is_tensor(t) else t
                                     for t in batch)
    qe, de = head(q), head(d)
    scores = maxsim(qe, q_mask, de, d_mask)
    loss = symmetric_nce(scores, tau)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    return float(loss.detach())


def demo(devices: list[str]) -> int:
    rows = load_split(ROOT / "data" / "splits" / "scope_train.tsv")
    store = EmbeddingStore()
    rng = random.Random(0)
    batch = sample_pair_batch(rows, store, 16, rng)
    metrics: dict[str, float] = {}
    ok_all = True
    for dev in devices:
        torch.manual_seed(0)
        head = Projection().to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        before = head.proj.weight.detach().clone()
        losses = [train_step(head, opt, batch, tau=0.1) for _ in range(2)]
        moved = not torch.equal(before, head.proj.weight.detach())
        ok = all(np.isfinite(l) for l in losses) and moved
        metrics[f"{dev}_loss_1"] = losses[0]
        metrics[f"{dev}_loss_2"] = losses[1]
        print(f"{dev}: losses={[round(l, 4) for l in losses]} "
              f"weight_updated={moved} -> {'OK' if ok else 'FAIL'}")
        ok_all &= ok
    log_run("T2.1", {"batch": 16, "tau": 0.1, "lr": 1e-3,
                     "devices": devices, "pairs": "same-superfamily, scope_train"},
            metrics, seed=0)
    return 0 if ok_all else 1


def overfit(steps: int = 500, batch_size: int = 16, tau: float = 1.0,
            lr: float = 3e-3) -> int:
    """T2.2: memorize one fixed batch; exit when loss < 0.05 within `steps`."""
    rows = load_split(ROOT / "data" / "splits" / "scope_train.tsv")
    batch = sample_pair_batch(rows, EmbeddingStore(), batch_size, random.Random(0))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    head = Projection().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    first = last = None
    hit = None
    for step in range(1, steps + 1):
        loss = train_step(head, opt, batch, tau=tau)
        first = loss if first is None else first
        last = loss
        if loss < 0.05 and hit is None:
            hit = step
        if step % 100 == 0 or (hit and step == hit):
            print(f"step {step}: loss={loss:.4f}")
    ok = hit is not None
    print(f"overfit {dev}: first={first:.4f} last={last:.4f} "
          f"first_below_0.05={hit} -> {'OK' if ok else 'FAIL'}")
    log_run("T2.2", {"batch": batch_size, "tau": tau, "lr": lr,
                     "steps": steps, "device": dev,
                     "criterion": "loss<0.05 within 500 steps"},
            {"loss_first": round(first, 4), "loss_last": round(last, 4),
             "first_below_005": hit}, seed=0)
    return 0 if ok else 1


# ============================ M3: partial fine-tune ==========================

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def _esmc():
    from esm.models.esmc.model import EsmcForMaskedLM
    from esm.models.esmc.tokenizer import EsmcTokenizer

    m = EsmcForMaskedLM.from_pretrained("biohub/ESMC-300M", dtype=torch.float32)
    return m.esmc, EsmcTokenizer()


def m3_cache_dir(k: int) -> Path:
    return ROOT / "cache" / "m3" / f"k{k}"


class Suffix(nn.Module):
    """transformer.blocks[cut:] + final norm (cut = 30-k), fed by the prefix
    cache. Attention mask mirrors EsmcModel.forward's non-flash branch:
    sequence_id = bool_mask[:, None, None, :]."""

    def __init__(self, esmc: nn.Module, cut: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(list(esmc.transformer.blocks)[cut:])
        self.norm = esmc.transformer.norm

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_id = None if mask is None or bool(mask.all()) else mask[:, None, None, :]
        for b in self.blocks:
            x, _ = b(x, seq_id)
        return self.norm(x)


def all_scope_rows() -> list[dict]:
    rows = []
    for split in ("scope_train", "scope_val", "scope_test"):
        rows.extend(load_split(ROOT / "data" / "splits" / f"{split}.tsv"))
    seen: set[str] = set()
    return [r for r in rows if not (r["id"] in seen or seen.add(r["id"]))]


def build_prefix_cache(k: int) -> None:
    """Cache the input to block 30-k for every scope domain, once."""
    cut = 30 - k
    out = m3_cache_dir(k)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in all_scope_rows() if not (out / f"{r['id']}.npy").exists()]
    if not rows:
        print(f"prefix cache k={k}: already complete")
        return
    esmc, tok = _esmc()
    esmc = esmc.to(DEV).eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            batch = rows[i:i + 16]
            enc = tok([r["sequence"] for r in batch], padding=True,
                      return_tensors="pt")
            enc = {kk: vv.to(DEV) for kk, vv in enc.items()}
            x = esmc.embed(enc["input_ids"])
            mask = enc["attention_mask"].bool()
            seq_id = None if bool(mask.all()) else mask[:, None, None, :]
            for block in list(esmc.transformer.blocks)[:cut]:
                x, _ = block(x, seq_id)
            x = x.cpu()
            for j, r in enumerate(batch):
                t = int(mask[j].sum())
                np.save(out / f"{r['id']}.npy", x[j, :t].numpy())
            if i % 1024 == 0:
                print(f"  prefix k={k} {i}/{len(rows)}", flush=True)
    print(f"prefix cache k={k}: +{len(rows)} files in {time.time()-t0:.0f}s")


def check_suffix(k: int) -> int:
    """T3.1: with the k blocks unmodified, prefix-cache + suffix output must
    match the esmc.cpp embeddings (cache/scope): min cosine >= 0.999."""
    cut = 30 - k
    out = m3_cache_dir(k)
    ids = sorted(p.stem for p in out.glob("*.npy"))
    if not ids:
        build_prefix_cache(k)
        ids = sorted(p.stem for p in out.glob("*.npy"))
    esmc, _ = _esmc()
    suffix = Suffix(esmc, cut).to(DEV).eval()
    store = EmbeddingStore(out)
    min_cos = 1.0
    worst = ""
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(ids), 16):
            batch = ids[i:i + 16]
            embs = [store.get(r) for r in batch]
            L = max(len(e) for e in embs)
            x = np.zeros((len(batch), L, 960), dtype=np.float32)
            m = np.zeros((len(batch), L), dtype=bool)
            for j, e in enumerate(embs):
                x[j, :len(e)] = e
                m[j, :len(e)] = True
            mt = torch.from_numpy(m)
            h = suffix(torch.from_numpy(x).to(DEV), mt.to(DEV)).cpu().numpy()
            for j, rid in enumerate(batch):
                res_rows = np.where(m[j])[0][1:-1]  # drop CLS/EOS token rows
                pred = h[j, res_rows]
                ref = np.load(CACHE / f"{rid}.npy")
                if len(pred) != len(ref):
                    raise RuntimeError(f"{rid}: len {len(pred)} vs esmc.cpp {len(ref)}")
                pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
                rn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
                cos = float((pn * rn).sum(axis=1).min())
                if cos < min_cos:
                    min_cos, worst = cos, rid
    print(f"T3.1 check k={k}: min cosine {min_cos:.6f} (worst {worst}) "
          f"over {len(ids)} ids in {time.time()-t0:.0f}s -> "
          f"{'OK' if min_cos >= 0.999 else 'FAIL'}")
    log_run("T3.1", {"k": k, "cut": 30 - k, "device": DEV,
                     "criterion": "min cosine vs esmc.cpp >= 0.999"},
            {"min_cos": round(min_cos, 6), "n_ids": len(ids),
             "worst_id": worst}, seed=0, wall_s=time.time() - t0)
    return 0 if min_cos >= 0.999 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true",
                    help="one train step each on CPU and MPS (T2.1 exit)")
    ap.add_argument("--overfit", action="store_true",
                    help="memorize one batch of 16 pairs (T2.2 exit)")
    ap.add_argument("--finetune", type=int, choices=[2, 3], metavar="K",
                    help="M3 T3.1: build prefix cache for last-k blocks and "
                         "verify suffix reproduces esmc.cpp embeddings")
    args = ap.parse_args()
    if args.finetune:
        build_prefix_cache(args.finetune)
        return check_suffix(args.finetune)
    if args.demo:
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        return demo(devices)
    if args.overfit:
        return overfit()
    ap.error("pass --demo, --overfit or --finetune K")
    return 2


if __name__ == "__main__":
    sys.exit(main())
