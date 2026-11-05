"""M8 CLI: build, search, benchmark a local `.esmci` index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from esmc_li.indexes import INDEXES, resolve_index
from esmc_li.models import DEFAULT_MODEL
from esmc_li.protein_index import ProteinIndex, _read_fasta


def _cmd_build(args: argparse.Namespace) -> int:
    idx = ProteinIndex.from_fasta(
        args.fasta, args.output, model=args.model,
        embed_bin=args.embed_bin, head=args.head,
    )
    print(json.dumps({"output": str(args.output), "n": len(idx),
                      "ids": idx.ids[:3]}, indent=2))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    idx = ProteinIndex.load(args.index, model=args.model,
                            embed_bin=args.embed_bin, head=args.head)
    if bool(args.sequence) == bool(args.query):
        raise SystemExit("pass exactly one of --sequence or --query")
    if args.sequence:
        hits = idx.search(args.sequence, k=args.top_k)
        print(json.dumps(hits, indent=2))
        return 0
    rows = _read_fasta(Path(args.query))
    results = idx.search_batch([r["sequence"] for r in rows], k=args.top_k)
    with open(args.output, "w") as f:
        f.write("query\trank\tid\tscore\n")
        for r, hits in zip(rows, results):
            for rank, h in enumerate(hits, start=1):
                f.write(f"{r['id']}\t{rank}\t{h['id']}\t{h['score']:.6f}\n")
    print(f"{len(results)} queries -> {args.output}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    idx = ProteinIndex.load(args.index, model=args.model,
                            embed_bin=args.embed_bin, head=args.head)
    print(json.dumps(idx.benchmark(nq=args.nq, k=args.k), indent=2))
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    if args.action == "list":
        print("\n".join(INDEXES))
        return 0
    print(resolve_index(args.name))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="esmc-li")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a .esmci index from FASTA")
    b.add_argument("fasta")
    b.add_argument("--output", required=True)
    b.add_argument("--model", default=DEFAULT_MODEL,
                   help="fine-tuned ESM-C GGUF or registry name")
    b.add_argument("--embed-bin", default=None, help="esmc-embed binary")
    b.add_argument("--head", default=None, help="projection head .npy")
    b.set_defaults(func=_cmd_build)

    s = sub.add_parser("search", help="search an index")
    s.add_argument("index")
    s.add_argument("--sequence", default=None)
    s.add_argument("--query", default=None)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--output", default="results.tsv")
    s.add_argument("--model", default=None)
    s.add_argument("--embed-bin", default=None)
    s.add_argument("--head", default=None)
    s.set_defaults(func=_cmd_search)

    m = sub.add_parser("benchmark", help="benchmark exact search")
    m.add_argument("index")
    m.add_argument("--nq", type=int, default=10)
    m.add_argument("--k", type=int, default=10)
    m.add_argument("--model", default=None)
    m.add_argument("--embed-bin", default=None)
    m.add_argument("--head", default=None)
    m.set_defaults(func=_cmd_benchmark)

    i = sub.add_parser("index", help="list or download prebuilt indexes")
    i.add_argument("action", choices=["list", "pull"])
    i.add_argument("name", nargs="?", default="")
    i.set_defaults(func=_cmd_index)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
