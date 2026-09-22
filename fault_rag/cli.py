"""fault-rag CLI — JSON-on-stdout interface for agents."""

import argparse
import json
import sys
import time


def _print(obj):
    json.dump(obj, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="fault-rag",
        description="Hybrid BM25+bge-m3 retrieval over the inference/training "
                    "fault catalogs. All output is JSON.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="hybrid retrieval over catalog entries")
    s.add_argument("query")
    s.add_argument("--repo", choices=["inference", "training"])
    s.add_argument("--category", help="NUM / HW / cat1 / 1 / KV1 / neg_rejected ...")
    s.add_argument("--confidence", choices=["verified", "documented", "speculative"])
    s.add_argument("--trigger", help="inference 触发 tag, e.g. yes/partial/no")
    s.add_argument("--topk", type=int, default=8)
    s.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    s.add_argument("--no-neg", action="store_true",
                   help="exclude negative-doc chunks (rejected/blindspots/appendix)")
    s.add_argument("--include-pointers", action="store_true",
                   help="include pointer/meta entries in full-text search")
    s.add_argument("--full", action="store_true",
                   help="include complete fields instead of summary")

    l = sub.add_parser("lookup", help="exact id / snake_name / alias lookup")
    l.add_argument("query")

    r = sub.add_parser("related", help="walk cross-reference links of an entry")
    r.add_argument("query")
    r.add_argument("--depth", type=int, default=1)

    ri = sub.add_parser("reindex", help="re-parse catalogs and re-embed changes")
    ri.add_argument("--full", action="store_true", help="re-embed everything")

    sub.add_parser("stats", help="index counts by repo/category/confidence")
    sub.add_parser("taxonomy", help="list categories with entry counts")

    args = p.parse_args(argv)

    if args.cmd == "reindex":
        from . import index as index_mod

        meta = index_mod.build(full=args.full)
        _print(meta)
        return 0

    from . import search as search_mod

    if args.cmd == "search":
        t0 = time.time()
        res = search_mod.search(
            args.query, topk=args.topk, mode=args.mode, repo=args.repo,
            category=args.category, confidence=args.confidence,
            trigger=args.trigger, include_neg=not args.no_neg,
            include_pointers=args.include_pointers,
        )
        if args.full:
            from .search import _full, get_index

            idx = get_index()
            by_key = {(e["repo"], e["id"]): e for e in idx.entries}
            for rec in res:
                rec["fields"] = _full(by_key[(rec["repo"], rec["id"])])["fields"]
        _print({"query": args.query, "elapsed_ms": int((time.time() - t0) * 1000),
                "results": res})
    elif args.cmd == "lookup":
        _print(search_mod.lookup(args.query))
    elif args.cmd == "related":
        _print(search_mod.related(args.query, depth=args.depth))
    elif args.cmd == "stats":
        _print(search_mod.stats())
    elif args.cmd == "taxonomy":
        st = search_mod.stats()
        _print(st["by_category"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
