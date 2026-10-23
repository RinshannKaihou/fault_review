#!/usr/bin/env python3
"""Web quality gate: run the eval cases through the site's own JS retrieval (rank.mjs on a built
web/dist) and apply eval/run_eval.py's metrics and gate rules to the rankings.

  python3 web/eval/check_web.py                     # gate every tier the build has
  python3 web/eval/check_web.py --tier bm25
  python3 web/eval/check_web.py --write-baselines   # after a reviewed snapshot / retrieval change

Same rules as the CLI gate, against web-own baselines (the CLI baseline includes the reranker,
which no browser tier has):
  bench   eval/cases.jsonl          aggregate: R@5/R@10 may not drop, MRR/nDCG tolerance 0.02
  indy    eval/indy_cases.jsonl     per case: a baseline pass at R@5/R@10 must keep passing
  heldout eval/heldout_cases.jsonl  reported only
It also checks that src/search.js tokenizes every case query and corpus text exactly like
build_site.tokenize, since the index side is built in Python and the query side runs in JS.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
ROOT = os.path.dirname(WEB)
sys.path.insert(0, ROOT)
sys.path.insert(0, WEB)

from eval.run_eval import (check_report, evaluate_cases, load_cases, load_fix_list,  # noqa: E402
                           rank_churn, resolve_legacy_ids, validate_cases)
from build_site import tokenize                     # noqa: E402

SUITES = (   # name, cases file, per-case gate, gating
    ("bench", "cases.jsonl", False, True),
    ("indy", "indy_cases.jsonl", True, True),
    ("heldout", "heldout_cases.jsonl", False, False),
)
BASELINES = os.path.join(HERE, "baselines")


def rank_with_js(dist, tier, cases, texts):
    payload = {"tier": tier, "texts": texts,
               "cases": [{"query": c["query"], "filters": c["filters"]} for c in cases]}
    proc = subprocess.run(["node", os.path.join(HERE, "rank.mjs"), dist],
                          input=json.dumps(payload, ensure_ascii=False), capture_output=True,
                          text=True, check=False)
    if proc.returncode:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"rank.mjs failed ({tier})")
    return json.loads(proc.stdout)


def tokenizer_parity(texts, js_tokens):
    bad = [t for t, js in zip(texts, js_tokens) if tokenize(t) != js]
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dist", default=os.path.join(WEB, "dist"))
    ap.add_argument("--tier", choices=("bm25", "hybrid"), action="append",
                    help="default: bm25, plus hybrid when the build has vectors")
    ap.add_argument("--write-baselines", action="store_true")
    ap.add_argument("--fix-list", default=os.path.join(HERE, "fix_list.json"))
    ap.add_argument("--report", help="write every report to this JSON file")
    args = ap.parse_args(argv)

    data = os.path.join(args.dist, "data")
    with open(os.path.join(data, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(data, "docs.json"), encoding="utf-8") as f:
        docs = json.load(f)
    tiers = args.tier or (["bm25", "hybrid"] if meta.get("semantic") else ["bm25"])

    corpus_keys = {(d["repo"], d["id"]) for d in docs}
    legacy = {(d["repo"], old): (d["repo"], d["id"]) for d in docs for old in d["legacy_ids"] or []}
    suites = []
    for name, fname, per_case, gating in SUITES:
        cases = resolve_legacy_ids(load_cases(os.path.join(ROOT, "eval", fname)), corpus_keys, legacy)
        validate_cases(cases, corpus_keys)
        suites.append((name, cases, per_case, gating))

    from fault_rag.index import bm25_text
    from fault_rag.parse import parse_all
    texts = [c["query"] for _, cases, _, _ in suites for c in cases]
    texts += [bm25_text(e) for e in parse_all()]

    fix_list = set(load_fix_list(args.fix_list)) if os.path.exists(args.fix_list) else set()
    failures, reports = [], {}
    for tier in tiers:
        for n, (name, cases, per_case, gating) in enumerate(suites):
            parity = n == 0 and tier == tiers[0]
            out = rank_with_js(args.dist, tier, cases, texts if parity else [])
            if parity:
                bad = tokenizer_parity(texts, out["tokens"])
                if bad:
                    failures.append(f"tokenizer parity: {len(bad)} texts differ, e.g. {bad[0][:80]!r}")
            by_query = {}
            for c, ranked in zip(cases, out["rankings"]):
                by_query[(c["query"], json.dumps(c["filters"], sort_keys=True))] = ranked

            def search_fn(query, topk, mode, **filters):
                ranked = by_query[(query, json.dumps(filters, sort_keys=True))]
                return [{"repo": r, "id": i} for r, i in ranked[:topk]]

            report = evaluate_cases(cases, search_fn=search_fn)
            report["tier"] = tier
            path = os.path.join(BASELINES, f"{name}_{tier}.json")
            baseline = None
            if os.path.exists(path) and not args.write_baselines:
                with open(path, encoding="utf-8") as f:
                    baseline = json.load(f)
            gate = []
            if gating:
                gate = check_report(report, baseline=baseline, per_case=per_case,
                                    fix_list=fix_list if per_case else None)
                if baseline is None and not args.write_baselines:
                    gate.append(f"no baseline at {os.path.relpath(path, ROOT)}")
                failures += [f"{name}/{tier}: {g}" for g in gate]
            if per_case and baseline is not None:
                report["churn"] = rank_churn(report, baseline)
            if args.write_baselines and gating:
                os.makedirs(BASELINES, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=1)
                    f.write("\n")
            reports[f"{name}_{tier}"] = report
            a = report["aggregate"]
            churn = report.get("churn")
            print(f"{name:8s} {tier:7s} n={report['case_count']:3d}  "
                  + "  ".join(f"{k}={a[k]:.4f}" for k in ("recall_at_5", "recall_at_10",
                                                          "mrr_at_10", "ndcg_at_10"))
                  + ("  gate=" + ("pass" if not gate else "FAIL") if gating else "  (observe)")
                  + (f"  slipped={[s['case_id'] for s in churn['slipped']]}" if churn else ""))

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=1)
    for failure in failures:
        print(f"WEB QUALITY GATE: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
