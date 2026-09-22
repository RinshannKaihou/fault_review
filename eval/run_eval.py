#!/usr/bin/env python3
"""Evaluate fault-rag retrieval against source-verified relevance labels."""

import argparse
import json
import math
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


METRIC_NAMES = ("recall_at_5", "mrr_at_10", "ndcg_at_10")
ALLOWED_MODES = {"hybrid", "dense", "bm25"}
ALLOWED_FILTERS = {
    "repo", "category", "confidence", "trigger", "include_neg",
    "include_pointers",
}


def _dedupe_ranked(ranked):
    seen = set()
    out = []
    for key in ranked:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def metrics_for_ranking(ranked, relevant, recall_k=5, rank_k=10):
    """Compute binary Recall@K, MRR@K and nDCG@K for one query."""
    relevant = set(relevant)
    if not relevant:
        raise ValueError("relevant set must not be empty")
    ranked = _dedupe_ranked(ranked)

    recall_hits = sum(key in relevant for key in ranked[:recall_k])
    recall = recall_hits / len(relevant)

    reciprocal_rank = 0.0
    for rank, key in enumerate(ranked[:rank_k], start=1):
        if key in relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, key in enumerate(ranked[:rank_k], start=1)
        if key in relevant
    )
    ideal_count = min(len(relevant), rank_k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    ndcg = dcg / ideal_dcg

    return {
        "recall_at_5": recall,
        "mrr_at_10": reciprocal_rank,
        "ndcg_at_10": ndcg,
    }


def validate_cases(cases, corpus_keys):
    """Reject malformed or unauditable benchmark labels."""
    corpus_keys = set(corpus_keys)
    seen_ids = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"{case_id}: query must be a non-empty string")
        if case.get("mode") not in ALLOWED_MODES:
            raise ValueError(f"{case_id}: unsupported mode {case.get('mode')!r}")
        filters = case.get("filters")
        if not isinstance(filters, dict):
            raise ValueError(f"{case_id}: filters must be an object")
        unknown_filters = set(filters) - ALLOWED_FILTERS
        if unknown_filters:
            raise ValueError(
                f"{case_id}: unsupported filters {sorted(unknown_filters)}"
            )
        relevant = case.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"{case_id}: relevant must be a non-empty list")
        for item in relevant:
            key = (item.get("repo"), item.get("id"))
            if key not in corpus_keys:
                raise ValueError(f"{case_id}: unknown relevant entry {key}")
        tags = case.get("tags")
        if not isinstance(tags, list) or not tags or not all(
            isinstance(tag, str) and tag for tag in tags
        ):
            raise ValueError(f"{case_id}: tags must be non-empty strings")
        if not isinstance(case.get("rationale"), str) or not case["rationale"].strip():
            raise ValueError(f"{case_id}: rationale must be a non-empty string")
    return cases


def _mean_metrics(records):
    if not records:
        raise ValueError("cannot average an empty metric list")
    return {
        name: sum(record[name] for record in records) / len(records)
        for name in METRIC_NAMES
    }


def evaluate_cases(cases, search_fn=None):
    """Run cases through one in-process index and return aggregate diagnostics."""
    if search_fn is None:
        from fault_rag.search import search as search_fn

    case_reports = []
    for case in cases:
        results = search_fn(
            case["query"],
            topk=10,
            mode=case["mode"],
            **case["filters"],
        )
        ranked = [(row["repo"], row["id"]) for row in results]
        relevant = {
            (item["repo"], item["id"]) for item in case["relevant"]
        }
        metrics = metrics_for_ranking(ranked, relevant)
        case_reports.append({
            "case_id": case["case_id"],
            "query": case["query"],
            "tags": case["tags"],
            "relevant": [list(key) for key in sorted(relevant)],
            "top_10": [list(key) for key in ranked[:10]],
            "metrics": metrics,
        })

    aggregate = _mean_metrics([row["metrics"] for row in case_reports])
    tags = sorted({tag for case in cases for tag in case["tags"]})
    slices = {}
    for tag in tags:
        rows = [row["metrics"] for row in case_reports if tag in row["tags"]]
        slices[tag] = {"case_count": len(rows), **_mean_metrics(rows)}
    failures = [
        row for row in case_reports if row["metrics"]["recall_at_5"] < 1.0
    ]
    return {
        "case_count": len(case_reports),
        "aggregate": aggregate,
        "slices": slices,
        "failures": failures,
        "cases": case_reports,
    }


def check_report(report, baseline=None, regression_tolerance=0.02):
    """Return human-readable quality-gate failures; an empty list means pass."""
    floors = {
        "recall_at_5": 0.85,
        "mrr_at_10": 0.70,
        "ndcg_at_10": 0.75,
    }
    failures = []
    aggregate = report["aggregate"]
    for name, floor in floors.items():
        value = aggregate[name]
        if value < floor:
            failures.append(f"{name} floor: {value:.6f} < {floor:.6f}")

    negative = report.get("slices", {}).get("negative")
    if negative is None:
        failures.append("negative slice is missing")
    elif negative["recall_at_5"] < 0.75:
        failures.append(
            "negative recall_at_5 floor: "
            f"{negative['recall_at_5']:.6f} < 0.750000"
        )

    if baseline is not None:
        for name in METRIC_NAMES:
            drop = baseline["aggregate"][name] - aggregate[name]
            if drop > regression_tolerance + 1e-12:
                failures.append(
                    f"{name} regressed by {drop:.6f} "
                    f"(limit {regression_tolerance:.6f})"
                )
    return failures


def load_cases(path):
    cases = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
    return cases


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Evaluate fault-rag retrieval quality."
    )
    parser.add_argument("--cases", default=os.path.join(here, "cases.jsonl"))
    parser.add_argument("--output", help="write the passing report to this path")
    parser.add_argument("--check", help="compare against a checked-in baseline")
    parser.add_argument("--min-cases", type=int, default=36)
    args = parser.parse_args(argv)

    from fault_rag.index import get_index

    cases = load_cases(args.cases)
    if len(cases) < args.min_cases:
        raise SystemExit(
            f"benchmark requires at least {args.min_cases} cases; got {len(cases)}"
        )
    idx = get_index()
    corpus_keys = {(e["repo"], e["id"]) for e in idx.entries}
    validate_cases(cases, corpus_keys)
    report = evaluate_cases(cases)

    baseline = None
    if args.check:
        with open(args.check, encoding="utf-8") as f:
            baseline = json.load(f)
    gate_failures = check_report(report, baseline=baseline)
    report["quality_gate"] = {
        "passed": not gate_failures,
        "failures": gate_failures,
    }

    json.dump(report, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
    if gate_failures:
        for failure in gate_failures:
            print(f"QUALITY GATE: {failure}", file=sys.stderr)
        return 1

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
