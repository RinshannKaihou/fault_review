#!/usr/bin/env python3
"""Merge all LLM extraction outputs into the structured entries.

Precedence per entry, lowest to highest:
  1. pilot 3-way majority (pilot/out_{A,B,C}*.jsonl) on the primary subsystem
  2. full-run first pass (run/out/*.jsonl)
  3. full-run adjudication / audit (run/judged/*.jsonl)

Each merged entry gets a `migration` block recording how its semantic fields
were decided, so later audits can filter by provenance.

    python3 normalize/finalize.py
"""

import glob
import json
import os
import sys
from collections import Counter

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from apply_extraction import merge  # noqa: E402
from convert import dump_yaml  # noqa: E402

ENTRIES = os.path.join(HERE, "out", "entries")
EXTRACTOR = "opus v1.1"


def read_jsonl(pattern):
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, pattern))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r
    return out


def pilot_majority():
    passes = {p: read_jsonl(f"pilot/out_{p}[0-9].jsonl") for p in "ABC"}
    chosen = {}
    for eid in passes["A"]:
        votes = {p: passes[p][eid]["cause"]["objects"][0]["subsystem"] for p in "ABC" if eid in passes[p]}
        top, n = Counter(votes.values()).most_common(1)[0]
        winner = next((p for p in "ABC" if votes.get(p) == top), "A") if n >= 2 else "A"
        rec = dict(passes[winner][eid])
        rec["uncertain"] = "" if n >= 2 else rec.get("uncertain", "")
        rec["_migration"] = {"method": "pilot-majority", "votes": f"{n}/{len(votes)}", "pass": winner}
        chosen[eid] = rec
    return chosen


def main():
    records = pilot_majority()
    first = read_jsonl("run/out/*.jsonl")
    judged = read_jsonl("run/judged/*.jsonl")
    for eid, rec in first.items():
        rec["_migration"] = {"method": "extract"}
        records[eid] = rec
    for eid, rec in judged.items():
        j = rec.get("judge", {})
        rec["_migration"] = {
            "method": "adjudicated",
            "reason": j.get("reason", []),
            "changed": j.get("changed", []),
            "note": j.get("why", ""),
            "audit_issues": j.get("audit_issues", []),
        }
        records[eid] = rec

    stats = Counter()
    for eid, rec in records.items():
        path = os.path.join(ENTRIES, f"{eid}.yaml")
        with open(path, encoding="utf-8") as fh:
            entry = yaml.safe_load(fh)
        entry = merge(entry, rec)
        entry["migration"] = {"extractor": EXTRACTOR, **rec["_migration"]}
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(dump_yaml(entry))
        stats[rec["_migration"]["method"]] += 1
    print(f"merged {len(records)} entries:", dict(stats))


if __name__ == "__main__":
    main()
