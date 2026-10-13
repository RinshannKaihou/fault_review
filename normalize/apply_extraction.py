#!/usr/bin/env python3
"""Merge LLM extraction results (JSONL) into the structured entry files.

Only the semantic fields are written: title, kind, cause, effect. Mechanical
fields (IDs, labels, evidence, relations, original) and mitigation are never
touched, so re-running convert.py + this script is reproducible.

    python3 normalize/apply_extraction.py pilot/out_A1.jsonl pilot/out_A2.jsonl
"""

import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert import dump_yaml  # noqa: E402


def merge(entry, ext):
    entry["title"] = ext["title"]
    entry["kind"] = ext["kind"]
    entry["cause"] = {"objects": ext["cause"]["objects"], "behavior": ext["cause"]["behavior"]}
    effect = ext["effect"]
    entry["effect"] = {
        "objects": effect["objects"] if isinstance(effect.get("objects"), list) else effect.get("objects", "待定"),
        "detail": effect.get("detail", "待定"),
        "visibility": effect.get("visibility", "待定"),
    }
    notes = [n for n in (ext.get("uncertain"), ext.get("vocab_gap")) if n]
    if notes:
        entry["remarks"] = "；".join(notes)
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--entries", default=os.path.join(HERE, "out", "entries"))
    args = ap.parse_args()

    n = 0
    for path in args.jsonl:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                ext = json.loads(line)
                target = os.path.join(args.entries, f"{ext['id']}.yaml")
                with open(target, encoding="utf-8") as fh_in:
                    entry = yaml.safe_load(fh_in)
                with open(target, "w", encoding="utf-8") as fh_out:
                    fh_out.write(dump_yaml(merge(entry, ext)))
                n += 1
    print(f"merged {n} extraction records into {args.entries}")


if __name__ == "__main__":
    main()
