#!/usr/bin/env python3
"""Build judge items and the prior (pre-audit) values from the frozen baseline.

items.jsonl  — what judges see: original text fields (task C) plus the structured
               text fields under audit (task F). No prior classification.
prior.jsonl  — the baseline's classification + text fields, used only by merge/apply.

Reads the baseline tarball (verified against baseline.md5), never out/entries.
Prints counts only.
"""

import hashlib
import io
import json
import os
import tarfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "baseline_v1.1_2026-10-12.tar.gz")
ORIGINAL_FIELDS = ("mechanism", "behavior", "source", "trigger_surface", "signature", "repro", "discovery", "locus")


def verify_baseline():
    want = open(os.path.join(HERE, "baseline.md5")).read().split()[0]
    got = hashlib.md5(open(BASELINE, "rb").read()).hexdigest()
    if want != got:
        raise SystemExit(f"baseline md5 mismatch: {got} != {want}")


def load_baseline():
    verify_baseline()
    entries = {}
    with tarfile.open(BASELINE) as tar:
        for m in tar.getmembers():
            if m.name.endswith(".yaml"):
                e = yaml.safe_load(io.TextIOWrapper(tar.extractfile(m), encoding="utf-8"))
                entries[e["id"]] = e
    return entries


def text_fields(e):
    objs = e["cause"]["objects"] if isinstance(e["cause"]["objects"], list) else []
    return {
        "title": e["title"],
        "component": "；".join(str(o.get("component", "")) for o in objs),
        "behavior": e["cause"]["behavior"],
        "detail": e["effect"]["detail"],
    }


def first(v):
    return v[0] if isinstance(v, list) and v else v


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-out", default="items.jsonl",
                    help="v1 (items.jsonl) omitted the metadata line; v2 runs write items_v2.jsonl")
    ap.add_argument("--with-meta", action="store_true", help="show the entry's metadata line to judges (v2)")
    a = ap.parse_args()
    entries = load_baseline()
    with open(os.path.join(HERE, a.items_out), "w", encoding="utf-8") as fi, \
         open(os.path.join(HERE, "prior.jsonl"), "w", encoding="utf-8") as fp:
        for eid in sorted(entries):
            e = entries[eid]
            original = {k: e["original"][k] for k in ORIGINAL_FIELDS if e["original"].get(k)}
            if a.with_meta and e["original"].get("meta"):
                # the metadata line carries facts (its parenthetical notes) that the extraction saw
                original = {"meta": e["original"]["meta"], **original}
            fi.write(json.dumps({"id": eid, "original": original, "structured": text_fields(e)}, ensure_ascii=False) + "\n")
            objs = e["cause"]["objects"] if isinstance(e["cause"]["objects"], list) else []
            fp.write(json.dumps({
                "id": eid,
                "kind": e["kind"],
                "subsystem": objs[0]["subsystem"] if objs else e["cause"]["objects"],
                "effect": first(e["effect"]["objects"]),
                "visibility": e["effect"]["visibility"],
                **text_fields(e),
            }, ensure_ascii=False) + "\n")
    print(f"items: {len(entries)}")


if __name__ == "__main__":
    main()
