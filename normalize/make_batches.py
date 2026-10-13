#!/usr/bin/env python3
"""Split entries still awaiting extraction into LLM batches and build the run manifest.

Batches follow document order (similar entries share context). Every 3 batches
form one adjudication group; ~5% of entries are drawn as a faithfulness audit
sample, deterministically, so the workflow script needs no randomness.
"""

import json
import os
import random

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRIES = os.path.join(HERE, "out", "entries")
RUN = os.path.join(HERE, "run")
BATCH, GROUP, AUDIT_RATE, SEED = 25, 3, 0.05, 20261012


def to_input(e):
    o = e["original"]
    keep = {k: o[k] for k in ("mechanism", "behavior", "source", "trigger_surface", "signature", "discovery", "locus") if k in o}
    return {
        "id": e["id"], "legacy_id": e["legacy_ids"][0], "slug": e["slug"],
        "repo": "inference" if e["id"].startswith("INF") else "training",
        "old_section": o["section"], "meta": o["meta"],
        "stack_hint": e["cause"]["objects"][0]["stack"], "original": keep,
    }


def main():
    pending = []
    for name in sorted(os.listdir(ENTRIES)):
        e = yaml.safe_load(open(os.path.join(ENTRIES, name), encoding="utf-8"))
        if e["title"] == "待定":
            pending.append(e)
    rng = random.Random(SEED)
    audit = set(rng.sample([e["id"] for e in pending], round(len(pending) * AUDIT_RATE)))

    batches = []
    for i in range(0, len(pending), BATCH):
        name = f"b{len(batches) + 1:03d}"
        path_in = os.path.join(RUN, "in", f"{name}.jsonl")
        with open(path_in, "w", encoding="utf-8") as fh:
            for e in pending[i:i + BATCH]:
                fh.write(json.dumps(to_input(e), ensure_ascii=False) + "\n")
        ids = [e["id"] for e in pending[i:i + BATCH]]
        batches.append({"name": name, "in": path_in, "out": os.path.join(RUN, "out", f"{name}.jsonl"), "ids": ids})

    groups = []
    for j in range(0, len(batches), GROUP):
        gname = f"g{len(groups) + 1:02d}"
        members = batches[j:j + GROUP]
        groups.append({
            "group": gname,
            "batches": [{k: b[k] for k in ("name", "in", "out")} for b in members],
            "audit_ids": sorted(i for b in members for i in b["ids"] if i in audit),
            "judged": os.path.join(RUN, "judged", f"{gname}.jsonl"),
            "scratch": os.path.join(RUN, "scratch", f"{gname}.md"),
        })
    with open(os.path.join(RUN, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"groups": groups}, fh, ensure_ascii=False, indent=1)
    print(f"{len(pending)} pending entries → {len(batches)} batches → {len(groups)} groups; audit sample {len(audit)}")


if __name__ == "__main__":
    main()
