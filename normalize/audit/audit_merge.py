#!/usr/bin/env python3
"""Fold judge votes into per-field audit decisions (pattern: llm-judge-labeling langmix_common.decide).

  python3 audit_merge.py stats          # counts per field/outcome + judge agreement with final decisions
  python3 audit_merge.py select-v2      # freeze the v2 re-judgment subsets (see below)
  python3 audit_merge.py export-panel   # every non-final item -> panel_items_{C,F}.jsonl
  python3 audit_merge.py final          # decisions.jsonl (re-derived from all stored votes; idempotent)

Categorical fields (task C): base stage needs a quorum of 2 valid votes and unanimity, otherwise
'split'; with panel votes the top value needs >= 5 valid votes and a >= 2/3 share ('panel7').
Text fields (task F): binary issue/no-issue with the same quorum and majority rule.
Weak decisions keep the baseline value and are listed as uncertain — never silently changed.

v2 (2026-10-12): the v1 judge input omitted each entry's metadata line, so 736 of 850 v1 'issue'
fields quoted text that only exists there. Re-judged with items_v2.jsonl (metadata included):
task F on entries with any field not 'ok' (an 'ok' without the metadata stays 'ok' with it), task C
on entries with any field not 'confirmed'. For an id in a frozen v2 subset, only its v2 base votes
count (no mixing of v1 and v2 inputs for one id); panel votes are always on v2 items.
Prints ids/labels/counts only.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
VOTES = os.path.join(HERE, "votes")
BASE = ("kimi-k3", "qwen3.8-max", "glm-5.3")
PANEL = ("doubao-seed-2.1-pro", "deepseek-v4-pro", "MiniMax-M3", "hy3")
CAT_FIELDS = ("kind", "subsystem", "effect", "visibility")
TEXT_FIELDS = ("title", "component", "behavior", "detail")
QUORUM_BASE, QUORUM_PANEL, MAJ = 2, 5, 2 / 3


def safe(model):
    return re.sub(r"[.\-/]", "_", model)


def load_votes(task, tag, models):
    """{id: {model: record}} keeping the latest non-failed record per (id, model)."""
    out = defaultdict(dict)
    for m in models:
        for suffix in ("_progress.jsonl", "_retry_progress.jsonl"):
            path = os.path.join(VOTES, f"{task}{tag}_{safe(m)}{suffix}")
            if not os.path.exists(path):
                continue
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("parse_failed") or r["id"] not in out or out[r["id"]].get(m, {}).get("parse_failed", True):
                    out[r["id"]][m] = r
    return out


def decide_cat(values, panel):
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None, "unresolved", 0
    top, k = Counter(vals).most_common(1)[0]
    if panel:
        return top, ("panel7" if n >= QUORUM_PANEL and k / n >= MAJ else "uncertain"), n
    if n < QUORUM_BASE:
        return top, "uncertain", n
    return top, ("unanimous" if k == n else "split"), n


def decide_bin(values, panel):
    vals = [v for v in values if v is not None]
    t, n = sum(vals), len(vals)
    if n == 0:
        return False, "unresolved", 0
    final = t > n - t
    if panel:
        return final, ("panel7" if n >= QUORUM_PANEL and max(t, n - t) / n >= MAJ else "uncertain"), n
    if n < QUORUM_BASE:
        return final, "uncertain", n
    return final, ("unanimous" if t in (0, n) else "split"), n


FINAL = ("unanimous", "panel7")


def v2_subsets():
    path = os.path.join(HERE, "v2_subsets.json")
    return json.load(open(path)) if os.path.exists(path) else {"C": [], "F": []}


def base_votes(task):
    v1, v2, subset = load_votes(task, "", BASE), load_votes(task, "_v2", BASE), set(v2_subsets()[task])
    return {i: (v2.get(i, {}) if i in subset else v) for i, v in v1.items()} | {i: v2.get(i, {}) for i in subset}


def build():
    prior = {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "prior.jsonl"), encoding="utf-8"))}
    C = {"base": base_votes("C"), "panel": load_votes("C", "_panel", PANEL)}
    F = {"base": base_votes("F"), "panel": load_votes("F", "_panel", PANEL)}
    decisions = {}
    for eid, p in prior.items():
        d = {"id": eid, "fields": {}}
        for f in CAT_FIELDS:
            base = [r.get(f) if not r.get("parse_failed") else None for r in C["base"].get(eid, {}).values()]
            pan = [r.get(f) if not r.get("parse_failed") else None for r in C["panel"].get(eid, {}).values()]
            value, status, n = decide_cat(base + pan, panel=bool([v for v in pan if v is not None]))
            if status in FINAL:
                outcome = "confirmed" if value == p[f] else "corrected"
            else:
                value, outcome = p[f], "uncertain"
            d["fields"][f] = {"prior": p[f], "final": value, "status": status, "n": n, "outcome": outcome,
                              "votes": {m: (r.get(f) if not r.get("parse_failed") else None)
                                        for src in ("base", "panel") for m, r in C[src].get(eid, {}).items()}}
        for f in TEXT_FIELDS:
            base = [r["votes"].get(f) if not r.get("parse_failed") else None for r in F["base"].get(eid, {}).values()]
            pan = [r["votes"].get(f) if not r.get("parse_failed") else None for r in F["panel"].get(eid, {}).values()]
            issue, status, n = decide_bin(base + pan, panel=bool([v for v in pan if v is not None]))
            outcome = ("issue" if issue else "ok") if status in FINAL else "uncertain"
            notes = [r["notes"][f] | {"model": m} for src in ("base", "panel") for m, r in F[src].get(eid, {}).items()
                     if not r.get("parse_failed") and f in r.get("notes", {})]
            d["fields"][f] = {"status": status, "n": n, "outcome": outcome, "notes": notes}
        decisions[eid] = d
    return decisions, C, F


def stats(decisions, C, F):
    n = len(decisions)
    print(f"entries {n} | task C voted: base {len(C['base'])} panel {len(C['panel'])} | task F voted: base {len(F['base'])} panel {len(F['panel'])}")
    for f in CAT_FIELDS + TEXT_FIELDS:
        c = Counter(d["fields"][f]["outcome"] for d in decisions.values())
        s = Counter(d["fields"][f]["status"] for d in decisions.values())
        print(f"  {f:<10} outcome {dict(c)} | status {dict(s)}")
    # judge agreement with final decisions (only where the decision is final)
    agree = defaultdict(lambda: [0, 0])
    for d in decisions.values():
        for f in CAT_FIELDS:
            x = d["fields"][f]
            if x["status"] in FINAL:
                for m, v in x["votes"].items():
                    if v is not None:
                        agree[(m, "C")][0] += v == x["final"]
                        agree[(m, "C")][1] += 1
    for (m, t), (k, tot) in sorted(agree.items()):
        print(f"  judge {m:<22} task {t} agrees with final {k}/{tot} ({k / tot:.0%})")


def export_panel(decisions):
    for task, fields in (("C", CAT_FIELDS), ("F", TEXT_FIELDS)):
        ids = sorted(i for i, d in decisions.items() if any(d["fields"][f]["status"] not in FINAL for f in fields))
        items = {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "items_v2.jsonl"), encoding="utf-8"))}
        with open(os.path.join(HERE, f"panel_items_{task}.jsonl"), "w", encoding="utf-8") as fh:
            for i in ids:
                fh.write(json.dumps(items[i], ensure_ascii=False) + "\n")
        print(f"panel items task {task}: {len(ids)}")


def select_v2(decisions):
    path = os.path.join(HERE, "v2_subsets.json")
    if os.path.exists(path):
        sys.exit("v2_subsets.json already frozen; delete it deliberately to re-select")
    subsets = {
        "C": sorted(i for i, d in decisions.items() if any(d["fields"][f]["outcome"] != "confirmed" for f in CAT_FIELDS)),
        "F": sorted(i for i, d in decisions.items() if any(d["fields"][f]["outcome"] != "ok" for f in TEXT_FIELDS)),
    }
    json.dump(subsets, open(path, "w"), indent=0)
    items = {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "items_v2.jsonl"), encoding="utf-8"))}
    for task, ids in subsets.items():
        with open(os.path.join(HERE, f"items_v2_{task}.jsonl"), "w", encoding="utf-8") as fh:
            for i in ids:
                fh.write(json.dumps(items[i], ensure_ascii=False) + "\n")
        print(f"v2 subset task {task}: {len(ids)} entries")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    decisions, C, F = build()
    if cmd == "stats":
        stats(decisions, C, F)
    elif cmd == "export-panel":
        export_panel(decisions)
    elif cmd == "select-v2":
        select_v2(decisions)
    elif cmd == "final":
        with open(os.path.join(HERE, "decisions.jsonl"), "w", encoding="utf-8") as fh:
            for eid in sorted(decisions):
                fh.write(json.dumps(decisions[eid], ensure_ascii=False, sort_keys=True) + "\n")
        stats(decisions, C, F)
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
