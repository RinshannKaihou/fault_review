#!/usr/bin/env python3
"""Apply-only: rebuild out/entries from the frozen baseline plus audit decisions and accepted fixes.

  python3 audit_apply.py            # writes ../out/entries/*.yaml, prints counts
  python3 audit_apply.py --check    # rebuild into a temp dir and compare byte-for-byte with out/entries

Never re-runs judges or re-merges votes. Categorical fields take the final gateway decision
('unanimous' / 'panel7'); weak decisions keep the baseline value and are listed under
migration.audit.uncertain. Text fields take only fixes that passed verification.
"""

import argparse
import filecmp
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from build_items import load_baseline  # noqa: E402
from convert import dump_yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "entries")
ROUND = "gateway-audit-2026-10-12"
JUDGES = {"base": ["kimi-k3", "qwen3.8-max", "glm-5.3"], "panel": ["doubao-seed-2.1-pro", "deepseek-v4-pro", "MiniMax-M3", "hy3"],
          "fixer": "qwen3.8-max", "fix_verifiers": ["kimi-k3", "glm-5.3"]}


def read_jsonl(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return {}
    return {r["id"]: r for r in map(json.loads, (l for l in open(path, encoding="utf-8") if l.strip()))}


def apply_entry(e, d, fix):
    f = d["fields"]
    corrected, uncertain = {}, []
    for name in ("kind", "subsystem", "effect", "visibility"):
        x = f[name]
        if x["outcome"] == "corrected":
            corrected[name] = {"from": x["prior"], "to": x["final"], "status": x["status"]}
        elif x["outcome"] == "uncertain":
            uncertain.append(name)
    if "kind" in corrected:
        e["kind"] = f["kind"]["final"]
    if "subsystem" in corrected and isinstance(e["cause"]["objects"], list):
        e["cause"]["objects"][0]["subsystem"] = f["subsystem"]["final"]
    if "effect" in corrected:
        new = f["effect"]["final"]
        old = e["effect"]["objects"] if isinstance(e["effect"]["objects"], list) else []
        e["effect"]["objects"] = ([new] + [o for o in old if o != new])[:3]
    if "visibility" in corrected:
        e["effect"]["visibility"] = f["visibility"]["final"]

    rewritten = sorted(fix.get("fixed", {})) if fix else []
    for name, text in (fix or {}).get("fixed", {}).items():
        if name == "title":
            e["title"] = text
        elif name == "component":
            e["cause"]["objects"][0]["component"] = text
        elif name == "behavior":
            e["cause"]["behavior"] = text
        elif name == "detail":
            e["effect"]["detail"] = text
    issues = [n for n in ("title", "component", "behavior", "detail") if f[n]["outcome"] == "issue"]
    unfixed = [n for n in issues if n not in rewritten]
    text_uncertain = [n for n in ("title", "component", "behavior", "detail") if f[n]["outcome"] == "uncertain"]

    audit = {"round": ROUND}
    if corrected:
        audit["corrected"] = corrected
    if rewritten:
        audit["rewritten"] = rewritten
    if unfixed:
        audit["issue_unfixed"] = unfixed
    if uncertain or text_uncertain:
        audit["uncertain"] = uncertain + text_uncertain
    e.setdefault("migration", {})["audit"] = audit
    return e


def build(out_dir):
    baseline = load_baseline()
    decisions = read_jsonl("decisions.jsonl")
    fixes = read_jsonl("accepted_fixes.jsonl")
    if set(decisions) != set(baseline):
        raise SystemExit(f"decisions cover {len(decisions)} of {len(baseline)} baseline entries; run audit_merge.py final first")
    os.makedirs(out_dir, exist_ok=True)
    for eid in sorted(baseline):
        e = apply_entry(baseline[eid], decisions[eid], fixes.get(eid))
        with open(os.path.join(out_dir, f"{eid}.yaml"), "w", encoding="utf-8") as fh:
            fh.write(dump_yaml(e))
    return baseline, decisions, fixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        with tempfile.TemporaryDirectory() as tmp:
            build(tmp)
            names = sorted(os.listdir(tmp))
            _, mismatch, errors = filecmp.cmpfiles(tmp, OUT, names, shallow=False)
            print(f"idempotence check: {len(names)} files, {len(mismatch)} differ, {len(errors)} missing")
            sys.exit(1 if mismatch or errors else 0)
    baseline, decisions, fixes = build(OUT)
    print(f"applied {len(decisions)} decisions and {len(fixes)} accepted fixes onto {len(baseline)} baseline entries")


if __name__ == "__main__":
    main()
