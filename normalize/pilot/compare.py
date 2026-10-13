#!/usr/bin/env python3
"""Agreement report between two independent extraction passes (A vs B)."""

import glob
import json
import os
from collections import Counter

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = yaml.safe_load(open(os.path.join(HERE, "..", "vocab.yaml"), encoding="utf-8"))
PH = set(VOCAB["placeholders"])


def load(pass_name):
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, f"out_{pass_name}[0-9].jsonl"))):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def subsystems(r):
    objs = r["cause"]["objects"]
    return [o["subsystem"] for o in objs] if isinstance(objs, list) else [objs]


def stacks(r):
    objs = r["cause"]["objects"]
    return [o.get("stack") for o in objs] if isinstance(objs, list) else [objs]


def effects(r):
    objs = r["effect"]["objects"]
    return objs if isinstance(objs, list) else [objs]


def vocab_errors(r):
    errs = []
    if r["kind"] not in VOCAB["kind"] and r["kind"] not in PH:
        errs.append(f"kind={r['kind']}")
    errs += [f"subsystem={s}" for s in subsystems(r) if s not in VOCAB["subsystem"] and s not in PH]
    errs += [f"effect={e}" for e in effects(r) if e not in VOCAB["effect_object"] and e not in PH]
    v = r["effect"].get("visibility")
    if v not in VOCAB["visibility"] and v not in PH:
        errs.append(f"visibility={v}")
    return errs


def main():
    import sys
    pa, pb = (sys.argv[1:3] if len(sys.argv) >= 3 else ("A", "B"))
    A, B = load(pa), load(pb)
    print(f"pass {pa} vs pass {pb}")
    ids = sorted(set(A) & set(B))
    print(f"entries: A={len(A)} B={len(B)} both={len(ids)}\n")

    checks = {
        "kind": lambda r: r["kind"],
        "subsystem (primary)": lambda r: subsystems(r)[0],
        "subsystem (set)": lambda r: frozenset(subsystems(r)),
        "stack (primary)": lambda r: stacks(r)[0],
        "effect (primary)": lambda r: effects(r)[0],
        "effect (set)": lambda r: frozenset(effects(r)),
        "visibility": lambda r: r["effect"].get("visibility"),
    }
    disagreements = {}
    print("| 字段 | A/B 一致 |\n|---|---|")
    for name, fn in checks.items():
        same = [i for i in ids if fn(A[i]) == fn(B[i])]
        disagreements[name] = [i for i in ids if i not in same]
        print(f"| {name} | {len(same)}/{len(ids)} ({len(same) / len(ids):.0%}) |")
    overlap = sum(1 for i in ids if set(effects(A[i])) & set(effects(B[i])))
    print(f"| effect (至少一项重合) | {overlap}/{len(ids)} ({overlap / len(ids):.0%}) |")

    for p, R in ((pa, A), (pb, B)):
        bad = {i: vocab_errors(r) for i, r in R.items() if vocab_errors(r)}
        print(f"\n词表外取值 pass {p}: {len(bad)} 条 {bad if bad else ''}")
        multi = sum(1 for r in R.values() if len(subsystems(r)) > 1)
        print(f"多对象 cause pass {p}: {multi}/{len(R)}")

    print("\n## subsystem 分歧")
    for i in disagreements["subsystem (primary)"]:
        print(f"- {i}: A={subsystems(A[i])} B={subsystems(B[i])}  《{A[i]['title']}》")
    print("\n## effect 主对象分歧")
    for i in disagreements["effect (primary)"]:
        print(f"- {i}: A={effects(A[i])} B={effects(B[i])}")
    print("\n## kind 分歧")
    for i in disagreements["kind"]:
        print(f"- {i}: A={A[i]['kind']} B={B[i]['kind']}")

    print("\n## 旧分类 → 新 subsystem（pass A）")
    inputs = {}
    for path in glob.glob(os.path.join(HERE, "in_A*.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            inputs[r["id"]] = r
    for i in ids:
        print(f"- {i} {inputs[i]['legacy_id']:>9}  {inputs[i]['old_section'][:22]:<24} → {subsystems(A[i])}")

    print("\n## vocab_gap 提议")
    gaps = Counter()
    for p, R in ((pa, A), (pb, B)):
        for i, r in R.items():
            if r.get("vocab_gap"):
                print(f"- [{p}] {i}: {r['vocab_gap']}")
                gaps[p] += 1
    print(f"\n共 {pa}={gaps[pa]} {pb}={gaps[pb]} 条提出词表缺口")


if __name__ == "__main__":
    main()
