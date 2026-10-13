#!/usr/bin/env python3
"""Summarise the full migration run and draw the human form-review sheet.

Writes run/REPORT.md (rates, distributions, audit error estimate, vocab gaps)
and run/form_review.md (a stratified sample for format/form checks only —
human reviewers do not judge technical correctness).

    python3 normalize/run_report.py
"""

import glob
import json
import math
import os
import random
from collections import Counter, defaultdict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run")
ENTRIES = os.path.join(HERE, "out", "entries")
FORM_SAMPLE, SEED = 30, 20261012


def read_jsonl(pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(RUN, pattern))):
        with open(path, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def refs(e):
    ev = e["evidence"] if isinstance(e["evidence"], list) else []
    shown = [x["url"] if x.get("url") else f"（引用）{x.get('text', '')}" for x in ev[:3]]
    more = f" 等 {len(ev)} 个" if len(ev) > 3 else ""
    return "；".join(shown) + more if shown else "（缺）"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(k, n):
    return f"{k}/{n}（{k / n:.0%}）" if n else "0/0"


def main():
    first = read_jsonl("out/*.jsonl")
    judged = read_jsonl("judged/*.jsonl")
    entries = [yaml.safe_load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(ENTRIES, "*.yaml")))]
    filled = [e for e in entries if e["title"] != "待定"]

    lines = ["# 全量迁移运行报告", ""]
    lines += ["## 覆盖", ""]
    methods = Counter(e.get("migration", {}).get("method", "未合并") for e in entries)
    lines += [f"- 条目总数 {len(entries)}，语义字段已填 {len(filled)}", f"- 来源：{dict(methods)}"]
    tbd = Counter()
    for e in filled:
        if e["kind"] == "待定":
            tbd["kind"] += 1
        if all(o.get("subsystem") == "待定" for o in e["cause"]["objects"]):
            tbd["原因子系统"] += 1
        if e["effect"]["objects"] in ("待定", ["待定"]):
            tbd["影响对象"] += 1
        if e["effect"].get("visibility") == "待定":
            tbd["可见性"] += 1
    lines += [f"- 已填条目中仍为 `待定` 的字段（原文缺信息）：{dict(tbd) or '无'}", ""]

    unc = [r for r in first if r.get("uncertain")]
    adjudicated = [r for r in judged if "uncertain" in r.get("judge", {}).get("reason", [])]
    changed = [r for r in adjudicated if r["judge"].get("changed")]
    field_changes = Counter(f for r in changed for f in r["judge"]["changed"])
    lines += ["## 第一遍抽取与裁定", ""]
    lines += [
        f"- 第一遍标为拿不准：{pct(len(unc), len(first))}",
        f"- 裁定后改动：{pct(len(changed), len(adjudicated))}，改动字段 {dict(field_changes)}",
        "",
    ]

    audited = [r for r in judged if "audit" in r.get("judge", {}).get("reason", [])]
    with_issue = [r for r in audited if r["judge"].get("audit_issues")]
    issue_types = Counter(i.get("type") for r in audited for i in r["judge"].get("audit_issues", []))
    lo, hi = wilson(len(with_issue), len(audited))
    lines += ["## 忠实性审计（随机 5%）", ""]
    lines += [
        f"- 审计 {len(audited)} 条，发现问题并已改正 {pct(len(with_issue), len(audited))}；"
        f"按 95% Wilson 区间，未审计条目的问题率估计在 {lo:.0%}–{hi:.0%}",
        f"- 问题类型：{dict(issue_types) or '无'}",
    ]
    for r in with_issue:
        for i in r["judge"]["audit_issues"]:
            lines.append(f"  - {r['id']} `{i.get('field')}` [{i.get('type')}] {i.get('detail')}")
    lines.append("")

    lines += ["## 分布", ""]
    sub = Counter(e["cause"]["objects"][0]["subsystem"] for e in filled if isinstance(e["cause"]["objects"], list))
    lines += ["| 原因子系统 | 条数 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sub.most_common()] + [""]
    eff = Counter(e["effect"]["objects"][0] for e in filled if isinstance(e["effect"]["objects"], list))
    lines += ["| 影响主对象 | 条数 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in eff.most_common()] + [""]
    kinds = Counter(e["kind"] for e in filled)
    vis = Counter(e["effect"].get("visibility") for e in filled)
    lines += [f"- kind：{dict(kinds)}", f"- 可见性：{dict(vis)}", ""]

    cross = defaultdict(Counter)
    for e in filled:
        cross[e["original"]["category"]][e["cause"]["objects"][0]["subsystem"]] += 1
    lines += ["## 旧分类 → 新原因子系统", "", "| 旧分类 | 条数 | 新子系统分布（前 4） |", "|---|---|---|"]
    for cat, c in sorted(cross.items(), key=lambda kv: -sum(kv[1].values())):
        top = "、".join(f"{k} {v}" for k, v in c.most_common(4))
        lines.append(f"| {cat} | {sum(c.values())} | {top} |")
    lines.append("")

    gaps = [(r["id"], r["vocab_gap"]) for r in first if r.get("vocab_gap")]
    lines += [f"## 词表缺口（第一遍提出 {len(gaps)} 条）", ""] + [f"- {i}: {g}" for i, g in gaps] + [""]
    with open(os.path.join(RUN, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # form review sheet: stratified by old category, deterministic
    rng = random.Random(SEED)
    by_cat = defaultdict(list)
    for e in filled:
        by_cat[e["original"]["category"]].append(e)
    pool = [rng.choice(v) for _, v in sorted(by_cat.items())]
    rest = [e for e in filled if e not in pool]
    sample = (pool + rng.sample(rest, max(0, FORM_SAMPLE - len(pool))))[:FORM_SAMPLE]
    sheet = [
        "# 人工形式检查清单",
        "",
        "只检查格式和写法，**不需要判断技术内容是否正确**（正确性已由独立裁定和忠实性审计负责）。",
        "每条在不符合的项后标 ✗：",
        "",
        "- A 标题是一句话、能看懂在说什么",
        "- B 原因行为只写缺陷本身，没有混入后果",
        "- C 影响详情只写外部表现和触发条件，没有重复原因",
        "- D 原因对象/影响对象的取值看起来与文字描述相称（凭字面判断即可）",
        "- E 没有乱码、截断、中英混排失控等形式问题",
        "- F 有来源链接或文字引用",
        "",
    ]
    for e in sample:
        objs = "；".join(f"{o['subsystem']} / {o.get('stack')} / {o.get('component')}" for o in e["cause"]["objects"])
        eff = "、".join(e["effect"]["objects"]) if isinstance(e["effect"]["objects"], list) else e["effect"]["objects"]
        sheet += [
            f"### {e['id']}（旧号 {e['legacy_ids'][0]}）{e['title']}",
            "",
            f"- 原因对象：{objs}",
            f"- 原因行为：{e['cause']['behavior']}",
            f"- 影响对象：{eff}（{e['effect'].get('visibility')}）",
            f"- 影响详情：{e['effect']['detail']}",
            f"- 来源：{refs(e)}",
            "- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐",
            "",
        ]
    with open(os.path.join(RUN, "form_review.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sheet))
    print(f"wrote {os.path.join(RUN, 'REPORT.md')} and form_review.md ({len(sample)} entries)")


if __name__ == "__main__":
    main()
