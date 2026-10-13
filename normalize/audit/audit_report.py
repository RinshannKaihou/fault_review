#!/usr/bin/env python3
"""Write audit/REPORT.md from stored votes and decisions — counts and ids only, no record text.

    python3 audit_report.py
"""

import itertools
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_merge import BASE, CAT_FIELDS, FINAL, PANEL, TEXT_FIELDS, build, v2_subsets  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def read_jsonl(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return {}
    return {r["id"]: r for r in map(json.loads, (l for l in open(path, encoding="utf-8") if l.strip()))}


def pct(k, n):
    return f"{k}/{n}（{k / n:.0%}）" if n else "0/0"


def main():
    decisions, C, F = build()
    n = len(decisions)
    subs = v2_subsets()
    fixes = read_jsonl("fixes.jsonl")
    accepted = read_jsonl("accepted_fixes.jsonl")
    L = ["# 网关模型审计报告", ""]

    L += ["## 过程", "",
          f"- 基线：`baseline_v1.1_2026-10-12.tar.gz`（{n} 条，md5 见 `baseline.md5`）。审计只从基线增量应用，可重复执行。",
          f"- 第一层判定：{', '.join(BASE)}；评审团：{', '.join(PANEL)}。全程不用 Claude，编排者不读原文与判定内容。",
          "- 任务 C：只看原文、按 v1.2 词表盲分类；任务 F：对照原文检查标题/组件/原因行为/影响详情，引用片段由程序定位。",
          "- 投票规则：第一层需 ≥2 张有效票且一致；否则送评审团，7 票中至少 5 张有效且多数 ≥2/3。弱结论保留基线值并标为不确定。",
          f"- v1 缺陷：审核输入漏了条目元数据行，v1 有 736/850 个“有问题”字段引用的正是该行内容。v2 补上后重审：C 任务 {len(subs['C'])} 条、F 任务 {len(subs['F'])} 条；这些条目只计 v2 票。",
          "- glm-5.3（火山方舟 glm-5-3-260814）在 v1 慢速重试阶段按用户决定停止；kimi-k3 只接受 temperature=1。",
          ""]

    L += ["## 结论分布", "", "| 字段 | 确认 | 纠正 | 有问题 | 没问题 | 不确定 | 其中经评审团定论 |", "|---|---|---|---|---|---|---|"]
    for f in CAT_FIELDS + TEXT_FIELDS:
        c = Counter(d["fields"][f]["outcome"] for d in decisions.values())
        p7 = sum(1 for d in decisions.values() if d["fields"][f]["status"] == "panel7")
        L.append(f"| {f} | {c['confirmed'] or '—'} | {c['corrected'] or '—'} | {c['issue'] or '—'} | {c['ok'] or '—'} | {c['uncertain']} | {p7} |")
    L.append("")

    L += ["## 分类纠正（基线值 → 审计值，前 15）", ""]
    for f in ("subsystem", "effect", "visibility", "kind"):
        moves = Counter((d["fields"][f]["prior"], d["fields"][f]["final"]) for d in decisions.values()
                        if d["fields"][f]["outcome"] == "corrected")
        if moves:
            L.append(f"**{f}**（共 {sum(moves.values())}）：" + "；".join(f"{a} → {b} ×{k}" for (a, b), k in moves.most_common(15)))
            L.append("")

    n_issue = sum(1 for d in decisions.values() for f in TEXT_FIELDS if d["fields"][f]["outcome"] == "issue")
    n_fixed = sum(len(r.get("fixed", {})) for r in fixes.values() if not r.get("parse_failed"))
    n_acc = sum(len(r["fixed"]) for r in accepted.values())
    L += ["## 文本字段改写", "",
          f"- 被确认有问题的字段 {n_issue} 个；qwen3.8-max 产出改写 {n_fixed} 个；kimi-k3 + glm-5.3 一致复核通过并采纳 {n_acc} 个。",
          "- 未采纳的保留原文，在条目 `migration.audit.issue_unfixed` 中列出。", ""]

    L += ["## 判定模型一致性", "", "第一层两两一致率（任务 C，两者都有有效票的字段）：", ""]
    for a, b in itertools.combinations(BASE, 2):
        same = tot = 0
        for v in C["base"].values():
            ra, rb = v.get(a), v.get(b)
            if ra and rb and not ra.get("parse_failed") and not rb.get("parse_failed"):
                for f in CAT_FIELDS:
                    tot += 1
                    same += ra[f] == rb[f]
        L.append(f"- {a} × {b}：{pct(same, tot)}")
    L += ["", "评审团阶段各模型与 7 票多数的一致率（任务 C，仅 panel7 定论的字段）：", ""]
    agree = defaultdict(lambda: [0, 0])
    for d in decisions.values():
        for f in CAT_FIELDS:
            x = d["fields"][f]
            if x["status"] == "panel7":
                for m, v in x["votes"].items():
                    if v is not None:
                        agree[m][0] += v == x["final"]
                        agree[m][1] += 1
    for m, (k, t) in sorted(agree.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        L.append(f"- {m}：{pct(k, t)}")
    L.append("")

    unc = defaultdict(list)
    for eid, d in sorted(decisions.items()):
        for f in CAT_FIELDS + TEXT_FIELDS:
            if d["fields"][f]["outcome"] == "uncertain":
                unc[f].append(eid)
    L += ["## 仍不确定的字段（保留基线值）", ""]
    for f in CAT_FIELDS + TEXT_FIELDS:
        if unc[f]:
            L.append(f"- **{f}**（{len(unc[f])}）：{', '.join(unc[f])}")
    L.append("")
    with open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {os.path.join(HERE, 'REPORT.md')}")


if __name__ == "__main__":
    main()
