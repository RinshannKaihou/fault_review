#!/usr/bin/env python3
"""Validate structured fault entries against SCHEMA.md and vocab.yaml.

Errors (exit 1): malformed IDs, duplicate IDs/legacy IDs, values outside the
strict vocabularies, dangling relation targets. Warnings: semi-controlled
stack values, over-long text, unresolved legacy relation references.

    python3 normalize/validate.py [entries_dir] [--quiet]

Always prints how many entries still carry a placeholder in each semantic
field, so fill progress is visible across migration passes.
"""

import argparse
import os
import re
import sys
from collections import Counter

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TBD = "待定"
ID_RE = re.compile(r"^(INF|TRN)-\d{4,}$")
DATE_RE = re.compile(r"^20\d\d-\d\d-\d\d$")
TOP_KEYS = ("id", "legacy_ids", "slug", "title", "kind", "status", "cause", "effect",
            "mitigation", "labels", "remarks", "evidence", "relations", "original")
LABEL_ENUMS = {
    "family": ("infra_off", "model_on", "ambiguous"),
    "trigger": ("config", "yes", "partial", "other_stack", "hw_specific", "no", "not_target"),
    "confidence": ("verified", "documented", "speculative"),
    "coverage": ("NEW", "var", "rationale", "existing"),
}
STAGES = ("pretrain", "sft", "rl", "shared")
STATUSES = ("active", "merged", "retired")
EVIDENCE_KINDS = ("github_issue", "github_pr", "arxiv", "paper", "other")


def load_vocab():
    with open(os.path.join(HERE, "vocab.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []
        self.unfilled = Counter()

    def err(self, eid, msg):
        self.errors.append(f"{eid}: {msg}")

    def warn(self, eid, msg):
        self.warnings.append(f"{eid}: {msg}")


def check_entry(e, fname, vocab, rep):
    ph = set(vocab["placeholders"])
    eid = e.get("id", fname)

    def is_ph(value):
        return isinstance(value, str) and value in ph

    def enum(value, allowed, field):
        if not is_ph(value) and not (isinstance(value, str) and value in allowed):
            rep.err(eid, f"{field}={value!r} 不在词表内")

    def unfilled(value, field):
        # only 待定 counts as unfilled; N/A is a deliberate answer
        if value == TBD or value == [] or value == [TBD]:
            rep.unfilled[field] += 1
            return True
        return False

    missing = [k for k in TOP_KEYS if k not in e]
    if missing:
        rep.err(eid, f"缺少字段 {missing}")
        return
    if not ID_RE.match(str(e["id"])):
        rep.err(eid, "id 格式应为 INF-0001 / TRN-0001")
    if fname != f"{e['id']}.yaml":
        rep.err(eid, f"文件名 {fname} 与 id 不一致")
    if e["kind"] != "待定":
        enum(e["kind"], vocab["kind"], "kind")
    unfilled(e["kind"], "kind")
    enum(e["status"], STATUSES, "status")
    if e["status"] == "merged" and not e.get("merged_into"):
        rep.err(eid, "status=merged 但缺 merged_into")
    if unfilled(e["title"], "title") is False and len(e["title"]) > 40:
        rep.warn(eid, f"title 超过 40 字（{len(e['title'])}）")

    # cause
    objs = e["cause"].get("objects")
    if is_ph(objs):
        rep.unfilled["cause.objects.subsystem"] += 1
    elif not isinstance(objs, list) or not objs:
        rep.err(eid, "cause.objects 应为非空列表或占位符")
    else:
        for o in objs:
            enum(o.get("subsystem"), vocab["subsystem"], "cause.objects.subsystem")
            if not is_ph(o.get("stack")) and o.get("stack") not in vocab["stack"]:
                rep.warn(eid, f"stack={o.get('stack')!r} 不在词表内（半受控）")
        if all(o.get("subsystem") == TBD for o in objs):
            rep.unfilled["cause.objects.subsystem"] += 1
    beh = e["cause"].get("behavior")
    if not unfilled(beh, "cause.behavior") and len(beh) > 120:
        rep.warn(eid, f"cause.behavior 超过 120 字（{len(beh)}）")

    # effect
    eobjs = e["effect"].get("objects")
    if is_ph(eobjs):
        rep.unfilled["effect.objects"] += 1
    elif not isinstance(eobjs, list) or not 1 <= len(eobjs) <= 3:
        rep.err(eid, "effect.objects 应为 1–3 项列表或占位符")
    else:
        for o in eobjs:
            enum(o, vocab["effect_object"], "effect.objects")
        if all(o == TBD for o in eobjs):
            rep.unfilled["effect.objects"] += 1
    det = e["effect"].get("detail")
    if not unfilled(det, "effect.detail") and len(det) > 150:
        rep.warn(eid, f"effect.detail 超过 150 字（{len(det)}）")
    enum(e["effect"].get("visibility"), vocab["visibility"], "effect.visibility")
    for k in ("detect", "fix"):
        if k not in e["mitigation"]:
            rep.err(eid, f"缺少 mitigation.{k}")

    # labels
    lab = e["labels"]
    for k, allowed in LABEL_ENUMS.items():
        enum(lab.get(k), allowed, f"labels.{k}")
    stage = lab.get("stage")
    if not is_ph(stage) and (not isinstance(stage, list) or not set(stage) <= set(STAGES)):
        rep.err(eid, f"labels.stage={stage!r} 非法")
    disc = lab.get("discovered")
    if not is_ph(disc) and not DATE_RE.match(str(disc)):
        rep.err(eid, f"labels.discovered={disc!r} 不是日期")

    # evidence: every entry needs at least one reference — a link, or a text citation when none exists
    ev = e["evidence"]
    if is_ph(ev) or not ev:
        rep.err(eid, "evidence 为空：至少需要一个来源链接或文字引用")
    else:
        for item in ev:
            if item.get("kind") == "citation":
                if not str(item.get("text", "")).strip():
                    rep.err(eid, f"citation 缺少 text: {item}")
            elif item.get("kind") not in EVIDENCE_KINDS or not str(item.get("url", "")).startswith("http"):
                rep.err(eid, f"evidence 项非法: {item}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="?", default=os.path.join(HERE, "out", "entries"))
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args()

    vocab = load_vocab()
    rep = Report()
    entries = {}
    for fname in sorted(os.listdir(args.entries)):
        if not fname.endswith(".yaml"):
            continue
        with open(os.path.join(args.entries, fname), encoding="utf-8") as fh:
            e = yaml.safe_load(fh)
        check_entry(e, fname, vocab, rep)
        entries[e.get("id")] = e

    legacy = Counter((eid[:3], old) for eid, e in entries.items() for old in e.get("legacy_ids", []))
    for (ns, old), n in legacy.items():
        if n > 1:
            rep.err(ns, f"旧编号 {old} 出现在 {n} 个条目中")
    for eid, e in entries.items():
        for r in e.get("relations", []):
            if r.get("kind") not in ("var", "cross"):
                rep.err(eid, f"relation kind 非法: {r}")
            if "target" in r and r["target"] not in entries:
                rep.err(eid, f"relation 目标不存在: {r['target']}")
            if "target_raw" in r:
                rep.warn(eid, f"{r['kind']} 目标无法解析: {r['target_raw']}")

    if not args.quiet:
        for m in rep.errors:
            print("ERROR", m)
        for m in rep.warnings:
            print("WARN ", m)
    n = len(entries)
    print(f"\n{n} entries · {len(rep.errors)} errors · {len(rep.warnings)} warnings")
    print("仍为待定:", ", ".join(f"{k} {v}/{n}" for k, v in rep.unfilled.most_common()))
    sys.exit(1 if rep.errors else 0)


if __name__ == "__main__":
    main()
