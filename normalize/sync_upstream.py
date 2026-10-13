#!/usr/bin/env python3
"""Bring Markdown entries the daily scan added after the migration into the upstream structured catalog.

  python3 normalize/sync_upstream.py --repo inference [--use-pilot] [--dry-run]

For every entry in the upstream Markdown master whose legacy id is not yet in catalog/id_registry.tsv:
  1. structured fields come from glm-5.3 (the daily-scan model), prompted only with the repo's own
     catalog/SCHEMA.md + vocab — exactly what the scan agent will read after cut-over
     (--use-pilot seeds the first draft with scan_pilot/outputs_repaired.jsonl when available);
  2. the narrative ("original") is taken verbatim from the Markdown entry, never from the model;
  3. the draft goes through `catalog_tool.py new --dry-run`; its errors are fed back to the model
     (up to --max-rounds), then the entry is written for real with --legacy-id (status pending_audit).
Prints counts only.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "audit"))
sys.path.insert(0, os.path.join(HERE, "scan_pilot"))
from fault_rag.parse import parse_master  # noqa: E402
from convert import RETIRED, coverage_of, meta_line  # noqa: E402

REPOS = {
    "inference": ("/workspace/inference_error_review", "/workspace/inference_error_review/FAULT_MASTER_REFERENCE.zh.md", "INF"),
    "training": ("/workspace/training_error_review", "/workspace/training_error_review/catalog/FAULT_MASTER_REFERENCE.zh.md", "TRN"),
}
FIELD_KEYS = {"机制": "mechanism", "来源": "source", "行为效应": "behavior", "触发面": "trigger_surface",
              "隐藏态签名": "signature", "复现方式": "repro", "发现来源": "discovery"}


def tool(root, *args, stdin=None):
    p = subprocess.run([sys.executable, os.path.join(root, "tools", "catalog_tool.py"), *args],
                       input=stdin, capture_output=True, text=True)
    try:
        return p.returncode, json.loads(p.stdout)
    except json.JSONDecodeError:
        return p.returncode, {"ok": False, "errors": [p.stdout[-500:] + p.stderr[-500:]]}


def system_prompt(root):
    import yaml
    voc = yaml.safe_load(open(os.path.join(root, "catalog", "vocab.yaml"), encoding="utf-8"))
    defs = lambda k: "\n".join(f"- {a}：{b}" for a, b in voc[k].items())
    return (open(os.path.join(root, "catalog", "SCHEMA.md"), encoding="utf-8").read()
            + f"\n\n## 受控词表\n\n### kind\n{defs('kind')}\n\n### subsystem\n{defs('subsystem')}\n\n### stack\n{', '.join(voc['stack'])}"
            + f"\n\n### effect.objects\n{defs('effect_object')}\n\n### visibility\n{defs('visibility')}\n\n### 判定规则\n"
            + "\n".join(f"- {r}" for r in voc["rules"])
            + "\n\n你现在要为下面这条已整理好的故障写结构化 JSON。`original` 由脚本从原文填写，你不用写。只输出一个 JSON 对象。")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", choices=list(REPOS), required=True)
    ap.add_argument("--model", default="zhipu/glm-5.3")
    ap.add_argument("--extra", default='{"reasoning_effort":"high"}')
    ap.add_argument("--use-pilot", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import audit_judge as J
    import run_pilot as P
    root, md, ns = REPOS[a.repo]
    J.MAX_TOKENS, J.EXTRA = 32768, json.loads(a.extra)
    registry = {}
    with open(os.path.join(root, "catalog", "id_registry.tsv"), encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            new_id, legacy = (line.rstrip("\n").split("\t") + [""])[:2]
            if legacy:
                registry[legacy] = new_id
    pilot = {}
    if a.use_pilot:
        path = os.path.join(HERE, "scan_pilot", "outputs_repaired.jsonl")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("error"):
                pilot[r["id"]] = r["entry"]

    system = system_prompt(root)
    stats, rounds = Counter(), Counter()
    for e in parse_master(md, a.repo):
        if e["id"] in registry or e["id"] in RETIRED:
            continue
        original = {"meta": meta_line(e["raw"])}
        original.update({FIELD_KEYS.get(k, k): v for k, v in e["fields"].items()})
        _, ref = coverage_of(e)
        hint = ""
        if ref:
            target = registry.get(ref)
            if not target:
                _, res = tool(root, "find", ref)
                target = next((h["id"] for h in res.get("hits", []) if h["slug"] == ref), None)
            hint = f"\n\n这条是已有条目 `{ref}` 的变体，coverage 写 var，relations 写 [{{\"kind\": \"var\", \"target\": \"{target or '待定'}\"}}]。"
        user = f"编目：{'推理' if a.repo == 'inference' else '训练'}\n\n下面是这条故障的来源整理（Markdown）：\n\n{e['raw']}{hint}"
        draft = pilot.get(f"PILOT-{ns}-{e['id']}")
        if draft is None:
            content, _ = J.call_api(a.model, [{"role": "system", "content": system}, {"role": "user", "content": user}])
            draft, _ = P.parse_json(content)
        ok = False
        for rnd in range(1, a.max_rounds + 1):
            if draft is None:
                break
            body = {k: v for k, v in draft.items() if k not in ("id", "evidence", "status", "migration", "original")}
            body["original"] = original
            code, res = tool(root, "new", "--dry-run", "--legacy-id", e["id"], "--allow-duplicate-source",
                             stdin=json.dumps(body, ensure_ascii=False, default=str))
            if code == 0:
                ok = True
                rounds[rnd] += 1
                break
            msg = (user + "\n\n你上次输出的 JSON：\n" + json.dumps(body, ensure_ascii=False, default=str)
                   + "\n\n`catalog_tool.py new` 校验未通过：\n" + "\n".join(f"- {m}" for m in res.get("errors", []))
                   + "\n\n只修正这些问题，输出完整的 JSON 对象。")
            content, _ = J.call_api(a.model, [{"role": "system", "content": system}, {"role": "user", "content": msg}])
            draft, _ = P.parse_json(content)
        if not ok:
            stats["failed"] += 1
            print(f"FAILED {e['id']}", file=sys.stderr)
            continue
        if a.dry_run:
            stats["would_write"] += 1
            continue
        code, res = tool(root, "new", "--legacy-id", e["id"], "--allow-duplicate-source",
                         stdin=json.dumps(body, ensure_ascii=False, default=str))
        stats["written" if code == 0 else "write_failed"] += 1
    print(json.dumps({"repo": a.repo, **stats, "passed_on_round": dict(sorted(rounds.items()))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
