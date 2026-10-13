#!/usr/bin/env python3
"""Offline pilot: can the daily-scan model write new catalog entries in the structured YAML format?

  python3 run_pilot.py build                     # inputs.jsonl: 21 fresh upstream entries + 40 audited reference entries
  python3 run_pilot.py run [--model zhipu/glm-5.3] [--workers 8] [--extra '{"reasoning_effort":"medium"}']
                         [--limit N] [--out outputs.jsonl]            # resumable by id
  python3 run_pilot.py repair [--model ...]       # one validator-feedback round -> outputs_repaired.jsonl
  python3 run_pilot.py judge-items               # f_items.jsonl / c_items_fresh.jsonl for audit_judge.py
  python3 run_pilot.py evaluate                  # REPORT.md: format, labels, agreement, faithfulness (counts only)

Input per entry is the Markdown entry exactly as the daily scan wrote it (heading, metadata line,
fields). The model writes the semantic part of the new YAML entry; mechanical fields (id, evidence,
original text) are filled by the harness, as a helper script would in production.
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NORM = os.path.dirname(HERE)
sys.path.insert(0, NORM)
sys.path.insert(0, os.path.dirname(NORM))
sys.path.insert(0, os.path.join(NORM, "audit"))
from fault_rag.parse import parse_master  # noqa: E402
from convert import SOURCES, coverage_of, evidence_of, load_registry, meta_line  # noqa: E402

SEED, N_REF = 20261013, 40
VOCAB = yaml.safe_load(open(os.path.join(NORM, "vocab.yaml"), encoding="utf-8"))


def read_jsonl(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return {}
    return {r["id"]: r for r in map(json.loads, (l for l in open(path, encoding="utf-8") if l.strip()))}


def render_vocab():
    sec = lambda k: "\n".join(f"- {a}：{b}" for a, b in VOCAB[k].items())
    return (f"## 受控词表\n\n### kind\n{sec('kind')}\n\n### subsystem（判定口诀：问“修复要改哪里”）\n{sec('subsystem')}\n\n"
            f"### stack\n{', '.join(VOCAB['stack'])}\n\n### effect.objects\n{sec('effect_object')}\n\n"
            f"### visibility\n{sec('visibility')}\n\n### 判定规则\n" + "\n".join(f"- {r}" for r in VOCAB["rules"]))


def cmd_build():
    registry = load_registry()
    by_slug = {}
    for p in glob.glob(os.path.join(NORM, "out", "entries", "*.yaml")):
        e = yaml.safe_load(open(p, encoding="utf-8"))
        by_slug[e["slug"]] = e["id"]
    decisions = read_jsonl("../audit/decisions.jsonl")
    fresh, pool = [], []
    for ns, repo, path in SOURCES:
        for e in parse_master(path, repo):
            new_id = registry.get((ns, e["id"]))
            cov, ref = coverage_of(e)
            hint = None
            if ref:
                hint = registry.get((ns, ref)) or by_slug.get(ref)
            row = {"repo": repo, "legacy_id": e["id"], "markdown": e["raw"], "var_ref": ref, "var_target": hint}
            if new_id is None:
                fresh.append({"id": f"PILOT-{ns}-{e['id']}", "group": "fresh", **row})
            else:
                d = decisions.get(new_id)
                if d and all(d["fields"][f]["status"] in ("unanimous", "panel7") for f in ("subsystem", "effect", "visibility")):
                    pool.append({"id": new_id, "group": "reference", **row})
    # reference: the most recently discovered audited entries (closest to what the scan writes today);
    # a seeded shuffle breaks ties within a date
    def discovered(r):
        m = re.search(r"20\d\d-\d\d-\d\d", r["markdown"].split("发现来源", 1)[-1])
        return m.group(0) if m else ""
    random.Random(SEED).shuffle(pool)
    pool.sort(key=discovered, reverse=True)
    ref = [r for r in pool if r["repo"] == "training"][:8] + [r for r in pool if r["repo"] == "inference"][:N_REF - 8]
    with open(os.path.join(HERE, "inputs.jsonl"), "w", encoding="utf-8") as fh:
        for r in fresh + ref:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"inputs: fresh {len(fresh)}, reference {len(ref)}")


def user_msg(r):
    hint = ""
    if r["var_ref"]:
        hint = (f"\n\n该条目是已有条目 `{r['var_ref']}` 的变体，对应新编号："
                f"{r['var_target'] or '待定（未找到对应条目）'}。")
    return f"编目：{'推理' if r['repo'] == 'inference' else '训练'}\n\n下面是这条故障的来源整理（Markdown）：\n\n{r['markdown']}{hint}"


def parse_json(content):
    if not content:
        return None, "no_content"
    m = re.search(r"```(?:json)?\s*\n(.*?)```", content, re.S)
    text = m.group(1) if m else content[content.find("{"): content.rfind("}") + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"json_error: {exc.msg} line {exc.lineno} col {exc.colno}"
    return (obj, None) if isinstance(obj, dict) else (None, "not_mapping")


def parse_yaml(content):
    if not content:
        return None, "no_content"
    m = re.search(r"```(?:yaml)?\s*\n(.*?)```", content, re.S)
    text = m.group(1) if m else content
    try:
        obj = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" line {mark.line + 1} col {mark.column + 1}" if mark else ""
        return None, f"yaml_error: {getattr(exc, 'problem', exc)}{where}"
    return (obj, None) if isinstance(obj, dict) else (None, "not_mapping")


def cmd_run(model, workers, extra, limit, out_name, fmt):
    import audit_judge as J
    prompt_file = "entry_prompt_json.md" if fmt == "json" else "entry_prompt.md"
    system = open(os.path.join(HERE, prompt_file), encoding="utf-8").read().replace("{{VOCAB}}", render_vocab())
    parser = parse_json if fmt == "json" else parse_yaml
    inputs = read_jsonl("inputs.jsonl")
    done = read_jsonl(out_name)
    todo = [r for i, r in inputs.items() if i not in done or done[i].get("error")][:limit]
    J.MAX_TOKENS, J.EXTRA = 32768, extra
    import time
    lock, st = threading.Lock(), Counter()
    print(f"run {model}: todo {len(todo)}")

    def work(r):
        t0 = time.time()
        content, usage = J.call_api(model, [{"role": "system", "content": system}, {"role": "user", "content": user_msg(r)}])
        obj, err = parser(content)
        rec = {"id": r["id"], "model": model, "extra": extra, "format": fmt, "error": err, "entry": obj, "seconds": round(time.time() - t0),
               "raw": content if err and content else None,   # kept only for format diagnosis
               "out_tokens": usage.get("completion_tokens"), "finish": usage.get("_finish"),
               "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")   # YAML turns bare dates into date objects
            fh.flush()
            st[err or "ok"] += 1

    with open(os.path.join(HERE, out_name), "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(work, r) for r in todo]):
            f.result()
    print(dict(st))


def assemble(r, out, seq):
    """Full entry as production would store it: model output + harness-filled mechanical fields.
    Pilot entries get placeholder ids INF-9xxxx / TRN-9xxxx so they can never collide with real ones."""
    ns = "INF" if r["repo"] == "inference" else "TRN"
    entry = {"id": f"{ns}-{90000 + seq}", "legacy_ids": [], "aliases": [],
             "status": "active", "remarks": "", "relations": [], "mitigation": {"detect": "待定", "fix": "待定"}}
    entry.update({k: v for k, v in (out or {}).items() if k not in ("id", "evidence", "original")})
    fields = parse_master_fields(r["markdown"])
    entry["evidence"] = evidence_of(fields.get("来源", "")) or "待定"
    entry["original"] = {"meta": meta_line(r["markdown"]), **fields}
    return entry


def parse_master_fields(md):
    out, cur = {}, None
    for line in md.splitlines():
        m = re.match(r"^- \*\*(.+?)\*\*[：:]\s*(.*)$", line)
        if m:
            cur = m.group(1).strip()
            out[cur] = m.group(2).strip()
        elif cur and line.strip() and not line.startswith("###"):
            out[cur] += "\n" + line.rstrip()
    return out


def text_fields(entry):
    objs = entry.get("cause", {}).get("objects") if isinstance(entry.get("cause"), dict) else None
    objs = objs if isinstance(objs, list) else []
    return {
        "title": str(entry.get("title", "")),
        "component": "；".join(str(o.get("component", "")) for o in objs if isinstance(o, dict)),
        "behavior": str((entry.get("cause") or {}).get("behavior", "")),
        "detail": str((entry.get("effect") or {}).get("detail", "")),
    }


LIMITS = (("title", 40), ("component", 25), ("behavior", 120), ("detail", 150))


def violations(r, entry, seq=1):
    """What the production helper script would bounce back to the agent: schema/vocab errors + length overruns."""
    import validate as V
    rep = V.Report()
    e = assemble(r, entry, seq)
    V.check_entry(e, f"{e['id']}.yaml", V.load_vocab(), rep)
    out = [m.split(": ", 1)[1] for m in rep.errors]
    for f, v in text_fields(entry).items():
        lim = dict(LIMITS)[f]
        for part in (v.split("；") if f == "component" else [v]):
            if len(part) > lim:
                out.append(f"{f} 长 {len(part)} 字，上限 {lim} 字")
    if not (entry.get("effect") or {}).get("visibility"):
        out.append("缺少 effect.visibility")
    return sorted(set(out))


def cmd_repair(model, extra):
    import audit_judge as J
    import time
    system = open(os.path.join(HERE, "entry_prompt_json.md"), encoding="utf-8").read().replace("{{VOCAB}}", render_vocab())
    inputs, outputs = read_jsonl("inputs.jsonl"), read_jsonl("outputs.jsonl")
    done = read_jsonl("outputs_repaired.jsonl")
    J.MAX_TOKENS, J.EXTRA = 32768, extra
    lock, st = threading.Lock(), Counter()

    def work(i):
        r, o = inputs[i], outputs[i]
        problems = violations(r, o["entry"]) if not o.get("error") else ["上次输出无法解析"]
        if not problems:
            rec = dict(o, repaired=False, problems_before=[])
        else:
            t0 = time.time()
            msg = (user_msg(r) + "\n\n你上次输出的 JSON：\n" + json.dumps(o.get("entry"), ensure_ascii=False, default=str)
                   + "\n\n辅助脚本校验未通过：\n" + "\n".join(f"- {p}" for p in problems)
                   + "\n\n只修正这些问题（缩短时保留关键事实与触发条件，不新增事实），输出完整的 JSON 对象。")
            content, usage = J.call_api(model, [{"role": "system", "content": system}, {"role": "user", "content": msg}])
            obj, err = parse_json(content)
            rec = {"id": i, "model": model, "extra": extra, "format": "json", "error": err, "entry": obj if not err else o.get("entry"),
                   "seconds": round(time.time() - t0), "out_tokens": usage.get("completion_tokens"),
                   "repaired": True, "problems_before": problems}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            st["repaired" if rec["repaired"] else "clean"] += 1

    todo = [i for i in inputs if i in outputs and i not in done]
    with open(os.path.join(HERE, "outputs_repaired.jsonl"), "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(work, i) for i in todo]):
            f.result()
    print(dict(st))


def cmd_judge_items(out_name="outputs.jsonl", suffix=""):
    inputs, outputs = read_jsonl("inputs.jsonl"), read_jsonl(out_name)
    with open(os.path.join(HERE, f"f_items{suffix}.jsonl"), "w", encoding="utf-8") as ff, \
         open(os.path.join(HERE, f"c_items_fresh{suffix}.jsonl"), "w", encoding="utf-8") as fc:
        for i, r in inputs.items():
            o = outputs.get(i)
            fields = parse_master_fields(r["markdown"])
            original = {"meta": meta_line(r["markdown"]), **fields}
            if o and not o.get("error"):
                ff.write(json.dumps({"id": i, "original": original, "structured": text_fields(o["entry"])}, ensure_ascii=False) + "\n")
            if r["group"] == "fresh":
                fc.write(json.dumps({"id": i, "original": original, "structured": {}}, ensure_ascii=False) + "\n")
    print(f"wrote f_items{suffix}.jsonl and c_items_fresh{suffix}.jsonl")


def cmd_evaluate(out_name="outputs.jsonl", f_tag="_scanpilot", report="REPORT.md"):
    import validate as V
    from audit_merge import decide_bin, decide_cat, load_votes
    inputs, outputs = read_jsonl("inputs.jsonl"), read_jsonl(out_name)
    decisions = read_jsonl("../audit/decisions.jsonl")
    vocab = V.load_vocab()
    L = [f"# 每日扫描结构化写入：离线试点报告（{out_name}）", ""]
    over = Counter(p.split(" 长")[0] for i, o in outputs.items() if not o.get("error") for p in violations(inputs[i], o["entry"]) if " 长 " in p)
    miss_vis = sum(1 for o in outputs.values() if not o.get("error") and not (o["entry"].get("effect") or {}).get("visibility"))
    errs = Counter(o.get("error") or "ok" for o in outputs.values())
    toks = sorted(o["out_tokens"] for o in outputs.values() if o.get("out_tokens"))
    L += [f"- 模型：{next(iter(outputs.values()))['model'] if outputs else '?'}；条目 {len(inputs)}（新鲜 {sum(r['group'] == 'fresh' for r in inputs.values())}、参考 {sum(r['group'] == 'reference' for r in inputs.values())}）",
          f"- 输出可解析：{dict(errs)}；输出 token 中位数 {toks[len(toks) // 2] if toks else '-'}，最大 {toks[-1] if toks else '-'}", ""]

    # schema / vocab / length
    rep, n_ok = V.Report(), 0
    for seq, (i, o) in enumerate(sorted(outputs.items()), 1):
        if o.get("error"):
            continue
        e = assemble(inputs[i], o["entry"], seq)
        before = len(rep.errors)
        V.check_entry(e, f"{e['id']}.yaml", vocab, rep)
        n_ok += len(rep.errors) == before
    err_kinds = Counter(re.sub(r"=.*", "", m.split(": ", 1)[1]) for m in rep.errors)
    warn_kinds = Counter(re.sub(r"（.*|=.*", "", m.split(": ", 1)[1]) for m in rep.warnings)
    L += ["## 格式与词表", "", f"- 通过校验（0 错误）：{n_ok}/{sum(1 for o in outputs.values() if not o.get('error'))}",
          f"- 超长字段数：{dict(over) or '无'}；缺 visibility：{miss_vis}",
          f"- 错误类型：{dict(err_kinds) or '无'}", f"- 告警类型（含超长）：{dict(warn_kinds) or '无'}", ""]

    # labels vs the scan's own metadata line (mechanical truth)
    lab_ok, lab_tot, lab_bad = 0, 0, Counter()
    for i, o in outputs.items():
        if o.get("error"):
            continue
        r = inputs[i]
        meta = meta_line(r["markdown"])
        lab = (o["entry"] or {}).get("labels") or {}
        conf = re.search(r"置信度\s*`([^`]+)`", meta)
        expect = {"confidence": conf.group(1) if conf else None,
                  "coverage": ("var" if "var:" in meta else "NEW" if "`NEW" in meta else None)}
        for k, v in expect.items():
            if v is None:
                continue
            lab_tot += 1
            if lab.get(k) == v:
                lab_ok += 1
            else:
                lab_bad[k] += 1
    L += ["## 标签（与扫描自己写的元数据行对照）", "", f"- confidence / coverage 一致：{lab_ok}/{lab_tot}；不一致 {dict(lab_bad) or '无'}", ""]

    # classification agreement with audited values (reference group)
    def first(x):
        return x[0] if isinstance(x, list) and x else x
    agree = Counter()
    nref = 0
    for i, o in outputs.items():
        r = inputs[i]
        if r["group"] != "reference" or o.get("error"):
            continue
        nref += 1
        d, e = decisions[i]["fields"], o["entry"] or {}
        objs = (e.get("cause") or {}).get("objects") or [{}]
        got = {"kind": e.get("kind"), "subsystem": (objs[0] or {}).get("subsystem"),
               "effect": first((e.get("effect") or {}).get("objects")), "visibility": (e.get("effect") or {}).get("visibility")}
        for f, v in got.items():
            if d[f]["status"] in ("unanimous", "panel7"):
                agree[(f, "tot")] += 1
                agree[(f, "ok")] += v == d[f]["final"]
    L += ["## 分类与审计结论的一致率（参考组）", ""] + [
        f"- {f}：{agree[(f, 'ok')]}/{agree[(f, 'tot')]}（{agree[(f, 'ok')] / max(agree[(f, 'tot')], 1):.0%}）"
        for f in ("kind", "subsystem", "effect", "visibility")] + [""]

    # fresh group: agreement with the 3-judge gateway decision (task C, tag _scanpilot)
    cv = load_votes("C", "_scanpilot", ("qwen3.8-max", "kimi-k3", "doubao-seed-2.1-pro"))
    fa = Counter()
    for i, o in outputs.items():
        if inputs[i]["group"] != "fresh" or o.get("error") or i not in cv:
            continue
        e = o["entry"] or {}
        objs = (e.get("cause") or {}).get("objects") or [{}]
        got = {"kind": e.get("kind"), "subsystem": (objs[0] or {}).get("subsystem"),
               "effect": first((e.get("effect") or {}).get("objects")), "visibility": (e.get("effect") or {}).get("visibility")}
        for f, v in got.items():
            final, status, _ = decide_cat([r.get(f) for r in cv[i].values() if not r.get("parse_failed")], panel=False)
            if status == "unanimous":
                fa[(f, "tot")] += 1
                fa[(f, "ok")] += v == final
    L += ["## 分类与网关三票一致结论的一致率（新鲜组，仅三票一致的字段）", ""] + [
        f"- {f}：{fa[(f, 'ok')]}/{fa[(f, 'tot')]}" for f in ("kind", "subsystem", "effect", "visibility")] + [""]

    # faithfulness of the written text fields (task F, tag _scanpilot)
    fv = load_votes("F", f_tag, ("qwen3.8-max", "kimi-k3", "doubao-seed-2.1-pro"))
    fc = Counter()
    for i, votes in fv.items():
        for f in ("title", "component", "behavior", "detail"):
            issue, status, _ = decide_bin([r["votes"].get(f) if not r.get("parse_failed") else None for r in votes.values()], panel=False)
            fc[(f, "issue" if status == "unanimous" and issue else "ok" if status == "unanimous" else "split")] += 1
    L += ["## 文本字段忠实性（网关三票）", "", "| 字段 | 一致没问题 | 一致有问题 | 分歧 |", "|---|---|---|---|"] + [
        f"| {f} | {fc[(f, 'ok')]} | {fc[(f, 'issue')]} | {fc[(f, 'split')]} |" for f in ("title", "component", "behavior", "detail")] + [""]
    with open(os.path.join(HERE, report), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "repair", "judge-items", "evaluate"])
    ap.add_argument("--model", default="zhipu/glm-5.3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra", default="{}", help="JSON merged into each request, e.g. reasoning settings")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="outputs.jsonl")
    ap.add_argument("--format", choices=["json", "yaml"], default="json", help="what the model is asked to emit")
    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build()
    elif a.cmd == "run":
        cmd_run(a.model, a.workers, json.loads(a.extra), a.limit, a.out, a.format)
    elif a.cmd == "repair":
        cmd_repair(a.model, json.loads(a.extra))
    elif a.cmd == "judge-items":
        cmd_judge_items(a.out, "_repaired" if "repaired" in a.out else "")
    else:
        repaired = "repaired" in a.out
        cmd_evaluate(a.out, "_scanpilot2" if repaired else "_scanpilot", "REPORT_repaired.md" if repaired else "REPORT.md")


if __name__ == "__main__":
    main()
