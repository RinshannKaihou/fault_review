#!/usr/bin/env python3
"""解析编目里悬空的 var 关系目标（阶段 A）。

45 条 `var` 关系的 `target` 为空，但都留着 `target_raw`——迁移时写下的 snake_case
目标名，没能解析成条目编号。`validate` 一直把它们报成 warning
（`<id>: var 目标无法解析 <target_raw>`）。

本工具按 llm-judge-labeling 协议判定「`target_raw` 指的是不是候选条目，且 var
（变体）关系成立」：3 个基础判定模型投票（法定 2、多数 ≥2/3），分歧交 4 个评审团
模型（7 票、法定 5）。编排者只看计数，不读条目正文和判定理由。

写入不由本工具完成：`apply` 只生成 `catalog_tool.py update --relation` 命令，
真源仍只经 catalog_tool 修改。

    items   从两个编目构建待判条目，按匹配形态分桶
    judge   跑判定（可按桶、可续跑，按 (条目, 模型) 去重）
    decide  汇总投票 → 决定与状态
    apply   生成（或执行）catalog_tool 命令
"""

import argparse
import collections
import difflib
import glob
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

API = "https://st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com/v1/chat/completions"
BASE = ("qwen3.8-max", "kimi-k3", "doubao-seed-2.1-pro")
PANEL = ("deepseek-v4-pro", "MiniMax-M3", "hy3", "glm-5.3")
# kimi-k3 只接受 temperature=1；doubao 的 reasoning 预算要压低才不吃满 max_tokens。
MODEL_OPTS = {"kimi-k3": {"temperature": 1.0}, "doubao-seed-2.1-pro": {"reasoning_effort": "low"}}
QUORUM_BASE, QUORUM_PANEL, MAJ = 2, 5, 2 / 3
FINAL = ("unanimous", "panel7")
REPOS = {"inf": "/workspace/inference_error_review", "trn": "/workspace/training_error_review"}
HERE = Path(__file__).resolve().parent

SYSTEM = """你在核对一份「LLM 推理/训练故障模式编目」里的变体关系。

编目里有一条故障条目 S 被标为某条目的**变体**（var），迁移时只留下了目标的
snake_case 名字（target_raw），没留编号。现在给你一个候选条目 T，请判断：
**target_raw 指的就是 T，并且「S 是 T 的变体」这个断言成立吗？**

判 yes 的条件（两条都要满足）：
1. target_raw 这个名字描述的就是 T 这条条目；
2. S 与 T 共享**同一个具体失效机制**——同一处代码/同一个错误假设/同一条数据通路，
   S 只是它在不同触发面、不同栈、不同配置下的另一种显形。

**不要**因为下列理由判 yes（这些都是近似但不够的情形）：
- 仅仅是 slug 字符串像。名字相近但机制不同的条目在本编目里是存在的；
- 仅仅同属一个宽泛主题（都关于 KV 缓存、都关于量化、都是「版本契约」问题）；
- 仅仅是同一个仓库、同一个算子家族、同一类症状（都输出乱码、都静默错值）；
- 两条互为**因果**或**上下游**，而不是同一机制的变体；
- T 看起来比 S 更一般或更特殊，但两者的失效点并不是同一处。

拿不准就填 unsure，不要猜。宁可 unsure 也不要把不同机制连成变体。

只输出一个 JSON 对象，不要任何其它文字：
{"verdict": "yes|no|unsure", "shared_mechanism": "<两条共享的具体机制，一句话；no/unsure 时写两者的关键差异>"}"""


SYSTEM_LEGACY = """你在核对一份「LLM 推理/训练故障模式编目」里的变体关系。

编目里有一条故障条目 S 被标为某条目的**变体**（var），但目标编号丢了。候选条目 T
是 S 自己的原文里引用过的条目。请判断：**「S 是 T 的变体」这个断言成立吗？**

判 yes 的条件：S 与 T 共享**同一个具体失效机制**——同一处代码/同一个错误假设/
同一条数据通路，S 只是它在不同触发面、不同栈、不同配置下的另一种显形。

**不要**因为下列理由判 yes（这些都是近似但不够的情形）：
- S 的原文只是**提到**或**对比**了 T（「与 T 不同」「区别于 T」「T 的根因不同」）
  ——被引用不等于是其变体，这一条最常见，务必先排除；
- 仅仅同属一个宽泛主题（都关于 KV 缓存、都关于量化、都是「版本契约」问题）；
- 仅仅是同一个仓库、同一个算子家族、同一类症状（都输出乱码、都静默错值）；
- 两条互为**因果**或**上下游**，而不是同一机制的变体；
- T 看起来比 S 更一般或更特殊，但两者的失效点并不是同一处。

拿不准就填 unsure，不要猜。宁可 unsure 也不要把不同机制连成变体。

只输出一个 JSON 对象，不要任何其它文字：
{"verdict": "yes|no|unsure", "shared_mechanism": "<两条共享的具体机制，一句话；no/unsure 时写两者的关键差异>"}"""


# ------------------------------------------------------------------ catalog

def load():
    cat, slug2 = {}, {}
    for repo, path in REPOS.items():
        for f in sorted(glob.glob(path + "/catalog/entries/*.yaml")):
            e = yaml.safe_load(open(f))
            e["_repo"], e["_file"] = repo, f
            cat[e["id"]] = e
            if e.get("slug"):
                slug2.setdefault(e["slug"], []).append(e["id"])
    return cat, slug2


def render(e):
    """条目的判定视图：结构化摘要 + 迁移/扫描写下的原始叙述。"""
    s, c, eff = [], e.get("cause") or {}, e.get("effect") or {}
    s.append(f"编号: {e['id']}   名称: {e.get('slug')}")
    s.append(f"标题: {e.get('title')}")
    obj = (c.get("objects") or [{}])[0]
    s.append(f"原因: 子系统={obj.get('subsystem')} 栈={obj.get('stack')} 部件={obj.get('component')}")
    s.append(f"原因行为: {c.get('behavior')}")
    s.append(f"影响: {'/'.join(str(x) for x in (eff.get('objects') or []))}  可见性: {eff.get('visibility')}")
    s.append(f"影响详情: {eff.get('detail')}")
    for k, v in (e.get("original") or {}).items():
        if k in ("mechanism", "behavior", "trigger_surface"):
            s.append(f"原始{k}: {v}")
    return "\n".join(str(x) for x in s)


def user_msg(cat, src, cand, raw=None):
    head = f"target_raw（迁移时写下的目标名）: {raw}\n\n" if raw else ""
    return (head + f"<<<条目 S（被标为变体的那条）\n{render(cat[src])}\nS>>>\n\n"
            f"<<<候选条目 T\n{render(cat[cand])}\nT>>>")


def system_for(source):
    return SYSTEM if source == "raw" else SYSTEM_LEGACY


# ------------------------------------------------------------------ items

LEGACY_NUM = re.compile(r"(?<![\d.])(\d{1,2}\.\d{1,3})(?![\d.])|(?<![A-Z.])([A-Z]{2,4}\.\d{1,3})(?![\d.])")


def legacy_index(cat):
    idx = {}
    for i, e in cat.items():
        for l in e.get("legacy_ids") or []:
            idx.setdefault(str(l), []).append(i)
    return idx


def legacy_candidates(cat, idx, src, raw):
    """候选 = 条目自己原文里引用的旧编号。

    迁移编造了 target_raw（39 个都只以 target_raw 身份存在过，从不是任何条目的
    slug），但原文里的旧编号是从原始 Markdown 抄来的，可经 legacy_ids 解析。
    """
    e = cat[src]
    text = " ".join(str(v) for v in (e.get("original") or {}).values()) + " " + raw
    found = {m[0] or m[1] for m in LEGACY_NUM.findall(text)}
    return sorted({idx[n][0] for n in found if n in idx} - {src})


def cmd_items(a):
    cat, slug2 = load()
    allslugs = list(slug2)
    lidx = legacy_index(cat)
    rows = []
    for i, e in sorted(cat.items()):
        for n, r in enumerate(e.get("relations") or []):
            if str(r.get("target") or ""):
                continue
            raw = str(r.get("target_raw") or "")
            exact = slug2.get(raw, [])
            close = exact or difflib.get_close_matches(raw, allslugs, n=2, cutoff=0.75)
            cands = exact or [slug2[c][0] for c in close]
            stack = lambda s: s.split("_")[0]
            if exact:
                bucket = "exact"
            elif cands:
                bucket = "fuzzy_same" if any(stack(c) == stack(raw) for c in close) else "fuzzy_cross"
            else:
                bucket = "nomatch"
            rows.append(dict(item=f"{i}#{n}", src=i, repo=e["_repo"], kind=r.get("kind"),
                             target_raw=raw, candidates=cands, bucket=bucket,
                             legacy_cands=legacy_candidates(cat, lidx, i, raw)))
    out = HERE / "items.jsonl"
    with open(out, "w") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(rows),
                      "by_bucket": collections.Counter(r["bucket"] for r in rows),
                      "by_repo": collections.Counter(r["repo"] for r in rows),
                      "unique_target_raw": len({r["target_raw"] for r in rows}),
                      "candidate_count": collections.Counter(len(r["candidates"]) for r in rows),
                      "legacy_candidate_count": collections.Counter(len(r["legacy_cands"]) for r in rows),
                      "out": str(out)}, ensure_ascii=False, default=dict, indent=1))


def items(bucket=None):
    rows = [json.loads(l) for l in open(HERE / "items.jsonl")]
    return [r for r in rows if not bucket or r["bucket"] == bucket]


# ------------------------------------------------------------------ calls

def call(model, system, user, key):
    body = {"model": model, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}],
            "max_tokens": 32768, "temperature": 0.0, **MODEL_OPTS.get(model, {})}
    for attempt in range(4):
        try:
            r = requests.post(API, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=900)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"].get("content") or ""
            time.sleep((10 if r.status_code == 429 else 5) * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def parse(content):
    if not content:
        return None
    c = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        obj = json.loads(c[c.find("{"): c.rfind("}") + 1])
    except Exception:
        return None
    if not isinstance(obj, dict) or obj.get("verdict") not in ("yes", "no", "unsure"):
        return None
    return {"verdict": obj["verdict"], "shared_mechanism": str(obj.get("shared_mechanism") or "")[:600]}


def votes_path(tag):
    return HERE / f"votes_{tag}.jsonl"


def read_votes(tag):
    p, out = votes_path(tag), {}
    if p.exists():
        for line in open(p):
            try:
                v = json.loads(line)
            except Exception:
                continue
            if v.get("result"):
                out[v["job"]] = v["result"]
    return out


def judge_all(jobs, key, tag, system=None):
    """jobs: [(job_name, model, user)]；逐条一次调用，边跑边落盘，按 job 名可续跑。"""
    res, lock = {}, threading.Lock()
    log = open(votes_path(tag), "a")

    def work(job):
        name, model, user = job
        parsed = parse(call(model, system or SYSTEM, user, key))
        with lock:
            res[name] = parsed
            log.write(json.dumps({"job": name, "model": model, "result": parsed},
                                 ensure_ascii=False) + "\n")
            log.flush()

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(work, jobs))
    log.close()
    return res


def cands_of(r, source):
    return r["legacy_cands"] if source == "legacy" else r["candidates"]


def cmd_judge(a):
    key = os.environ.get("JUDGE_API_KEY") or sys.exit("JUDGE_API_KEY not set")
    cat, _ = load()
    rows = [r for r in items(a.bucket) if cands_of(r, a.source)]
    if a.max_cands:
        rows = [r for r in rows if len(cands_of(r, a.source)) <= a.max_cands]
    if a.limit:
        rows = rows[:a.limit]
    done = read_votes(a.tag)
    models = PANEL if a.panel else BASE
    jobs = []
    pairs = 0
    for r in rows:
        for cand in cands_of(r, a.source):
            pairs += 1
            user = user_msg(cat, r["src"], cand,
                            raw=r["target_raw"] if a.source == "raw" else None)
            for m in models:
                name = f"{r['item']}|{cand}|{m}"
                if name not in done:
                    jobs.append((name, m, user))
    print(json.dumps({"items": len(rows), "pairs": pairs, "models": list(models),
                      "calls_planned": len(jobs), "already_done": pairs * len(models) - len(jobs)},
                     ensure_ascii=False))
    if a.dry_run or not jobs:
        return
    t0 = time.time()
    got = judge_all(jobs, key, a.tag, system_for(a.source))
    ok = sum(1 for v in got.values() if v)
    print(json.dumps({"calls": len(jobs), "parsed_ok": ok, "parse_fail": len(jobs) - ok,
                      "minutes": round((time.time() - t0) / 60, 1)}, ensure_ascii=False))


# ------------------------------------------------------------------ decide

def tally(vals):
    vals = [v for v in vals if v]
    n = len(vals)
    c = collections.Counter(vals)
    top, k = (c.most_common(1)[0] if c else (None, 0))
    return n, top, k, dict(c)


def cmd_decide(a):
    votes = read_votes(a.tag)
    rows = items(a.bucket)
    out = []
    for r in rows:
      for cand in cands_of(r, a.source):
        base = [votes.get(f"{r['item']}|{cand}|{m}", {}).get("verdict") for m in BASE]
        pan = [votes.get(f"{r['item']}|{cand}|{m}", {}).get("verdict") for m in PANEL]
        if any(pan):
            n, top, k, dist = tally(base + pan)
            status = "panel7" if n >= QUORUM_PANEL and k / n >= MAJ else "uncertain"
        else:
            n, top, k, dist = tally(base)
            status = ("uncertain" if n < QUORUM_BASE else
                      "unanimous" if k == n else "split")
        out.append(dict(item=r["item"], src=r["src"], repo=r["repo"], bucket=r["bucket"],
                        target_raw=r["target_raw"], candidate=cand, verdict=top,
                        status=status, votes=n, dist=dist))
    (HERE / f"decisions_{a.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    summ = collections.Counter((d["status"], d["verdict"]) for d in out)
    agree = [d for d in out if d["status"] in FINAL]
    print(json.dumps({
        "decided": len(out),
        "by_status_verdict": {f"{s}/{v}": n for (s, v), n in sorted(summ.items(), key=lambda x: str(x[0]))},
        "final (unanimous|panel7)": len(agree),
        "needs_panel (split)": sum(1 for d in out if d["status"] == "split"),
        "apply_now (final & yes)": sum(1 for d in agree if d["verdict"] == "yes"),
        "calls_per_item": round(sum(d["votes"] for d in out) / max(len(out), 1), 2),
    }, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ apply

def cmd_apply(a):
    decisions = json.loads((HERE / f"decisions_{a.tag}.json").read_text())
    todo = [d for d in decisions if d["status"] in FINAL and d["verdict"] == "yes"]
    cmds = []
    for d in todo:
        tool = REPOS[d["repo"]] + "/tools/catalog_tool.py"
        cmds.append(["/usr/bin/python3", tool, "update", d["src"],
                     "--relation", f"var:{d['candidate']}",
                     "--note", f"relink: target_raw {d['target_raw']} -> {d['candidate']} "
                               f"({d['status']}, {d['votes']} votes)"])
    print(json.dumps({"final_yes": len(todo), "commands": len(cmds)}, ensure_ascii=False))
    for c in cmds:
        print("  " + " ".join(f"'{x}'" if " " in x else x for x in c))
    if not a.execute:
        print("\n(dry-run；加 --execute 才会真的改真源)")
        return
    import subprocess
    fails = 0
    for c in cmds:
        p = subprocess.run(c, capture_output=True, text=True, cwd=os.path.dirname(c[1]))
        if p.returncode:
            fails += 1
            print(f"FAIL {c[3]}: {p.stderr.strip()[:200]}")
    print(json.dumps({"applied": len(cmds) - fails, "failed": fails}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("items"); s.set_defaults(func=cmd_items)
    s = sub.add_parser("judge")
    s.add_argument("--bucket", choices=["exact", "fuzzy_same", "fuzzy_cross", "nomatch"])
    s.add_argument("--tag", required=True)
    s.add_argument("--panel", action="store_true", help="跑评审团 4 模型（只对 split 项有意义）")
    s.add_argument("--source", choices=["raw", "legacy"], default="legacy",
                   help="候选来源：legacy=原文里的旧编号（默认）；raw=迁移写下的 target_raw（已证实为编造）")
    s.add_argument("--max-cands", type=int, help="只跑候选数 <= N 的条目")
    s.add_argument("--limit", type=int)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_judge)
    s = sub.add_parser("decide")
    s.add_argument("--tag", required=True)
    s.add_argument("--source", choices=["raw", "legacy"], default="legacy")
    s.add_argument("--bucket", choices=["exact", "fuzzy_same", "fuzzy_cross", "nomatch"])
    s.set_defaults(func=cmd_decide)
    s = sub.add_parser("apply")
    s.add_argument("--tag", required=True)
    s.add_argument("--execute", action="store_true")
    s.set_defaults(func=cmd_apply)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
