#!/usr/bin/env python3
"""给模式生成 title 与 shared_mechanism（阶段 B 第 1 步）。

代表条目的标题不能当模式名——它带着实例的特指（`PAT-0113` 的代表标题写「vLLM
Kimi-K3 parser…」，而该族 16 个成员跨 vllm 与 sglang、涵盖多个不同解析器）。
所以按**成员集**生成，再由 2 票复核。

沿用 `catalog_audit.py` 的分工：qwen3.8-max 写，kimi-k3 + doubao-seed-2.1-pro 复核。

    stars          只列纯星形模式（链深 <=1，不假设 var 的传递性）
    gen --limit N  生成 + 复核，按模式 id 去重可续跑
    files          把通过复核的模式写成 YAML（默认写到 out/ 预览目录，不碰上游）
"""

import argparse
import collections
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_patterns as B  # noqa: E402

API = "https://st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com/v1/chat/completions"
WRITER = "qwen3.8-max"
VERIFIERS = ("kimi-k3", "doubao-seed-2.1-pro")
MODEL_OPTS = {"kimi-k3": {"temperature": 1.0}, "doubao-seed-2.1-pro": {"reasoning_effort": "low"}}
HERE = Path(__file__).resolve().parent

WRITE_SYS = """你在给一份「LLM 推理/训练故障模式编目」里的一个**故障模式**命名并写出它的共享机制。

我给你这个模式的全部成员条目。这些条目被编目作者标注为彼此的变体，也就是同一个
底层问题的不同显形。请写出覆盖**全部**成员的模式名与共享机制。

要求：
- `title`：不超过 40 字，描述**机制**而不是某个实例。**不要**写只有部分成员才有的
  具体栈名、模型名、函数名（例如成员里既有 vllm 又有 sglang，就不能只写 vllm）。
  如果全部成员确实共享某个栈或部件，才可以写进去。
- `shared_mechanism`：1–3 句，说明这些条目共同的失效机制是什么——哪一处假设不成立、
  哪条数据通路出错。只写**所有**成员都成立的内容。
- 如果这些成员其实并不共享一个机制，`title` 照写你能概括的最大公因，并把
  `coherent` 填 false，在 `note` 里说明分歧在哪。不要为了凑一个名字而编造共性。

只输出一个 JSON 对象，不要任何其它文字：
{"title": "...", "shared_mechanism": "...", "coherent": true, "note": ""}"""

CHECK_SYS = """你在复核一个「故障模式」的命名与共享机制写得对不对。

我给你该模式的全部成员条目，以及待复核的 title 与 shared_mechanism。请检查三件事：
1. **是否对每个成员都成立**——有任何一个成员不符合，就是不通过；
2. **是否过度泛化**——泛到「都是数值错误」「都和 KV 有关」这种程度就是不通过；
3. **是否夹带了只有部分成员才有的特指**（某个栈名、模型名、函数名）——夹带了就是不通过。

只输出一个 JSON 对象，不要任何其它文字：
{"verdict": "ok|no", "problem": "<不通过时指出是上面哪一条、具体哪个成员>"}"""


FIX_SYS = """你在修改一个「故障模式」的命名与共享机制。上一版被复核否掉了，请按意见改写。

要求与上一版相同：`title` 不超过 40 字、描述机制而非实例、不夹带只有部分成员才有的
栈名/模型名/函数名；`shared_mechanism` 1–3 句、只写**所有**成员都成立的内容。

针对复核意见：如果意见指出某个成员不符合，就把共性收窄到真正覆盖全部成员的那一层，
**不要**靠加泛化词（「某些情况下」「可能」）来糊弄；如果确实无法用一个机制覆盖全部成员，
把 `coherent` 填 false 并在 `note` 里说明该拆成哪几组。

只输出一个 JSON 对象，不要任何其它文字：
{"title": "...", "shared_mechanism": "...", "coherent": true, "note": ""}"""


def member_view(cat, mid):
    e = cat[mid]
    c, eff = e.get("cause") or {}, e.get("effect") or {}
    obj = (c.get("objects") or [{}])[0]
    return (f"- {mid}  {e.get('title')}\n"
            f"    子系统={obj.get('subsystem')} 栈={obj.get('stack')} 部件={obj.get('component')}\n"
            f"    原因行为: {c.get('behavior')}\n"
            f"    影响详情: {eff.get('detail')}")


def pattern_view(cat, pat):
    head = f"模式成员共 {pat['size']} 条（代表 {pat['representative']}）：\n"
    return head + "\n".join(member_view(cat, m) for m in pat["members"])


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


def parse(content, keys):
    if not content:
        return None
    c = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        obj = json.loads(c[c.find("{"): c.rfind("}") + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) and all(k in obj for k in keys) else None


def depth_of(out, n):
    d, seen = 0, set()
    while n in out and n not in seen:
        seen.add(n)
        n = out[n]
        d += 1
    return d


def stars(cat):
    """纯星形模式：树内最大链深 <=1，不依赖 var 的传递性。"""
    out = {a: b for a, b in B.var_edges(cat)}
    return [p for p in B.patterns(cat)
            if max(depth_of(out, m) for m in p["members"]) <= 1]


def results_path():
    return HERE / "generated.jsonl"


def read_results():
    out = {}
    if results_path().exists():
        for line in open(results_path()):
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[r["id"]] = r
    return out


def cmd_stars(a):
    cat = B.load()
    ps = stars(cat)
    print(json.dumps({"star_patterns": len(ps), "covered_entries": sum(p["size"] for p in ps),
                      "size_distribution": dict(sorted(collections.Counter(p["size"] for p in ps).items()))},
                     ensure_ascii=False, indent=1))


def cmd_gen(a):
    key = os.environ.get("JUDGE_API_KEY") or sys.exit("JUDGE_API_KEY not set")
    cat = B.load()
    ps = sorted(stars(cat), key=lambda p: (-p["size"], p["id"]))
    done = read_results()
    todo = [p for p in ps if p["id"] not in done][:a.limit or len(ps)]
    print(json.dumps({"star_patterns": len(ps), "already_done": len(done), "this_run": len(todo),
                      "calls": len(todo) * (1 + len(VERIFIERS))}, ensure_ascii=False))
    if a.dry_run or not todo:
        return
    lock = threading.Lock()
    log = open(results_path(), "a")
    t0 = time.time()

    def work(p):
        view = pattern_view(cat, p)
        wrote = parse(call(WRITER, WRITE_SYS, view, key), ("title", "shared_mechanism"))
        rec = dict(id=p["id"], representative=p["representative"], members=p["members"],
                   size=p["size"], written=wrote, checks={})
        if wrote:
            cu = (view + "\n\n待复核：\ntitle: " + str(wrote.get("title"))
                  + "\nshared_mechanism: " + str(wrote.get("shared_mechanism")))
            for m in VERIFIERS:
                rec["checks"][m] = parse(call(m, CHECK_SYS, cu, key), ("verdict",))
        with lock:
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, todo))
    log.close()
    res = read_results()
    fresh = [res[p["id"]] for p in todo if p["id"] in res]
    ok = sum(1 for r in fresh if passed(r))
    print(json.dumps({"generated": len(fresh), "both_verifiers_ok": ok,
                      "write_failed": sum(1 for r in fresh if not r["written"]),
                      "minutes": round((time.time() - t0) / 60, 1)}, ensure_ascii=False))


def passed(r):
    """两票复核都通过，且写手没有自报「这些成员并不共享一个机制」。

    写手把 coherent 填 false 时，它只是概括了最大公因——复核可能照样放行那个标题，
    但那等于把一个不自洽的族当成模式。这类归入待复核，不自动采用。
    """
    w = r.get("written")
    if not w or w.get("coherent") is False:
        return False
    v = [(r["checks"].get(m) or {}).get("verdict") for m in VERIFIERS]
    return v.count("ok") == len(VERIFIERS)


def cmd_files(a):
    res = read_results()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for pid, r in sorted(res.items()):
        if a.only_passed and not passed(r):
            continue
        doc = {"id": pid, "title": r["written"]["title"],
               "shared_mechanism": r["written"]["shared_mechanism"],
               "representative": r["representative"], "members": r["members"]}
        (out / f"{pid}.yaml").write_text(
            yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8")
        n += 1
    print(json.dumps({"written": n, "dir": str(out),
                      "skipped_not_passed": sum(1 for r in res.values() if not passed(r))},
                     ensure_ascii=False))


def cmd_fix(a):
    """把复核意见喂回去重写，再用同样 2 票复核（与 audit_fix.py 同一套做法）。"""
    key = os.environ.get("JUDGE_API_KEY") or sys.exit("JUDGE_API_KEY not set")
    cat = B.load()
    by_id = {p["id"]: p for p in stars(cat)}
    bad = [r for r in read_results().values() if not passed(r) and r["id"] in by_id
           and not r.get("fixed")]
    if a.limit:
        bad = bad[:a.limit]
    print(json.dumps({"to_fix": len(bad), "calls": len(bad) * (1 + len(VERIFIERS))},
                     ensure_ascii=False))
    if a.dry_run or not bad:
        return
    lock = threading.Lock()
    log = open(results_path(), "a")

    def work(r):
        p = by_id[r["id"]]
        view = pattern_view(cat, p)
        objections = "\n".join(
            f"- {m}: {(c or {}).get('problem')}" for m, c in r["checks"].items()
            if (c or {}).get("verdict") != "ok")
        u = (view + "\n\n上一版：\ntitle: " + str(r["written"]["title"])
             + "\nshared_mechanism: " + str(r["written"]["shared_mechanism"])
             + "\n\n复核意见：\n" + objections)
        wrote = parse(call(WRITER, FIX_SYS, u, key), ("title", "shared_mechanism"))
        rec = dict(id=r["id"], representative=p["representative"], members=p["members"],
                   size=p["size"], written=wrote or r["written"], checks={}, fixed=True)
        if wrote:
            cu = (view + "\n\n待复核：\ntitle: " + str(wrote["title"])
                  + "\nshared_mechanism: " + str(wrote["shared_mechanism"]))
            for m in VERIFIERS:
                rec["checks"][m] = parse(call(m, CHECK_SYS, cu, key), ("verdict",))
        with lock:
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, bad))
    log.close()
    res = read_results()
    print(json.dumps({"fixed_attempted": len(bad),
                      "now_passing": sum(1 for r in bad if passed(res[r["id"]]))},
                     ensure_ascii=False))


def cmd_report(a):
    res = read_results()
    ok = [r for r in res.values() if passed(r)]
    rej = collections.Counter()
    seen = collections.Counter()
    for r in res.values():
        for m, c in r["checks"].items():
            if c:
                seen[m] += 1
                if c.get("verdict") != "ok":
                    rej[m] += 1
    print(json.dumps({
        "patterns_generated": len(res),
        "passed_both_verifiers": len(ok),
        "pass_rate": round(len(ok) / max(len(res), 1), 3),
        "fixed_round_records": sum(1 for r in res.values() if r.get("fixed")),
        "write_failed": sum(1 for r in res.values() if not r.get("written")),
        "verifier_rejection_rate": {m: f"{rej[m]}/{seen[m]}" for m in VERIFIERS},
        "coherent_false": sum(1 for r in res.values()
                              if (r.get("written") or {}).get("coherent") is False),
        "covered_entries": sum(r["size"] for r in ok),
    }, ensure_ascii=False, indent=1))


def cmd_apply(a):
    """把生成并通过复核的 title / shared_mechanism 写进上游模式文件。

    **按成员集匹配，不按编号**：离线 `build_patterns.py` 的 PAT 编号与上游
    `pattern-sync` 分配的编号不同（前者按树根排序、后者按规模分配），只有成员集是稳定键。
    """
    import subprocess
    res = {frozenset(r["members"]): r for r in read_results().values() if passed(r)}
    applied = unmatched = failed = 0
    for repo, root in B.REPOS.items():
        pdir = Path(root) / "catalog" / "patterns"
        if not pdir.is_dir():
            continue
        for f in sorted(pdir.glob("PAT-*.yaml")):
            doc = yaml.safe_load(open(f))
            if doc.get("status") == "merged":
                continue
            r = res.get(frozenset(doc.get("members") or []))
            if not r:
                continue
            if doc.get("title") not in ("待定", None) and not a.force:
                continue
            cmd = ["/usr/bin/python3", str(Path(root) / "tools" / "catalog_tool.py"),
                   "pattern-set", doc["id"], "--title", r["written"]["title"],
                   "--shared-mechanism", r["written"]["shared_mechanism"]]
            if a.dry_run:
                cmd.append("--dry-run")
            pr = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
            if pr.returncode:
                failed += 1
                print(f"FAIL {doc['id']}: {pr.stderr.strip()[:160] or pr.stdout.strip()[:160]}")
            else:
                applied += 1
    matched = {frozenset(yaml.safe_load(open(f)).get("members") or [])
               for repo, root in B.REPOS.items()
               for f in (Path(root) / "catalog" / "patterns").glob("PAT-*.yaml")}
    unmatched = sum(1 for k in res if k not in matched)
    print(json.dumps({"passed_results": len(res), "applied": applied, "failed": failed,
                      "unmatched_member_sets": unmatched, "dry_run": bool(a.dry_run)},
                     ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stars"); s.set_defaults(func=cmd_stars)
    s = sub.add_parser("gen")
    s.add_argument("--limit", type=int)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_gen)
    s = sub.add_parser("fix")
    s.add_argument("--limit", type=int)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_fix)
    s = sub.add_parser("report"); s.set_defaults(func=cmd_report)
    s = sub.add_parser("apply")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--force", action="store_true", help="覆盖已有的非占位 title")
    s.set_defaults(func=cmd_apply)
    s = sub.add_parser("files")
    s.add_argument("--out", default=str(HERE / "out"))
    s.add_argument("--only-passed", action="store_true", default=True)
    s.set_defaults(func=cmd_files)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
