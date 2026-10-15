#!/usr/bin/env python3
"""从已解析的 var 关系森林推导模式层（阶段 B 第 1 步）。

425 条 var 边里每个条目最多一条出边，所以这是一座森林：每棵树就是一个候选模式。
本脚本只做离线推导与预览，**不写上游**；写入由 catalog_tool 负责。

代表条目（representative）的选法：树里入度最高的条目；并列时取编号最小的那个，
保证可复现。入度最高 = 最多条目声明自己是它的变体，也就是事实上的家族代表。

    preview   推导森林、按树输出模式预览（含代表、成员、跨库情况）
    stats     只打印分布统计
"""

import argparse
import collections
import glob
import json
from pathlib import Path

import yaml

REPOS = {"inf": "/workspace/inference_error_review", "trn": "/workspace/training_error_review"}
HERE = Path(__file__).resolve().parent


def load():
    cat = {}
    for repo, path in REPOS.items():
        for f in sorted(glob.glob(path + "/catalog/entries/*.yaml")):
            e = yaml.safe_load(open(f))
            e["_repo"] = repo
            cat[e["id"]] = e
    return cat


def var_edges(cat):
    """(源, 目标) 边；目标可能写的是旧编号，按 legacy_ids 换算。"""
    legacy = {}
    for i, e in cat.items():
        for l in e.get("legacy_ids") or []:
            legacy.setdefault(str(l), []).append(i)
    edges = []
    for i, e in sorted(cat.items()):
        for r in e.get("relations") or []:
            if r.get("kind") != "var":
                continue
            t = str(r.get("target") or "")
            if not t:
                continue  # 阶段 A：45 条目标不可恢复，已就地留痕
            tid = t if t in cat else (legacy.get(t) or [None])[0]
            if tid and tid in cat:
                edges.append((i, tid))
    return edges


def forest(edges):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    trees = collections.defaultdict(list)
    for n in list(parent):
        trees[find(n)].append(n)
    return trees


def patterns(cat):
    edges = var_edges(cat)
    indeg = collections.Counter(b for _, b in edges)
    out = []
    for n, (_, members) in enumerate(sorted(forest(edges).items()), start=1):
        members = sorted(members)
        rep = min(members, key=lambda i: (-indeg[i], i))
        repos = sorted({cat[i]["_repo"] for i in members})
        subs = collections.Counter(
            ((cat[i].get("cause") or {}).get("objects") or [{}])[0].get("subsystem") for i in members)
        out.append(dict(id=f"PAT-{n:04d}", representative=rep, members=members,
                        size=len(members), repos=repos,
                        title_from_rep=cat[rep].get("title"),
                        subsystems=dict(subs),
                        subsystem_agreement=round(subs.most_common(1)[0][1] / len(members), 2)))
    return out


def cmd_preview(a):
    cat = load()
    pats = patterns(cat)
    (HERE / "patterns_preview.json").write_text(
        json.dumps(pats, ensure_ascii=False, indent=1), encoding="utf-8")
    covered = sum(p["size"] for p in pats)
    print(json.dumps({
        "patterns": len(pats),
        "covered_entries": covered,
        "uncovered_entries": len(cat) - covered,
        "size_distribution": dict(sorted(collections.Counter(p["size"] for p in pats).items())),
        "cross_repo_patterns": sum(1 for p in pats if len(p["repos"]) > 1),
        "subsystem_agreement": {
            "==1.0 (全员同子系统)": sum(1 for p in pats if p["subsystem_agreement"] == 1.0),
            ">=0.6": sum(1 for p in pats if 0.6 <= p["subsystem_agreement"] < 1.0),
            "<0.6": sum(1 for p in pats if p["subsystem_agreement"] < 0.6)},
        "out": str(HERE / "patterns_preview.json"),
    }, ensure_ascii=False, indent=1))
    if a.show:
        for p in sorted(pats, key=lambda x: -x["size"])[:a.show]:
            print(f"\n{p['id']}  size={p['size']}  repos={','.join(p['repos'])}  "
                  f"子系统一致度={p['subsystem_agreement']}")
            print(f"  代表 {p['representative']}: {p['title_from_rep']}")
            print(f"  子系统: {p['subsystems']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("preview")
    s.add_argument("--show", type=int, default=0, help="额外打印最大的 N 个模式")
    s.set_defaults(func=cmd_preview)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
