#!/usr/bin/env python3
"""抽样对比：条目正文 vs 来源 issue（正文 + 全部评论）的长度。只读，只打印分位数。

    python3 probe_ratio.py --n 80 --seed 7      # 与 probe_closure.py 同一批样本
"""
import argparse, json, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from probe_closure import sample

Q = """query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){issue(number:$n){
 body comments(first:100){totalCount nodes{body}}}}}"""
CODE = re.compile(r"```.*?```", re.S)


def fetch(args):
    y, (o, r, num) = args
    p = subprocess.run(["gh", "api", "graphql", "-f", "query=" + Q, "-f", f"o={o}", "-f", f"r={r}", "-F", f"n={num}"],
                       capture_output=True, text=True)
    try:
        iss = json.loads(p.stdout)["data"]["repository"]["issue"]
    except Exception:
        return None
    texts = [iss["body"] or ""] + [c["body"] or "" for c in iss["comments"]["nodes"]]
    src = sum(len(t) for t in texts)
    prose = sum(len(CODE.sub("", t)) for t in texts)
    o_ = y.get("original") or {}
    entry = sum(len(str(v)) for k, v in o_.items() if k not in ("category", "section"))
    return dict(entry=entry, src=src, prose=prose, comments=iss["comments"]["totalCount"])


def q(xs):
    xs = sorted(xs)
    return [round(xs[int(len(xs) * p)], 2) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    picked, _ = sample(a.n, a.seed)
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(fetch, picked) if r]
    print(json.dumps({
        "ok": len(rows),
        "quantiles": "p10/p25/p50/p75/p90",
        "entry_chars": q([r["entry"] for r in rows]),
        "source_chars_issue+comments": q([r["src"] for r in rows]),
        "source_chars_without_code_blocks": q([r["prose"] for r in rows]),
        "entry/source": q([r["entry"] / max(r["src"], 1) for r in rows]),
        "entry/source_prose": q([r["entry"] / max(r["prose"], 1) for r in rows]),
        "entry_longer_than_source_prose": sum(r["entry"] > r["prose"] for r in rows),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
