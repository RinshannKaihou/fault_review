#!/usr/bin/env python3
"""抽样测量：条目的来源 issue 在上游现在闭环到哪一步、条目自己记录到哪一步。

只读上游 GitHub（gh api graphql，用 gh 自己的登录态），不写任何仓库。只打印计数。

    python3 probe_closure.py --n 80 --seed 7
"""
import argparse, glob, json, random, re, subprocess, collections as C
from concurrent.futures import ThreadPoolExecutor
import yaml

REPOS = {"inf": "/workspace/inference_error_review", "trn": "/workspace/training_error_review"}
ISSUE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/issues/(\d+)")
Q = """query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){issue(number:$n){
 state stateReason comments{totalCount}
 closedByPullRequestsReferences(first:5,includeClosedPrs:true){nodes{number merged}}
 timelineItems(itemTypes:[CROSS_REFERENCED_EVENT],first:30){nodes{... on CrossReferencedEvent{
   source{... on PullRequest{number merged repository{nameWithOwner}}}}}}}}}"""


def sample(n, seed):
    pool = []
    for repo, path in REPOS.items():
        for f in sorted(glob.glob(path + "/catalog/entries/*.yaml")):
            y = yaml.safe_load(open(f))
            if y.get("status") in ("merged", "retired") or y.get("kind") != "故障":
                continue
            urls = [e.get("url") or "" for e in y.get("evidence") or [] if e.get("kind") == "github_issue"]
            m = next((ISSUE.search(u) for u in urls if ISSUE.search(u)), None)
            if m:
                pool.append((y, m.groups()))
    random.Random(seed).shuffle(pool)
    return pool[:n], len(pool)


def fetch(args):
    y, (o, r, num) = args
    out = subprocess.run(["gh", "api", "graphql", "-f", "query=" + Q, "-f", f"o={o}", "-f", f"r={r}",
                          "-F", f"n={num}"], capture_output=True, text=True)
    try:
        iss = json.loads(out.stdout)["data"]["repository"]["issue"]
    except Exception:
        return y, None
    closing = {p["number"] for p in iss["closedByPullRequestsReferences"]["nodes"] if p["merged"]}
    prs = closing | {s["source"]["number"] for s in iss["timelineItems"]["nodes"]
            if s.get("source", {}).get("merged") and s["source"]["repository"]["nameWithOwner"].lower() == f"{o}/{r}".lower()}
    return y, dict(state=iss["state"], reason=iss["stateReason"], comments=iss["comments"]["totalCount"], prs=prs, closing=closing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    picked, pool = sample(a.n, a.seed)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(fetch, picked))
    c = C.Counter()
    for y, s in res:
        if s is None:
            c["fetch_failed"] += 1
            continue
        text = json.dumps(y, ensure_ascii=False)
        fix = str((y.get("mitigation") or {}).get("fix", ""))
        recorded = any(re.search(rf"(?:pull/|#){p}\b", text) for p in s["prs"])
        c["ok"] += 1
        c[f"state={s['state']}/{s['reason']}"] += 1
        c["upstream_has_merged_pr"] += bool(s["prs"])
        c["upstream_closed_by_merged_pr(strict)"] += bool(s["closing"])
        c["merged_pr_recorded_in_entry"] += bool(s["prs"]) and recorded
        c["merged_pr_NOT_in_entry"] += bool(s["prs"]) and not recorded
        c["entry_fix_is_待定"] += fix.startswith("待定")
        c["entry_fix_待定_but_upstream_merged_pr"] += fix.startswith("待定") and bool(s["prs"])
        c["comments>=5"] += s["comments"] >= 5
    print(json.dumps({"pool_entries_with_github_issue": pool, "sampled": len(picked), **dict(sorted(c.items()))},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
