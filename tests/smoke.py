#!/usr/bin/env python3
"""Smoke battery for fault-rag. Runs in-process (single model load).

Usage: python3 tests/smoke.py
Exits non-zero if any check fails.
"""

import sys
import time

sys.path.insert(0, ".")  # run from repo root

from fault_rag import search as S  # noqa: E402

FAILURES = []


def check(label, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)


def top_ids(res):
    return [(r["repo"], r["id"], r["category"]) for r in res]


def cats(res):
    return {r["category"] for r in res}


# 1. index counts -----------------------------------------------------------
# Catalogs are vendored snapshots (catalogs/) that grow on every sync; floor
# checks catch parse regressions (lost entries) without breaking on growth.
st = S.stats()
check("inference catalog >= 1184", st["by_category"] and
      sum(v for k, v in st["by_category"].items()
          if k.startswith("inference:") and "neg" not in k) >= 1184)
check("training catalog >= 221",
      sum(v for k, v in st["by_category"].items()
          if k.startswith("training:") and "neg" not in k) >= 221)
n_neg = sum(v for k, v in st["by_category"].items() if ":neg_" in k)
check("negative-doc chunks >= 14", n_neg >= 14, f"got {n_neg}")
check("entry types: fault >= 1404, negative >= 14, pointer == 1",
      st["by_entry_type"]["fault"] >= 1404
      and st["by_entry_type"]["negative"] >= 14
      and st["by_entry_type"]["pointer"] == 1,
      str(st["by_entry_type"]))

idx = S.get_index()
bad_categories = []
for e in idx.entries:
    if e["neg"]:
        continue
    if e["repo"] == "inference":
        expected = ("KV" + e["id"][1:].split(".")[0]
                    if e["id"].startswith("K")
                    else "cat" + e["id"].split(".")[0])
    else:
        expected = e["id"].rsplit(".", 1)[0]
    if e["category"] != expected:
        bad_categories.append((e["repo"], e["id"], e["category"], expected))
check("all catalog categories follow entry ID grammar",
      not bad_categories, str(bad_categories[:5]))
check("regular faults all have confidence",
      all(e.get("confidence") for e in idx.entries
          if e.get("entry_type") == "fault"))

# 2. lookup -----------------------------------------------------------------
r = S.lookup("HW.01")
check("lookup HW.01", len(r) == 1 and
      r[0]["name"] == "training_sdc_unhealthy_node_optima_shift")
r = S.lookup("minp_typical_misconfig")
check("lookup alias -> 11.5", any(x["id"] == "11.5" for x in r))
r = S.lookup("10.4")
check("lookup pointer 10.4 is explicit and non-negative",
      len(r) == 1 and r[0]["entry_type"] == "pointer" and not r[0]["neg"])
r = S.lookup("vllm_prefix_cache_silent_noop_mamba_gdn_hybrid")
check("duplicate snake name is explicit",
      {x["id"] for x in r} == {"6.28", "6.33"} and
      all(x["name_ambiguous"] for x in r),
      str([(x["id"], x.get("same_name_ids")) for x in r]))

# 3. hybrid searches --------------------------------------------------------
t0 = time.time()
res = S.search("loss spike 但训练没有 crash", topk=5)
check("search 'loss spike' -> LOSS.* or cat* in top5",
      any(c.startswith("LOSS") or c.startswith("cat") for c in cats(res)),
      str(top_ids(res)[:5]))

res = S.search("GRPO advantage std zero 组内无对比", topk=5)
check("search GRPO std=0 -> RL-ADV in top5",
      any(c == "RL-ADV" for c in cats(res)), str(top_ids(res)[:5]))

res = S.search("vllm prefix caching gibberish tokens", topk=5)
check("search vllm prefix caching -> KV* in top5",
      any(c.startswith("KV") for c in cats(res)), str(top_ids(res)[:5]))

res = S.search("训练 reward 上涨但能力下降", topk=5)
check("search zh reward-hacking -> RL-RWD in top5",
      any(c == "RL-RWD" for c in cats(res)), str(top_ids(res)[:5]))

res = S.search("checkpoint resume loads wrong weights silently", topk=5)
check("search ckpt resume -> CKPT.* or cat12 in top5",
      any(c.startswith("CKPT") or c == "cat12" for c in cats(res)),
      str(top_ids(res)[:5]))

res = S.search("OOM 算静默故障吗", topk=6)
check("negative recall: rejected.md surfaces for OOM question",
      any(r["neg"] for r in res), str(top_ids(res)))

# Six source-labelled benchmark representatives -----------------------------
benchmark_cases = [
    ("Blackhole 11-wide grid batch 32 RoPE cos sin shards do not match the query shard grid",
     {"repo": "inference", "category": "cat9"}, {"9.27"}),
    ("答案已经正确但模型认不出结束 token，持续重复直到 max_new_tokens",
     {"repo": "inference", "category": "cat10"}, {"10.5"}),
    ("训练时一次 SDC 坏更新进入发布 checkpoint，serving 前向本身完全干净",
     {"repo": "training", "category": "HW"}, {"HW.05"}),
    ("packed samples share one sequence but attention mask does not cut document boundaries",
     {"repo": "training", "category": "DATA"}, {"DATA.02"}),
    ("monitor JSONL file exists but is empty and has no hidden-state samples",
     {"repo": "training"}, {"NEG-BLINDSPOTS-4"}),
    ("chat_template_faults cited for completeness not a new entry",
     {"repo": "inference", "include_pointers": True}, {"10.4"}),
]
for query, kwargs, expected_ids in benchmark_cases:
    res = S.search(query, topk=5, **kwargs)
    check(f"labelled retrieval: {sorted(expected_ids)} in top5",
          bool(expected_ids & {x["id"] for x in res}), str(top_ids(res)))

# 4. filters ----------------------------------------------------------------
res = S.search("bf16 mixed precision", topk=5, repo="training", category="NUM")
check("filter repo=training category=NUM",
      bool(res) and all(r["repo"] == "training" and r["category"] == "NUM"
                        for r in res), str(top_ids(res)))

res = S.search("silent data corruption", topk=5, confidence="verified")
check("filter confidence=verified",
      bool(res) and all(r["confidence"] == "verified" for r in res))

res = S.search("bf16", topk=5, repo="training", category="NUM",
               include_neg=False)
check("--no-neg excludes negative chunks",
      all(not r["neg"] for r in res))
res = S.search("zzzxqv_nonexistent_token_94731", topk=3, mode="bm25",
               include_neg=False)
check("BM25 zero-hit query returns no fabricated rows", res == [], str(res))

# 5. related ----------------------------------------------------------------
rel = S.related("HW.05")
check("related HW.05 -> cross link to inference 1.6",
      any(r["repo"] == "inference" and r["id"] == "1.6" for r in rel),
      str([(r["repo"], r["id"], r.get("link_kind")) for r in rel]))

# 6. latency ----------------------------------------------------------------
t1 = time.time()
S.search("embedding latency probe", topk=3)
warm_ms = (time.time() - t1) * 1000
check("warm single-query latency < 3000ms", warm_ms < 3000, f"{warm_ms:.0f}ms")
print(f"total query wall time (10 queries): {time.time() - t0:.1f}s")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES:", *FAILURES, sep="\n  - ")
    sys.exit(1)
print("all smoke checks passed")
