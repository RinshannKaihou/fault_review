"""Hybrid retrieval: BM25 + dense cosine, fused with Reciprocal Rank Fusion."""
import re
import sys

import numpy as np

from . import embed as embed_mod
from . import rerank as rerank_mod
from .index import get_index, tokenize

RRF_K = 60
# Old Markdown ids (inference 1.44 / K9.3, training HW.01 / RL-RO.01), retired 2026-10-26.
LEGACY_ID_RE = re.compile(r"(?:\d+|[A-Z][A-Z0-9-]*)\.\d+")


def _entry_type(e):
    """Read the schema field with compatibility for pre-migration indexes."""
    return e.get("entry_type", "negative" if e.get("neg") else "fault")


def _passes(e, repo=None, category=None, confidence=None,
            neg=None, include_pointers=False, pattern=None):
    if _entry_type(e) == "pointer" and not include_pointers:
        return False
    if repo and e["repo"] != repo:
        return False
    if category and not _category_matches(e, category):
        return False
    if confidence and (e.get("confidence") or "") != confidence:
        return False
    if pattern and (e.get("pattern") or "") != pattern:
        return False
    if neg is not None and bool(e["neg"]) != neg:
        return False
    return True


def _category_matches(e, category):
    """--category is the cause subsystem (e.g. KV缓存) or a negative kind (neg_rejected)."""
    return e["category"].lower() == category.lower()


def _ranks(scores, eligible):
    """Return {row: rank} over eligible rows, best score = rank 0."""
    order = sorted(eligible, key=lambda i: -scores[i])
    return {i: r for r, i in enumerate(order)}


def _positive_ranks(scores, eligible):
    """Rank only rows with positive evidence (used for lexical scores)."""
    matched = [i for i in eligible if scores[i] > 0]
    return _ranks(scores, matched)


def _rrf_scores(*rankings):
    """Fuse rank maps without inventing candidates absent from every ranker."""
    fused = {}
    for ranking in rankings:
        for i, rank in ranking.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
    return fused


def _rerank(query, entries, order, fused, pool):
    """Re-score the hybrid head with the cross-encoder and blend the two ranks (RRF).
    Returns (new order, scores to report, {row: cross-encoder rank})."""
    head = order[:pool]
    s = rerank_mod.scores(query, [entries[i] for i in head])
    by_score = sorted(range(len(head)), key=lambda n: -s[n])
    rerank_r = {head[n]: r for r, n in enumerate(by_score)}
    blended = _rrf_scores({i: r for r, i in enumerate(head)}, rerank_r)
    head = sorted(head, key=lambda i: -blended[i])
    return head + order[pool:], {**fused, **blended}, rerank_r


def search(query, topk=8, mode="hybrid", repo=None, category=None,
           confidence=None, include_neg=True,
           include_pointers=False, pattern=None, rerank=True):
    idx = get_index()
    neg_flag = None if include_neg else False

    eligible = [
        i for i, e in enumerate(idx.entries)
        if _passes(e, repo, category, confidence, neg=neg_flag,
                   include_pointers=include_pointers, pattern=pattern)
    ]
    if not eligible:
        if category and not any(_category_matches(e, category) for e in idx.entries):
            print(f"fault-rag: no entry has category {category!r}. --category takes a cause subsystem "
                  "(e.g. KV缓存) or neg_rejected / neg_blindspot / neg_appendix; `stats` lists them "
                  "under by_category. Old categories (cat1 / KV9 / HW ...) were retired 2026-10-26.",
                  file=sys.stderr)
        return []

    dense_r = {}
    if mode in ("hybrid", "dense"):
        qv = embed_mod.embed_one(query)
        sims = idx.mat @ qv  # unit vectors -> cosine
        dense_r = _ranks(sims, eligible)

    bm25_r = {}
    if mode in ("hybrid", "bm25"):
        bscores = idx.bm25.scores(tokenize(query))
        bm25_r = _positive_ranks(bscores, eligible)

    fused = _rrf_scores(dense_r, bm25_r)
    order = sorted(fused, key=lambda i: -fused[i])

    rerank_r = {}
    if rerank and mode == "hybrid":
        if rerank_mod.available():
            order, fused, rerank_r = _rerank(query, idx.entries, order, fused,
                                             max(rerank_mod.POOL, topk))
        else:
            rerank_mod.warn_unavailable()

    top = order[:topk]
    out = []
    for rank, i in enumerate(top, start=1):
        e = idx.entries[i]
        out.append({
            "rank": rank,
            "score": round(fused[i], 6),
            "dense_rank": dense_r.get(i),
            "bm25_rank": bm25_r.get(i),
            "rerank_rank": rerank_r.get(i),
            "id": e["id"],
            "name": e["name"],
            "title": e.get("title"),
            "summary": _summary(e),
            "repo": e["repo"],
            "category": e["category"],
            "confidence": e.get("confidence"),
            "status": e.get("status"),
            "pattern": e.get("pattern"),
            "neg": e["neg"],
            "entry_type": _entry_type(e),
            "name_ambiguous": e.get("name_ambiguous", False),
            "same_name_ids": e.get("same_name_ids", []),
            "file": e["file"],
            "line": e["line"],
            "links": e["links"][:6],
        })
    return out


def lookup(q):
    """Exact id, name / alias, then name substring for non-ID queries. Old Markdown ids are not
    resolved: they return nothing, with a pointer to the id registry on stderr."""
    idx = get_index()
    rows = []
    if q in idx.by_id:
        rows = idx.by_id[q]
    elif q in idx.by_name:
        rows = idx.by_name[q]
    elif LEGACY_ID_RE.fullmatch(q):
        repo = "inference" if re.fullmatch(r"K?\d+\.\d+", q) else "training"
        print(f"fault-rag: {q} looks like an old Markdown id; those were retired 2026-10-26 and are not "
              f"resolved. Find its INF-/TRN- id in catalogs/{repo}/id_registry.tsv, or with "
              f"`tools/catalog_tool.py find {q}` in the upstream catalog repo (also covers merged ids).",
              file=sys.stderr)
    elif re.fullmatch(r"(?:INF|TRN)-\d+", q):
        # An unknown catalog ID must not partially match an unrelated ID.
        rows = []
    else:
        ql = q.lower()
        rows = [i for i, e in enumerate(idx.entries)
                if ql in e["name"].lower() or ql in e["id"].lower()][:10]
    return [_full(idx.entries[i]) for i in rows]


def related(q, depth=1):
    """Lookup an entry and walk its cross-reference links."""
    idx = get_index()
    seeds = lookup(q)
    if not seeds:
        return []
    by_key = {(e["repo"], e["id"]): e for e in idx.entries}
    seen = {(s["repo"], s["id"]) for s in seeds}
    frontier = list(seeds)
    out = []
    for _ in range(max(1, depth)):
        nxt = []
        for s in frontier:
            for link in by_key[(s["repo"], s["id"])].get("links", []):
                key = (link["repo"], link["id"])
                if key in seen or key not in by_key:
                    continue
                seen.add(key)
                rec = _full(by_key[key])
                rec["link_kind"] = link["kind"]
                out.append(rec)
                nxt.append(rec)
        frontier = nxt
    return out


def _summary(e):
    """给 agent 判断相关性用的一行摘要：只做展示，不进 BM25 也不进嵌入文本，
    所以加它不改变任何排序。来源是条目已有的压缩字段（effect.detail 写现象与条件，
    cause.behavior 写原因），零生成成本；条目将来有了叙述式导语再换成导语。"""
    f = e["fields"]
    if e["neg"]:
        return "负结果：" + " ".join(f.get("content", "").split())[:200]
    fix = str(f.get("mitigation_fix") or "").strip()
    fix = "未记录" if not fix or fix.startswith("待定") else fix[:120]
    return (f"现象：{f.get('effect_detail') or '待定'}｜原因：{f.get('cause_behavior') or '待定'}"
            f"｜修复：{fix}")


def _full(e):
    return {
        "id": e["id"],
        "name": e["name"],
        "title": e.get("title"),
        "repo": e["repo"],
        "category": e["category"],
        "kind": e.get("kind"),
        "status": e.get("status"),
        "confidence": e.get("confidence"),
        "coverage": e.get("coverage"),
        "pattern": e.get("pattern"),
        "stage": e.get("stage"),
        "neg": e["neg"],
        "entry_type": _entry_type(e),
        "name_ambiguous": e.get("name_ambiguous", False),
        "same_name_ids": e.get("same_name_ids", []),
        "fields": e["fields"],
        "file": e["file"],
        "line": e["line"],
        "line_end": e["line_end"],
        "links": e["links"],
        "relations": e.get("relations", []),
    }


def stats():
    idx = get_index()
    import collections

    by_repo = collections.Counter(e["repo"] for e in idx.entries)
    by_cat = collections.Counter(
        (e["repo"], e["category"]) for e in idx.entries)
    by_conf = collections.Counter(
        (e["repo"], e.get("confidence") or ("neg" if e["neg"] else "?"))
        for e in idx.entries)
    by_entry_type = collections.Counter(_entry_type(e) for e in idx.entries)
    by_status = collections.Counter(
        (e["repo"], e.get("status") or ("negative" if e["neg"] else "?")) for e in idx.entries)
    name_collisions = [
        {"repo": repo, "name": name, "ids": ids}
        for (repo, name), ids in sorted(idx.name_collisions.items())
    ]
    return {
        "total": len(idx.entries),
        "by_repo": dict(by_repo),
        "by_category": {f"{r}:{c}": n for (r, c), n in sorted(by_cat.items())},
        "by_confidence": {f"{r}:{c}": n for (r, c), n in sorted(by_conf.items())},
        "by_entry_type": dict(sorted(by_entry_type.items())),
        "by_status": {f"{r}:{s}": n for (r, s), n in sorted(by_status.items())},
        "name_collisions": name_collisions,
        "links": sum(len(e["links"]) for e in idx.entries),
    }
