"""Hybrid retrieval: BM25 + dense cosine, fused with Reciprocal Rank Fusion."""

import numpy as np

from . import embed as embed_mod
from .index import get_index, tokenize

RRF_K = 60


def _entry_type(e):
    """Read the schema field with compatibility for pre-migration indexes."""
    return e.get("entry_type", "negative" if e.get("neg") else "fault")


def _passes(e, repo=None, category=None, confidence=None, trigger=None,
            neg=None, include_pointers=False):
    if _entry_type(e) == "pointer" and not include_pointers:
        return False
    if repo and e["repo"] != repo:
        return False
    if category:
        cat = category.lower().lstrip("cat")
        ec = e["category"].lower()
        # accept NUM / cat1 / 1 / KV1 / kv1 / neg_rejected forms
        if not (ec == category.lower() or ec == "cat" + cat or ec == "kv" + cat
                or ec.lstrip("catkv") == cat):
            return False
    if confidence and (e.get("confidence") or "") != confidence:
        return False
    if trigger and (e.get("trigger") or "") != trigger:
        return False
    if neg is not None and bool(e["neg"]) != neg:
        return False
    return True


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


def search(query, topk=8, mode="hybrid", repo=None, category=None,
           confidence=None, trigger=None, include_neg=True,
           include_pointers=False):
    idx = get_index()
    neg_flag = None if include_neg else False

    eligible = [
        i for i, e in enumerate(idx.entries)
        if _passes(e, repo, category, confidence, trigger, neg=neg_flag,
                   include_pointers=include_pointers)
    ]
    if not eligible:
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

    top = sorted(fused, key=lambda i: -fused[i])[:topk]
    out = []
    for rank, i in enumerate(top, start=1):
        e = idx.entries[i]
        out.append({
            "rank": rank,
            "score": round(fused[i], 6),
            "dense_rank": dense_r.get(i),
            "bm25_rank": bm25_r.get(i),
            "id": e["id"],
            "name": e["name"],
            "repo": e["repo"],
            "category": e["category"],
            "confidence": e.get("confidence"),
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
    """Exact id / name / alias, then substring fallback on names."""
    idx = get_index()
    rows = []
    if q in idx.by_id:
        rows = idx.by_id[q]
    elif q in idx.by_name:
        rows = idx.by_name[q]
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


def _full(e):
    return {
        "id": e["id"],
        "name": e["name"],
        "repo": e["repo"],
        "category": e["category"],
        "confidence": e.get("confidence"),
        "coverage": e.get("coverage"),
        "stage": e.get("stage"),
        "axis": e.get("axis"),
        "trigger": e.get("trigger"),
        "neg": e["neg"],
        "entry_type": _entry_type(e),
        "name_ambiguous": e.get("name_ambiguous", False),
        "same_name_ids": e.get("same_name_ids", []),
        "fields": e["fields"],
        "file": e["file"],
        "line": e["line"],
        "line_end": e["line_end"],
        "links": e["links"],
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
        "name_collisions": name_collisions,
        "links": sum(len(e["links"]) for e in idx.entries),
    }
