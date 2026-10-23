"""Index build/load: entries.jsonl + embeddings.npy + bm25.pkl + meta.json.

Incremental: embeddings are reused whenever the exact derived embedding text is
unchanged (matched by its hash, so a renumbering such as the 2026-10 move to
INF-/TRN- ids does not force re-embedding) and only recomputed when it changes. `reindex` re-parses the catalogs
(read-only) and reuses all unchanged vectors; `reindex --full` re-embeds
everything.
"""

import json
import os
import pickle
import re
import time

import numpy as np

from . import embed as embed_mod
from . import parse as parse_mod

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
ENTRIES_JSONL = os.path.join(DATA, "entries.jsonl")
EMB_NPY = os.path.join(DATA, "embeddings.npy")
BM25_PKL = os.path.join(DATA, "bm25.pkl")
META_JSON = os.path.join(DATA, "meta.json")


# ---------------------------------------------------------------------------
# tokenization (shared by BM25 index and query side)
# ---------------------------------------------------------------------------

_LATIN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-/#]*")
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize(text):
    """jieba for CJK + regex for code/latin tokens, lowercased.

    Also emits CJK bigrams so zh terms jieba splits badly still match.
    """
    text = text.lower()
    toks = _LATIN_RE.findall(text)
    cjk_runs = re.findall(r"[一-鿿]+", text)
    import jieba

    for run in cjk_runs:
        toks.extend(w for w in jieba.lcut(run) if w.strip())
        toks.extend(run[i : i + 2] for i in range(len(run) - 1))
    return [t for t in toks if t.strip()]


def bm25_text(e):
    """Full-text for lexical indexing (broader than the embedding text)."""
    f = e["fields"]
    parts = [
        e["id"],
        " ".join(e.get("legacy_ids", [])),
        e["name"],
        e["name"].replace("_", " "),
        e["category"],
        e.get("legacy_category") or "",
    ]
    # Not the other structured fields (cause_behavior / effect_detail / component /
    # effect_objects): their repeated technical terms let near-topic entries outrank
    # the right one (independent set Recall@5 0.952 -> 0.937). Revisit notes stay in.
    for key in ("title", "mechanism", "behavior", "signature", "trigger_surface",
                "repro", "source", "notes", "content"):
        if f.get(key):
            parts.append(f[key])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Okapi BM25 (inline, no dependency)
# ---------------------------------------------------------------------------


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, docs_tokens):
        self.n_docs = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = sum(self.dl) / max(1, self.n_docs)
        self.df = {}
        self.tf = []
        for toks in docs_tokens:
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
        return self

    def scores(self, query_tokens):
        import math

        n = max(1, self.n_docs)
        out = np.zeros(self.n_docs, dtype=np.float64)
        for qt in set(query_tokens):
            df = self.df.get(qt, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, counts in enumerate(self.tf):
                f = counts.get(qt, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out


# ---------------------------------------------------------------------------
# build / load
# ---------------------------------------------------------------------------


def _key(e):
    return f"{e['repo']}:{e['id']}"


def build(full=False, progress=True):
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    entries = parse_mod.parse_all()

    # load previous vectors for incremental reuse
    old, by_hash = {}, {}
    if not full and os.path.exists(EMB_NPY) and os.path.exists(ENTRIES_JSONL):
        with open(ENTRIES_JSONL, encoding="utf-8") as f:
            old_entries = [json.loads(line) for line in f]
        old_mat = np.load(EMB_NPY)
        for i, oe in enumerate(old_entries):
            if i < len(old_mat):
                # Pre-embed_hash indexes already persist embed_text. Derive the
                # old fingerprint from that text so schema migration only
                # re-embeds entries whose actual model input changed.
                old_embed_hash = oe.get("embed_hash")
                if old_embed_hash is None and "embed_text" in oe:
                    old_embed_hash = parse_mod.embedding_hash(oe["embed_text"])
                old[_key(oe)] = (old_embed_hash, old_mat[i])
                by_hash[old_embed_hash] = old_mat[i]

    texts = [e["embed_text"] for e in entries]
    dim = None
    if old:
        dim = next(iter(old.values()))[1].shape[0]

    need = [i for i, e in enumerate(entries) if e["embed_hash"] not in by_hash]
    mat = np.zeros((len(entries), dim or 1024), dtype=np.float32)
    for i, e in enumerate(entries):
        if e["embed_hash"] in by_hash:
            mat[i] = by_hash[e["embed_hash"]]

    if need:
        if progress:
            print(f"embedding {len(need)} new/changed entries "
                  f"({len(entries) - len(need)} reused)...", flush=True)
        new_vecs = embed_mod.embed(
            [texts[i] for i in need], show_progress=progress
        )
        if dim is None:
            mat = np.zeros((len(entries), new_vecs.shape[1]), dtype=np.float32)
            for i, e in enumerate(entries):
                if e["embed_hash"] in by_hash:
                    mat[i] = by_hash[e["embed_hash"]]
        for row, i in zip(new_vecs, need):
            mat[i] = row
    elif dim is None:
        # nothing to embed and no cache: still need the model for dim
        v = embed_mod.embed_one("dim probe")
        mat = np.zeros((len(entries), v.shape[0]), dtype=np.float32)

    bm25 = BM25().fit([tokenize(bm25_text(e)) for e in entries])

    with open(ENTRIES_JSONL, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    np.save(EMB_NPY, mat)
    with open(BM25_PKL, "wb") as f:
        pickle.dump(bm25, f)
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_entries": len(entries),
        "n_reembedded": len(need),
        "model": "BAAI/bge-m3",
        "dim": int(mat.shape[1]),
        "sources": {
            "inference_entries": parse_mod.INFERENCE_ENTRIES,
            "training_entries": parse_mod.TRAINING_ENTRIES,
            "inference_appendix_negatives": parse_mod.INFERENCE_MASTER,
            "training_rejected": parse_mod.TRAINING_REJECTED,
            "training_blindspots": parse_mod.TRAINING_BLINDSPOTS,
        },
        "by_repo": {r: sum(e["repo"] == r for e in entries) for r in ("inference", "training")},
        "pending_audit": sum(e.get("status") == "pending_audit" for e in entries),
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    if progress:
        print(f"index built: {len(entries)} entries, dim={mat.shape[1]}, "
              f"{len(need)} re-embedded, {time.time() - t0:.1f}s")
    return meta


def mark_name_collisions(entries):
    """Flag same-catalog name clashes in place (name_ambiguous / same_name_ids) and return
    {(repo, name): sorted ids}. Cross-catalog clashes are deliberately not flagged.
    Shared with web/build_site.py so the static site marks exactly what the CLI marks."""
    rows = {}
    for e in entries:
        rows.setdefault((e["repo"], e["name"]), []).append(e["id"])
    collisions = {key: sorted(ids) for key, ids in rows.items() if len(ids) > 1}
    for e in entries:
        ids = collisions.get((e["repo"], e["name"]), [])
        e["name_ambiguous"] = bool(ids)
        e["same_name_ids"] = ids
    return collisions


class Index:
    """Lazy-loaded in-memory index used by search/lookup/related."""

    def __init__(self):
        with open(ENTRIES_JSONL, encoding="utf-8") as f:
            self.entries = [json.loads(line) for line in f]
        self.mat = np.load(EMB_NPY)
        with open(BM25_PKL, "rb") as f:
            self.bm25 = pickle.load(f)
        self.tokens = [None] * len(self.entries)  # filled on demand if needed
        self.by_id = {}
        self.by_legacy = {}
        self.by_name = {}
        for i, e in enumerate(self.entries):
            self.by_id.setdefault(e["id"], []).append(i)
            for old in e.get("legacy_ids", []):
                self.by_legacy.setdefault(old, []).append(i)
            self.by_name.setdefault(e["name"], []).append(i)
            for alias in e.get("aliases", []):
                self.by_name.setdefault(alias, []).append(i)
        self.name_collisions = mark_name_collisions(self.entries)


_INDEX = None


def get_index():
    global _INDEX
    if _INDEX is None:
        if not os.path.exists(ENTRIES_JSONL):
            raise SystemExit(
                "index not built yet — run `fault-rag reindex` first"
            )
        _INDEX = Index()
    return _INDEX


def reset():
    global _INDEX
    _INDEX = None
