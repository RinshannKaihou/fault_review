#!/usr/bin/env python3
"""Build the static (GitHub Pages) fault-rag site into web/dist.

Reads the vendored catalog snapshot through fault_rag.parse — read-only, the same entries
`fault-rag reindex` indexes — and writes:

  data/meta.json                 snapshot, counts, filter vocabularies, semantic-tier config
  data/docs.json                 one slim row per entry: what results, filters, lookup, related need
  data/bm25.json                 bigram BM25 postings (tokenize() below; mirrored in src/search.js)
  data/entry/<repo>/<id>.json    full entry for the detail view, fetched on demand
  data/vectors.f32               multilingual-e5-small corpus vectors (embed_corpus.mjs), optional

and copies src/ plus the vendored transformers.js runtime (from web/node_modules).

Why bigram BM25 instead of the CLI's jieba tokenizer: it needs no segmenter in the browser and,
measured 2026-10-23 on the 2117+364 snapshot, scores held-out colloquial queries higher
(R@5 0.80 vs 0.68) at equal indy R@5 (0.905). See web/README.md.

Usage:
  python3 web/build_site.py                  # keyword tier + semantic vectors (needs node + model)
  python3 web/build_site.py --no-vectors     # keyword tier only
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from fault_rag import parse as parse_mod                     # noqa: E402
from fault_rag.index import BM25, bm25_text, mark_name_collisions   # noqa: E402
from fault_rag.search import _entry_type, _full, _summary    # noqa: E402

TOKENIZER = "bigram-v1"
_LATIN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-/#]*")
_CJK_RUN_RE = re.compile(r"[一-鿿]+")

# web/node_modules files the browser's semantic tier loads (see src/app.js).
VENDOR = {
    "transformers.min.js": "@huggingface/transformers/dist/transformers.min.js",
    "ort-wasm-simd-threaded.mjs": "onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm": "onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
}


def tokenize(text):
    """Latin/code tokens plus overlapping CJK bigrams (a lone CJK char stands for itself).
    Must stay byte-identical to tokenize() in src/search.js; web/eval/check_web.py checks it."""
    text = text.lower()
    toks = _LATIN_RE.findall(text)
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def entry_yaml(e):
    """The entry's own YAML (fault and pointer entries), minus the pipeline audit trail that the
    source file keeps; None for negative chunks, which come from Markdown."""
    import yaml

    if not e["file"].endswith(".yaml"):
        return None
    y = yaml.safe_load(e["raw"]) or {}
    y.pop("migration", None)
    return y


def doc_row(e, y):
    """What search results, filters, browsing, lookup and related read; mirrors fault_rag.search
    output plus the first-seen date the browse view sorts by."""
    discovered = str(((y or {}).get("labels") or {}).get("discovered") or "")
    return {
        "id": e["id"],
        "name": e["name"],
        "title": e.get("title"),
        "summary": _summary(e),
        # null (not []) when the entry has no legacy_ids key: _category_matches reads that.
        "legacy_ids": e.get("legacy_ids"),
        "aliases": e.get("aliases", []),
        "repo": e["repo"],
        "category": e["category"],
        "legacy_category": e.get("legacy_category"),
        "confidence": e.get("confidence"),
        "status": e.get("status"),
        "pattern": e.get("pattern"),
        "trigger": e.get("trigger"),
        "neg": e["neg"],
        "entry_type": _entry_type(e),
        "name_ambiguous": e.get("name_ambiguous", False),
        "same_name_ids": e.get("same_name_ids", []),
        "file": rel(e["file"]),
        "line": e["line"],
        "line_end": e.get("line_end"),
        "links": e["links"],
        "discovered": discovered if re.fullmatch(r"\d{4}-\d{2}-\d{2}", discovered) else None,
    }


def entry_detail(e, y):
    d = _full(e)
    d["file"] = rel(e["file"])
    d["summary"] = _summary(e)
    d["yaml"] = y
    return d


def bm25_payload(entries):
    bm = BM25().fit([tokenize(bm25_text(e)) for e in entries])
    postings = {}
    for i, counts in enumerate(bm.tf):
        for term, n in counts.items():
            postings.setdefault(term, []).extend((i, n))
    return {"tokenizer": TOKENIZER, "k1": bm.k1, "b": bm.b, "n": bm.n_docs,
            "avgdl": bm.avgdl, "dl": bm.dl, "postings": postings}


def read_sources():
    out = []
    with open(os.path.join(ROOT, "catalogs", "SOURCES.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, commit, date, *rest = line.split()
            entries = next((r.split("=", 1)[1] for r in rest if r.startswith("entries=")), None)
            out.append({"repo": name, "commit": commit, "date": date,
                        "entries": int(entries) if entries else None})
    return out


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", ROOT, *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def repo_url():
    if os.environ.get("GITHUB_REPOSITORY"):
        return f"https://github.com/{os.environ['GITHUB_REPOSITORY']}"
    url = git("remote", "get-url", "origin") or ""
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    return f"https://github.com/{m.group(1)}" if m else None


def counted(values):
    counts = {}
    for v in values:
        if v:
            counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def write_json(path, obj, pretty=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        else:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "dist"))
    ap.add_argument("--no-vectors", action="store_true",
                    help="skip the semantic tier (no node / model needed)")
    args = ap.parse_args(argv)

    with open(os.path.join(HERE, "src", "model.json"), encoding="utf-8") as f:
        model = json.load(f)

    entries = parse_mod.parse_all()
    mark_name_collisions(entries)

    out = os.path.abspath(args.out)
    if os.path.isdir(out):
        previous_build = os.path.exists(os.path.join(out, "data", "meta.json"))
        if os.listdir(out) and not previous_build:
            raise SystemExit(f"refusing to replace {out}: not empty and not a previous site build")
        shutil.rmtree(out)
    shutil.copytree(os.path.join(HERE, "src"), out)
    open(os.path.join(out, ".nojekyll"), "w").close()
    data = os.path.join(out, "data")

    ys = [entry_yaml(e) for e in entries]
    write_json(os.path.join(data, "docs.json"), [doc_row(e, y) for e, y in zip(entries, ys)])
    write_json(os.path.join(data, "bm25.json"), bm25_payload(entries))
    for e, y in zip(entries, ys):
        write_json(os.path.join(data, "entry", e["repo"], e["id"] + ".json"), entry_detail(e, y))

    semantic = None
    if not args.no_vectors:
        nm = os.path.join(HERE, "node_modules")
        missing = [src for src in VENDOR.values() if not os.path.exists(os.path.join(nm, src))]
        if missing:
            raise SystemExit(f"missing {missing}: run `npm ci` in web/ (or pass --no-vectors)")
        os.makedirs(os.path.join(out, "vendor"))
        for name, src in VENDOR.items():
            shutil.copy(os.path.join(nm, src), os.path.join(out, "vendor", name))
        work = os.path.join(HERE, ".build")
        os.makedirs(work, exist_ok=True)
        passages = os.path.join(work, "passages.json")
        write_json(passages, [e["embed_text"] for e in entries])
        vec_path = os.path.join(data, "vectors.f32")
        subprocess.run(["node", os.path.join(HERE, "embed_corpus.mjs"), passages, vec_path],
                       check=True)
        dim = os.path.getsize(vec_path) // 4 // len(entries)
        semantic = {**model, "dim": dim, "n": len(entries), "vectors": "data/vectors.f32"}

    commit = os.environ.get("GITHUB_SHA") or git("rev-parse", "HEAD")
    meta = {
        "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "site_commit": commit,
        "repo_url": repo_url(),
        "sources": read_sources(),
        "counts": {
            "total": len(entries),
            "by_repo": dict(counted(e["repo"] for e in entries)),
            "by_entry_type": dict(counted(_entry_type(e) for e in entries)),
            "pending_audit": sum(e.get("status") == "pending_audit" for e in entries),
        },
        "filters": {
            "category": {r: counted(e["category"] for e in entries if e["repo"] == r and not e["neg"])
                         for r in ("inference", "training")},
            "confidence": [c for c, _ in counted(e.get("confidence") for e in entries)],
            "trigger": [t for t, _ in counted(e.get("trigger") for e in entries)],
        },
        "tokenizer": TOKENIZER,
        "semantic": semantic,
    }
    write_json(os.path.join(data, "meta.json"), meta, pretty=True)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print(f"site built: {out} ({len(entries)} entries, {size / 2**20:.1f} MB, "
          f"semantic={'on' if semantic else 'off'})")


if __name__ == "__main__":
    main()
