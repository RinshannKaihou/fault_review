"""Parse the vendored catalog snapshots (+ negative docs) into entries.jsonl.

Reads the snapshots under ``catalogs/`` in this repo:
  - catalogs/inference/FAULT_MASTER_REFERENCE.zh.md  (inference master)
  - catalogs/training/FAULT_MASTER_REFERENCE.zh.md   (training master)
  - catalogs/training/rejected.md
  - catalogs/training/blindspots.md

The snapshots are read-only vendored copies; the live sources of truth are
the two catalog repos (/workspace/inference_error_review,
/workspace/training_error_review). `scripts/sync_catalogs.sh` refreshes the
snapshots and rebuilds the index.

Entry header grammar (validated against the catalogs' own stated counts):
  inference:  ### 1.1 `snake_name`        ### K1.1 `snake_name`
  training:   ### HW.01 `snake_name`

Anything under a `###` header that does not match the entry grammar
(appendix subsections, KV § intros) is treated as section prose, and only
appendix C/D subsections are indexed, as negative/blindspot chunks.
"""

import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CATALOGS = os.path.join(ROOT, "catalogs")

INFERENCE_MASTER = os.path.join(CATALOGS, "inference", "FAULT_MASTER_REFERENCE.zh.md")
TRAINING_MASTER = os.path.join(CATALOGS, "training", "FAULT_MASTER_REFERENCE.zh.md")
TRAINING_REJECTED = os.path.join(CATALOGS, "training", "rejected.md")
TRAINING_BLINDSPOTS = os.path.join(CATALOGS, "training", "blindspots.md")

# Entry headers: `### 1.1 `name``  /  `### K1.1 `name``  /  `### HW.01 `name``
# Tolerates trailing notes（（arxiv）/ ⚑§9）, uppercase in names, and dual-name
# headers like `### 11.5 `name_a` / `name_b`` (second name kept as alias).
ENTRY_RE = re.compile(
    r"^### ([A-Z][A-Z0-9-]*\.\d+|\d+\.\d+) `([A-Za-z0-9_]+)`(?:\s*/\s*`([A-Za-z0-9_]+)`)?"
)
# `## 第 1 类 · 硬件 / 静默数据损坏（SDC）`  /  `## KV §1 value-corruption（值损坏）`  /  `## HW`
SECTION_RE = re.compile(r"^## (.+?)\s*$")
# - **机制**：...   (fullwidth colon; tolerate ascii colon)
FIELD_RE = re.compile(r"^- \*\*(.+?)\*\*[：:]\s*(.*)$")

# Field name normalization across the two schemas
FIELD_MAP = {
    "机制": "mechanism",
    "来源": "source",
    "行为效应": "behavior",
    "触发面": "trigger_surface",
    "隐藏态签名": "signature",
    "复现方式": "repro",
    "发现来源": "discovery",
    "观测信号": "observed_signal",
    "Locus": "locus",
    "优先级": "priority",
}

CONFIDENCES = ("verified", "documented", "speculative")
POINTER_MARKERS = ("不是新故障", "cited for completeness; not a new entry")


def _category_from_id(entry_id, repo):
    """Derive the stable category from the entry ID grammar.

    Section headings are retained as provenance, but cannot define categories:
    inference masters contain nested ``## §...`` headings inside a numbered
    category, and treating those as category boundaries loses ``catN``.
    """
    if repo == "inference":
        m = re.fullmatch(r"(\d+)\.\d+", entry_id)
        if m:
            return f"cat{m.group(1)}"
        m = re.fullmatch(r"K(\d+)\.\d+", entry_id)
        if m:
            return f"KV{m.group(1)}"
    elif repo == "training":
        m = re.fullmatch(r"([A-Z][A-Z0-9-]*)\.\d+", entry_id)
        if m:
            return m.group(1)
    raise ValueError(f"unsupported entry id: repo={repo!r} id={entry_id!r}")


def _parse_meta(meta_line, repo):
    """Parse the backtick metadata line following an entry header.

    training:  stage: `pretrain/shared` · Cov: `NEW` · 置信度 `documented`
    inference: `infra_off` · Cov: `var:1.212` · 触发 `yes` · 置信度 `verified`
    """
    meta = {"stage": None, "axis": None, "coverage": None, "trigger": None, "confidence": None}
    tokens = re.findall(r"`([^`]+)`", meta_line)
    if repo == "training":
        m = re.search(r"stage\s*[：:]\s*`([^`]+)`", meta_line)
        if m:
            meta["stage"] = m.group(1)
    else:
        # first backtick token is the detector-axis label (may carry a ⚠ note outside)
        m = re.match(r"^\s*`([^`]+)`", meta_line)
        if m:
            meta["axis"] = m.group(1)
    m = re.search(r"Cov\s*[：:]\s*`([^`]+)`", meta_line)
    if m:
        meta["coverage"] = m.group(1)
    m = re.search(r"触发\s*`([^`]+)`", meta_line)
    if m:
        meta["trigger"] = m.group(1)
    for c in CONFIDENCES:
        if f"`{c}`" in meta_line:
            meta["confidence"] = c
            break
    if not tokens and meta_line.strip():
        meta["unparsed"] = meta_line.strip()
    return meta


def _classify_catalog_entry(e):
    """Classify a catalog header without conflating pointers and negatives."""
    raw_lower = e.get("raw", "").lower()
    is_explicit_pointer = (
        (e.get("coverage") or "").lower() == "existing"
        and (e.get("trigger") or "").lower() == "n/a"
        and not e.get("confidence")
        and any(marker in raw_lower for marker in POINTER_MARKERS)
    )
    return "pointer" if is_explicit_pointer else "fault"


def parse_master(path, repo):
    """Yield entry dicts from one master catalog file."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    entries = []
    section = ""
    section_line = 0
    cur = None          # current entry dict being accumulated
    cur_field = None    # current normalized field name for continuation lines
    fence = None

    def flush():
        nonlocal cur, cur_field
        if cur is not None:
            cur["raw"] = "".join(cur["_raw_lines"]).rstrip("\n")
            del cur["_raw_lines"]
            cur["entry_type"] = _classify_catalog_entry(cur)
            entries.append(cur)
            cur = None
            cur_field = None

    def chapter_boundary(i):
        # i is the index immediately after '---'; only a confirmed chapter flushes.
        if i >= len(lines) or lines[i].strip():
            return False
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        heading = re.fullmatch(r"# ([A-Z][A-Z0-9-]*)(?: · .+)?\s*", lines[j]) if j < len(lines) else None
        if not heading:
            return False
        chapter = heading.group(1)
        j += 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j == len(lines) or lines[j].strip() != f"## {chapter}":
            return False
        j += 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        entry = ENTRY_RE.match(lines[j]) if j < len(lines) else None
        return bool(entry and entry.group(1).startswith(f"{chapter}."))

    for i, line in enumerate(lines, start=1):
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence_match:
            marker, rest = fence_match.groups()
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not rest.strip():
                fence = None
        if (
            repo == "training" and cur is not None and fence is None
            and line.startswith("---") and line.strip() == "---"
            and chapter_boundary(i)
        ):
            flush()
            continue
        sec = SECTION_RE.match(line)
        ent = ENTRY_RE.match(line)

        if sec and not ent:
            flush()
            section = sec.group(1)
            section_line = i
            continue

        if ent:
            flush()
            cur = {
                "id": ent.group(1),
                "name": ent.group(2),
                "aliases": [ent.group(3)] if ent.group(3) else [],
                "repo": repo,
                "category": _category_from_id(ent.group(1), repo),
                "section": section,
                "section_line": section_line,
                "file": path,
                "line": i,
                "neg": False,
                "fields": {},
                "_raw_lines": [line],
            }
            cur_field = None
            continue

        if cur is not None:
            cur["_raw_lines"].append(line)
            fm = FIELD_RE.match(line)
            if fm:
                zh, val = fm.group(1).strip(), fm.group(2).strip()
                key = FIELD_MAP.get(zh, zh)
                cur["fields"][key] = val
                cur_field = key
            elif line.strip().startswith("- **") and cur_field:
                # a new bold field we did not map; FIELD_RE should have caught it
                pass
            elif cur_field and line.strip() and not line.startswith("###"):
                # continuation of a multi-line field value
                cur["fields"][cur_field] += "\n" + line.rstrip("\n")
            # metadata line: the first non-empty line after the header that is
            # not a field and carries backticks / stage:
            if (
                "stage:" in line or "置信度" in line or "· Cov" in line
            ) and "stage" not in cur:
                cur.update(_parse_meta(line, repo))
    flush()

    # entries missing the metadata line still need the keys
    for e in entries:
        for k in ("stage", "axis", "coverage", "trigger", "confidence"):
            e.setdefault(k, None)
        e["line_end"] = e["line"] + e["raw"].count("\n")
    return entries


def parse_negative_doc(path, repo, kind):
    """Chunk a negative-result doc on its `##`/`###` subsections.

    kind: 'rejected' | 'blindspot' | 'appendix'
    Returns one chunk per subsection; a doc with no subsections becomes one chunk.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    base = os.path.basename(path).replace(".md", "").upper().replace("_", "-")
    chunks = []
    cur_title, cur_start, cur_lines = None, 1, []

    def flush(end_line):
        nonlocal cur_title, cur_lines
        text = "".join(cur_lines).strip()
        if cur_title is not None and text:
            chunks.append({
                "id": f"NEG-{base}-{len(chunks) + 1}",
                "name": f"{kind}:{cur_title.strip()}"[:120],
                "repo": repo,
                "category": f"neg_{kind}",
                "section": cur_title.strip(),
                "section_line": cur_start,
                "file": path,
                "line": cur_start,
                "line_end": end_line,
                "neg": True,
                "entry_type": "negative",
                "stage": None, "axis": None, "coverage": None,
                "trigger": None, "confidence": None,
                "fields": {"content": text},
                "raw": text,
            })
        cur_lines = []

    for i, line in enumerate(lines, start=1):
        if re.match(r"^#{2,3} ", line):
            flush(i - 1)
            cur_title = line.lstrip("#").strip()
            cur_start = i
            cur_lines = [line]
        elif cur_title is None:
            # doc head (title + intro) — start an implicit first chunk
            cur_title = f"{base} head"
            cur_start = 1
            cur_lines = [line]
        else:
            cur_lines.append(line)
    flush(len(lines))
    return chunks


def parse_appendix(path, repo, letters=("C", "D")):
    """Chunk inference-master appendix C/D subsections as negative/blindspot docs."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    in_appendix = False
    cur_title, cur_start, cur_lines = None, 0, []

    def flush(end_line):
        nonlocal cur_lines
        text = "".join(cur_lines).strip()
        if cur_title is not None and text:
            m = re.match(r"([CD])\.(\d+)", cur_title)
            cid = f"NEG-INF-{m.group(1)}{m.group(2)}" if m else f"NEG-INF-X{len(chunks)}"
            chunks.append({
                "id": cid,
                "name": f"appendix:{cur_title.strip()}"[:120],
                "repo": repo,
                "category": "neg_appendix",
                "section": cur_title.strip(),
                "section_line": cur_start,
                "file": path,
                "line": cur_start,
                "line_end": end_line,
                "neg": True,
                "entry_type": "negative",
                "stage": None, "axis": None, "coverage": None,
                "trigger": None, "confidence": None,
                "fields": {"content": text},
                "raw": text,
            })
        cur_lines = []

    for i, line in enumerate(lines, start=1):
        if line.startswith("# 附录"):
            in_appendix = True
        if not in_appendix:
            continue
        sec = re.match(r"^### ([CD])\.\d+ ", line)
        if sec:
            flush(i - 1)
            cur_title = line.lstrip("#").strip()
            cur_start = i
            cur_lines = [line]
        elif cur_title is not None:
            if line.startswith("### ") or line.startswith("## "):
                flush(i - 1)
                cur_title = None
            else:
                cur_lines.append(line)
    if cur_title is not None:
        flush(len(lines))
    return chunks


def build_links(entries):
    """Wire cross-references between entries.

    Sources: `var:<name>` coverage, `交叉：<repo> <id>` pointers, and exact
    mentions of known ids / snake_names in the raw text. Only ids/names that
    actually exist in the corpus are kept (no false positives from version
    numbers like 2.7.1).
    """
    by_id = {}
    by_name = {}
    for e in entries:
        by_id.setdefault(e["id"], []).append(e)
        by_name.setdefault(e["name"], []).append(e)
        for alias in e.get("aliases", []):
            by_name.setdefault(alias, []).append(e)

    # long ids first so `RL-RO.01` wins over a bare `1.1` style prefix clash
    known_ids = sorted(by_id, key=len, reverse=True)
    id_pat = re.compile(
        r"(?<![\w.])(" + "|".join(re.escape(i) for i in known_ids) + r")(?![\w.])"
    ) if known_ids else None

    for e in entries:
        links = {}  # (repo, id) -> kind, strongest wins
        prio = {"var": 3, "cross": 2, "mentions": 1}

        def add(kind, t):
            if t is e:
                return
            key = (t["repo"], t["id"])
            if prio[kind] > prio.get(links.get(key), 0):
                links[key] = kind

        cov = e.get("coverage") or ""
        if cov.startswith("var:"):
            target = cov[4:]
            # training: var:<snake_name>; inference: var:<id> (e.g. var:1.212)
            for t in by_name.get(target, []) + by_id.get(target, []):
                add("var", t)
        for m in re.finditer(r"交叉[：:]\s*(\S+)\s+([A-Z]*\d+\.\d+)", e["raw"]):
            for t in by_id.get(m.group(2), []):
                add("cross", t)
        if id_pat:
            for m in id_pat.finditer(e["raw"]):
                for t in by_id.get(m.group(1), []):
                    add("mentions", t)
        e["links"] = sorted(
            ({"kind": k, "repo": r, "id": i} for (r, i), k in links.items()),
            key=lambda d: (d["kind"], d["repo"], d["id"]),
        )
    return entries


def content_hash(e):
    h = hashlib.sha1()
    h.update(e["raw"].encode("utf-8"))
    return h.hexdigest()[:16]


def embedding_hash(text):
    """Fingerprint exactly the derived text consumed by the embedder."""
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def embed_text(e):
    """Composed text fed to the embedding model."""
    f = e["fields"]
    if e["neg"]:
        parts = [e["name"], f.get("content", "")]
    else:
        parts = [
            e["name"].replace("_", " "),
            f"category {e['category']}",
            f.get("mechanism", ""),
            f.get("behavior", ""),
            f.get("signature", ""),
            f.get("trigger_surface", ""),
        ]
    return "\n".join(p for p in parts if p)[:4000]


def parse_all():
    entries = []
    entries += parse_master(INFERENCE_MASTER, "inference")
    entries += parse_master(TRAINING_MASTER, "training")
    entries += parse_negative_doc(TRAINING_REJECTED, "training", "rejected")
    entries += parse_negative_doc(TRAINING_BLINDSPOTS, "training", "blindspot")
    entries += parse_appendix(INFERENCE_MASTER, "inference")
    entries = build_links(entries)
    for e in entries:
        e["hash"] = content_hash(e)
        e["embed_text"] = embed_text(e)
        e["embed_hash"] = embedding_hash(e["embed_text"])
    return entries


if __name__ == "__main__":
    import sys

    es = parse_all()
    by_repo = {}
    for e in es:
        by_repo.setdefault(e["repo"], []).append(e)
    for repo, lst in sorted(by_repo.items()):
        neg = sum(1 for e in lst if e["neg"])
        print(f"{repo}: {len(lst)} entries ({neg} negative-doc chunks)")
    n_links = sum(len(e["links"]) for e in es)
    print(f"total: {len(es)} entries, {n_links} cross-reference links")
    if "--json" in sys.argv:
        json.dump(es, sys.stdout, ensure_ascii=False, indent=1)
