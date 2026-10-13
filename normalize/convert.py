#!/usr/bin/env python3
"""One-shot migration: upstream Markdown catalogs -> one YAML file per entry.

Only fields that can be derived mechanically are filled here (IDs, labels,
evidence links, relations, verbatim original text, a stack hint). Semantic
fields (title, kind, cause/effect) are left as the 待定 placeholder for the
LLM extraction pass; mitigation stays 待定 until its own later phase.

Reads the live catalogs read-only and writes under normalize/out/.
New IDs are frozen in id_registry.tsv: an entry keeps its ID across re-runs no
matter where upstream inserts new entries; unseen entries get max+1. Re-running
resets semantic fields to 待定, so run finalize.py afterwards to re-merge the
LLM outputs.
"""

import argparse
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from fault_rag.parse import parse_master  # noqa: E402

SOURCES = [
    ("INF", "inference", "/workspace/inference_error_review/FAULT_MASTER_REFERENCE.zh.md"),
    ("TRN", "training", "/workspace/training_error_review/catalog/FAULT_MASTER_REFERENCE.zh.md"),
]
REPO_NS = {"inference_error_review": "INF", "training_error_review": "TRN"}
REGISTRY = os.path.join(HERE, "id_registry.tsv")
# 2026-09-24 编目勘误：撤销的重复编号 -> 保留编号
RETIRED = {"1.70": "1.44", "11.30": "11.24"}

TBD, NA = "待定", "N/A"
FAMILIES = ("infra_off", "model_on", "ambiguous")
TRIGGERS = ("config", "yes", "partial", "other_stack", "hw_specific", "no", "not_target")
# source label `n/a` (非复现目标) is renamed so it cannot be confused with the N/A placeholder
TRIGGER_RENAME = {"n/a": "not_target"}
CONFIDENCES = ("verified", "documented", "speculative")
STAGE_ORDER = ("pretrain", "sft", "rl", "shared")

GITHUB_STACK = {
    "text-generation-inference": "tgi",
    "tensorrt-llm": "tensorrt-llm",
}
NAME_PREFIX_STACK = [
    ("ttmetal_", "tt-metal"), ("ttm_", "tt-metal"), ("ttnn_", "tt-metal"),
    ("triton_", "triton"), ("vllm_", "vllm"), ("sglang_", "sglang"),
    ("trtllm_", "tensorrt-llm"), ("flashinfer_", "flashinfer"), ("xpu_", "xpu"),
]
URL_RE = re.compile(r"https?://[^\s）)，。；、`>\]]+")
ARXIV_ID_RE = re.compile(r"arXiv[:：]\s*(\d{4}\.\d{4,5})", re.I)
CROSS_RE = re.compile(
    r"交叉[：:]\s*`?(inference_error_review|training_error_review)`?\s*\**\s*"
    r"([A-Z][A-Z0-9-]*\.\d+|K?\d+\.\d+)"
)
DATE_RE = re.compile(r"20\d\d-\d\d-\d\d")


class _Dumper(yaml.SafeDumper):
    pass


def _str_repr(dumper, s):
    style = "|" if "\n" in s else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", s, style=style)


_Dumper.add_representer(str, _str_repr)


def dump_yaml(obj):
    return yaml.dump(obj, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=10_000)


def meta_line(raw):
    for line in raw.splitlines()[1:6]:
        if "置信度" in line or "Cov" in line or "stage:" in line:
            return line.strip()
    return ""


def family_of(e):
    fam = (e.get("axis") or "").strip()
    if e["repo"] == "training":
        return NA
    if fam in FAMILIES:
        return fam
    return NA if fam == "n/a" else TBD


def trigger_of(e, meta):
    if e["repo"] == "training":
        return NA, NA
    trig = (e.get("trigger") or "").strip()
    trig = TRIGGER_RENAME.get(trig, trig)
    review = bool(re.search(r"触发\s*`[^`]+`\s*⚠", meta))
    return (trig if trig in TRIGGERS else TBD), review


def stage_of(e):
    if e["repo"] == "inference":
        return NA
    parts = {p.strip() for p in (e.get("stage") or "").split("/") if p.strip()}
    return [s for s in STAGE_ORDER if s in parts] or TBD


def coverage_of(e):
    cov = (e.get("coverage") or "").strip()
    if cov.startswith("var:"):
        return "var", cov[4:].strip()
    if cov.startswith("NEW"):
        return "NEW", None
    if cov.startswith("rationale"):
        return "rationale", None
    if cov == "existing":
        return "existing", None
    return TBD, None


def stack_hint(e, evidence):
    for ev in evidence:
        m = re.match(r"https?://github\.com/[\w.-]+/([\w.-]+)/", ev["url"])
        if m:
            repo = m.group(1).lower()
            return GITHUB_STACK.get(repo, repo)
    for prefix, stack in NAME_PREFIX_STACK:
        if e["name"].startswith(prefix):
            return stack
    return TBD


def evidence_of(source_text):
    urls = []
    for u in URL_RE.findall(source_text):
        u = u.rstrip(".,;:")
        if u not in urls:
            urls.append(u)
    for aid in ARXIV_ID_RE.findall(source_text):
        if not any(aid in u for u in urls):
            urls.append(f"https://arxiv.org/abs/{aid}")
    out = []
    for u in urls:
        if re.search(r"github\.com/[^/]+/[^/]+/pull/\d+", u):
            kind = "github_pr"
        elif re.search(r"github\.com/[^/]+/[^/]+/issues/\d+", u):
            kind = "github_issue"
        elif "arxiv.org" in u:
            kind = "arxiv"
        elif re.search(r"aclanthology|doi\.org|usenix|acm\.org|ieee|openreview|hpca|sigops", u, re.I):
            kind = "paper"
        else:
            kind = "other"
        out.append({"url": u, "kind": kind})
    return out


def load_registry(path=REGISTRY):
    """(ns, legacy_id) -> new_id for every ID ever assigned."""
    reg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                new_id, legacy = line.rstrip("\n").split("\t")
                reg[(new_id[:3], legacy)] = new_id
    return reg


def save_registry(reg, path=REGISTRY):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("new_id\tlegacy_id\n")
        for (_, legacy), new_id in sorted(reg.items(), key=lambda kv: kv[1]):
            fh.write(f"{new_id}\t{legacy}\n")


def build(entries_by_ns, registry):
    """entries_by_ns: list of (ns, repo, [parsed entries]) in document order.

    Mutates `registry` with IDs assigned to entries it has not seen before.
    """
    # pass 1: assign IDs (stable via the registry) and build lookup tables
    by_legacy, by_slug = {}, {}
    rows = []
    for ns, repo, entries in entries_by_ns:
        used = [int(v[4:]) for (n, _), v in registry.items() if n == ns]
        next_num = max(used, default=0) + 1
        for e in entries:
            new_id = registry.get((ns, e["id"]))
            if new_id is None:
                new_id = f"{ns}-{next_num:04d}"
                registry[(ns, e["id"])] = new_id
                next_num += 1
            rows.append((new_id, ns, e))
            by_legacy[(ns, e["id"])] = new_id
            for slug in [e["name"], *e.get("aliases", [])]:
                by_slug.setdefault((ns, slug), []).append(new_id)
    for old, kept in RETIRED.items():
        if ("INF", kept) in by_legacy:
            by_legacy[("INF", old)] = by_legacy[("INF", kept)]

    def resolve(ns, ref):
        if (ns, ref) in by_legacy:
            return [by_legacy[(ns, ref)]]
        if (ns, ref) in by_slug:
            return by_slug[(ns, ref)]
        other = "TRN" if ns == "INF" else "INF"
        return by_slug.get((other, ref), [])

    # pass 2: build records
    records = []
    for new_id, ns, e in rows:
        f = e["fields"]
        meta = meta_line(e["raw"])
        evidence = evidence_of(f.get("source", ""))
        trigger, review = trigger_of(e, meta)
        coverage, var_ref = coverage_of(e)

        relations = []
        if var_ref:
            targets = [t for t in resolve(ns, var_ref) if t != new_id]
            relations += [{"kind": "var", "target": t} for t in targets] or [
                {"kind": "var", "target_raw": var_ref}
            ]
        for repo_name, ref in CROSS_RE.findall(e["raw"]):
            tns = REPO_NS[repo_name]
            targets = [by_legacy[(tns, ref)]] if (tns, ref) in by_legacy else []
            relations += [{"kind": "cross", "target": t} for t in targets] or [
                {"kind": "cross", "target_raw": f"{repo_name} {ref}"}
            ]

        kind = TBD
        if e["entry_type"] == "pointer":
            kind = "指针"
        elif coverage == "rationale":
            kind = "结论"

        legacy = [e["id"]] + sorted(o for o, k in RETIRED.items() if ns == "INF" and k == e["id"])
        discovered = DATE_RE.search(f.get("discovery", "") or "")

        original = {"category": e["category"], "section": e["section"], "meta": meta}
        original.update(f)

        records.append({
            "id": new_id,
            "legacy_ids": legacy,
            "slug": e["name"],
            "aliases": e.get("aliases", []),
            "title": TBD,
            "kind": kind,
            "status": "active",
            "cause": {
                "objects": [{"subsystem": TBD, "stack": stack_hint(e, evidence), "component": TBD}],
                "behavior": TBD,
            },
            "effect": {"objects": [TBD], "detail": TBD, "visibility": TBD},
            "mitigation": {"detect": TBD, "fix": TBD},
            "labels": {
                "family": family_of(e),
                "trigger": trigger,
                "trigger_review": review,
                "trigger_note": f.get("trigger_surface", NA) if ns == "INF" else NA,
                "stage": stage_of(e),
                "confidence": e.get("confidence") or TBD,
                "coverage": coverage,
                "discovered": discovered.group(0) if discovered else TBD,
            },
            "remarks": "",
            "evidence": evidence or TBD,
            "relations": relations,
            "original": original,
        })
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()

    parsed = [(ns, repo, parse_master(path, repo)) for ns, repo, path in SOURCES]
    registry = load_registry()
    known = set(registry)
    records = build(parsed, registry)
    seen = {(ns, e["id"]) for ns, _, entries in parsed for e in entries}
    vanished = sorted(registry[k] for k in known - seen)
    if vanished:
        print(f"WARNING: {len(vanished)} registered IDs no longer in source: {vanished[:20]}")
    save_registry(registry)

    ent_dir = os.path.join(args.out, "entries")
    os.makedirs(ent_dir, exist_ok=True)
    for name in os.listdir(ent_dir):
        if name.endswith(".yaml"):
            os.remove(os.path.join(ent_dir, name))
    for r in records:
        with open(os.path.join(ent_dir, f"{r['id']}.yaml"), "w", encoding="utf-8") as fh:
            fh.write(dump_yaml(r))
    with open(os.path.join(args.out, "id_map.tsv"), "w", encoding="utf-8") as fh:
        fh.write("new_id\tlegacy_id\tslug\n")
        for r in records:
            for old in r["legacy_ids"]:
                fh.write(f"{r['id']}\t{old}\t{r['slug']}\n")
    counts = {ns: sum(r["id"].startswith(ns) for r in records) for ns, _, _ in SOURCES}
    print(f"wrote {len(records)} entries to {ent_dir} {counts}; {len(set(registry) - known)} new IDs registered")


if __name__ == "__main__":
    main()
