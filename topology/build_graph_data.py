#!/usr/bin/env python3
"""Build normalized client-side causal graph payload (`causal_graph.json`) from catalog entries.

Implements confirmed architectural decisions (without modifying upstream YAML):
  1. Deduplicates `cause.objects` subsystems and `effect.objects` carriers in projection.
  2. Normalizes inverted `effect.objects` sequences (`causal_inversion`) to canonical causal order,
     recording `norm_inverted: true` for transparency.
  3. Resolves dynamic `effective_stages` per entry so the 74 pure-training entries inside
     `inference_error_review` (`*_bw`, `moreh_adam`, `mse_loss`, etc.) mount onto training stages.

Zero third-party dependencies (runs with Python 3 standard library).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from typing import Any

from topology.audit_topology import (
    ENTRIES_JSONL,
    TOPOLOGY_PATH,
    classify_carrier_pair,
    load_entries,
    load_topology,
    rel_path,
    split_eff,
    split_semi,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(ROOT, "web", "dist", "data", "causal_graph.json")

STACK_GROUPS: list[tuple[str, str, list[str]]] = [
    ("vllm", "vLLM 生态 (vllm / vllm-ascend)", ["vllm", "vllm-ascend"]),
    ("sglang", "SGLang 生态 (sglang / sgl-kernel)", ["sglang", "sgl-deep-gemm", "sgl-kernel-npu"]),
    ("ascend", "Ascend / CANN 生态", ["cann", "mindspore", "torch_npu", "mindformers", "mindspeed", "mindspeed-llm", "mindspeed-rl", "torchair", "triton-ascend", "msmodelslim", "msprobe", "mssanitizer", "vllm-ascend", "sgl-kernel-npu"]),
    ("tt-metal", "Tenstorrent (tt-metal / sfpi)", ["tt-metal", "sfpi"]),
    ("nvidia_trt", "CUDA / TensorRT-LLM / cuDNN / Flash", ["tensorrt-llm", "cuda", "cudnn", "flashinfer", "flash-attention", "flashmla", "deepgemm", "deepep", "nccl"]),
    ("dist_train", "分布式训练 (Megatron / DeepSpeed / TorchTitan)", ["megatron-lm", "megatron-bridge", "deepspeed", "torchtitan"]),
    ("rl_post", "RL 与后训练 (TRL / veRL / OpenRLHF / SWIFT)", ["trl", "verl", "openrlhf", "ms-swift", "peft"]),
    ("hf_torch", "Transformers / PyTorch / Triton", ["transformers", "pytorch", "triton"]),
]


def resolve_effective_stages(
    repo: str,
    raw_stage: str | None,
    subs: list[str],
    effs: list[str],
) -> tuple[list[str], bool]:
    """Return (effective_stages, cross_mounted_from_inf)."""
    train_only_subs = {"RL训练回路", "优化器", "损失计算", "数据管线"}
    train_only_carriers = {"梯度", "优化器状态", "训练样本流", "训练信号", "训练轨迹"}
    serving_carriers = {"输出内容", "KV状态", "输出分布"}

    stages: list[str] = []
    cross_mounted = False

    if repo == "training":
        parts = [p.strip() for p in str(raw_stage or "shared").split("/") if p.strip()]
        if "shared" in parts or not parts:
            stages.extend(["pretrain", "sft", "rl"])
        else:
            for p in parts:
                if p in ("pretrain", "sft", "rl"):
                    stages.append(p)
        if any(s in {"权重/检查点", "量化"} for s in subs) and "模型权重" in effs:
            stages.append("ckpt_export")
        if any(s == "观测/评测" for s in subs) or any(c in {"测试判定", "指标/日志"} for c in effs):
            stages.append("eval_obs")
    else:
        touches_train = bool((set(subs) & train_only_subs) or (set(effs) & train_only_carriers))
        touches_serving = bool(set(effs) & serving_carriers)
        if touches_train and not touches_serving:
            cross_mounted = True
            if "RL训练回路" in subs:
                stages.append("rl")
            else:
                stages.extend(["pretrain", "sft", "rl"])
        else:
            stages.append("serving")
            if touches_train:
                stages.extend(["pretrain", "sft", "rl"])
            if "RL训练回路" in subs:
                stages.append("rl")
        if any(s in {"权重/检查点", "量化"} for s in subs) and "模型权重" in effs:
            stages.append("ckpt_export")
        if any(s == "观测/评测" for s in subs) or any(c in {"测试判定", "指标/日志"} for c in effs):
            stages.append("eval_obs")

    return (list(dict.fromkeys(stages)), cross_mounted)


def normalize_effect_sequence(
    raw_effs: list[str],
    entry: dict[str, Any],
    dedup_subs: list[str],
    topo_subs: dict[str, Any],
    topo_carriers: dict[str, Any],
    forward_set: set[tuple[str, str]],
    wildcard_sinks: set[str],
    coupled_set: set[tuple[str, str]],
    loops: list[dict[str, Any]],
) -> tuple[list[str], bool]:
    """Deduplicate placeholders/repeats and normalize inverted effect sequences by causal rank."""
    valid = list(dict.fromkeys(e for e in raw_effs if e in topo_carriers))
    if len(valid) <= 1:
        return (valid, False)

    has_inversion = False
    for i in range(len(valid) - 1):
        cls, _ = classify_carrier_pair(
            valid[i], valid[i + 1], entry, dedup_subs, forward_set, wildcard_sinks, coupled_set, loops
        )
        if cls == "causal_inversion":
            has_inversion = True
            break

    if not has_inversion:
        return (valid, False)

    # Determine whether 模型权重 acts as an initial loaded/attacked input (rank 12) vs backward update sink (90)
    has_bwd_update = bool(set(valid) & {"梯度", "优化器状态", "训练信号"}) or bool(
        set(dedup_subs) & {"优化器", "损失计算", "RL训练回路"}
    )
    primary_union: set[str] = set()
    for s in dedup_subs:
        if s in topo_subs:
            primary_union.update(topo_subs[s]["primary_carriers"])

    def sort_key(cid: str) -> int:
        eff_rank = topo_carriers[cid]["rank"]
        if cid == "模型权重" and not has_bwd_update and ("模型权重" in primary_union or not dedup_subs):
            eff_rank = 12
        return eff_rank

    reordered = sorted(valid, key=sort_key)
    return (reordered, reordered != valid)


def build_graph_payload(
    topo: dict[str, Any] | None = None,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    topo = topo or load_topology(TOPOLOGY_PATH)
    entries = entries if entries is not None else load_entries(ENTRIES_JSONL)

    topo_subs = {s["id"]: s for s in topo["subsystems"]}
    topo_carriers = {c["id"]: c for c in topo["carriers"]}
    forward_set = {(e["from"], e["to"]) for e in topo["canonical_forward_edges"] if e["from"] != "*"}
    wildcard_sinks = {e["to"] for e in topo["canonical_forward_edges"] if e["from"] == "*"}
    coupled_set = {(p["a"], p["b"]) for p in topo["intra_step_coupled_pairs"]}
    loops = topo["feedback_loops"]

    sub_list = [s["id"] for s in topo["subsystems"]]
    sub_idx = {sid: i for i, sid in enumerate(sub_list)}
    car_list = [c["id"] for c in topo["carriers"]]
    car_idx = {cid: i for i, cid in enumerate(car_list)}
    stage_list = [st["id"] for st in topo["stages"]]
    stage_idx = {stid: i for i, stid in enumerate(stage_list)}

    vis_list = ["静默错误", "质量退化", "数值异常", "不可复现", "性能退化", "挂起", "报错", "待定"]
    vis_idx = {v: i for i, v in enumerate(vis_list)}
    conf_list = ["verified", "documented", "speculative", "待定"]
    conf_idx = {c: i for i, c in enumerate(conf_list)}

    stack_vocab: list[str] = []
    stack_to_idx: dict[str, int] = {}

    def get_stack_idx(st_name: str) -> int:
        if st_name not in stack_to_idx:
            stack_to_idx[st_name] = len(stack_vocab)
            stack_vocab.append(st_name)
        return stack_to_idx[st_name]

    pat_vocab: list[dict[str, Any]] = []
    pat_to_idx: dict[str, int] = {}

    def get_pat_idx(pid: str | None, ptitle: str | None, repo: str) -> int:
        if not pid:
            return -1
        if pid not in pat_to_idx:
            pat_to_idx[pid] = len(pat_vocab)
            pat_vocab.append({"id": pid, "title": ptitle or "待定", "repo": repo})
        return pat_to_idx[pid]

    methods_by_carrier: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    compact_faults: list[list[Any]] = []
    norm_inverted_count = 0
    dedup_fixed_count = 0
    cross_mounted_count = 0

    for e in entries:
        if e.get("neg"):
            continue
        f = e.get("fields") or {}
        raw_subs = split_semi(f.get("subsystem"))
        raw_effs = split_eff(f.get("effect_objects"))
        dedup_subs = list(dict.fromkeys(s for s in raw_subs if s in topo_subs))

        if e.get("kind") == "方法":
            for ef in dict.fromkeys(x for x in raw_effs if x in topo_carriers):
                methods_by_carrier[ef].append({
                    "id": e["id"],
                    "repo": e["repo"],
                    "name": e["name"],
                    "title": e.get("title") or "",
                    "confidence": e.get("confidence") or "documented",
                    "file": rel_path(e.get("file", "")),
                })
            continue

        if e.get("kind") != "故障":
            continue

        if len(raw_subs) != len(dedup_subs) and "待定" not in raw_subs:
            dedup_fixed_count += 1

        norm_effs, was_inverted = normalize_effect_sequence(
            raw_effs, e, dedup_subs, topo_subs, topo_carriers,
            forward_set, wildcard_sinks, coupled_set, loops
        )
        if was_inverted:
            norm_inverted_count += 1
        if len(raw_effs) > len(set(raw_effs)):
            dedup_fixed_count += 1

        eff_stages, cross_mounted = resolve_effective_stages(
            e["repo"], e.get("stage"), dedup_subs, norm_effs
        )
        if cross_mounted:
            cross_mounted_count += 1

        stacks = list(dict.fromkeys(s.lower() for s in split_semi(f.get("stack")) if s and s not in ("待定", "N/A")))
        vis = (f.get("visibility") or "待定").strip()
        if vis not in vis_idx:
            vis = "待定"
        conf = e.get("confidence") or "待定"
        if conf not in conf_idx:
            conf = "待定"

        has_det = 1 if (f.get("mitigation_detect") or "").strip() not in ("", "待定", "N/A") else 0
        has_sig = 1 if (f.get("signature") or "").strip() not in ("", "待定", "N/A") else 0
        has_fix = 1 if (f.get("mitigation_fix") or "").strip() not in ("", "待定", "N/A") else 0
        has_cross = 1 if any(rel.get("kind") == "cross" for rel in (e.get("relations") or [])) else 0

        flags = (
            (1 if was_inverted else 0)
            | (2 if cross_mounted else 0)
            | (4 if has_det else 0)
            | (8 if has_sig else 0)
            | (16 if has_fix else 0)
            | (32 if has_cross else 0)
            | (64 if not dedup_subs else 0)
        )

        # Compact tuple schema (14 elements):
        # [0:id, 1:repo_bit(0=inf,1=trn), 2:name, 3:title, 4:sub_indices, 5:norm_car_indices,
        #  6:stage_indices, 7:stack_indices, 8:vis_idx, 9:conf_idx, 10:pat_idx, 11:flags,
        #  12:cause_behavior, 13:effect_detail]
        cause_snip = (f.get("cause_behavior") or "")[:120]
        effect_snip = (f.get("effect_detail") or "")[:150]
        compact_faults.append([
            e["id"],
            0 if e["repo"] == "inference" else 1,
            e["name"],
            e.get("title") or "",
            [sub_idx[s] for s in dedup_subs],
            [car_idx[c] for c in norm_effs],
            [stage_idx[st] for st in eff_stages if st in stage_idx],
            [get_stack_idx(st) for st in stacks],
            vis_idx[vis],
            conf_idx[conf],
            get_pat_idx(e.get("pattern"), e.get("pattern_title"), e["repo"]),
            flags,
            cause_snip,
            effect_snip,
        ])

    return {
        "schema_version": "1.0.0",
        "normalization_stats": {
            "total_faults": len(compact_faults),
            "inverted_sequences_normalized": norm_inverted_count,
            "duplicate_enums_cleaned": dedup_fixed_count,
            "inf_entries_mounted_to_train": cross_mounted_count,
        },
        "topology": {
            "stages": topo["stages"],
            "carriers": topo["carriers"],
            "subsystems": topo["subsystems"],
            "canonical_forward_edges": topo["canonical_forward_edges"],
            "intra_step_coupled_pairs": topo["intra_step_coupled_pairs"],
            "feedback_loops": topo["feedback_loops"],
        },
        "dictionaries": {
            "subsystems": sub_list,
            "carriers": car_list,
            "stages": stage_list,
            "visibilities": vis_list,
            "confidences": conf_list,
            "stacks": stack_vocab,
            "stack_groups": [
                {"id": gid, "label": glabel, "members": mems}
                for gid, glabel, mems in STACK_GROUPS
            ],
            "patterns": pat_vocab,
        },
        "methods_by_carrier": dict(methods_by_carrier),
        "faults_schema": [
            "id", "repo_bit", "name", "title", "sub_idxs", "car_idxs",
            "stage_idxs", "stack_idxs", "vis_idx", "conf_idx", "pat_idx", "flags",
            "cause_behavior", "effect_detail"
        ],
        "flag_bits": {
            "1": "inverted_normalized",
            "2": "cross_mounted_from_inf",
            "4": "has_detect",
            "8": "has_signature",
            "16": "has_fix",
            "32": "has_cross_link",
            "64": "unresolved_subsystem",
        },
        "faults": compact_faults,
    }


def build_standalone_html(payload: dict[str, Any], out_html: str) -> None:
    """Bundle standalone template + style.css + inline JSON into one zero-dependency offline HTML."""
    tpl_dir = os.path.join(HERE, "standalone_template")
    src_dir = os.path.join(ROOT, "web", "src")
    with open(os.path.join(tpl_dir, "causal.html"), encoding="utf-8") as fh:
        html = fh.read()
    with open(os.path.join(src_dir, "style.css"), encoding="utf-8") as fh:
        style_css = re.sub(r"@font-face\s*\{[^}]*\}", "", fh.read()).strip()
    with open(os.path.join(tpl_dir, "causal.css"), encoding="utf-8") as fh:
        causal_css = fh.read().strip()
    with open(os.path.join(tpl_dir, "causal.js"), encoding="utf-8") as fh:
        causal_js = fh.read()

    # Escape </script> if any string ever contains it
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    head_inject = (
        f"<style>\n{style_css}\n\n{causal_css}\n</style>"
    )
    body_inject = (
        f"<script>window.__INLINE_CAUSAL_GRAPH__ = {data_json};</script>\n"
        f"<script type=\"module\">\n{causal_js}\n</script>"
    )
    html = html.replace('  <link rel="stylesheet" href="style.css">\n  <link rel="stylesheet" href="causal.css">\n  <script type="module" src="causal.js"></script>', head_inject)
    html = html.replace("</body>", f"{body_inject}\n</body>")
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)


def enrich_site_docs(
    entries: list[dict[str, Any]],
    docs_rows: list[dict[str, Any]],
    topo: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach normalized `causal` metadata onto `docs.json` rows in-place and return `meta.causal`."""
    topo = topo or load_topology(TOPOLOGY_PATH)
    topo_subs = {s["id"]: s for s in topo["subsystems"]}
    topo_carriers = {c["id"]: c for c in topo["carriers"]}
    forward_set = {(e["from"], e["to"]) for e in topo["canonical_forward_edges"] if e["from"] != "*"}
    wildcard_sinks = {e["to"] for e in topo["canonical_forward_edges"] if e["from"] == "*"}
    coupled_set = {(p["a"], p["b"]) for p in topo["intra_step_coupled_pairs"]}
    loops = topo["feedback_loops"]

    entry_by_key = {(e["repo"], e["id"]): e for e in entries}
    norm_inverted_count = 0
    dedup_fixed_count = 0
    cross_mounted_count = 0
    fault_count = 0

    for row in docs_rows:
        if row.get("neg") or row.get("entry_type") == "pointer":
            row.pop("causal", None)
            continue
        e = entry_by_key.get((row["repo"], row["id"]))
        if not e:
            continue
        f = e.get("fields") or {}
        raw_subs = split_semi(f.get("subsystem"))
        raw_effs = split_eff(f.get("effect_objects"))
        dedup_subs = list(dict.fromkeys(s for s in raw_subs if s in topo_subs))

        if e.get("kind") == "方法":
            valid_effs = list(dict.fromkeys(x for x in raw_effs if x in topo_carriers))
            row["causal"] = {"kind": "method", "effs": valid_effs}
            continue

        if e.get("kind") != "故障":
            row.pop("causal", None)
            continue

        fault_count += 1
        if len(raw_subs) != len(dedup_subs) and "待定" not in raw_subs:
            dedup_fixed_count += 1

        norm_effs, was_inverted = normalize_effect_sequence(
            raw_effs, e, dedup_subs, topo_subs, topo_carriers,
            forward_set, wildcard_sinks, coupled_set, loops
        )
        if was_inverted:
            norm_inverted_count += 1
        if len(raw_effs) > len(set(raw_effs)):
            dedup_fixed_count += 1

        eff_stages, cross_mounted = resolve_effective_stages(
            e["repo"], e.get("stage"), dedup_subs, norm_effs
        )
        if cross_mounted:
            cross_mounted_count += 1

        stacks = list(dict.fromkeys(
            s.lower() for s in split_semi(f.get("stack")) if s and s not in ("待定", "N/A")
        ))
        vis = (f.get("visibility") or "待定").strip()
        has_det = 1 if (f.get("mitigation_detect") or "").strip() not in ("", "待定", "N/A") else 0
        has_sig = 1 if (f.get("signature") or "").strip() not in ("", "待定", "N/A") else 0
        has_fix = 1 if (f.get("mitigation_fix") or "").strip() not in ("", "待定", "N/A") else 0
        has_cross = 1 if any(rel.get("kind") == "cross" for rel in (e.get("relations") or [])) else 0

        flags = (
            (1 if was_inverted else 0)
            | (2 if cross_mounted else 0)
            | (4 if has_det else 0)
            | (8 if has_sig else 0)
            | (16 if has_fix else 0)
            | (32 if has_cross else 0)
            | (64 if not dedup_subs else 0)
        )
        row["causal"] = {
            "subs": dedup_subs,
            "effs": norm_effs,
            "stages": eff_stages,
            "stacks": stacks,
            "vis": vis,
            "flags": flags,
        }

    return {
        "normalization_stats": {
            "total_faults": fault_count,
            "inverted_sequences_normalized": norm_inverted_count,
            "duplicate_enums_cleaned": dedup_fixed_count,
            "inf_entries_mounted_to_train": cross_mounted_count,
        },
        "stack_groups": [
            {"id": gid, "label": glabel, "members": mems}
            for gid, glabel, mems in STACK_GROUPS
        ],
    }


def patch_built_dist(dist_dir: str) -> dict[str, Any]:
    """Enrich an existing web/dist/data/{docs,meta}.json in-place from data/entries.jsonl."""
    entries = load_entries(ENTRIES_JSONL)
    docs_path = os.path.join(dist_dir, "data", "docs.json")
    meta_path = os.path.join(dist_dir, "data", "meta.json")
    with open(docs_path, encoding="utf-8") as fh:
        docs_rows = json.load(fh)
    with open(meta_path, encoding="utf-8") as fh:
        meta_obj = json.load(fh)

    causal_meta = enrich_site_docs(entries, docs_rows)
    meta_obj["causal"] = causal_meta

    with open(docs_path, "w", encoding="utf-8") as fh:
        json.dump(docs_rows, fh, ensure_ascii=False, separators=(",", ":"))
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_obj, fh, ensure_ascii=False, indent=1)

    old_cg = os.path.join(dist_dir, "data", "causal_graph.json")
    if os.path.exists(old_cg):
        os.remove(old_cg)
    return causal_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="Optional standalone JSON output path")
    ap.add_argument("--patch-dist", default=None, help="Enrich an existing web/dist directory in-place")
    ap.add_argument("--standalone-html", default=None, help="Also emit a zero-dependency offline single-file HTML")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    if args.patch_dist:
        cmeta = patch_built_dist(args.patch_dist)
        print(f"Patched {args.patch_dist}/data/{{docs,meta}}.json | stats={cmeta['normalization_stats']}")

    if args.out or args.standalone_html or not args.patch_dist:
        payload = build_graph_payload()
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as fh:
                if args.pretty:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
                else:
                    json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
            size_kb = os.path.getsize(args.out) / 1024.0
            print(f"Wrote {args.out} ({size_kb:.1f} KB) | stats={payload['normalization_stats']}")
        if args.standalone_html:
            build_standalone_html(payload, args.standalone_html)
            hsize_kb = os.path.getsize(args.standalone_html) / 1024.0
            print(f"Wrote standalone offline HTML: {args.standalone_html} ({hsize_kb:.1f} KB)")


if __name__ == "__main__":
    main()
