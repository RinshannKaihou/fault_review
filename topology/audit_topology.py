#!/usr/bin/env python3
"""Audit all structured fault catalog entries against causal_topology.json.

Zero third-party dependencies (runs with pure Python 3 standard library).
Reads:
  - topology/causal_topology.json
  - catalogs/inference/vocab.yaml (validates enum parity)
  - data/entries.jsonl (or parses catalogs/*/entries/*.yaml fallback)

Usage:
  python3 topology/audit_topology.py                 # JSON summary to stdout
  python3 topology/audit_topology.py --write-json topology/health_metrics.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOPOLOGY_PATH = os.path.join(HERE, "causal_topology.json")
ENTRIES_JSONL = os.path.join(ROOT, "data", "entries.jsonl")
VOCAB_PATH = os.path.join(ROOT, "normalize", "vocab.yaml")
UPSTREAM_INF_VOCAB = os.path.join(os.path.dirname(ROOT), "inference_error_review", "catalog", "vocab.yaml")


def parse_vocab_enums_stdlib(vocab_file: str) -> dict[str, list[str]]:
    """Extract top-level mapping keys under subsystem/effect_object/visibility/kind without PyYAML."""
    sections: dict[str, list[str]] = {"subsystem": [], "effect_object": [], "visibility": [], "kind": []}
    current: str | None = None
    with open(vocab_file, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not line.startswith(" "):
                key = stripped.rstrip(":")
                current = key if key in sections else None
            elif current and line.startswith("  ") and not line.startswith("    "):
                if ":" in stripped and not stripped.startswith("-"):
                    k = stripped.split(":", 1)[0].strip()
                    sections[current].append(k)
    return sections


def split_semi(val: Any) -> list[str]:
    if not val:
        return []
    return [x.strip() for x in str(val).split("；") if x.strip()]


def split_eff(val: Any) -> list[str]:
    if not val:
        return []
    return [x.strip() for x in re.split(r"[、；]", str(val)) if x.strip()]


def rel_path(fp: str) -> str:
    try:
        return os.path.relpath(fp, ROOT)
    except ValueError:
        return fp


def cite_entry(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": e["id"],
        "repo": e["repo"],
        "name": e["name"],
        "confidence": e.get("confidence") or "待定",
        "status": e.get("status") or "active",
        "file": rel_path(e.get("file", "")),
        "title": e.get("title", ""),
    }


def load_topology(path: str = TOPOLOGY_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_entries(jsonl_path: str = ENTRIES_JSONL) -> list[dict[str, Any]]:
    entries = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("neg"):
                entries.append(r)
    return entries


def classify_carrier_pair(
    e1: str,
    e2: str,
    entry: dict[str, Any],
    subs: list[str],
    forward_set: set[tuple[str, str]],
    wildcard_sinks: set[str],
    coupled_set: set[tuple[str, str]],
    loops: list[dict[str, Any]],
) -> tuple[str, str]:
    """Classify an adjacent (effect.objects[i], effect.objects[i+1]) transition."""
    if e1 == "待定" or e2 == "待定":
        return ("placeholder", "contains_tbd")
    if e1 == e2:
        return ("self_duplicate", f"{e1}=={e2}")

    # 1. Intra-step coupled pairs (check before wildcard sink so 可用性/进度<->性能/资源 stays coupled)
    if (e1, e2) in coupled_set or (e2, e1) in coupled_set:
        return ("intra_step_coupled", f"{e1}<->{e2}")

    # 2. Explicit canonical forward edges (excluding wildcard when source itself is observation sink)
    if (e1, e2) in forward_set:
        return ("canonical_forward", f"{e1}->{e2}")

    # Wildcard sinks: any non-verdict state can project into 可用性/进度, 性能/资源, 指标/日志, 测试判定
    # Respect observation hierarchy: {可用性/进度, 性能/资源} -> 指标/日志 -> 测试判定 is forward,
    # whereas 测试判定 -> anything or 指标/日志 -> state is an inversion!
    obs_ranks = {"可用性/进度": 100, "性能/资源": 100, "指标/日志": 110, "测试判定": 115}
    if e2 in wildcard_sinks:
        r1 = obs_ranks.get(e1, 50)
        r2 = obs_ranks[e2]
        if r1 < r2:
            return ("canonical_forward", f"{e1}->{e2}(sink)")

    # 3. Check explicit feedback loops
    is_rl_or_trn = (
        entry["repo"] == "training"
        or "rl" in str(entry.get("stage") or "")
        or "RL训练回路" in subs
    )
    for lp in loops:
        cond = lp.get("condition", "always")
        if cond == "rl_or_training" and not is_rl_or_trn:
            continue
        if e1 in lp["from"] and e2 in lp["to"]:
            return ("feedback_loop", lp["id"])

    # 4. Special runtime/scheduler fault-tolerance transitions:
    # When scheduler/runtime/communication abort, timeout, or OOM recovery itself corrupts state
    if e1 == "可用性/进度" and any(s in {"调度/服务", "运行时/执行", "集合通信", "重计算/卸载", "KV缓存", "权重/检查点"} for s in subs):
        return ("abort_recovery_corruption", f"{e1}->{e2}")

    # 5. Check if the reverse (e2 -> e1) IS a valid forward or loop edge -> strict causal inversion!
    rev_is_forward = (e2, e1) in forward_set or (
        e1 in wildcard_sinks and obs_ranks.get(e2, 50) < obs_ranks[e1]
    )
    rev_in_loop = any(e2 in lp["from"] and e1 in lp["to"] for lp in loops)
    if rev_is_forward or rev_in_loop:
        return ("causal_inversion", f"expected {e2}->{e1}, got {e1}->{e2}")

    return ("unmodeled_transition", f"{e1}->{e2}")


def run_audit(topo: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    vocab_file = UPSTREAM_INF_VOCAB if os.path.exists(UPSTREAM_INF_VOCAB) else VOCAB_PATH
    vocab_enums = parse_vocab_enums_stdlib(vocab_file)

    topo_subs = {s["id"]: s for s in topo["subsystems"]}
    topo_carriers = {c["id"]: c for c in topo["carriers"]}

    vocab_parity = {
        "vocab_source": vocab_file,
        "missing_subsystems_in_topology": sorted(set(vocab_enums["subsystem"]) - set(topo_subs)),
        "extra_subsystems_in_topology": sorted(set(topo_subs) - set(vocab_enums["subsystem"])),
        "missing_carriers_in_topology": sorted(set(vocab_enums["effect_object"]) - set(topo_carriers)),
        "extra_carriers_in_topology": sorted(set(topo_carriers) - set(vocab_enums["effect_object"])),
    }

    forward_set: set[tuple[str, str]] = set()
    wildcard_sinks: set[str] = set()
    for edge in topo["canonical_forward_edges"]:
        if edge["from"] == "*":
            wildcard_sinks.add(edge["to"])
        else:
            forward_set.add((edge["from"], edge["to"]))

    coupled_set = {(p["a"], p["b"]) for p in topo["intra_step_coupled_pairs"]}
    loops = topo["feedback_loops"]

    faults = [e for e in entries if e.get("kind") == "故障"]
    non_faults = [e for e in entries if e.get("kind") != "故障"]

    # Check 1: Coverage & TBD mysteries
    fully_bound = 0
    partial_tbd_effect = []
    unresolved_sub_mysteries = []
    unresolved_eff_mysteries = []

    # Build empirical P(subsystem | carrier reached, repo) on clean entries for mystery attribution
    emp_sub_given_eff = collections.defaultdict(collections.Counter)
    for e in faults:
        subs = [s for s in split_semi(e["fields"].get("subsystem")) if s not in ("待定", "N/A")]
        effs = [x for x in split_eff(e["fields"].get("effect_objects")) if x not in ("待定", "N/A")]
        if subs and effs:
            for s in dict.fromkeys(subs):
                for ef in dict.fromkeys(effs):
                    emp_sub_given_eff[(e["repo"], ef)][s] += 1

    # Check 2: Injection fidelity
    inj_counts = collections.Counter()
    inj_matrix = collections.Counter()
    boundary_violations = []
    shortcut_examples = collections.defaultdict(list)

    # Check 3: Carrier transition order
    trans_class_counts = collections.Counter()
    trans_pair_by_class = collections.defaultdict(collections.Counter)
    inversion_entries = []
    duplicate_effect_entries = []
    unmodeled_entries = []

    # Check 4: Multi-object hygiene
    true_multi_sub = []
    duplicate_sub_entries = []

    # Check 5: Stage & Cross-repo alignment
    train_only_subs = {"RL训练回路", "优化器", "损失计算", "数据管线"}
    train_only_carriers = {"梯度", "优化器状态", "训练样本流", "训练信号", "训练轨迹"}
    serving_carriers = {"输出内容", "KV状态", "输出分布"}
    inf_pure_train_footprint = []

    # Check 7: Observability & Silent risk per carrier and per subsystem
    carrier_stats = {cid: {"total": 0, "primary": 0, "silent": 0, "detect": 0, "signature": 0, "methods": []} for cid in topo_carriers}
    sub_stats = {sid: {"total": 0, "inf": 0, "trn": 0, "silent": 0, "detect": 0, "signature": 0, "fix": 0} for sid in topo_subs}

    for e in non_faults:
        if e.get("kind") == "方法":
            effs = [x for x in split_eff(e["fields"].get("effect_objects")) if x in carrier_stats]
            for ef in effs:
                carrier_stats[ef]["methods"].append(cite_entry(e))

    for e in faults:
        f = e["fields"]
        raw_subs = split_semi(f.get("subsystem"))
        raw_effs = split_eff(f.get("effect_objects"))
        vis = (f.get("visibility") or "").strip()
        has_det = (f.get("mitigation_detect") or "").strip() not in ("", "待定", "N/A")
        has_fix = (f.get("mitigation_fix") or "").strip() not in ("", "待定", "N/A")
        has_sig = (f.get("signature") or "").strip() not in ("", "待定", "N/A")
        is_silent = vis == "静默错误"

        valid_subs = [s for s in raw_subs if s in topo_subs]
        valid_effs = [x for x in raw_effs if x in topo_carriers]

        # Multi-sub hygiene
        if len(raw_subs) > 1 and len(valid_subs) != len(set(valid_subs)):
            duplicate_sub_entries.append({**cite_entry(e), "raw_subsystems": raw_subs})
        elif len(set(valid_subs)) > 1:
            true_multi_sub.append({**cite_entry(e), "subsystems": valid_subs, "effects": valid_effs})

        # Coverage accounting
        sub_has_tbd = (not valid_subs) or ("待定" in raw_subs)
        eff_has_tbd = (not valid_effs) or ("待定" in raw_effs)
        if not sub_has_tbd and not eff_has_tbd:
            fully_bound += 1
        else:
            if sub_has_tbd:
                pri_eff = valid_effs[0] if valid_effs else None
                priors = emp_sub_given_eff[(e["repo"], pri_eff)].most_common(4) if pri_eff else []
                tot_p = sum(emp_sub_given_eff[(e["repo"], pri_eff)].values()) or 1
                unresolved_sub_mysteries.append({
                    **cite_entry(e),
                    "effects": raw_effs,
                    "visibility": vis,
                    "top_candidate_subsystems": [
                        {"subsystem": s, "count": c, "prior_prob": round(c / tot_p, 3)} for s, c in priors
                    ],
                })
            if not valid_effs:
                unresolved_eff_mysteries.append({**cite_entry(e), "subsystems": raw_subs, "raw_effects": raw_effs})
            elif "待定" in raw_effs:
                partial_tbd_effect.append({**cite_entry(e), "subsystems": raw_subs, "raw_effects": raw_effs})

        # Subsystem & injection stats
        dedup_subs = list(dict.fromkeys(valid_subs))
        for s in dedup_subs:
            st = sub_stats[s]
            st["total"] += 1
            st["inf" if e["repo"] == "inference" else "trn"] += 1
            if is_silent:
                st["silent"] += 1
            if has_det:
                st["detect"] += 1
            if has_sig:
                st["signature"] += 1
            if has_fix:
                st["fix"] += 1

            if valid_effs:
                e0 = valid_effs[0]
                inj_matrix[(s, e0)] += 1
                spec = topo_subs[s]
                if e0 in spec["primary_carriers"]:
                    inj_counts["direct_injection"] += 1
                elif e0 in spec["secondary_carriers"]:
                    inj_counts["shortcut_injection"] += 1
                    if len(shortcut_examples[(s, e0)]) < 3:
                        shortcut_examples[(s, e0)].append(cite_entry(e))
                elif len(dedup_subs) > 1 and any(
                    ef in spec["primary_carriers"] or ef in spec["secondary_carriers"] for ef in valid_effs[1:]
                ):
                    inj_counts["joint_paired_injection"] += 1
                else:
                    inj_counts["boundary_violation"] += 1
                    boundary_violations.append({
                        **cite_entry(e),
                        "subsystem": s,
                        "primary_effect": e0,
                        "all_effects": valid_effs,
                    })

        # Carrier stats
        for idx, ef in enumerate(dict.fromkeys(valid_effs)):
            cst = carrier_stats[ef]
            cst["total"] += 1
            if idx == 0:
                cst["primary"] += 1
            if is_silent:
                cst["silent"] += 1
            if has_det:
                cst["detect"] += 1
            if has_sig:
                cst["signature"] += 1

        # Adjacent effect transitions
        entry_inversions = []
        for i in range(len(raw_effs) - 1):
            e1, e2 = raw_effs[i], raw_effs[i + 1]
            cls, detail = classify_carrier_pair(
                e1, e2, e, dedup_subs, forward_set, wildcard_sinks, coupled_set, loops
            )
            trans_class_counts[cls] += 1
            trans_pair_by_class[cls][f"{e1} -> {e2}"] += 1
            if cls == "causal_inversion":
                entry_inversions.append({"pair": f"{e1} -> {e2}", "detail": detail})
            elif cls == "self_duplicate":
                duplicate_effect_entries.append({**cite_entry(e), "raw_effects": raw_effs})
            elif cls == "unmodeled_transition":
                unmodeled_entries.append({**cite_entry(e), "pair": f"{e1} -> {e2}", "subsystems": dedup_subs, "raw_effects": raw_effs})
        if entry_inversions:
            inversion_entries.append({
                **cite_entry(e),
                "subsystems": dedup_subs,
                "raw_effects": raw_effs,
                "inversions": entry_inversions,
            })

        # Stage alignment check for inference repo
        if e["repo"] == "inference":
            touches_train = bool((set(dedup_subs) & train_only_subs) or (set(valid_effs) & train_only_carriers))
            touches_serving = bool(set(valid_effs) & serving_carriers)
            has_cross = any(rel.get("kind") == "cross" for rel in (e.get("relations") or []))
            if touches_train and not touches_serving:
                inf_pure_train_footprint.append({
                    **cite_entry(e),
                    "subsystems": dedup_subs,
                    "effects": valid_effs,
                    "stack": f.get("stack", ""),
                    "has_cross_link": has_cross,
                })

    # Cross-repo link topology classification
    by_id = {e["id"]: e for e in entries}
    seen_cross: set[tuple[str, str]] = set()
    cross_taxonomy = collections.Counter()
    cross_samples = collections.defaultdict(list)
    for e in entries:
        for rel in (e.get("relations") or []):
            if rel.get("kind") != "cross":
                continue
            tgt_id = rel.get("target")
            if not tgt_id or tgt_id not in by_id:
                continue
            pair = tuple(sorted([e["id"], tgt_id]))
            if pair in seen_cross:
                continue
            seen_cross.add(pair)
            a, b = by_id[pair[0]], by_id[pair[1]]
            if a["repo"] == "inference" and b["repo"] == "training":
                a, b = b, a
            sa = set(split_semi(a["fields"].get("subsystem")))
            sb = set(split_semi(b["fields"].get("subsystem")))
            ea = set(split_eff(a["fields"].get("effect_objects")))
            eb = set(split_eff(b["fields"].get("effect_objects")))

            if a.get("kind") != "故障" or b.get("kind") != "故障":
                ckind = "meta_method_pair"
            elif "RL训练回路" in (sa | sb) or ("rl" in str(a.get("stage") or "") and (sa | sb) & {"KV缓存", "解码/采样", "输出解析", "调度/服务"}):
                ckind = "rl_rollout_coupling"
            elif (ea & {"模型权重", "训练轨迹"}) and (eb & {"模型权重", "输出内容", "输出分布"}) and ((sa | sb) & {"权重/检查点", "硬件"} or a["name"] == b["name"]):
                ckind = "state_inheritance"
            else:
                ckind = "common_substrate"
            cross_taxonomy[ckind] += 1
            if len(cross_samples[ckind]) < 5:
                cross_samples[ckind].append({
                    "trn": cite_entry(a),
                    "inf": cite_entry(b),
                    "trn_sub": sorted(sa),
                    "inf_sub": sorted(sb),
                })

    # Check 6: Pattern topological cohesion
    pat_groups = collections.defaultdict(list)
    for e in faults:
        if e.get("pattern"):
            pat_groups[e["pattern"]].append(e)

    cohesive_pats = 0
    cross_cutting_pats = []
    for pid, mems in sorted(pat_groups.items()):
        sc = collections.Counter()
        ec = collections.Counter()
        planes = set()
        for m in mems:
            ms = [s for s in split_semi(m["fields"].get("subsystem")) if s in topo_subs]
            me = [x for x in split_eff(m["fields"].get("effect_objects")) if x in topo_carriers]
            if ms:
                sc[ms[0]] += 1
                planes.add(topo_subs[ms[0]]["plane"])
            if me:
                ec[me[0]] += 1
        sub_purity = sc.most_common(1)[0][1] / len(mems) if sc else 1.0
        eff_purity = ec.most_common(1)[0][1] / len(mems) if ec else 1.0
        if sub_purity >= 0.70 and eff_purity >= 0.55:
            cohesive_pats += 1
        elif len(mems) >= 3:
            cross_cutting_pats.append({
                "pattern": pid,
                "title": mems[0].get("pattern_title") or "待定",
                "size": len(mems),
                "sub_purity": round(sub_purity, 3),
                "eff_purity": round(eff_purity, 3),
                "planes_spanned": sorted(planes),
                "subsystems": dict(sc.most_common()),
                "primary_effects": dict(ec.most_common()),
            })

    # Compute blindspot / naked risk score for carriers:
    # Risk = total * (silent_rate) * (1 - observable_probe_rate)
    carrier_risk_ranking = []
    for cid, st in carrier_stats.items():
        tot = st["total"]
        if tot == 0:
            continue
        silent_rate = st["silent"] / tot
        # Entries with either detect or signature
        covered = 0
        for e in faults:
            f = e["fields"]
            if cid in split_eff(f.get("effect_objects")):
                if (f.get("mitigation_detect") or "").strip() not in ("", "待定", "N/A") or (f.get("signature") or "").strip() not in ("", "待定", "N/A"):
                    covered += 1
        probe_cov = covered / tot
        naked_silent = max(0, st["silent"] - covered)
        carrier_risk_ranking.append({
            "carrier": cid,
            "track": topo_carriers[cid]["track"],
            "rank": topo_carriers[cid]["rank"],
            "total_entries": tot,
            "primary_entries": st["primary"],
            "silent_entries": st["silent"],
            "silent_rate": round(silent_rate, 3),
            "detect_or_sig_coverage": round(probe_cov, 3),
            "method_entries_count": len(st["methods"]),
            "unmonitored_silent_exposure": round(tot * silent_rate * (1.0 - probe_cov), 1),
        })
    carrier_risk_ranking.sort(key=lambda x: -x["unmonitored_silent_exposure"])

    return {
        "snapshot_summary": {
            "total_structured_entries": len(entries),
            "fault_entries": len(faults),
            "non_fault_entries": len(non_faults),
            "inference_faults": sum(1 for e in faults if e["repo"] == "inference"),
            "training_faults": sum(1 for e in faults if e["repo"] == "training"),
            "active_patterns": len(pat_groups),
        },
        "vocab_parity": vocab_parity,
        "check_1_binding_coverage": {
            "fully_bound_faults": fully_bound,
            "fully_bound_rate": round(fully_bound / len(faults), 5),
            "unresolved_subsystem_count": len(unresolved_sub_mysteries),
            "unresolved_effect_count": len(unresolved_eff_mysteries),
            "partial_tbd_effect_count": len(partial_tbd_effect),
            "unresolved_subsystem_entries": unresolved_sub_mysteries,
            "unresolved_effect_entries": unresolved_eff_mysteries,
            "partial_tbd_effect_entries": partial_tbd_effect,
        },
        "check_2_injection_fidelity": {
            "total_subsystem_injections": sum(inj_counts.values()),
            "counts_by_tier": dict(inj_counts),
            "rates_by_tier": {
                k: round(v / max(sum(inj_counts.values()), 1), 4) for k, v in inj_counts.items()
            },
            "distinct_nonzero_cells": len(inj_matrix),
            "possible_matrix_cells": len(topo_subs) * len(topo_carriers),
            "matrix_density": round(len(inj_matrix) / (len(topo_subs) * len(topo_carriers)), 4),
            "boundary_violations": boundary_violations,
            "top_shortcut_injections": [
                {
                    "subsystem": s,
                    "shortcut_carrier": e0,
                    "count": cnt,
                    "examples": shortcut_examples[(s, e0)],
                }
                for (s, e0), cnt in inj_matrix.most_common()
                if e0 in topo_subs[s]["secondary_carriers"]
            ][:15],
        },
        "check_3_propagation_topology": {
            "total_adjacent_transitions": sum(trans_class_counts.values()),
            "counts_by_class": dict(trans_class_counts),
            "rates_by_class": {
                k: round(v / max(sum(trans_class_counts.values()), 1), 4)
                for k, v in trans_class_counts.items()
            },
            "top_pairs_by_class": {
                cls: dict(cnts.most_common(15)) for cls, cnts in trans_pair_by_class.items()
            },
            "causal_inversion_entry_count": len(inversion_entries),
            "causal_inversion_entries_sample": inversion_entries[:30],
            "duplicate_effect_entries": duplicate_effect_entries,
            "unmodeled_transitions": unmodeled_entries,
        },
        "check_4_multi_object_hygiene": {
            "true_multi_subsystem_hyperedges_count": len(true_multi_sub),
            "true_multi_subsystem_hyperedges": true_multi_sub,
            "duplicate_subsystem_schema_defects_count": len(duplicate_sub_entries),
            "duplicate_subsystem_schema_defects": duplicate_sub_entries,
        },
        "check_5_lifecycle_stage_and_cross": {
            "inf_entries_with_pure_train_footprint_count": len(inf_pure_train_footprint),
            "inf_pure_train_without_cross_count": sum(1 for x in inf_pure_train_footprint if not x["has_cross_link"]),
            "inf_pure_train_by_stack": dict(collections.Counter(x["stack"] for x in inf_pure_train_footprint).most_common(12)),
            "inf_pure_train_sample": inf_pure_train_footprint[:20],
            "unique_cross_repo_pairs": len(seen_cross),
            "cross_pair_taxonomy": dict(cross_taxonomy),
            "cross_pair_samples": dict(cross_samples),
        },
        "check_6_pattern_cohesion": {
            "total_patterns": len(pat_groups),
            "cohesive_single_locus_patterns": cohesive_pats,
            "cross_cutting_or_drifted_patterns_size_ge_3": len(cross_cutting_pats),
            "cross_cutting_or_drifted_list": sorted(
                cross_cutting_pats, key=lambda x: (x["sub_purity"], -x["size"])
            )[:20],
        },
        "check_7_observability_and_blindspots": {
            "carrier_risk_ranking": carrier_risk_ranking,
            "subsystem_stats": {
                sid: {
                    **st,
                    "silent_rate": round(st["silent"] / max(st["total"], 1), 3),
                    "detect_rate": round(st["detect"] / max(st["total"], 1), 3),
                    "signature_rate": round(st["signature"] / max(st["total"], 1), 3),
                    "fix_rate": round(st["fix"] / max(st["total"], 1), 3),
                }
                for sid, st in sorted(sub_stats.items(), key=lambda kv: -kv[1]["total"])
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topology", default=TOPOLOGY_PATH)
    ap.add_argument("--entries", default=ENTRIES_JSONL)
    ap.add_argument("--write-json", default=None)
    ap.add_argument("--check", action="store_true", help="Exit non-zero if topology/vocab parity or structural invariants fail")
    args = ap.parse_args()

    topo = load_topology(args.topology)
    entries = load_entries(args.entries)
    report = run_audit(topo, entries)

    if args.check:
        vp = report["vocab_parity"]
        errors = []
        for k in ("missing_subsystems_in_topology", "extra_subsystems_in_topology", "missing_carriers_in_topology", "extra_carriers_in_topology"):
            if vp[k]:
                errors.append(f"{k}: {vp[k]}")
        if errors:
            sys.stderr.write("TOPOLOGY CHECK FAILED:\n  " + "\n  ".join(errors) + "\n")
            sys.exit(1)

    out = json.dumps(report, ensure_ascii=False, indent=2)
    if args.write_json:
        with open(args.write_json, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
    else:
        sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
