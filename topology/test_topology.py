#!/usr/bin/env python3
"""Unit tests for causal_topology.json and audit_topology.py (stdlib unittest)."""

import os
import unittest

from topology.audit_topology import (
    ENTRIES_JSONL,
    TOPOLOGY_PATH,
    classify_carrier_pair,
    load_entries,
    load_topology,
    run_audit,
)
from topology.build_graph_data import (
    build_graph_payload,
    enrich_site_docs,
    normalize_effect_sequence,
    resolve_effective_stages,
)


class TestCausalTopologySpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topo = load_topology(TOPOLOGY_PATH)
        cls.carriers = {c["id"]: c for c in cls.topo["carriers"]}
        cls.subsystems = {s["id"]: s for s in cls.topo["subsystems"]}

    def test_counts_and_unique_ids(self) -> None:
        self.assertEqual(len(self.topo["carriers"]), 17)
        self.assertEqual(len(self.carriers), 17)
        self.assertEqual(len(self.topo["subsystems"]), 26)
        self.assertEqual(len(self.subsystems), 26)
        self.assertEqual(len(self.topo["stages"]), 6)

    def test_carrier_backbone_is_strict_dag_by_rank(self) -> None:
        """Every explicit canonical_forward_edge must strictly increase topological rank."""
        for edge in self.topo["canonical_forward_edges"]:
            src, dst = edge["from"], edge["to"]
            self.assertIn(dst, self.carriers)
            if src == "*":
                self.assertGreaterEqual(self.carriers[dst]["rank"], 100)
                continue
            self.assertIn(src, self.carriers)
            self.assertLess(
                self.carriers[src]["rank"],
                self.carriers[dst]["rank"],
                msg=f"Forward edge {src} -> {dst} violates strict rank order",
            )

    def test_subsystem_carriers_valid_and_disjoint(self) -> None:
        valid_planes = {
            "pipeline_train",
            "pipeline_shared",
            "pipeline_serving",
            "substrate_compute",
            "substrate_distributed",
            "cross_cutting_meta",
        }
        for sid, spec in self.subsystems.items():
            self.assertIn(spec["plane"], valid_planes, msg=sid)
            pri = set(spec["primary_carriers"])
            sec = set(spec["secondary_carriers"])
            self.assertTrue(pri, msg=f"{sid} has empty primary_carriers")
            self.assertTrue(pri <= set(self.carriers), msg=f"{sid} invalid primary: {pri - set(self.carriers)}")
            self.assertTrue(sec <= set(self.carriers), msg=f"{sid} invalid secondary: {sec - set(self.carriers)}")
            self.assertEqual(pri & sec, set(), msg=f"{sid} overlaps primary & secondary: {pri & sec}")

    def test_loops_and_coupled_pairs_reference_valid_carriers(self) -> None:
        for p in self.topo["intra_step_coupled_pairs"]:
            self.assertIn(p["a"], self.carriers)
            self.assertIn(p["b"], self.carriers)
            self.assertNotEqual(p["a"], p["b"])
        for lp in self.topo["feedback_loops"]:
            self.assertTrue(set(lp["from"]) <= set(self.carriers))
            self.assertTrue(set(lp["to"]) <= set(self.carriers))

    def test_transition_classifier_semantics(self) -> None:
        fwd = {(e["from"], e["to"]) for e in self.topo["canonical_forward_edges"] if e["from"] != "*"}
        sinks = {e["to"] for e in self.topo["canonical_forward_edges"] if e["from"] == "*"}
        coupled = {(p["a"], p["b"]) for p in self.topo["intra_step_coupled_pairs"]}
        loops = self.topo["feedback_loops"]

        inf_entry = {"repo": "inference", "stage": None}
        rl_entry = {"repo": "training", "stage": "rl"}

        # Forward
        self.assertEqual(
            classify_carrier_pair("KV状态", "输出内容", inf_entry, ["KV缓存"], fwd, sinks, coupled, loops)[0],
            "canonical_forward",
        )
        # Inversion in serving: 输出内容 -> 模型权重
        self.assertEqual(
            classify_carrier_pair("输出内容", "模型权重", inf_entry, ["外部攻击"], fwd, sinks, coupled, loops)[0],
            "causal_inversion",
        )
        # Loop-1 in serving: 模型权重 -> 输出内容
        self.assertEqual(
            classify_carrier_pair("模型权重", "输出内容", inf_entry, ["权重/检查点"], fwd, sinks, coupled, loops)[0],
            "feedback_loop",
        )
        # Loop-3 gated by RL/training: 输出内容 -> 训练信号 is loop in RL, inversion/unmodeled in pure serving
        self.assertEqual(
            classify_carrier_pair("输出内容", "训练信号", rl_entry, ["输出解析"], fwd, sinks, coupled, loops)[0],
            "feedback_loop",
        )
        # Self duplicate
        self.assertEqual(
            classify_carrier_pair("数值结果", "数值结果", rl_entry, ["算子/内核"], fwd, sinks, coupled, loops)[0],
            "self_duplicate",
        )

    @unittest.skipUnless(os.path.exists(ENTRIES_JSONL), "entries.jsonl not built")
    def test_live_catalog_audit_partition(self) -> None:
        entries = load_entries(ENTRIES_JSONL)
        rep = run_audit(self.topo, entries)
        vp = rep["vocab_parity"]
        self.assertEqual(vp["missing_subsystems_in_topology"], [])
        self.assertEqual(vp["extra_subsystems_in_topology"], [])
        self.assertEqual(vp["missing_carriers_in_topology"], [])
        self.assertEqual(vp["extra_carriers_in_topology"], [])

        c1 = rep["check_1_binding_coverage"]
        # Exact partition of fault entries when no entry fails both subsystem=待定 and effect=待定 simultaneously
        total_faults = rep["snapshot_summary"]["fault_entries"]
        self.assertEqual(
            c1["fully_bound_faults"]
            + c1["unresolved_subsystem_count"]
            + c1["unresolved_effect_count"]
            + c1["partial_tbd_effect_count"],
            total_faults,
        )

    def test_normalization_and_dynamic_stage_mounting(self) -> None:
        fwd = {(e["from"], e["to"]) for e in self.topo["canonical_forward_edges"] if e["from"] != "*"}
        sinks = {e["to"] for e in self.topo["canonical_forward_edges"] if e["from"] == "*"}
        coupled = {(p["a"], p["b"]) for p in self.topo["intra_step_coupled_pairs"]}
        loops = self.topo["feedback_loops"]

        # 1. BFA weight attack inverted sequence: ['输出内容', '模型权重'] -> ['模型权重', '输出内容']
        norm_bfa, inv_bfa = normalize_effect_sequence(
            ["输出内容", "模型权重"],
            {"repo": "inference", "stage": None},
            ["外部攻击"],
            self.subsystems,
            self.carriers,
            fwd,
            sinks,
            coupled,
            loops,
        )
        self.assertTrue(inv_bfa)
        self.assertEqual(norm_bfa, ["模型权重", "输出内容"])

        # 2. Training trajectory inverted sequence: ['训练轨迹', '梯度'] -> ['梯度', '训练轨迹']
        norm_trn, inv_trn = normalize_effect_sequence(
            ["训练轨迹", "梯度"],
            {"repo": "training", "stage": "pretrain"},
            ["模型定义"],
            self.subsystems,
            self.carriers,
            fwd,
            sinks,
            coupled,
            loops,
        )
        self.assertTrue(inv_trn)
        self.assertEqual(norm_trn, ["梯度", "训练轨迹"])

        # 3. Duplicate carrier cleanup: ['数值结果', '数值结果'] -> ['数值结果']
        norm_dup, inv_dup = normalize_effect_sequence(
            ["数值结果", "数值结果"],
            {"repo": "training", "stage": "shared"},
            ["算子/内核"],
            self.subsystems,
            self.carriers,
            fwd,
            sinks,
            coupled,
            loops,
        )
        self.assertFalse(inv_dup)
        self.assertEqual(norm_dup, ["数值结果"])

        # 4. Dynamic mounting of INF backward operator onto training stages
        stages, mounted = resolve_effective_stages("inference", None, ["算子/内核"], ["梯度"])
        self.assertTrue(mounted)
        self.assertEqual(stages, ["pretrain", "sft", "rl"])

        if os.path.exists(ENTRIES_JSONL):
            entries = load_entries(ENTRIES_JSONL)
            payload = build_graph_payload(self.topo, entries)
            self.assertEqual(payload["normalization_stats"]["inf_entries_mounted_to_train"], 74)
            self.assertGreaterEqual(payload["normalization_stats"]["inverted_sequences_normalized"], 230)

            stub_docs = [
                {"id": e["id"], "repo": e["repo"], "neg": e.get("neg", False), "entry_type": e.get("entry_type")}
                for e in entries[:50]
            ]
            cmeta = enrich_site_docs(entries[:50], stub_docs, self.topo)
            self.assertGreater(cmeta["normalization_stats"]["total_faults"], 40)
            self.assertTrue(any("causal" in d and "subs" in d["causal"] for d in stub_docs))


if __name__ == "__main__":
    unittest.main()
