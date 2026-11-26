#!/usr/bin/env python3
"""Unit tests for scripts/rag_auto_sync.py (staging, quarantine, feedback loop, topology QA)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rag_auto_sync as RAS  # noqa: E402
from eval.run_eval import load_fix_list  # noqa: E402


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@local"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)


def _make_entry(eid: str, status: str = "active", effects: list[str] | None = None) -> dict:
    return {
        "id": eid,
        "slug": f"slug_{eid.lower().replace('-', '_')}",
        "title": f"Title {eid}",
        "kind": "故障",
        "status": status,
        "cause": {
            "objects": [{"subsystem": "算子/内核", "stack": "cann", "component": "aclnnTestOp"}],
            "behavior": "参数未校验直接传入核函数",
        },
        "effect": {
            "objects": effects or ["数值结果", "输出内容"],
            "detail": "输出静默错误",
            "visibility": "静默错误",
        },
        "labels": {"confidence": "documented", "discovered": "2026-11-26"},
        "evidence": [{"url": f"https://example.com/{eid}", "kind": "github_issue"}],
        "relations": [],
        "original": {"mechanism": "m", "behavior": "b", "source": "s"},
        "migration": {"scanned": "2026-11-26"},
    }


class RagAutoSyncTests(unittest.TestCase):
    def test_stage_catalogs_skips_new_pending_and_preserves_baseline_reaudit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            src_inf = base / "inference_error_review"
            src_tr = base / "training_error_review"
            base_cats = base / "baseline_catalogs"
            dst_cats = base / "dst_catalogs"

            for repo_dir, prefix in ((src_inf, "INF"), (src_tr, "TRN")):
                _init_git_repo(repo_dir)
                (repo_dir / "catalog" / "entries").mkdir(parents=True)
                (repo_dir / "catalog" / "patterns").mkdir(parents=True)
                (repo_dir / "catalog" / "rejected.md").write_text("# Rejected\n", encoding="utf-8")
                (repo_dir / "catalog" / "blindspots.md").write_text("# Blindspots\n", encoding="utf-8")
                if prefix == "INF":
                    (repo_dir / "FAULT_MASTER_REFERENCE.zh.md").write_text("# Master\n", encoding="utf-8")

            # Baseline has INF-0001 (active) and TRN-0001 (active)
            (base_cats / "inference" / "entries").mkdir(parents=True)
            (base_cats / "training" / "entries").mkdir(parents=True)
            (base_cats / "inference" / "patterns").mkdir(parents=True)
            (base_cats / "training" / "patterns").mkdir(parents=True)
            (base_cats / "inference" / "entries" / "INF-0001.yaml").write_text(
                yaml.safe_dump(_make_entry("INF-0001", status="active"), allow_unicode=True),
                encoding="utf-8",
            )
            (base_cats / "training" / "entries" / "TRN-0001.yaml").write_text(
                yaml.safe_dump(_make_entry("TRN-0001", status="active"), allow_unicode=True),
                encoding="utf-8",
            )

            # Upstream inference has:
            # - INF-0001 temporarily in pending_audit (reaudit) -> should preserve baseline active YAML!
            # - INF-0002 active (graduated) -> should sync!
            # - INF-0003 pending_audit (brand new held) -> should skip!
            # - INF-0004 active but quarantined -> should skip!
            for eid, st in (("INF-0001", "pending_audit"), ("INF-0002", "active"), ("INF-0003", "pending_audit"), ("INF-0004", "active")):
                (src_inf / "catalog" / "entries" / f"{eid}.yaml").write_text(
                    yaml.safe_dump(_make_entry(eid, status=st), allow_unicode=True),
                    encoding="utf-8",
                )
            (src_inf / "catalog" / "id_registry.tsv").write_text(
                "id\tslug\nINF-0001\ts1\nINF-0002\ts2\nINF-0003\ts3\nINF-0004\ts4\n",
                encoding="utf-8",
            )
            # Pattern with 1 active + 1 new pending -> keeps active member; pattern with only new pending -> dropped
            (src_inf / "catalog" / "patterns" / "PAT-INF-0001.yaml").write_text(
                yaml.safe_dump({"id": "PAT-INF-0001", "title": "P1", "members": ["INF-0002", "INF-0003"]}, allow_unicode=True),
                encoding="utf-8",
            )
            (src_inf / "catalog" / "patterns" / "PAT-INF-0002.yaml").write_text(
                yaml.safe_dump({"id": "PAT-INF-0002", "title": "P2", "members": ["INF-0003"]}, allow_unicode=True),
                encoding="utf-8",
            )

            (src_tr / "catalog" / "entries" / "TRN-0001.yaml").write_text(
                yaml.safe_dump(_make_entry("TRN-0001", status="active"), allow_unicode=True),
                encoding="utf-8",
            )
            (src_tr / "catalog" / "id_registry.tsv").write_text("id\tslug\nTRN-0001\ts1\n", encoding="utf-8")

            for repo_dir in (src_inf, src_tr):
                subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
                subprocess.run(["git", "-C", str(repo_dir), "commit", "-q", "-m", "init"], check=True)

            stats = RAS.stage_catalogs(
                src_inf=src_inf,
                src_tr=src_tr,
                baseline_catalogs=base_cats,
                dst_catalogs=dst_cats,
                quarantined={"inference": {"INF-0004"}, "training": set()},
            )

            self.assertEqual(stats["added_ids"]["inference"], ["INF-0002"])
            self.assertEqual(stats["preserved_baseline_pending"]["inference"], ["INF-0001"])
            self.assertEqual([x["id"] for x in stats["excluded_pending"]["inference"]], ["INF-0003"])
            self.assertEqual(stats["excluded_quarantined"]["inference"], ["INF-0004"])
            self.assertEqual(stats["active_counts"]["inference"], 2)

            staged_inf = sorted(os.listdir(dst_cats / "inference" / "entries"))
            self.assertEqual(staged_inf, ["INF-0001.yaml", "INF-0002.yaml"])
            preserved_doc = yaml.safe_load((dst_cats / "inference" / "entries" / "INF-0001.yaml").read_text(encoding="utf-8"))
            self.assertEqual(preserved_doc["status"], "active")

            reg_txt = (dst_cats / "inference" / "id_registry.tsv").read_text(encoding="utf-8")
            self.assertIn("INF-0002", reg_txt)
            self.assertNotIn("INF-0003", reg_txt)
            self.assertNotIn("INF-0004", reg_txt)

            self.assertTrue((dst_cats / "inference" / "patterns" / "PAT-INF-0001.yaml").exists())
            self.assertFalse((dst_cats / "inference" / "patterns" / "PAT-INF-0002.yaml").exists())
            p1 = yaml.safe_load((dst_cats / "inference" / "patterns" / "PAT-INF-0001.yaml").read_text(encoding="utf-8"))
            self.assertEqual(p1["members"], ["INF-0002"])

    def test_find_candidate_culprits_pinpoints_failing_case_only(self):
        added = {("inference", "INF-2599"), ("inference", "INF-2600")}
        gate_res = {
            "failures": ["cli/indy: indy-025 dropped out of recall_at_5"],
            "cli_indy": {
                "cases": [
                    {
                        "case_id": "indy-001",
                        "metrics": {"recall_at_5": 1.0, "recall_at_10": 1.0},
                        "top_10": [["inference", "INF-2600"], ["inference", "INF-0001"]],
                    },
                    {
                        "case_id": "indy-025",
                        "metrics": {"recall_at_5": 0.0, "recall_at_10": 1.0},
                        "top_10": [["inference", "INF-2599"], ["inference", "INF-0042"]],
                    },
                ]
            },
        }
        culprits = RAS.find_candidate_culprits(gate_res, added)
        self.assertEqual(culprits, [(("inference", "INF-2599"), "indy-025")])

    def test_tier1_feedback_dedup_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            fb_dir = Path(tmp) / "rag_feedback"
            tid1, created1 = RAS.enqueue_tier1_todo(
                fb_dir,
                repo="inference",
                entry_id="INF-2600",
                kind="topology_inversion",
                dedup_key="输出内容,KV状态",
                instruction="Fix order",
            )
            self.assertTrue(created1)
            # Duplicate enqueue must be suppressed immediately
            tid2, created2 = RAS.enqueue_tier1_todo(
                fb_dir,
                repo="inference",
                entry_id="INF-2600",
                kind="topology_inversion",
                dedup_key="输出内容,KV状态",
                instruction="Fix order",
            )
            self.assertEqual(tid1, tid2)
            self.assertFalse(created2)

            todo_lines = (fb_dir / "inference_todo.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(todo_lines), 1)

            # Mark done via CLI
            rc = RAS.main(["--feedback-dir", str(fb_dir), "done-todo", "--id", tid1, "--note", "reordered via catalog_tool"])
            self.assertEqual(rc, 0)
            self.assertEqual((fb_dir / "inference_todo.jsonl").read_text(encoding="utf-8"), "")

            # Even after completion, same signature must NEVER re-enqueue (loop prevention!)
            _, created3 = RAS.enqueue_tier1_todo(
                fb_dir,
                repo="inference",
                entry_id="INF-2600",
                kind="topology_inversion",
                dedup_key="输出内容,KV状态",
                instruction="Fix order",
            )
            self.assertFalse(created3)

    def test_human_inbox_quarantine_and_approve_exemption_cli_and_web(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag_root = Path(tmp)
            (rag_root / "eval").mkdir(parents=True)
            (rag_root / "web" / "eval").mkdir(parents=True)
            (rag_root / "eval" / "fix_list.json").write_text(json.dumps({"fix_list": []}), encoding="utf-8")
            (rag_root / "web" / "eval" / "fix_list.json").write_text(json.dumps({"fix_list": []}), encoding="utf-8")

            store = RAS.load_pending_human(rag_root)
            item_id = "quarantine-inference-INF-2599-indy-025"
            RAS.upsert_human_item(store, {
                "item_id": item_id,
                "tier": 3,
                "kind": "gate_quarantined_entry",
                "repo": "inference",
                "entry_id": "INF-2599",
                "case_id": "indy-025",
                "gate_surfaces": ["cli", "web"],
                "pre_sync_ranks": {"hybrid": 5, "bm25": 8, "dense": 12},
            })
            RAS.save_pending_human(rag_root, store)
            self.assertEqual(RAS.active_quarantined_ids(RAS.load_pending_human(rag_root))["inference"], {"INF-2599"})

            rc = RAS.main([
                "--rag-root", str(rag_root),
                "resolve", item_id,
                "--action", "approve_exemption",
                "--reason", "Benign sibling INF-2599 shares exact PD prefix cache symptom.",
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(RAS.active_quarantined_ids(RAS.load_pending_human(rag_root))["inference"], set())

            # Both fix_list.json files must pass eval/run_eval.py's strict load_fix_list validator
            self.assertEqual(load_fix_list(str(rag_root / "eval" / "fix_list.json")), {"indy-025": "accepted_regression"})
            self.assertEqual(load_fix_list(str(rag_root / "web" / "eval" / "fix_list.json")), {"indy-025": "accepted_regression"})

    def test_scrub_identifiers_and_topology_inversion_detection(self):
        q = RAS.scrub_identifiers_from_blind_query(
            "在 fp16 下调用 aclnnKvRmsNormRopeCacheV2 后缓存错位",
            "cann_kv_rms_norm_rope_cache_v2_index_rank_unchecked",
            "aclnnKvRmsNormRopeCacheV2",
        )
        self.assertNotIn("aclnnKvRmsNormRopeCacheV2", q)
        self.assertIn("fp16", q)

        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            shutil.copytree(ROOT / "topology", wt / "topology")
            (wt / "data").mkdir(parents=True)
            rows = [
                # Inverted: 输出内容 -> KV状态 (exact INF-2600 post-audit shape in entries.jsonl)
                {
                    "repo": "inference",
                    "id": "INF-2600",
                    "neg": False,
                    "kind": "故障",
                    "stage": "serving",
                    "fields": {"subsystem": "运行时/执行", "effect_objects": "输出内容、KV状态"},
                },
                # Valid LOOP-2: KV状态 -> 数值结果 (INF-2588 shape)
                {
                    "repo": "inference",
                    "id": "INF-2588",
                    "neg": False,
                    "kind": "故障",
                    "stage": "serving",
                    "fields": {"subsystem": "KV缓存", "effect_objects": "KV状态、数值结果"},
                },
                # Valid LOOP-1: 模型权重 -> 数值结果 (INF-2594 / TRN-0813 shape)
                {
                    "repo": "training",
                    "id": "TRN-0813",
                    "neg": False,
                    "kind": "故障",
                    "stage": "pretrain",
                    "fields": {"subsystem": "量化", "effect_objects": "模型权重、数值结果、训练轨迹"},
                },
            ]
            with (wt / "data" / "entries.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

            defects = RAS.inspect_new_entry_topology(
                wt, {("inference", "INF-2600"), ("inference", "INF-2588"), ("training", "TRN-0813")}
            )
            self.assertEqual(len(defects), 1)
            self.assertEqual(defects[0]["entry_id"], "INF-2600")
            self.assertEqual(defects[0]["defect"], "causal_inversion")


if __name__ == "__main__":
    unittest.main()
