"""Unit tests for collect.py per-scope cursors (`scope_cursors`) & gap-free delta views."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

COLLECT = Path(__file__).parent / "fault_scan_sources" / "collect.py"


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


class CollectWatermarkTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("collect_under_test", COLLECT)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "cache"
        self.reg = self.root / "sources.json"
        self.reg.write_text(
            json.dumps({
                "cache_hours": 3,
                "window_days": 7,
                "sources": [
                    {
                        "platform": "github",
                        "repo": "vllm-project/vllm-ascend",
                        "web_url": "https://github.com/vllm-project/vllm-ascend",
                        "scopes": ["inference", "training"],
                    }
                ],
            }),
            encoding="utf-8",
        )

    def _fake_item(self, num: int, updated: str):
        return {
            "number": num,
            "title": f"precision bug #{num}",
            "body": f"silent wrong output {num}",
            "state": "open",
            "created_at": updated,
            "updated_at": updated,
        }

    def test_staggered_training_after_cache_expiry_keeps_inference_fetched_items(self):
        """Reproduces the 2026-11-17 gap: inference runs at T1 (23:00), training runs >3h
        later at T2 (04:00, cache expired). Training delta must still include T0..T1 items!"""
        items_all = [
            self._fake_item(101, "2026-11-16T10:00:00Z"),  # baseline day
            self._fake_item(102, "2026-11-16T22:30:00Z"),  # between T0 and T1 (fetched by inference)
            self._fake_item(103, "2026-11-17T03:30:00Z"),  # between T1 and T2 (fetched by training)
        ]

        def fake_request(url, raw):
            raw.parent.mkdir(parents=True, exist_ok=True)
            # Filter items by `since` query param just like GitHub REST API
            import urllib.parse
            qs = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            since = _ts(qs["since"][0])
            matched = [it for it in items_all if _ts(it["updated_at"]) >= since]
            return matched, {}

        with mock.patch.object(self.mod, "request", side_effect=fake_request):
            # Baseline T0: both scopes finish & mark consumed at 2026-11-16T12:00:00Z
            r0 = self.mod.run("inference", None, self.out, registry=self.reg, now=_ts("2026-11-16T12:00:00Z"))
            self.assertTrue(r0["complete"])
            self.mod.mark_consumed("inference", "2026-11-16T12:00:00Z", self.out)
            self.mod.mark_consumed("training", "2026-11-16T12:00:00Z", self.out)

            # T1 (23:00): inference runs (>3h after T0 -> fetches #102, advances fetch watermark to 23:00)
            r_inf = self.mod.run("inference", None, self.out, registry=self.reg, now=_ts("2026-11-16T23:00:00Z"))
            self.assertFalse(r_inf["cached"])
            inf_delta = [json.loads(x)["number"] for x in Path(r_inf["delta_candidates"]).read_text().splitlines()]
            self.assertIn("102", inf_delta)
            self.mod.mark_consumed("inference", r_inf["scope_window"]["until"], self.out)

            # T2 (04:00 next day, 5h after T1 -> cache expired, new fetch only pulls >= 22:00Z -> #102 & #103,
            # or even if #102 was at 15:00Z outside the 1h fetch overlap!)
            items_all.append(self._fake_item(104, "2026-11-16T15:00:00Z"))
            # Re-simulate properly: #104 was already in T1's normalized.jsonl if fetched at T1; let's verify #102 (at 22:30)
            # AND let's test an item at 15:00Z by running a clean sequence below!
            r_trn = self.mod.run("training", None, self.out, registry=self.reg, now=_ts("2026-11-17T04:00:00Z"))
            self.assertFalse(r_trn["cached"])
            # Training cursor is still 2026-11-16T12:00:00Z (since -> 11:00:00Z)
            self.assertEqual(r_trn["scope_window"]["since"], "2026-11-16T11:00:00Z")
            trn_delta = [json.loads(x)["number"] for x in Path(r_trn["delta_candidates"]).read_text().splitlines()]
            self.assertEqual(set(trn_delta), {"102", "103"})

    def test_unmarked_failed_run_automatically_recovers_window_next_day(self):
        items_all = [
            self._fake_item(201, "2026-11-16T14:00:00Z"),
            self._fake_item(202, "2026-11-17T14:00:00Z"),
        ]

        def fake_request(url, raw):
            raw.parent.mkdir(parents=True, exist_ok=True)
            import urllib.parse
            qs = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            since = _ts(qs["since"][0])
            return [it for it in items_all if _ts(it["updated_at"]) >= since], {}

        with mock.patch.object(self.mod, "request", side_effect=fake_request):
            self.mod.run("inference", None, self.out, registry=self.reg, now=_ts("2026-11-16T10:00:00Z"))
            self.mod.mark_consumed("inference", "2026-11-16T10:00:00Z", self.out)

            # Day 1 night: inference collects #201, but agent hits rate-limit / git block and NEVER calls mark-consumed
            r1 = self.mod.run("inference", None, self.out, registry=self.reg, now=_ts("2026-11-16T23:00:00Z"))
            self.assertEqual([json.loads(x)["number"] for x in Path(r1["delta_candidates"]).read_text().splitlines()], ["201"])

            # Day 2 night: inference runs again -> must see BOTH #201 and #202!
            r2 = self.mod.run("inference", None, self.out, registry=self.reg, now=_ts("2026-11-17T23:00:00Z"))
            nums = [json.loads(x)["number"] for x in Path(r2["delta_candidates"]).read_text().splitlines()]
            self.assertEqual(set(nums), {"201", "202"})

            # Now agent succeeds and marks consumed -> Day 3 sees empty delta when nothing new arrives
            self.mod.mark_consumed("inference", r2["scope_window"]["until"], self.out)
            r3 = self.mod.run("inference", None, self.out, registry=self.reg, now=_ts("2026-11-18T23:00:00Z"))
            self.assertEqual(r3["delta_count"], 0)

    def test_legacy_state_json_without_scope_cursors_seeds_safely(self):
        items_all = [self._fake_item(301, "2026-11-17T01:00:00Z")]

        def fake_request(url, raw):
            raw.parent.mkdir(parents=True, exist_ok=True)
            return items_all, {}

        with mock.patch.object(self.mod, "request", side_effect=fake_request):
            r0 = self.mod.run("training", None, self.out, registry=self.reg, now=_ts("2026-11-17T03:56:32Z"))
            # Simulate pre-upgrade state.json by stripping `scope_cursors`
            state_path = self.out / "state.json"
            legacy_state = json.loads(state_path.read_text(encoding="utf-8"))
            legacy_state.pop("scope_cursors", None)
            state_path.write_text(json.dumps(legacy_state), encoding="utf-8")

            # Upgrade run 3 days later for inference -> seeds both cursors from old batch's earliest `since`
            r_up = self.mod.run("inference", None, self.out, registry=self.reg, now=_ts("2026-11-20T23:00:00Z"))
            upgraded_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn("inference", upgraded_state["scope_cursors"])
            self.assertIn("training", upgraded_state["scope_cursors"])
            self.assertEqual(r_up["delta_count"], 1)


if __name__ == "__main__":
    unittest.main()
