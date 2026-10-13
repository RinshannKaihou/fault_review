import copy
import os
import sys
import unittest

os.environ.setdefault("JUDGE_API_KEY", "test")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_apply import apply_entry  # noqa: E402
from audit_merge import decide_bin, decide_cat  # noqa: E402


class VoteRuleTest(unittest.TestCase):
    def test_categorical_base_stage(self):
        self.assertEqual(decide_cat(["量化", "量化", "量化"], panel=False)[:2], ("量化", "unanimous"))
        self.assertEqual(decide_cat(["量化", "量化", "算子/内核"], panel=False)[1], "split")
        self.assertEqual(decide_cat(["量化", None, None], panel=False)[1], "uncertain")   # below quorum
        self.assertEqual(decide_cat(["量化", "量化", None], panel=False)[1], "unanimous")  # quorum 2
        self.assertEqual(decide_cat([None, None], panel=False)[1], "unresolved")

    def test_categorical_panel_stage(self):
        five_of_seven = ["量化"] * 5 + ["算子/内核"] * 2
        self.assertEqual(decide_cat(five_of_seven, panel=True)[:2], ("量化", "panel7"))
        four_of_seven = ["量化"] * 4 + ["算子/内核"] * 3
        self.assertEqual(decide_cat(four_of_seven, panel=True)[1], "uncertain")
        self.assertEqual(decide_cat(["量化"] * 4, panel=True)[1], "uncertain")  # panel quorum 5

    def test_binary(self):
        self.assertEqual(decide_bin([False, False, False], panel=False)[:2], (False, "unanimous"))
        self.assertEqual(decide_bin([True, True, False], panel=False)[1], "split")
        self.assertEqual(decide_bin([True] * 5 + [False] * 2, panel=True)[:2], (True, "panel7"))


def entry():
    return {"id": "INF-0001", "kind": "故障", "title": "t",
            "cause": {"objects": [{"subsystem": "调度/服务", "stack": "vllm", "component": "c"}], "behavior": "b"},
            "effect": {"objects": ["可用性/进度", "KV状态"], "detail": "d", "visibility": "静默错误"},
            "migration": {"method": "extract"}}


def decision(**overrides):
    d = {"fields": {f: {"outcome": "confirmed", "prior": None, "final": None, "status": "unanimous"}
                    for f in ("kind", "subsystem", "effect", "visibility")}}
    d["fields"].update({f: {"outcome": "ok"} for f in ("title", "component", "behavior", "detail")})
    for k, v in overrides.items():
        d["fields"][k] = v
    return d


class ApplyTest(unittest.TestCase):
    def test_corrections_and_fix(self):
        d = decision(subsystem={"outcome": "corrected", "prior": "调度/服务", "final": "输出解析", "status": "unanimous"},
                     effect={"outcome": "corrected", "prior": "可用性/进度", "final": "KV状态", "status": "panel7"},
                     behavior={"outcome": "issue"}, detail={"outcome": "issue"})
        e = apply_entry(copy.deepcopy(entry()), d, {"fixed": {"behavior": "b2"}})
        self.assertEqual(e["cause"]["objects"][0]["subsystem"], "输出解析")
        self.assertEqual(e["effect"]["objects"], ["KV状态", "可用性/进度"])   # moved to front, no duplicate
        self.assertEqual(e["cause"]["behavior"], "b2")
        self.assertEqual(e["effect"]["detail"], "d")                         # unverified fix not applied
        self.assertEqual(e["migration"]["audit"]["issue_unfixed"], ["detail"])
        self.assertEqual(e["migration"]["method"], "extract")                # earlier provenance kept

    def test_uncertain_keeps_baseline(self):
        d = decision(visibility={"outcome": "uncertain", "prior": "静默错误", "final": "静默错误", "status": "uncertain"})
        e = apply_entry(copy.deepcopy(entry()), d, None)
        self.assertEqual(e["effect"]["visibility"], "静默错误")
        self.assertEqual(e["migration"]["audit"]["uncertain"], ["visibility"])


if __name__ == "__main__":
    unittest.main()
