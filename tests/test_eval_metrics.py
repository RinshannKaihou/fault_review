import math
import os
import subprocess
import sys
import unittest

from eval.run_eval import (
    check_report,
    evaluate_cases,
    metrics_for_ranking,
    validate_cases,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_rank_one_single_relevant_is_perfect(self):
        got = metrics_for_ranking(
            [("inference", "1.1"), ("training", "HW.01")],
            {("inference", "1.1")},
        )
        self.assertEqual(got, {"recall_at_5": 1.0, "mrr_at_10": 1.0,
                               "ndcg_at_10": 1.0})

    def test_multiple_relevant_documents_use_binary_ndcg(self):
        got = metrics_for_ranking(
            [("x", "irrelevant"), ("x", "a"), ("x", "b")],
            {("x", "a"), ("x", "b")},
        )
        expected_dcg = 1 / math.log2(3) + 1 / math.log2(4)
        ideal_dcg = 1 + 1 / math.log2(3)
        self.assertEqual(got["recall_at_5"], 1.0)
        self.assertEqual(got["mrr_at_10"], 0.5)
        self.assertAlmostEqual(got["ndcg_at_10"], expected_dcg / ideal_dcg)

    def test_no_relevant_document_returns_zero_metrics(self):
        got = metrics_for_ranking(
            [("inference", "1.1")],
            {("training", "HW.01")},
        )
        self.assertEqual(got, {"recall_at_5": 0.0, "mrr_at_10": 0.0,
                               "ndcg_at_10": 0.0})

    def test_duplicate_ranked_id_does_not_consume_an_extra_rank(self):
        got = metrics_for_ranking(
            [("x", "a"), ("x", "a"), ("x", "b")],
            {("x", "b")},
        )
        self.assertEqual(got["mrr_at_10"], 0.5)

    def test_negative_entry_can_be_a_relevant_document(self):
        got = metrics_for_ranking(
            [("training", "NEG-REJECTED-1")],
            {("training", "NEG-REJECTED-1")},
        )
        self.assertEqual(got["recall_at_5"], 1.0)

    def test_empty_relevant_set_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "relevant set must not be empty"):
            metrics_for_ranking([("x", "a")], set())


class CaseValidationTests(unittest.TestCase):
    def _case(self, case_id="case-1", relevant=None):
        return {
            "case_id": case_id,
            "query": "query",
            "mode": "hybrid",
            "filters": {"repo": "training"},
            "relevant": relevant or [{"repo": "training", "id": "HW.01"}],
            "tags": ["training"],
            "rationale": "verified against the source entry",
        }

    def test_duplicate_case_id_is_rejected(self):
        case = self._case()
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            validate_cases([case, dict(case)], {("training", "HW.01")})

    def test_unknown_relevant_id_is_rejected(self):
        case = self._case(relevant=[{"repo": "training", "id": "MISSING.01"}])
        with self.assertRaisesRegex(ValueError, "unknown relevant entry"):
            validate_cases([case], {("training", "HW.01")})

    def test_valid_case_is_accepted(self):
        cases = [self._case()]
        self.assertEqual(
            validate_cases(cases, {("training", "HW.01")}), cases
        )


class EvaluationTests(unittest.TestCase):
    def test_evaluation_reports_aggregate_slices_and_failures(self):
        cases = [
            {
                "case_id": "hit",
                "query": "hit",
                "mode": "hybrid",
                "filters": {},
                "relevant": [{"repo": "training", "id": "HW.01"}],
                "tags": ["training", "zh-en"],
                "rationale": "test",
            },
            {
                "case_id": "miss",
                "query": "miss",
                "mode": "hybrid",
                "filters": {},
                "relevant": [{"repo": "training", "id": "NEG-REJECTED-1"}],
                "tags": ["negative"],
                "rationale": "test",
            },
        ]

        def fake_search(query, **kwargs):
            if query == "hit":
                return [{"repo": "training", "id": "HW.01"}]
            return [{"repo": "inference", "id": "1.1"}]

        report = evaluate_cases(cases, search_fn=fake_search)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["aggregate"]["recall_at_5"], 0.5)
        self.assertEqual(report["slices"]["training"]["recall_at_5"], 1.0)
        self.assertEqual(report["slices"]["negative"]["recall_at_5"], 0.0)
        self.assertEqual([x["case_id"] for x in report["failures"]], ["miss"])


class ScriptEntryPointTests(unittest.TestCase):
    def test_direct_script_execution_can_import_project_package(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.run(
            [sys.executable, "eval/run_eval.py", "--min-cases", "1000"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("benchmark requires at least 1000 cases", proc.stderr)
        self.assertNotIn("ModuleNotFoundError", proc.stderr)


class QualityGateTests(unittest.TestCase):
    def _report(self, recall=0.90, mrr=0.80, ndcg=0.82, negative=0.80):
        return {
            "aggregate": {
                "recall_at_5": recall,
                "mrr_at_10": mrr,
                "ndcg_at_10": ndcg,
            },
            "slices": {
                "negative": {
                    "case_count": 4,
                    "recall_at_5": negative,
                    "mrr_at_10": negative,
                    "ndcg_at_10": negative,
                }
            },
        }

    def test_absolute_metric_floor_is_enforced(self):
        failures = check_report(self._report(recall=0.80))
        self.assertTrue(any("recall_at_5 floor" in item for item in failures))

    def test_negative_slice_floor_is_enforced(self):
        failures = check_report(self._report(negative=0.50))
        self.assertTrue(any("negative recall_at_5 floor" in item for item in failures))

    def test_regression_over_tolerance_is_enforced(self):
        baseline = self._report(recall=0.90)
        failures = check_report(self._report(recall=0.87), baseline=baseline)
        self.assertTrue(any("recall_at_5 regressed" in item for item in failures))

    def test_passing_report_has_no_failures(self):
        baseline = self._report(recall=0.90)
        self.assertEqual(check_report(self._report(recall=0.89), baseline), [])


if __name__ == "__main__":
    unittest.main()
