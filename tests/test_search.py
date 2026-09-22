import unittest

import numpy as np

from fault_rag import search as S


class RankingTests(unittest.TestCase):
    def test_positive_ranks_excludes_zero_scores(self):
        scores = np.array([0.0, 2.0, 1.0])
        self.assertEqual(S._positive_ranks(scores, [0, 1, 2]), {1: 0, 2: 1})

    def test_positive_ranks_returns_empty_for_no_lexical_match(self):
        scores = np.array([0.0, 0.0, 0.0])
        self.assertEqual(S._positive_ranks(scores, [0, 1, 2]), {})

    def test_rrf_only_scores_rows_present_in_a_ranker(self):
        fused = S._rrf_scores({2: 0, 0: 1}, {})
        self.assertEqual(set(fused), {0, 2})
        ordered = sorted(fused, key=lambda i: -fused[i])
        self.assertEqual(ordered, [2, 0])


class EntryTypeFilterTests(unittest.TestCase):
    def _entry(self, entry_type, neg=False):
        return {
            "id": "10.4",
            "name": "chat_template_faults",
            "repo": "inference",
            "category": "cat10",
            "confidence": None,
            "trigger": "n/a",
            "entry_type": entry_type,
            "neg": neg,
        }

    def test_pointer_is_excluded_by_default(self):
        self.assertFalse(S._passes(self._entry("pointer")))

    def test_pointer_can_be_explicitly_included(self):
        self.assertTrue(
            S._passes(self._entry("pointer"), include_pointers=True)
        )

    def test_fault_and_negative_are_not_excluded_as_pointers(self):
        self.assertTrue(S._passes(self._entry("fault")))
        self.assertTrue(S._passes(self._entry("negative", neg=True)))


class IndexedPointerIntegrationTests(unittest.TestCase):
    def test_lookup_preserves_pointer_with_explicit_type(self):
        rows = S.lookup("10.4")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entry_type"], "pointer")
        self.assertFalse(rows[0]["neg"])

    def test_default_search_excludes_pointer_but_opt_in_restores_it(self):
        default = S.search("chat_template_faults", mode="bm25", topk=10)
        included = S.search(
            "chat_template_faults",
            mode="bm25",
            topk=10,
            include_pointers=True,
        )
        self.assertNotIn("10.4", {r["id"] for r in default})
        self.assertIn("10.4", {r["id"] for r in included})

    def test_stats_separates_fault_pointer_and_negative(self):
        counts = S.stats()["by_entry_type"]
        self.assertEqual(
            counts,
            {"fault": 1387, "negative": 14, "pointer": 1},
        )


class DuplicateNameIntegrationTests(unittest.TestCase):
    DUPLICATE = "vllm_prefix_cache_silent_noop_mamba_gdn_hybrid"

    def test_lookup_marks_every_duplicate_name_result_as_ambiguous(self):
        rows = S.lookup(self.DUPLICATE)
        self.assertEqual({r["id"] for r in rows}, {"6.28", "6.33"})
        for row in rows:
            self.assertTrue(row["name_ambiguous"])
            self.assertEqual(row["same_name_ids"], ["6.28", "6.33"])

    def test_unique_name_is_not_marked_ambiguous(self):
        row = S.lookup("HW.01")[0]
        self.assertFalse(row["name_ambiguous"])
        self.assertEqual(row["same_name_ids"], [])

    def test_stats_reports_canonical_name_collision(self):
        collisions = S.stats()["name_collisions"]
        self.assertIn(
            {
                "repo": "inference",
                "name": self.DUPLICATE,
                "ids": ["6.28", "6.33"],
            },
            collisions,
        )


class BM25IntegrationTests(unittest.TestCase):
    def test_bm25_returns_empty_for_unknown_token(self):
        results = S.search(
            "zzzxqv_nonexistent_token_94731",
            mode="bm25",
            topk=3,
            include_neg=False,
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
