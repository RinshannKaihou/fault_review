import contextlib
import io
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
    POINTER = "INF-0920"   # chat_template_faults

    def test_lookup_preserves_pointer_with_explicit_type(self):
        rows = S.lookup(self.POINTER)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], self.POINTER)
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
        self.assertNotIn(self.POINTER, {r["id"] for r in default})
        self.assertIn(self.POINTER, {r["id"] for r in included})

    def test_stats_separates_fault_pointer_and_negative(self):
        # Floor checks: vendored catalogs grow on every sync; the structural
        # invariant is that the three entry types stay separated.
        counts = S.stats()["by_entry_type"]
        self.assertGreaterEqual(counts["fault"], 1600)
        self.assertGreaterEqual(counts["negative"], 14)
        self.assertEqual(counts["pointer"], 1)


class DuplicateNameIntegrationTests(unittest.TestCase):
    DUPLICATE = "vllm_prefix_cache_silent_noop_mamba_gdn_hybrid"

    def test_lookup_marks_every_duplicate_name_result_as_ambiguous(self):
        rows = S.lookup(self.DUPLICATE)
        self.assertEqual({r["id"] for r in rows}, {"INF-0421", "INF-0426"})
        for row in rows:
            self.assertTrue(row["name_ambiguous"])
            self.assertEqual(row["same_name_ids"], ["INF-0421", "INF-0426"])

    def test_unique_name_is_not_marked_ambiguous(self):
        row = S.lookup("TRN-0001")[0]
        self.assertFalse(row["name_ambiguous"])
        self.assertEqual(row["same_name_ids"], [])

    def test_stats_reports_canonical_name_collision(self):
        collisions = S.stats()["name_collisions"]
        self.assertIn(
            {
                "repo": "inference",
                "name": self.DUPLICATE,
                "ids": ["INF-0421", "INF-0426"],
            },
            collisions,
        )


class CategoryFilterTests(unittest.TestCase):
    def test_subsystem_filters(self):
        rows = S.search("prefix cache stale KV reuse", topk=10, repo="inference", category="KV缓存")
        self.assertTrue(rows)
        self.assertTrue(all(r["category"] == "KV缓存" for r in rows))

    def test_negative_kind_filters(self):
        rows = S.search("OOM", topk=5, mode="bm25", category="neg_rejected")
        self.assertTrue(rows)
        self.assertTrue(all(r["neg"] and r["category"] == "neg_rejected" for r in rows))

    def test_retired_old_category_matches_nothing_and_says_so(self):
        # cat6 / KV6 / NUM were the old Markdown categories, retired 2026-10-26
        for old in ("cat6", "KV6", "NUM"):
            with self.subTest(old=old):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    rows = S.search("prefix cache stale KV reuse", topk=5, mode="bm25", category=old)
                self.assertEqual(rows, [])
                self.assertIn("retired", err.getvalue())

    def test_rows_carry_title_and_no_old_id_fields(self):
        row = S.search("topk single core ignores indices tensor", topk=3, mode="bm25", repo="inference")[0]
        self.assertTrue(row["title"])
        self.assertNotIn("legacy_ids", row)
        self.assertNotIn("legacy_category", row)


class BM25IntegrationTests(unittest.TestCase):
    def test_bm25_returns_empty_for_unknown_token(self):
        results = S.search(
            "zzzxqv_nonexistent_token_94731",
            mode="bm25",
            topk=3,
            include_neg=False,
        )
        self.assertEqual(results, [])


class RerankTests(unittest.TestCase):
    def setUp(self):
        from fault_rag import rerank as R
        self.R = R
        self._scores, self._available = R.scores, R.available

    def tearDown(self):
        self.R.scores, self.R.available = self._scores, self._available

    def test_blend_keeps_hybrid_signal(self):
        # cross-encoder alone would put row 3 first; the RRF blend with hybrid rank keeps row 0
        self.R.scores = lambda q, entries: [0.5, 0.1, 0.2, 0.9]
        order, scores, rr = S._rerank("q", [{}] * 5, [0, 1, 2, 3, 4], {i: 0.0 for i in range(5)}, pool=4)
        self.assertEqual(rr, {3: 0, 0: 1, 2: 2, 1: 3})
        self.assertEqual(order[:4], [0, 3, 1, 2])          # 1/61+1/63 edges out 1/62+1/62
        self.assertEqual(order[4], 4)                       # rows past the pool keep hybrid order
        self.assertAlmostEqual(scores[0], 1 / 60 + 1 / 61)

    def test_missing_model_falls_back_to_hybrid(self):
        self.R.available = lambda: False
        rows = S.search("loss spike", topk=3)
        self.assertTrue(rows)
        self.assertTrue(all(r["rerank_rank"] is None for r in rows))

    def test_single_channel_modes_are_not_reranked(self):
        rows = S.search("loss spike", topk=3, mode="bm25")
        self.assertTrue(all(r["rerank_rank"] is None for r in rows))

    def test_hybrid_search_is_reranked_when_model_present(self):
        if not self.R.available():
            self.skipTest("reranker model not downloaded")
        rows = S.search("loss spike", topk=3)
        self.assertTrue(all(r["rerank_rank"] is not None for r in rows))


if __name__ == "__main__":
    unittest.main()
