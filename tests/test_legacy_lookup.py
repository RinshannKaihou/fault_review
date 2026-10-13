import unittest

from fault_rag import search


class LegacyIdLookupTests(unittest.TestCase):
    """Old Markdown ids resolve to the entry that now carries them (catalog/entries legacy_ids)."""

    def test_old_ids_resolve_to_new_ids(self):
        self.assertEqual([r["id"] for r in search.lookup("1.44")], ["INF-0044"])
        self.assertEqual([r["id"] for r in search.lookup("HW.01")], ["TRN-0001"])
        self.assertEqual([r["id"] for r in search.lookup("INF-0044")], ["INF-0044"])

    def test_merged_old_ids_redirect_to_the_kept_entry(self):
        # 2026-09-24 erratum: 1.70 merged into 1.44, 11.30 into 11.24
        self.assertEqual([r["id"] for r in search.lookup("1.70")], ["INF-0044"])
        self.assertEqual([r["id"] for r in search.lookup("11.30")], ["INF-1048"])
        self.assertIn("1.70", search.lookup("1.70")[0]["legacy_ids"])

    def test_unknown_id_never_matches_another_id_by_substring(self):
        self.assertEqual([r["id"] for r in search.lookup("11.70")], ["INF-1093"])   # not 1.70's entry
        self.assertEqual(search.lookup("1.9999"), [])
        self.assertEqual(search.lookup("INF-9999"), [])

    def test_name_substring_still_works(self):
        self.assertTrue(search.lookup("ttmetal_topk_singlecore"))


if __name__ == "__main__":
    unittest.main()
