import contextlib
import io
import unittest

from fault_rag import search


def _lookup(q):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rows = search.lookup(q)
    return [r["id"] for r in rows], err.getvalue()


class RetiredLegacyIdTests(unittest.TestCase):
    """Old Markdown ids (1.44 / HW.01) were retired 2026-10-26: lookup does not resolve them,
    and says where to translate them instead."""

    def test_old_ids_are_not_resolved_but_point_to_the_registry(self):
        for old, registry in (("1.44", "catalogs/inference/id_registry.tsv"),
                              ("K9.3", "catalogs/inference/id_registry.tsv"),
                              ("HW.01", "catalogs/training/id_registry.tsv"),
                              ("RL-RO.01", "catalogs/training/id_registry.tsv")):
            with self.subTest(old=old):
                ids, err = _lookup(old)
                self.assertEqual(ids, [])
                self.assertIn(registry, err)

    def test_merged_old_ids_are_not_resolved_either(self):
        # 1.70 was merged into 1.44 (INF-0044) and 11.30 into 11.24 (INF-1048) before the cut-over
        self.assertEqual(_lookup("1.70")[0], [])
        self.assertEqual(_lookup("11.30")[0], [])

    def test_current_ids_and_names_still_resolve_without_a_hint(self):
        self.assertEqual(_lookup("INF-0044"), (["INF-0044"], ""))
        self.assertEqual(_lookup("TRN-0001"), (["TRN-0001"], ""))
        self.assertTrue(_lookup("ttmetal_topk_singlecore")[0])

    def test_unknown_id_never_matches_another_id_by_substring(self):
        self.assertEqual(_lookup("INF-9999"), ([], ""))
        self.assertEqual(_lookup("1.9999")[0], [])

    def test_lookup_rows_carry_no_old_id_fields(self):
        row = search.lookup("INF-0044")[0]
        self.assertNotIn("legacy_ids", row)
        self.assertNotIn("legacy_category", row)


if __name__ == "__main__":
    unittest.main()
