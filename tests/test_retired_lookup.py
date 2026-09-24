import unittest

from fault_rag import search


class RetiredIdLookupTests(unittest.TestCase):
    def test_missing_catalog_id_never_matches_another_id_by_substring(self):
        # 1.70 was merged into 1.44; 11.70 remains a different, valid fault.
        self.assertEqual(search.lookup("1.70"), [])
        self.assertEqual(search.lookup("11.30"), [])

    def test_existing_catalog_id_and_name_substring_still_work(self):
        self.assertEqual([row["id"] for row in search.lookup("11.70")], ["11.70"])
        self.assertTrue(search.lookup("ttmetal_topk_singlecore"))


if __name__ == "__main__":
    unittest.main()
