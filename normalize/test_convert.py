import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import build  # noqa: E402


def entry(eid, name):
    return {
        "id": eid, "name": name, "aliases": [], "repo": "inference", "category": "cat1",
        "section": "", "raw": f"### {eid} `{name}`\n", "fields": {}, "entry_type": "fault",
        "axis": None, "trigger": None, "confidence": None, "coverage": None, "stage": None,
    }


class StableIdTest(unittest.TestCase):
    def test_upstream_insertion_keeps_existing_ids(self):
        registry = {}
        build([("INF", "inference", [entry("1.1", "a"), entry("2.1", "b")])], registry)
        self.assertEqual(registry, {("INF", "1.1"): "INF-0001", ("INF", "2.1"): "INF-0002"})

        # upstream appends 1.2 at the end of category 1, i.e. before 2.1 in document order
        records = build([("INF", "inference", [entry("1.1", "a"), entry("1.2", "c"), entry("2.1", "b")])], registry)
        ids = {r["legacy_ids"][0]: r["id"] for r in records}
        self.assertEqual(ids, {"1.1": "INF-0001", "2.1": "INF-0002", "1.2": "INF-0003"})

    def test_namespaces_number_independently(self):
        registry = {("INF", "1.1"): "INF-0007"}
        build([("TRN", "training", [entry("HW.01", "x")])], registry)
        self.assertEqual(registry[("TRN", "HW.01")], "TRN-0001")


if __name__ == "__main__":
    unittest.main()
