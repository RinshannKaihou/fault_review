"""Unit tests for gh_api.py: .env token loading, host guard, pagination, and --out ledger."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("gh_api.py")


class _FakeResponse:
    def __init__(self, status: int, body_obj, headers: dict | None = None):
        self.status = status
        self._raw = json.dumps(body_obj).encode("utf-8")
        self.headers = headers or {}

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _RecordingOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, req, timeout=45):
        self.requests.append((req.full_url, dict(req.header_items()), timeout))
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class GhApiTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("gh_api_under_test", SCRIPT)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_rejects_non_github_host(self):
        with self.assertRaises(self.mod.GhApiError):
            self.mod.fetch_github("https://evil.example.com/search/issues", env_file=Path("/nonexistent"))

    def test_loads_token_and_paginates_search_without_leaking_secret(self):
        env_file = Path(self.tmp.name) / ".env"
        secret = "mock_super_secret_test_value_999"
        env_file.write_text(f"export GITHUB_TOKEN={secret}\n", encoding="utf-8")

        opener = _RecordingOpener([
            _FakeResponse(
                200,
                {"total_count": 3, "incomplete_results": False, "items": [{"number": 1}, {"number": 2}]},
                {
                    "Link": '<https://api.github.com/search/issues?q=NaN&per_page=2&page=2>; rel="next"',
                    "X-RateLimit-Limit": "30",
                    "X-RateLimit-Remaining": "29",
                    "X-RateLimit-Reset": "1795000000",
                },
            ),
            _FakeResponse(
                200,
                {"total_count": 3, "incomplete_results": False, "items": [{"number": 3}]},
                {
                    "X-RateLimit-Limit": "30",
                    "X-RateLimit-Remaining": "28",
                    "X-RateLimit-Reset": "1795000000",
                },
            ),
        ])

        old_tok = os.environ.pop("GITHUB_TOKEN", None)
        try:
            res = self.mod.fetch_github(
                "/search/issues",
                params=[("q", "NaN"), ("per_page", "2")],
                max_pages=3,
                env_file=env_file,
                opener=opener,
            )
        finally:
            if old_tok is not None:
                os.environ["GITHUB_TOKEN"] = old_tok

        self.assertTrue(res["ok"])
        self.assertTrue(res["authenticated"])
        self.assertTrue(res["complete"])
        self.assertFalse(res["has_more"])
        self.assertEqual(res["pages_fetched"], 2)
        self.assertEqual(res["total_count"], 3)
        self.assertEqual(res["fetched_count"], 3)
        self.assertEqual([x["number"] for x in res["data"]["items"]], [1, 2, 3])
        # Verify Authorization header was sent to api.github.com, yet secret is absent from output JSON
        self.assertEqual(opener.requests[0][1].get("Authorization"), f"Bearer {secret}")
        serialized = json.dumps(res)
        self.assertNotIn(secret, serialized)

    def test_unsafe_pagination_link_is_blocked(self):
        opener = _RecordingOpener([
            _FakeResponse(
                200,
                [{"id": 1}],
                {"Link": '<https://attacker.example.com/steal?page=2>; rel="next"'},
            ),
        ])
        old_tok = os.environ.pop("GITHUB_TOKEN", None)
        try:
            with self.assertRaises(self.mod.GhApiError):
                self.mod.fetch_github(
                    "/repos/vllm-project/vllm/issues/1/comments",
                    max_pages=2,
                    env_file=Path("/nonexistent"),
                    opener=opener,
                )
        finally:
            if old_tok is not None:
                os.environ["GITHUB_TOKEN"] = old_tok


if __name__ == "__main__":
    unittest.main()
