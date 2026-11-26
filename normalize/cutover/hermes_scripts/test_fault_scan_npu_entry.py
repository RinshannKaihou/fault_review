"""NPU Cron 前置入口测试：本地 Git remote + .env GITHUB_TOKEN 注入，不碰真实编目。"""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ENTRY = Path(__file__).with_name("fault_scan_npu_entry.py")


class EntryTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRY.exists(), "missing NPU cron entry script")
        spec = importlib.util.spec_from_file_location("entry_under_test", ENTRY)
        self.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.entry)
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.repo = root / "repo"
        remote = root / "remote.git"
        self.env = dict(os.environ, GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.invalid",
                        GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.invalid")
        self.git("init", "--bare", str(remote), cwd=root)
        self.git("init", "-b", "main", str(self.repo), cwd=root)
        self.git("commit", "--allow-empty", "-m", "test baseline")
        self.git("remote", "add", "origin", str(remote))
        self.git("push", "-u", "origin", "main")

    def git(self, *args, cwd=None):
        return subprocess.run(["git", *args], cwd=cwd or self.repo, env=self.env,
                              check=True, capture_output=True, text=True).stdout

    def test_clean_main_and_matching_remote_pass(self):
        self.entry.preflight(self.repo)

    def test_dirty_worktree_is_not_scanned_or_removed(self):
        dirty = self.repo / "unrelated.txt"
        dirty.write_text("leave this untouched", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "dirty"):
            self.entry.preflight(self.repo)
        self.assertEqual(dirty.read_text(encoding="utf-8"), "leave this untouched")

    def test_local_unpushed_commit_is_blocked(self):
        self.git("commit", "--allow-empty", "-m", "unpushed")
        with self.assertRaisesRegex(RuntimeError, "origin/main"):
            self.entry.preflight(self.repo)

    def test_non_main_branch_is_blocked(self):
        self.git("checkout", "-b", "not-main")
        with self.assertRaisesRegex(RuntimeError, "main"):
            self.entry.preflight(self.repo)

    def test_command_uses_system_python_shared_cache_and_scope(self):
        cmd = self.entry.collector_command("training")
        self.assertEqual(cmd[0], "/usr/bin/python3")
        self.assertEqual(cmd[cmd.index("--scope") + 1], "training")
        self.assertEqual(cmd[cmd.index("--output-dir") + 1],
                         "/root/.hermes/profiles/third/cache/fault_scan_sources")
        self.assertIn("--registry", cmd)

    def test_collector_env_loads_github_token_from_dotenv_without_overriding_explicit(self):
        dotenv = Path(self.tmp.name) / ".env"
        dotenv.write_text(
            "# comment\nOTHER_KEY=abc\nGITHUB_TOKEN='mock_token_123' # inline comment\n",
            encoding="utf-8",
        )
        injected = self.entry.collector_env(base_env={"PATH": "/usr/bin"}, env_file=dotenv)
        self.assertEqual(injected.get("GITHUB_TOKEN"), "mock_token_123")

        explicit = self.entry.collector_env(
            base_env={"PATH": "/usr/bin", "GITHUB_TOKEN": "explicit_override_token"},
            env_file=dotenv,
        )
        self.assertEqual(explicit.get("GITHUB_TOKEN"), "explicit_override_token")


if __name__ == "__main__":
    unittest.main()
