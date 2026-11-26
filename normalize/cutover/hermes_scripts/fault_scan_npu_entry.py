#!/usr/bin/env python3
"""两条故障扫描共用的 NPU 采集前置入口；不写编目、不处理存量 Git。"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess

PROFILE = Path("/root/.hermes/profiles/third")
SHARED = PROFILE / "scripts" / "fault_scan_sources"
CACHE = PROFILE / "cache" / "fault_scan_sources"
DOTENV = PROFILE / ".env"


def read_profile_env_var(name, env_file=DOTENV):
    """Safely parse a single variable from profile .env without shell eval or secret echo."""
    if not env_file or not Path(env_file).exists():
        return ""
    val = ""
    for raw in Path(env_file).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(rf"^(?:export\s+)?{re.escape(name)}=(.+)$", line)
        if m:
            try:
                toks = shlex.split(m.group(1), comments=True)
                val = toks[0].strip() if toks else ""
            except ValueError:
                val = m.group(1).strip().strip("\"'")
    return val


def collector_env(base_env=None, env_file=DOTENV):
    """Build subprocess environment for collect.py, injecting GITHUB_TOKEN from .env if absent."""
    env = dict(os.environ if base_env is None else base_env)
    if not env.get("GITHUB_TOKEN", "").strip():
        token = read_profile_env_var("GITHUB_TOKEN", env_file=env_file)
        if token:
            env["GITHUB_TOKEN"] = token
    return env


def preflight(repo):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True, timeout=90).strip()

    if git("rev-parse", "--abbrev-ref", "HEAD") != "main":
        raise RuntimeError(f"BLOCKED: {repo}: not on main")
    if git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}") != "origin/main":
        raise RuntimeError(f"BLOCKED: {repo}: upstream is not origin/main")
    if git("status", "--porcelain", "-uall"):
        raise RuntimeError(f"BLOCKED: {repo}: dirty worktree; existing files untouched")
    subprocess.run(["git", "fetch", "--quiet", "origin", "main"], cwd=repo, check=True, timeout=90)
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        raise RuntimeError(f"BLOCKED: {repo}: HEAD differs from origin/main")


def collector_command(scope):
    return ["/usr/bin/python3", str(SHARED / "collect.py"), "--scope", scope,
            "--registry", str(SHARED / "sources.json"), "--output-dir", str(CACHE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=["inference", "training"])
    args = parser.parse_args()
    preflight(Path("/workspace") / f"{args.scope}_error_review")
    subprocess.run(collector_command(args.scope), env=collector_env(), check=True)


if __name__ == "__main__":
    main()
