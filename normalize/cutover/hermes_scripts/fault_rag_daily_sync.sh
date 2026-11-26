#!/usr/bin/env bash
# hermes cron (--no-agent) job: safely sync audited upstream catalog entries into fault_review,
# run CLI/Web/Topology gates in an isolated worktree (with per-entry quarantine), run thin-layer
# read-only LLM retrievability/link QA, atomically update RAG index + push, and route follow-ups.
# Install as ~/.hermes/profiles/third/scripts/fault_rag_daily_sync.sh; stdout is delivered verbatim (Feishu).
set -uo pipefail

read_rc_var() {
  local name="$1"
  /usr/bin/python3 - "$name" "${HOME:-/root}/.bashrc" <<'PY'
import os, pwd, re, shlex, sys
name = sys.argv[1]
candidates = [sys.argv[2]]
try:
    candidates.append(os.path.join(pwd.getpwuid(os.getuid()).pw_dir, ".bashrc"))
except KeyError:
    pass
candidates.append("/root/.bashrc")
val = ""
seen = set()
for rc in candidates:
    if not rc or rc in seen or not os.path.exists(rc):
        continue
    seen.add(rc)
    for line in open(rc, encoding="utf-8", errors="ignore"):
        m = re.match(rf"^\s*export\s+{re.escape(name)}=(.+)$", line.strip())
        if m:
            try:
                toks = shlex.split(m.group(1), comments=True)
                if toks:
                    val = toks[0]
            except ValueError:
                pass
    if val:
        break
print(val, end="")
PY
}

export JUDGE_API_KEY="${JUDGE_API_KEY:-$(read_rc_var JUDGE_API_KEY)}"
export JUDGE_API_URL="${JUDGE_API_URL:-$(read_rc_var JUDGE_API_URL)}"

RAG_ROOT="${FAULT_RAG_ROOT:-/workspace/RAG/fault_review}"
exec /usr/bin/python3 "$RAG_ROOT/scripts/rag_auto_sync.py" sync "$@"
