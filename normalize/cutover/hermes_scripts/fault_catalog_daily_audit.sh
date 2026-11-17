#!/usr/bin/env bash
# hermes cron (--no-agent) job: audit entries the daily scans wrote, in both catalog repos.
# Install as ~/.hermes/profiles/third/scripts/fault_catalog_daily_audit.sh; stdout is delivered verbatim (Feishu).
set -uo pipefail
# Read JUDGE_API_KEY and JUDGE_API_URL safely without eval-ing shell lines from ~/.bashrc
read_rc_var() {
  local name="$1"
  python3 - "$name" "$HOME/.bashrc" <<'PY'
import os, re, shlex, sys
name, rc = sys.argv[1], sys.argv[2]
val = ""
if os.path.exists(rc):
    for line in open(rc, encoding="utf-8", errors="ignore"):
        m = re.match(rf"^\s*export\s+{re.escape(name)}=(.+)$", line.strip())
        if m:
            toks = shlex.split(m.group(1), comments=True)
            val = toks[0] if toks else ""
print(val, end="")
PY
}
export JUDGE_API_KEY="${JUDGE_API_KEY:-$(read_rc_var JUDGE_API_KEY)}"
export JUDGE_API_URL="${JUDGE_API_URL:-$(read_rc_var JUDGE_API_URL)}"
[ -n "${JUDGE_API_KEY:-}" ] || { echo "❌ 故障编目每日审计：JUDGE_API_KEY 未设置，未运行"; exit 1; }
[ -n "${JUDGE_API_URL:-}" ] || { echo "❌ 故障编目每日审计：JUDGE_API_URL 未设置，未运行"; exit 1; }

ROOT="${CATALOG_REPOS_ROOT:-/workspace}"   # override only for sandbox tests
echo "📋 故障编目每日审计 — $(date -u +%F) UTC"
for repo in inference_error_review training_error_review; do
  log=$(mktemp)
  bash "$ROOT/$repo/tools/daily_audit.sh" > "$log" 2>&1
  code=$?
  stats=$(python3 - "$log" <<'EOF'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"\{\s*\"ok\".*?\n\}", text, re.S)
if m:
    d = json.loads(m.group(0))
    short = f"（其中票数不足 {d['uncertain_short_fields']}）" if "uncertain_short_fields" in d else ""
    void = f"；空票 {d['void_replies']}" if "void_replies" in d else ""
    print(f"审计 {d.get('audited', 0)} 条（评审团 {d.get('panel', 0)}）；纠正 {d.get('corrected_fields', 0)}、改写 {d.get('rewritten_fields', 0)}、"
          f"未修复 {d.get('unfixed_fields', 0)}、不确定 {d.get('uncertain_fields', 0)} 个字段{short}{void}")
EOF
)
  last=$(grep -E "Z (OK|FAILED|BLOCKED|nothing pending|no changes)" "$log" | tail -n 1 | sed -E 's/^[0-9T:-]+Z //')
  [ $code -eq 0 ] && mark="✅" || mark="❌"
  echo "$mark ${repo%_error_review}：${last:-无输出}${stats:+；$stats}"
  [ $code -eq 0 ] || tail -n 15 "$log"
  rm -f "$log"
done
