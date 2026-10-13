#!/usr/bin/env bash
# Deterministic tail of the audit, run detached after the panel was launched:
#   nohup setsid bash finish_audit.sh < /dev/null >> logs/finish.log 2>&1 & disown
# wait panel -> final decisions -> rewrite confirmed issues (qwen) -> verify rewrites (kimi + glm, task F)
# -> accept unanimous passes -> apply-only rebuild from the frozen baseline -> idempotence check -> validate.
# Each step is resumable; prints counts only.
set -uo pipefail
cd "$(dirname "$0")"
step() { echo "$(date -u +%FT%TZ) == $*"; }

step "waiting for panel judges"
while pgrep -f "^python3 audit_judge.py --task [CF] --model .*--tag _panel" >/dev/null || pgrep -f "^bash run_task.sh [CF] .* _panel" >/dev/null; do sleep 60; done

step "final decisions"
python3 audit_merge.py final || exit 1

step "rewrite confirmed issues"
python3 audit_fix.py fix --model qwen3.8-max || exit 1
python3 audit_fix.py verify-items || exit 1

if [ -s fix_verify_items.jsonl ]; then
  step "verify rewrites"
  bash run_task.sh F kimi-k3 8 _fixverify fix_verify_items.jsonl &
  BATCH=1 bash run_task.sh F glm-5.3 8 _fixverify fix_verify_items.jsonl &
  wait
fi

step "accept verified rewrites"
python3 audit_fix.py accept || exit 1

step "apply onto the frozen baseline"
python3 audit_apply.py || exit 1
python3 audit_apply.py --check || exit 1

step "validate"
python3 ../validate.py --quiet
step "done"
