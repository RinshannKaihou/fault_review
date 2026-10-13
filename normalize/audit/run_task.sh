#!/usr/bin/env bash
# Run one (task, model) judge pass plus its --retry-solo pass, appending to logs/run_<task><tag>_<model>.log.
#   nohup setsid bash run_task.sh F kimi-k3 24 [tag] [items] < /dev/null & disown
# Per-model request options are applied here so no JSON has to survive nested shell quoting.
# Env: BATCH=<records per call> (default from audit_judge.py), RETRY_WORKERS (default 12).
set -uo pipefail
cd "$(dirname "$0")"
TASK=$1 MODEL=$2 WORKERS=$3 TAG=${4:-} ITEMS=${5:-items.jsonl}
LOG="logs/run_${TASK}${TAG}_${MODEL}.log"
EXTRA=()
case "$MODEL" in
  kimi-k3)             EXTRA=(--extra '{"temperature":1.0}' --max-tokens 32768) ;;
  doubao-seed-2.1-pro) EXTRA=(--extra '{"reasoning_effort":"low"}' --max-tokens 32768) ;;
  deepseek-v4-pro|MiniMax-M3|hy3) EXTRA=(--max-tokens 32768) ;;    # panel: avoid the truncation seen with glm at 16k
  glm-5.3)             EXTRA=(--max-tokens 32768) ;;                 # 16k truncated ~60% of 3-record F batches
esac
[ -n "${BATCH:-}" ] && EXTRA+=(--batch "$BATCH")
if pgrep -f "^python3 audit_judge.py --task $TASK --model $MODEL .*--tag ${TAG:-NONE}" >/dev/null \
   || { [ -z "$TAG" ] && pgrep -f "^python3 audit_judge.py --task $TASK --model $MODEL --items $ITEMS --workers" >/dev/null; }; then
  echo "$(date -u +%FT%TZ) $TASK$TAG $MODEL already running; refusing" >> "$LOG"; exit 1
fi
{
  python3 audit_judge.py --task "$TASK" --model "$MODEL" --items "$ITEMS" --workers "$WORKERS" --tag "$TAG" "${EXTRA[@]}" \
  && python3 audit_judge.py --task "$TASK" --model "$MODEL" --items "$ITEMS" --workers "${RETRY_WORKERS:-12}" --tag "$TAG" --retry-solo "${EXTRA[@]}" \
  && echo "$(date -u +%FT%TZ) OK $TASK$TAG $MODEL" || echo "$(date -u +%FT%TZ) FAILED $TASK$TAG $MODEL"
} >> "$LOG" 2>&1
