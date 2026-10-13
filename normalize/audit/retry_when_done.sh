#!/usr/bin/env bash
# Wait until the main pass of (task, model) has exited, then run its --retry-solo pass.
#   nohup setsid bash retry_when_done.sh F kimi-k3 [tag] [items] < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")"
TASK=$1 MODEL=$2 TAG=${3:-} ITEMS=${4:-items.jsonl} WORKERS=${RETRY_WORKERS:-12}
LOG="logs/run_${TASK}${TAG}_${MODEL}.log"
EXTRA=()
case "$MODEL" in
  kimi-k3)             EXTRA=(--extra '{"temperature":1.0}' --max-tokens 32768) ;;
  doubao-seed-2.1-pro) EXTRA=(--extra '{"reasoning_effort":"low"}') ;;
  glm-5.3)             EXTRA=(--max-tokens 32768) ;;
esac
while pgrep -f "^python3 audit_judge.py --task $TASK --model $MODEL --items $ITEMS" >/dev/null; do sleep 60; done
{
  python3 audit_judge.py --task "$TASK" --model "$MODEL" --items "$ITEMS" --workers "$WORKERS" --tag "$TAG" --retry-solo "${EXTRA[@]}" \
  && echo "$(date -u +%FT%TZ) OK retry $TASK$TAG $MODEL" || echo "$(date -u +%FT%TZ) FAILED retry $TASK$TAG $MODEL"
} >> "$LOG" 2>&1
