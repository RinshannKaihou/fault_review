#!/usr/bin/env bash
# Detached driver for the catalog audit (pattern: llm-judge-labeling langmix_v3.sh).
#   nohup setsid bash audit_run.sh base  >> logs/driver.log 2>&1 & disown
#   nohup setsid bash audit_run.sh panel >> logs/driver.log 2>&1 & disown
# base : BASE_MODELS (default kimi-k3 / qwen3.8-max / glm-5.3) on all items, tasks C and F, then a --retry-solo pass each.
# panel: audit_merge.py export-panel, then 4 panel models on the non-final items, then retry-solo.
# Per-model options live here, not in the judge script. Logs are appended, never truncated.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs votes
STAGE="${1:?usage: audit_run.sh base|panel}"
BASE_MODELS="${BASE_MODELS:-kimi-k3 qwen3.8-max glm-5.3}"
PANEL_MODELS="${PANEL_MODELS:-doubao-seed-2.1-pro deepseek-v4-pro MiniMax-M3 hy3}"
if [ "$STAGE" = base ]; then MODELS="$BASE_MODELS"; else MODELS="$PANEL_MODELS"; fi

for m in $MODELS; do   # never two runs of the same model on the same checkpoints; anchored so it cannot match a shell
  if pgrep -f "^python3 audit_judge.py --task [CF] --model $m " >/dev/null; then
    echo "$(date -u +%FT%TZ) $m already running; refusing to start" >&2
    exit 1
  fi
done

opts() {   # per-model request options
  case "$1" in
    kimi-k3)             echo "--extra {\"temperature\":1.0}" ;;     # model rejects any other temperature
    doubao-seed-2.1-pro) echo "--extra {\"reasoning_effort\":\"low\"}" ;;
    *)                   echo "" ;;
  esac
}

run_one() {   # task model items tag workers
  local task=$1 model=$2 items=$3 tag=$4 workers=$5 log="logs/run_${1}${4}_${2}.log"
  ( python3 audit_judge.py --task "$task" --model "$model" --items "$items" --tag "$tag" --workers "$workers" $(opts "$model") \
    && python3 audit_judge.py --task "$task" --model "$model" --items "$items" --tag "$tag" --workers 4 --retry-solo $(opts "$model") \
    && echo "$(date -u +%FT%TZ) OK $task$tag $model" || echo "$(date -u +%FT%TZ) FAILED $task$tag $model" ) >> "$log" 2>&1
}

echo "$(date -u +%FT%TZ) start stage $STAGE"
if [ "$STAGE" = base ]; then
  for m in $MODELS; do
    run_one C "$m" items.jsonl "" 8 &
    run_one F "$m" items.jsonl "" 12 &
  done
  wait
elif [ "$STAGE" = panel ]; then
  python3 audit_merge.py export-panel
  for m in $MODELS; do
    [ -s panel_items_C.jsonl ] && run_one C "$m" panel_items_C.jsonl _panel 6 &
    [ -s panel_items_F.jsonl ] && run_one F "$m" panel_items_F.jsonl _panel 8 &
  done
  wait
else
  echo "unknown stage $STAGE" >&2; exit 2
fi
grep -h -E "^[0-9T:-]+Z (OK|FAILED)" logs/run_*.log | sort | tail -n 24
echo "$(date -u +%FT%TZ) end stage $STAGE"
