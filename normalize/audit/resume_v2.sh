#!/usr/bin/env bash
# Resume the v2 re-judgment after a gateway quota pause (2026-10-12 13:50 UTC).
#   bash resume_v2.sh            # probe quota -> purge quota-caused failures -> relaunch 6 detached runs
# Aborts without touching anything if any base model still fails the quota probe.
set -uo pipefail
cd "$(dirname "$0")"
API=https://st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com/v1/chat/completions
TS=$(date -u +%Y%m%dT%H%M%SZ)

if pgrep -f "^python3 audit_judge.py" >/dev/null; then echo "judges already running; refusing"; exit 1; fi

# 1. quota probe: one tiny call per base model
for m in qwen3.8-max kimi-k3 glm-5.3; do
  temp=0.0; [ "$m" = kimi-k3 ] && temp=1.0
  code=$(curl -s -m 120 -o /tmp/quota_probe.json -w "%{http_code}" -H "Authorization: Bearer $JUDGE_API_KEY" \
    -H 'Content-Type: application/json' "$API" \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"回复OK\"}],\"max_tokens\":2048,\"temperature\":$temp}")
  echo "probe $m HTTP $code"
  if [ "$code" != 200 ]; then
    python3 -c "import json;print(json.load(open('/tmp/quota_probe.json')).get('error'))" 2>/dev/null
    echo "$(date -u +%FT%TZ) resume aborted: $m probe HTTP $code"; exit 2
  fi
done

# 2. back up and 3. purge failures caused by the quota outage (empty responses), so they are re-judged in batches
mkdir -p "votes/backup_pre_resume_$TS"
cp votes/*_v2_*progress.jsonl "votes/backup_pre_resume_$TS/" 2>/dev/null
python3 - <<'EOF'
import glob, json
for path in sorted(glob.glob("votes/*_v2_*progress.jsonl")):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    keep = [r for r in rows if not (r.get("parse_failed") and r.get("fail_reason") == "no_content")]
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
    print(f"{path}: purged {len(rows) - len(keep)} empty-response failures, kept {len(keep)}")
EOF

# 4. relaunch with the same settings as the paused runs
for f in logs/run_*_v2_*.log; do echo "$(date -u +%FT%TZ) RESUMED after quota pause" >> "$f"; done
for m in qwen3.8-max kimi-k3; do
  nohup setsid bash run_task.sh C "$m" 12 _v2 items_v2_C.jsonl < /dev/null > /dev/null 2>&1 & disown
  nohup setsid bash run_task.sh F "$m" 24 _v2 items_v2_F.jsonl < /dev/null > /dev/null 2>&1 & disown
done
BATCH=1 nohup setsid bash run_task.sh C glm-5.3 16 _v2 items_v2_C.jsonl < /dev/null > /dev/null 2>&1 & disown
BATCH=1 nohup setsid bash run_task.sh F glm-5.3 24 _v2 items_v2_F.jsonl < /dev/null > /dev/null 2>&1 & disown
sleep 15
echo "relaunched: $(pgrep -fc '^python3 audit_judge.py') judge processes"
