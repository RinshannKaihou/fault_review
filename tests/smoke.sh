#!/usr/bin/env bash
# Smoke battery: CLI entry-point check + in-process query battery.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== CLI entry point =="
./fault-rag stats > /dev/null && echo "[PASS] ./fault-rag stats runs"
./fault-rag search "loss spike" --topk 2 | python3 -c "
import json, sys
out = json.load(sys.stdin)
assert 'results' in out and len(out['results']) <= 2, out
print('[PASS] ./fault-rag search returns JSON with results')
"

echo
echo "== in-process battery =="
python3 tests/smoke.py
