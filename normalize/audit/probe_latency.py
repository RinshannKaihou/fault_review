#!/usr/bin/env python3
"""Measure one judge call on a single item: HTTP status, latency, token usage, finish reason, parse.

  python3 probe_latency.py --model kimi-k3 --task F [--extra '{"temperature":1.0}'] [--max-tokens 16384]

Prints numbers only (never the response text).
"""

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_judge as J  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--task", default="F", choices=list(J.TASKS))
ap.add_argument("--extra", default="{}")
ap.add_argument("--max-tokens", type=int, default=16384)
ap.add_argument("--timeout", type=int, default=1500)
a = ap.parse_args()

item = json.loads(open(os.path.join(J.HERE, "items.jsonl"), encoding="utf-8").readline())
sysm, userf = J.TASKS[a.task]
body = {"model": a.model, "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": userf([item])}],
        "max_tokens": a.max_tokens, "temperature": 0.0, **json.loads(a.extra)}
t0 = time.time()
try:
    r = requests.post(J.API, headers={"Authorization": f"Bearer {J.KEY}"}, json=body, timeout=a.timeout)
    d = r.json()
except Exception as exc:
    print(f"{a.model} {a.task} EXC after {time.time() - t0:.0f}s: {type(exc).__name__}")
    sys.exit(1)
choice = (d.get("choices") or [{}])[0]
content = choice.get("message", {}).get("content") or ""
u = d.get("usage", {})
print(f"{a.model} {a.task} HTTP {r.status_code} elapsed {time.time() - t0:.0f}s in={u.get('prompt_tokens')} "
      f"out={u.get('completion_tokens')} finish={choice.get('finish_reason')} parsed={J.parse(content, 1) is not None} "
      f"error={d.get('error')}")
