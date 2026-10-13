#!/usr/bin/env python3
"""Give every entry at least one reference: resolve "project #N" mentions to verified GitHub links,
otherwise add a text citation. Apply-only and idempotent; run after audit/audit_apply.py.

    python3 normalize/enrich_evidence.py            # rewrites out/entries/*.yaml, prints counts
    python3 normalize/enrich_evidence.py --dry-run

Only entries without any URL evidence are touched. A mention is resolved through an explicit
project-name -> repo table; an unnamed "PR #N" / "#N" inherits the nearest preceding named project
in the same source text (never the model-assigned stack). Every link is checked with the GitHub
API (`gh api repos/<repo>/issues/<n>`, which also covers PRs); results are cached in
evidence_cache.json so re-runs are offline and byte-identical.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert import dump_yaml  # noqa: E402

ENTRIES = os.path.join(HERE, "out", "entries")
CACHE = os.path.join(HERE, "evidence_cache.json")
REPOS = {
    "vllm": "vllm-project/vllm", "transformers": "huggingface/transformers", "sglang": "sgl-project/sglang",
    "llama.cpp": "ggml-org/llama.cpp", "tensorrt-llm": "NVIDIA/TensorRT-LLM", "trt-llm": "NVIDIA/TensorRT-LLM",
    "flash-attention": "Dao-AILab/flash-attention", "lmcache": "LMCache/LMCache", "exllamav2": "turboderp-org/exllamav2",
    "peft": "huggingface/peft", "tgi": "huggingface/text-generation-inference", "flashinfer": "flashinfer-ai/flashinfer",
    "mlx-lm": "ml-explore/mlx-lm", "pytorch": "pytorch/pytorch", "deepep": "deepseek-ai/DeepEP",
    "deepspeed": "deepspeedai/DeepSpeed", "megatron-lm": "NVIDIA/Megatron-LM", "megatron-deepspeed": "deepspeedai/Megatron-DeepSpeed",
    "axolotl": "axolotl-ai-cloud/axolotl", "unsloth": "unslothai/unsloth", "ctranslate2": "OpenNMT/CTranslate2",
    "chatglm2-6b": "THUDM/ChatGLM2-6B", "gpustack": "gpustack/gpustack", "mlx-vlm": "Blaizzy/mlx-vlm",
    "llama-cpp-python": "abetlen/llama-cpp-python", "llama.rn": "mybigday/llama.rn", "mooncake": "kvcache-ai/Mooncake",
    "dynamo": "ai-dynamo/dynamo", "openvino genai": "openvinotoolkit/openvino.genai", "aiter": "ROCm/aiter",
    "tt-metal": "tenstorrent/tt-metal", "triton": "triton-lang/triton", "trl": "huggingface/trl", "verl": "volcengine/verl",
    "openrlhf": "OpenRLHF/OpenRLHF", "ms-swift": "modelscope/ms-swift", "torchtitan": "pytorch/torchtitan",
    "xgrammar": "mlc-ai/xgrammar", "onnxruntime": "microsoft/onnxruntime",
}
NOT_GITHUB = ("hf forum", "discuss")       # forum / HF discussion numbers are not GitHub issues
REF_RE = re.compile(r"(?:([A-Za-z][\w.\-]*(?:\s+[A-Za-z][\w\-]*)?)\s+)?(?:PR\s*|issues?\s*|RFC\s*)?#(\d{3,6})")


def mentions(source):
    """[(repo, number, mention_text)] in order; unnamed refs inherit the last named project."""
    out, last = [], None
    for m in REF_RE.finditer(source):
        name = (m.group(1) or "").strip()
        key = re.sub(r"\s+(pr|rfc|issue|issues)$", "", name.lower())
        if any(key.startswith(x) for x in NOT_GITHUB):
            last = None
            continue
        if key in REPOS:
            last = REPOS[key]
        elif key not in ("", "pr", "issue", "issues"):
            last = None                       # an unknown project name breaks inheritance
            continue
        if last:
            out.append((last, int(m.group(2)), m.group(0).strip()))
    return out


def verify(repo, number, cache):
    key = f"{repo}#{number}"
    if key not in cache:
        r = subprocess.run(["gh", "api", f"repos/{repo}/issues/{number}", "--jq", "[.html_url, (.pull_request != null)]"],
                           capture_output=True, text=True, timeout=60)
        cache[key] = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip().startswith("[") else None
    return cache[key]


def citation(source):
    first = re.split(r"(?<=[。；;])\s*|\n", source.strip(), maxsplit=1)[0]
    return first[:200] if first else source.strip()[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    stats = {"touched": 0, "linked": 0, "links": 0, "citation": 0, "unverified_mentions": 0}
    for path in sorted(glob.glob(os.path.join(ENTRIES, "*.yaml"))):
        e = yaml.safe_load(open(path, encoding="utf-8"))
        ev = e["evidence"] if isinstance(e["evidence"], list) else []
        if any(str(x.get("url", "")).startswith("http") for x in ev):
            continue
        stats["touched"] += 1
        source = e["original"].get("source", "") or ""
        new, seen = [], set()
        for repo, number, text in mentions(source):
            hit = verify(repo, number, cache)
            if not hit:
                stats["unverified_mentions"] += 1
                continue
            url, is_pr = hit
            if url not in seen:
                seen.add(url)
                new.append({"url": url, "kind": "github_pr" if is_pr else "github_issue", "resolved_from": text})
        if new:
            stats["linked"] += 1
            stats["links"] += len(new)
        else:
            new = [{"kind": "citation", "text": citation(source)}]
            stats["citation"] += 1
        e["evidence"] = new
        if not a.dry_run:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(dump_yaml(e))
    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump(dict(sorted(cache.items())), fh, ensure_ascii=False, indent=0)
    print(stats)


if __name__ == "__main__":
    main()
