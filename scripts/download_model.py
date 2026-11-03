#!/usr/bin/env python3
"""Download the retrieval models into the project-local models/ directory.

bge-m3                        first stage, torch backend: dense query/entry vectors
bge-reranker-v2-m3            second stage, torch backend: cross-encoder reranking
bge-m3-onnx                   ONNX Runtime backend (torch-free install), fp32
bge-reranker-v2-m3-onnx       ONNX Runtime backend, fp32

The torch models serve the default torch backend; a torch-free install runs fully
on the two *-onnx models (FAULT_RAG_BACKEND selects; without torch installed it
is chosen automatically). int8/quantized exports are deliberately not offered:
reranker int8 flipped score order in 12/12 measured pools (eval/ACCEPTANCE.md,
2026-11-03).

Pass model names to fetch only some of them, e.g. `download_model.py bge-reranker-v2-m3`.
"""
import os
import sys

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODELS = {
    "bge-m3": ("BAAI/bge-m3", None),
    "bge-reranker-v2-m3": ("BAAI/bge-reranker-v2-m3", None),
    "bge-m3-onnx": ("onnx-community/bge-m3-ONNX",
                    ["onnx/model.onnx", "onnx/model.onnx_data", "tokenizer.json",
                     "tokenizer_config.json", "special_tokens_map.json", "config.json"]),
    "bge-reranker-v2-m3-onnx": ("onnx-community/bge-reranker-v2-m3-ONNX",
                                ["onnx/model.onnx", "onnx/model.onnx_data",
                                 "tokenizer.json", "tokenizer_config.json",
                                 "special_tokens_map.json", "config.json"]),
}


def fetch(name):
    repo, allow = MODELS[name]
    if allow:                       # keep the fp32 export only, not every quant variant
        kwargs = {"allow_patterns": allow}
    else:
        kwargs = {"ignore_patterns": ["*.onnx", "*.msgpack", "onnx/*", "*.h5", "*.tflite"]}
    path = snapshot_download(repo, local_dir=os.path.join(ROOT, "models", name), **kwargs)
    size = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, files in os.walk(path)
        for f in files
    )
    print(f"{name}: {path} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    for name in sys.argv[1:] or list(MODELS):
        if name not in MODELS:
            raise SystemExit(f"unknown model {name!r}; choose from {sorted(MODELS)}")
        fetch(name)
    sys.exit(0)
