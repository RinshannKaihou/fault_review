#!/usr/bin/env python3
"""Download the retrieval models into the project-local models/ directory.

bge-m3                first stage: dense query/entry vectors (required)
bge-reranker-v2-m3    second stage: cross-encoder reranking (search falls back to hybrid-only without it)

Pass model names to fetch only some of them, e.g. `download_model.py bge-reranker-v2-m3`.
"""
import os
import sys

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = {"bge-m3": "BAAI/bge-m3", "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3"}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(MODELS):
        path = snapshot_download(
            MODELS[name],
            local_dir=os.path.join(ROOT, "models", name),
            ignore_patterns=["*.onnx", "*.msgpack", "onnx/*", "*.h5", "*.tflite"],
        )
        size = sum(
            os.path.getsize(os.path.join(d, f))
            for d, _, files in os.walk(path)
            for f in files
        )
        print(f"{name}: {path} ({size / 1e9:.2f} GB)")
    sys.exit(0)
