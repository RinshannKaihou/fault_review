#!/usr/bin/env python3
"""Download BAAI/bge-m3 into the project-local models/ directory."""
import os
import sys

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST = os.path.join(ROOT, "models", "bge-m3")

if __name__ == "__main__":
    path = snapshot_download(
        "BAAI/bge-m3",
        local_dir=DEST,
        ignore_patterns=["*.onnx", "*.msgpack", "onnx/*", "*.h5", "*.tflite"],
    )
    print("downloaded to:", path)
    size = sum(
        os.path.getsize(os.path.join(d, f))
        for d, _, files in os.walk(path)
        for f in files
    )
    print(f"total size: {size / 1e9:.2f} GB")
    sys.exit(0)
