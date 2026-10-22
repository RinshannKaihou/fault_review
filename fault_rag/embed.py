"""bge-m3 dense embeddings on CPU via transformers (no sentence-transformers dep).

Model lives at <repo>/models/bge-m3 (see scripts/download_model.py).
Dense retrieval uses the CLS token, L2-normalized (bge-m3 convention).
"""

import contextlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models", "bge-m3")

MAX_LEN = 1024
_model = None
_tokenizer = None


def threads():
    """Measured 2026-10-22 on the 2x48-core box: 192 threads made a one-query forward take
    ~5 s (thread start-up and cross-socket sync dwarf the math); 48 was the fastest setting."""
    return int(os.environ.get("FAULT_RAG_THREADS", min(48, os.cpu_count() or 4)))


def fast_autocast():
    """bf16 autocast where the CPU has native bf16 (AMX / AVX512-BF16), else a no-op.
    Used for per-query work only (query vector, reranking): it cuts the first forward from
    ~4.7 s to 0.1 s, and bf16 query vectors match fp32 at cosine >= 0.9999. Corpus vectors
    stay fp32 so reindex output is unchanged."""
    import torch

    if torch.ops.mkldnn._is_mkldnn_bf16_supported():
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _load():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads())
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModel.from_pretrained(MODEL_DIR)
        _model.eval()
    return _tokenizer, _model


def embed(texts, batch_size=8, show_progress=False, fast=False):
    """Embed a list of strings -> (N, dim) float32 array of unit vectors.
    fast=True runs under bf16 autocast (query side only, see fast_autocast)."""
    import torch

    tok, model = _load()
    out = []
    rng = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm

        rng = tqdm(rng, desc="embed", unit="batch")
    with torch.no_grad(), (fast_autocast() if fast else contextlib.nullcontext()):
        for i in rng:
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            hidden = model(**enc).last_hidden_state  # (B, T, H)
            cls = hidden[:, 0].float()  # bge-m3 dense retrieval: CLS pooling
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            out.append(cls.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def embed_one(text):
    return embed([text], fast=True)[0]
