"""bge-m3 dense embeddings on CPU via transformers (no sentence-transformers dep).

Model lives at <repo>/models/bge-m3 (see scripts/download_model.py).
Dense retrieval uses the CLS token, L2-normalized (bge-m3 convention).
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models", "bge-m3")

MAX_LEN = 1024
_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(max(1, (os.cpu_count() or 4)))
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModel.from_pretrained(MODEL_DIR)
        _model.eval()
    return _tokenizer, _model


def embed(texts, batch_size=8, show_progress=False):
    """Embed a list of strings -> (N, dim) float32 array of unit vectors."""
    import torch

    tok, model = _load()
    out = []
    rng = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm

        rng = tqdm(rng, desc="embed", unit="batch")
    with torch.no_grad():
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
            cls = hidden[:, 0]  # bge-m3 dense retrieval: CLS pooling
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            out.append(cls.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def embed_one(text):
    return embed([text])[0]
