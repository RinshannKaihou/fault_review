"""Second-stage reranking with a cross-encoder (bge-reranker-v2-m3, CPU).

The hybrid stage scores the query and each entry separately; a cross-encoder reads the two
together, which is what lets a colloquial symptom ("简单加法题答案总是差一") meet the entry's
wording ("确定性 nearby-integer 替换"). It re-scores the hybrid top POOL rows. The final order
is an RRF blend of the hybrid rank and the cross-encoder rank, not the cross-encoder order
alone: on its own it ranks entries that merely share a surface symptom ("乱码") above the
specific-mechanism match (indy-017 dropped out of the top 10).

Model lives at <repo>/models/bge-reranker-v2-m3 (torch) or models/bge-reranker-v2-m3-onnx
(see onnx_backend.py — same logits, torch-free). Without either, search falls back to
hybrid-only and says so on stderr. Evidence, pool size and the latency
trade-off: eval/ACCEPTANCE.md, 2026-10-22.
"""

import os
import sys

from .embed import ROOT, backend, fast_autocast, threads

MODEL_DIR = os.path.join(ROOT, "models", "bge-reranker-v2-m3")
MAX_LEN = 1024   # 512 measured slightly worse (held-out R@5 0.86 vs 0.88) for ~0.3 s saved
POOL = 30        # 20 lost one indy R@5 hit and one held-out R@10 hit; 50 adds ~1 s for MRR only

_model = None
_tokenizer = None
_warned = False


def available():
    if backend() == "onnx":
        from . import onnx_backend

        return onnx_backend.rerank_available()
    return os.path.exists(os.path.join(MODEL_DIR, "config.json"))


def warn_unavailable():
    global _warned
    if not _warned:
        which = "models/bge-reranker-v2-m3-onnx" if backend() == "onnx" \
            else "models/bge-reranker-v2-m3"
        print(f"fault-rag: reranker model missing ({which}); results are "
              "hybrid-only. Run scripts/download_model.py.", file=sys.stderr)
        _warned = True


def _load():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(threads())
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
    return _tokenizer, _model


def rerank_text(e):
    """What the cross-encoder reads: title and observed effect first (a cross-encoder does not
    suffer the vector drift that kept them out of embed_text), then the narrative."""
    f = e["fields"]
    if e["neg"]:
        return e["name"] + "\n" + f.get("content", "")
    return "\n".join(x for x in (f.get("title"), f.get("effect_detail"), e["embed_text"]) if x)


def scores(query, entries, batch_size=16):
    """Relevance logits, in input order."""
    texts = [rerank_text(e) for e in entries]
    if backend() == "onnx":
        from . import onnx_backend

        if not onnx_backend.rerank_available():
            raise SystemExit(
                "fault-rag: ONNX backend selected but models/bge-reranker-v2-m3-onnx "
                "is missing. Run scripts/download_model.py bge-reranker-v2-m3-onnx "
                "(or unset FAULT_RAG_BACKEND)."
            )
        return onnx_backend.rerank_scores(query, texts, batch_size=batch_size)
    import torch

    tok, model = _load()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # less padding per batch
    out = [0.0] * len(texts)
    with torch.no_grad(), fast_autocast():
        for s in range(0, len(order), batch_size):
            rows = order[s:s + batch_size]
            enc = tok([query] * len(rows), [texts[i] for i in rows], padding=True,
                      truncation="only_second", max_length=MAX_LEN, return_tensors="pt")
            for i, x in zip(rows, model(**enc).logits[:, 0].float().tolist()):
                out[i] = x
    return out
