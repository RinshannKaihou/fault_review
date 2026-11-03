"""Optional ONNX Runtime inference backend: bge-m3 + bge-reranker-v2-m3 without torch.

Selected with FAULT_RAG_BACKEND=onnx (or automatically when torch is not installed,
see embed.backend()). Weight-identical to the torch backend: on 2026-11-03 (M5 Max,
onnxruntime 1.31.1, onnx-community fp32 exports, corpus rev f6bb540) the two agreed
to the last printed digit — dense query/corpus vectors cosine 1.000000 at 6 dp over a
130-text sample, and reranker logits max |Δ| = 0.0 with 0/12 rank flips over 12 eval
queries x 30 real pairs. Cached vectors therefore stay valid across backends; the
index meta records which backend produced them for diagnostics only.

Preprocessing replicates the torch path exactly, via `tokenizers` (no transformers):
  bge-m3   pad <pad>, truncation right at 1024, CLS pooling, L2 normalize
  reranker pair encoding with strategy='only_second' at 1024 (query never truncated)
input_ids were verified bit-equal to the transformers tokenizer on both models.

int8/quantized exports are deliberately not wired in: reranker int8 flipped the score
order in 12/12 pools (max |Δ| 0.84); see eval/ACCEPTANCE.md 2026-11-03 for numbers.

Model dirs: models/bge-m3-onnx, models/bge-reranker-v2-m3-onnx
(scripts/download_model.py bge-m3-onnx bge-reranker-v2-m3-onnx).
"""

import json
import os

import numpy as np

from .embed import MAX_LEN as EMBED_MAX_LEN, ROOT

DENSE_DIR = os.path.join(ROOT, "models", "bge-m3-onnx")
RERANK_DIR = os.path.join(ROOT, "models", "bge-reranker-v2-m3-onnx")
RERANK_MAX_LEN = 1024   # mirrors rerank.MAX_LEN

_embed_sess = None
_embed_tok = None
_rerank_sess = None
_rerank_tok = None


def threads():
    """ORT scaled worse past 8 threads on the 18-core M5 Max (30-pair rerank 7.2 s
    at 8 vs 8.3 s at 18), unlike torch which liked 48 on the 2x48-core upstream box."""
    return int(os.environ.get("FAULT_RAG_ORT_THREADS", min(8, os.cpu_count() or 4)))


def _load_tokenizer(model_dir, max_len, strategy="longest_first"):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    with open(os.path.join(model_dir, "tokenizer_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    pad_id = tok.token_to_id(cfg.get("pad_token") or "<pad>")
    tok.enable_truncation(max_length=max_len, strategy=strategy, direction="right")
    tok.enable_padding(pad_id=pad_id, pad_token=cfg.get("pad_token") or "<pad>")
    return tok


def _load_session(model_path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads()
    return ort.InferenceSession(model_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def dense_available():
    return os.path.exists(os.path.join(DENSE_DIR, "onnx", "model.onnx"))


def rerank_available():
    return os.path.exists(os.path.join(RERANK_DIR, "onnx", "model.onnx"))


def _load_embed():
    global _embed_sess, _embed_tok
    if _embed_sess is None:
        _embed_tok = _load_tokenizer(DENSE_DIR, EMBED_MAX_LEN)
        _embed_sess = _load_session(os.path.join(DENSE_DIR, "onnx", "model.onnx"))
    return _embed_sess, _embed_tok


def _feed(sess, encodings):
    names = {i.name for i in sess.get_inputs()}
    feed = {"input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64)}
    if "attention_mask" in names:
        feed["attention_mask"] = np.asarray([e.attention_mask for e in encodings],
                                            dtype=np.int64)
    if "token_type_ids" in names:
        feed["token_type_ids"] = np.asarray([e.type_ids for e in encodings],
                                            dtype=np.int64)
    return feed


def embed(texts, batch_size=8, show_progress=False, fast=False):
    """(N, dim) float32 unit vectors — same contract as embed.embed (fast is a no-op:
    this backend is always exact fp32)."""
    sess, tok = _load_embed()
    out = []
    rng = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm

        rng = tqdm(rng, desc="embed(ort)", unit="batch")
    for i in rng:
        enc = tok.encode_batch(list(texts[i:i + batch_size]))
        hidden = sess.run(None, _feed(sess, enc))[0]      # (B, T, H)
        cls = hidden[:, 0, :].astype(np.float32)          # CLS pooling, bge-m3 dense
        cls /= np.linalg.norm(cls, axis=1, keepdims=True)
        out.append(cls)
    return np.vstack(out).astype(np.float32)


def embed_one(text):
    return embed([text])[0]


def _load_rerank():
    global _rerank_sess, _rerank_tok
    if _rerank_sess is None:
        _rerank_tok = _load_tokenizer(RERANK_DIR, RERANK_MAX_LEN, strategy="only_second")
        _rerank_sess = _load_session(os.path.join(RERANK_DIR, "onnx", "model.onnx"))
    return _rerank_sess, _rerank_tok


def rerank_scores(query, texts, batch_size=16):
    """Cross-encoder logits in input order — same contract as rerank.scores.
    Batches are length-sorted like the torch path so padding matches."""
    sess, tok = _load_rerank()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [0.0] * len(texts)
    for s in range(0, len(order), batch_size):
        rows = order[s:s + batch_size]
        enc = tok.encode_batch([(query, texts[i]) for i in rows])
        logits = sess.run(None, _feed(sess, enc))[0]
        for i, x in zip(rows, logits[:, 0].tolist()):
            out[i] = x
    return out
