"""ONNX backend: dispatch, lazy dense matrix, and parity with the torch backend.

Parity tests skip unless both model formats are downloaded and torch is importable
(the repo's default install). The tokenizers-vs-transformers preprocessing test is
the cheap guard for the torch-free path's riskiest assumption.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from fault_rag import embed as E  # noqa: E402
from fault_rag import index as I  # noqa: E402

TORCH_OK = importlib.util.find_spec("torch") is not None
ONNX_DENSE = os.path.exists(os.path.join(E.ROOT, "models", "bge-m3-onnx", "onnx", "model.onnx"))
ONNX_RERANK = os.path.exists(
    os.path.join(E.ROOT, "models", "bge-reranker-v2-m3-onnx", "onnx", "model.onnx"))


class BackendSelectionTests(unittest.TestCase):
    def _backend_with(self, value):
        old = os.environ.get("FAULT_RAG_BACKEND")
        if value is None:
            os.environ.pop("FAULT_RAG_BACKEND", None)
        else:
            os.environ["FAULT_RAG_BACKEND"] = value
        try:
            return E.backend()
        finally:
            if old is None:
                os.environ.pop("FAULT_RAG_BACKEND", None)
            else:
                os.environ["FAULT_RAG_BACKEND"] = old

    def test_forced_backends(self):
        self.assertEqual(self._backend_with("onnx"), "onnx")
        self.assertEqual(self._backend_with("torch"), "torch")

    def test_invalid_backend_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._backend_with("tensorflow")


class LazyMatrixTests(unittest.TestCase):
    """bm25/lookup/stats must work without embeddings.npy on disk."""

    def setUp(self):
        self._old = I.EMB_NPY
        I.EMB_NPY = os.path.join(I.DATA, "embeddings.npy.definitely-absent")
        I.reset()

    def tearDown(self):
        I.EMB_NPY = self._old
        I.reset()

    def test_bm25_search_and_lookup_work_without_vectors(self):
        from fault_rag import search as S

        rows = S.search("loss spike 训练", topk=3, mode="bm25")
        self.assertTrue(rows)
        self.assertTrue(S.lookup("TRN-0001"))

    def test_dense_search_without_vectors_explains_itself(self):
        from fault_rag import search as S

        with self.assertRaises(SystemExit) as ctx:
            S.search("loss spike", topk=3, mode="dense")
        self.assertIn("reindex", str(ctx.exception))


@unittest.skipUnless(TORCH_OK and ONNX_DENSE, "needs torch + models/bge-m3-onnx")
class DenseParityTests(unittest.TestCase):
    def test_vectors_match_torch_backend(self):
        from fault_rag import onnx_backend

        texts = ["vllm prefix caching gibberish tokens",
                 "训练时 loss spike 但没有 crash，checkpoint 恢复后正常",
                 "x" * 3000]   # forces truncation at MAX_LEN
        os.environ.pop("FAULT_RAG_BACKEND", None)
        v_torch = E.embed(texts)
        v_onnx = onnx_backend.embed(texts)
        self.assertEqual(v_torch.shape, v_onnx.shape)
        cos = np.sum(v_torch * v_onnx, axis=1)
        self.assertGreaterEqual(float(cos.min()), 0.99999)


@unittest.skipUnless(TORCH_OK and ONNX_RERANK, "needs torch + reranker onnx model")
class RerankParityTests(unittest.TestCase):
    def test_logits_match_torch_backend(self):
        from fault_rag import onnx_backend, rerank as R

        entries = I.get_index().entries[:8]
        q = "答案已经正确但模型认不出结束 token，持续重复直到 max_new_tokens"
        os.environ.pop("FAULT_RAG_BACKEND", None)
        s_torch = np.asarray(R.scores(q, entries))
        s_onnx = np.asarray(onnx_backend.rerank_scores(q, [R.rerank_text(e) for e in entries]))
        self.assertLessEqual(float(np.abs(s_torch - s_onnx).max()), 1e-4)
        self.assertTrue(np.array_equal(np.argsort(-s_torch), np.argsort(-s_onnx)))


@unittest.skipUnless(ONNX_RERANK, "needs models/bge-reranker-v2-m3-onnx")
class TokenizerPairParityTests(unittest.TestCase):
    """The torch-free reranker path must tokenize (query, doc) exactly like the
    transformers tokenizer used by the torch path, including only_second truncation."""

    def test_pair_ids_match_transformers(self):
        from tokenizers import Tokenizer

        from fault_rag.onnx_backend import _load_tokenizer, RERANK_DIR

        tok = _load_tokenizer(RERANK_DIR, 1024, strategy="only_second")
        try:
            from transformers import AutoTokenizer

            tok_hf = AutoTokenizer.from_pretrained(
                os.path.join(E.ROOT, "models", "bge-reranker-v2-m3"))
        except Exception:
            self.skipTest("transformers not importable (lite install)")

        q = "训练时 loss spike 但没有 crash"
        docs = ["短文档", ("很长的文档 " * 900) + "结尾"]
        enc = tok.encode_batch([(q, d) for d in docs])
        hf = tok_hf([q] * 2, docs, padding=True, truncation="only_second", max_length=1024)
        self.assertEqual([e.ids for e in enc], [list(x) for x in hf["input_ids"]])
        self.assertEqual([e.attention_mask for e in enc],
                         [list(x) for x in hf["attention_mask"]])
        self.assertLessEqual(len(enc[1].ids), 1024)


if __name__ == "__main__":
    unittest.main()
