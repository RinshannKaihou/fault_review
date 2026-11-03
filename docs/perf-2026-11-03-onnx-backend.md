# fault-rag 安装体积与检索性能：基线 vs 候选（2026-11-03）

任务：在**不改变检索质量与 CLI 行为**的前提下优化安装体积与检索性能。本文是基线对
候选的完整测量记录；结论摘要见 `eval/ACCEPTANCE.md` 2026-11-03 节。

## 环境

| 项 | 值 |
|---|---|
| 硬件 | Apple M5 Max，18 核，128 GB（本机测量；**上游 2×48 核 8 卡机未测**） |
| 系统 / Python | macOS 25.5 (darwin arm64) / 3.11.4（framework build） |
| 仓库提交 | 起测 `06656be`，中途快进到 `f6bb540`（catalog sync，见下），全部最终数字基于 `f6bb540` |
| 语料 | inference `1034f13`（2372 条）+ training `b6addcb`（606 条），索引 2994 行（+340 vs 基线初测） |
| 关键版本 | torch 2.15.0、transformers 5.22.1、onnxruntime 1.31.1、tokenizers 0.23.3、numpy 2.4.6、jieba 0.42.1、bm25s 0.3.16、fastembed 0.9.1 |
| 模型 | BAAI/bge-m3、BAAI/bge-reranker-v2-m3（各 2.1 GB）；onnx-community fp32 导出（各 ~2.2 GB） |

基线建立后仓库收到 catalog sync `f6bb540`（+340 条）。整合方式：`git pull --ff-only` +
增量 `reindex`（2m03s，复用 2647 条缓存向量、新嵌 347 条），全部门禁在 `f6bb540` 上重跑。

## 基线（torch 栈，`f6bb540`）

**安装**（全新 venv，`pip install -r requirements.txt`）：**4.8 GB，97 s**。
模型下载（脚本原样）：**4.2 GB，8m20s**。`reindex --full`（2994 条）：**11m25s**；
增量 reindex：2m03s。索引落盘 38 MB（entries.jsonl 22.5 / embeddings.npy 12.3 / bm25.pkl 8.5）。

**分项 profile**（单进程，冷→热）：

| 阶段 | 耗时 | 备注 |
|---|---|---|
| import torch + transformers | 0.91 s | torch 0.45 + transformers 0.46 |
| Index 加载（2994 行） | 0.14 s | |
| BM25 打分（手写） | ~1.1 ms/条（5 token 查询） | |
| bge-m3 加载 | 1.86 s | |
| 查询向量（热） | 73 ms | 本机无 AMX bf16 探测 op，纯 fp32 |
| 重排模型加载 | ~1.9 s | |
| **重排 30 对（热）** | **8.4–9.4 s** | **占查询总延迟 ~98%** |
| hybrid 全程（热） | ~8.3 s/条 | |
| 冷进程端到端（重排 / --no-rerank） | 12.86 s / 3.61 s | 三次中位 |

质量基线：bench 门禁 1.0/1.0/0.9861/0.9875 ✓；indy 逐条门禁 0.9683/0.9683/0.9053/0.9213 ✓
（churn：下滑 001 1→2、021 1→2、065 6→出前10 [weak_gold 豁免]；上升 014→1、017、042、058、059、066）。
pytest 65+13sub ✓；smoke 除"热单查询 <3s"外全过（该阈值按上游 AMX 机器校准，本机 fp32 前向 5–8s 达不到，非质量回归）。

**兼容性修复（基线前置）**：`fault_rag/embed.py:fast_autocast()` 调用的私有
`torch.ops.mkldnn._is_mkldnn_bf16_supported` 在 torch 2.15（仓库声明范围 `torch>=2.7,<3` 内）
已不存在，本机 search/重排直接 AttributeError。改为 getattr 探测 + fp32 回退；
有该 op 的机器行为不变。

## 候选评估

### A. ONNX Runtime fp32 后端 —— **采纳**（可选后端，默认 auto 时无 torch 则启用）

实现：`fault_rag/onnx_backend.py`（onnxruntime + tokenizers，无 torch/transformers），
`FAULT_RAG_BACKEND=auto|torch|onnx` 选择；`scripts/download_model.py` 增名
`bge-m3-onnx` / `bge-reranker-v2-m3-onnx`。预处理逐项复刻（pad_id 取 tokenizer_config
的 `<pad>`；配对编码 `strategy="only_second"`、max_length 1024；CLS 池化 + L2 归一）。

等价证据（逐位）：

| 层 | 测量 | 结果 |
|---|---|---|
| 预处理 | input_ids/attention/type_ids 对 11 条探针（含长文截断） | 完全一致 |
| 稠密向量 | 100 语料 + 30 查询 | 余弦 min/mean = 1.000000/1.000000 |
| 稠密排名 | 全语料 2994 条 ONNX 重嵌 + 150 题查询 | top-10 与 top-100 重叠率均 **1.0** |
| 重排分数 | 12 题 × 30 对真实 rerank_text | logit 最大差 **0.0**，0/12 池换序 |
| 端到端 | bench + indy 两道门禁 | 指标与 torch **一字不差**，churn 相同 |

→ **切换后端不失效缓存向量**；`meta.json` 新增 `embed_backend` 仅作记录。

性能/体积（三次中位）：

| 场景 | torch 基线 | ONNX | 提升 |
|---|---|---|---|
| 冷进程 hybrid+重排 | 12.86 s | **7.78 s** | 1.65× |
| 冷进程 `--no-rerank` | 3.61 s | **1.06 s** | 3.4× |
| 重排 30 对（热） | 9.4 s | 5.3 s | 1.75× |
| 全量嵌入（每条折算） | 0.258 s/条（2653 条，旧快照） | 0.231 s/条（2994 条，f6bb540） | ~1.1× |
| 依赖安装 | 4.8 GB / 97 s | **~480 MB / 10 s** | 10× / 10× |
| 轻量安装（无 torch）冷启动 | 不可安装 | 8.34 s | 新能力 |
| 轻量安装 `--mode bm25` | — | **0.26 s** | 新能力 |

ORT 线程 8 最优（18 线程慢 0.9 s；`FAULT_RAG_ORT_THREADS` 可调）；fp32 是唯一既快又等价的精度档。

### B. int8 / fp16 量化 —— **否决**（证据）

| 变体 | 结果 |
|---|---|
| 重排 int8 | 12/12 池分数换序，logit 最大差 0.84；省时仅 1.3 s |
| 重排 fp16 | 本机更慢（8.95 vs 7.18 s，ORT 内部回转 fp32）+ 4/12 换序 |
| 稠密 int8 | 向量余弦 0.973–0.986，与 2026-10-23 网页版结论（丢 1 条 indy）一致 |

上游用 bf16 autocast 跑重排是**同值近似**（余弦 ≥0.9999 且当时过两道门禁）；int8 是换序级别的扰动，不可比。

### C. fastembed 0.9.1 作为集成层 —— **否决**（三个独立缺陷）

1. `TextEmbedding` 不支持 bge-m3（0.3.6–0.9.1 全版本实测无）；
2. `TextCrossEncoder` fp32 路径在其缓存布局下加载外部权重报错（"External data path escapes model directory"）；
3. 其重排预处理按模型 config 8192 联合截断，非本仓库 1024/only_second。
→ 改为直接 onnxruntime + tokenizers（A 方案），预处理逐位对齐。

### D. BM25S 0.3.16 —— **等价已证，默认不换**

`bm25s.BM25(method="atire", idf_method="lucene", k1=1.5, b=0.75, dtype="float64")` +
预分词语料 + **查询词 set() 去重**（bm25s 把重复查询词计双倍，这是与手写实现的唯一语义差）
= 手写打分完全等价：150 题最大分差 2.8e-14，排名与正分集合零差异。
打分 25.1ms → 0.08ms（300×），索引 +3 MB、加载 50→15 ms。但单查询省 ~25 ms（<0.3% 全程），
不抵新增依赖；结论留档备用（若未来做高频热循环再启用）。

### E. 常驻进程 —— **否决**（当前收益/复杂度比）

ONNX 后端下可省的只有导入+模型加载（~2–3 s/查询），5.3 s 前向省不掉；daemon 的代价
（生命周期、reindex 后索引失效、并发、stdout JSON 契约改造）与 sync 管线的一次性 CLI 假设冲突。
若未来量化/更小模型把前向压到 ~1s，固定开销重新占主导，再评估。

## 落地清单

- `fault_rag/onnx_backend.py`（新）：ORT fp32 双模型，tokenizers 预处理
- `fault_rag/embed.py`：`backend()`/`use_onnx()` 分发 + torch 2.15 mkldnn 兼容修复
- `fault_rag/rerank.py`：按后端分发；`available()`/缺失提示兼容两种模型目录
- `fault_rag/index.py`：`Index.mat` 惰性加载（词法路径免 embeddings.npy）；
  `build(lexical=True)` / `reindex --lexical` 无模型构建词法索引（评审后补充）；
  `meta.json` 记录 `embed_backend`
- `scripts/download_model.py`：新增 `bge-m3-onnx` / `bge-reranker-v2-m3-onnx`
- `tests/test_onnx_backend.py`（新，7 项）：后端选择、惰性矩阵、稠密/重排等价（双模型齐备时）、
  tokenizers 配对一致性
- `requirements.txt`：torch-free 安装说明；`AGENTS.md`：推理后端一节

## 复现命令

```bash
# 基线（torch）
python3 -m pytest tests/ -q && python3 tests/smoke.py
python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json
python3 eval/run_eval.py --cases eval/indy_cases.jsonl --check eval/baseline_indy.json --per-case
# ONNX 后端全套（同一索引，不重建）
FAULT_RAG_BACKEND=onnx python3 -m pytest tests/ -q
FAULT_RAG_BACKEND=onnx python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json
FAULT_RAG_BACKEND=onnx python3 eval/run_eval.py --cases eval/indy_cases.jsonl --check eval/baseline_indy.json --per-case
# 冷启动延迟（三次取中位）
time FAULT_RAG_BACKEND=onnx python3 -m fault_rag search "<query>" --topk 5
time FAULT_RAG_BACKEND=onnx python3 -m fault_rag search "<query>" --topk 5 --no-rerank
```

## 建议

1. 默认 `auto`（torch 优先）保持——上游 2×48 核 AMX 机器上 torch 重排 ~2s 可能仍快于 ORT，
   且该机未测。笔记本/新环境建议显式 `FAULT_RAG_BACKEND=onnx` 或直接用 torch-free 安装。
2. 上游机器切换前，先跑上述两道门禁 + 等价性抽查（应逐位一致；不一致即模型导出变了，须重验）。
3. smoke 的 3000 ms 热延迟档建议改为按硬件标注（本机两后端都过不了，属校准差异）。
