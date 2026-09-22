# fault-rag v1 稳定化实施计划

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 按既定顺序修复 `fault-rag` 的 category、BM25 零命中、pointer 条目、重复名称、检索评测和版本管理问题，使 v1 从“功能可用”进入“可回归验证、可追踪发布”的稳定状态。

**Architecture:** 保持当前“只读解析两个权威编目 → 派生本地索引 → CLI JSON 输出”的架构，不修改两个源编目仓库。修复以 TDD 为先：每一阶段先加入失败用例，再做最小实现，运行局部测试与完整 smoke；`data/` 和 `models/` 继续视为可再生且不进入版本控制。

**Tech Stack:** Python 3.10、stdlib `unittest`、NumPy、jieba、Transformers/BAAI bge-m3、Bash CLI、Git。

---

## 约束与当前基线

- 项目：`/workspace/RAG/fault_review`
- 权威源严格只读：
  - `/workspace/inference_error_review/FAULT_MASTER_REFERENCE.zh.md`
  - `/workspace/training_error_review/catalog/FAULT_MASTER_REFERENCE.zh.md`
  - `/workspace/training_error_review/catalog/rejected.md`
  - `/workspace/training_error_review/catalog/blindspots.md`
- 当前索引：1402 条、embedding `(1402, 1024)`、14 个 negative chunks、1532 条 links。
- 当前完整性：embedding 全 finite、零向量为 0、现有 smoke 全通过。
- 已确认缺陷：
  1. `9.26`、`9.27`、`9.28` 被错误归入 `§3.8 ...`，`--category cat9` 漏召回。
  2. BM25 全零得分仍按语料顺序返回 `1.1`、`1.2`、`1.3`。
  3. `10.4 chat_template_faults` 是 pointer，却被当成普通正向故障，且 `confidence=null`。
  4. `6.28` 与 `6.33` 共用 snake name `vllm_prefix_cache_silent_noop_mamba_gdn_hybrid`。
  5. 只有 smoke query，没有人工标注检索评测集及 Recall/MRR/nDCG。
  6. 目录尚未初始化 Git。
- 用户要求严格按上述 1→6 顺序实施。
- 因 Git 初始化排在第 6 步，前五步不能分阶段 commit；第 6 步做一次经过完整验证的 initial commit。此后再恢复小步提交。

## 实施前只读基线记录

在修改前保存实际输出到实施日志（不要写入 `data/` 之外的派生大文件）：

```bash
cd /workspace/RAG/fault_review
./tests/smoke.sh
./fault-rag stats
./fault-rag search 'zzzxqv_nonexistent_token_94731' --mode bm25 --topk 3 --no-neg
./fault-rag search 'RoPE dynamic scaling wrong angles' --repo inference --category cat9 --topk 10 --mode bm25
./fault-rag lookup 10.4
./fault-rag lookup vllm_prefix_cache_silent_noop_mamba_gdn_hybrid
```

预期：现有 smoke 通过，但后四条命令分别重现零命中伪结果、cat9 漏召回、pointer 误分类、同名多 ID。

---

### Task 1: 修复 category 解析与 embedding 缓存失效规则

**Objective:** 让正式条目的 category 由稳定的 ID grammar 决定，而不是被任意 `##` 子标题覆盖；同时保证派生的 embedding 输入变化时仅重嵌受影响条目。

**Files:**
- Modify: `fault_rag/parse.py:31-70,105-180,344-378`
- Modify: `fault_rag/index.py:114-163`
- Create: `tests/test_parse.py`
- Modify: `tests/smoke.py:33-50,79-92`

**Step 1: 写 category 失败测试**

在 `tests/test_parse.py` 使用 stdlib `unittest`，构造临时 inference master：顶层为 `## 第 9 类`，随后出现 `## §3.8 ...`，其下放置 `### 9.26`。断言：

```python
self.assertEqual(entries[0]["category"], "cat9")
self.assertEqual(entries[0]["section"], "§3.8 输入层故障（tokenizer / 模板 / RoPE）")
```

另加入 ID grammar 表驱动测试：

```python
cases = {
    ("inference", "9.26"): "cat9",
    ("inference", "K9.14"): "KV9",
    ("training", "RL-ADV.02"): "RL-ADV",
    ("training", "HW.01"): "HW",
}
```

`section` 仍保留最近的标题，仅 `category` 改为从 ID 推导。

**Step 2: 验证测试先失败**

```bash
python3 -m unittest tests.test_parse -v
```

Expected: `9.26` 当前得到截断的 `§3.8 ...`，测试失败。

**Step 3: 做最小 category 实现**

在 `fault_rag/parse.py` 增加单一 helper，例如：

```python
def _category_from_id(entry_id, repo):
    if repo == "inference":
        m = re.fullmatch(r"(\d+)\.\d+", entry_id)
        if m:
            return f"cat{m.group(1)}"
        m = re.fullmatch(r"K(\d+)\.\d+", entry_id)
        if m:
            return f"KV{m.group(1)}"
    else:
        m = re.fullmatch(r"([A-Z][A-Z0-9-]*)\.\d+", entry_id)
        if m:
            return m.group(1)
    raise ValueError(f"unsupported entry id: repo={repo!r} id={entry_id!r}")
```

创建 entry 时用该 helper 设置 `category`。不要 fallback 到截断 section；未知 ID 应显式失败，防止未来 parser drift 静默污染索引。

**Step 4: 加全语料一致性测试**

在 `tests/smoke.py` 增加：

- `9.26`、`9.27`、`9.28` 都是 `cat9`；
- 所有非 negative inference 数字 ID `N.x` 都是 `catN`；
- 所有非 negative inference `KN.x` 都是 `KVN`；
- training `PREFIX.x` 都是 `PREFIX`。

**Step 5: 修复 embedding cache fingerprint**

当前 `hash` 只覆盖 `raw`，但 `embed_text` 包含 category。若只改 category，旧向量会被错误复用。增加 `embed_hash = sha1(embed_text)`，并让 `index.py` 按 `embed_hash` 判断向量复用。

迁移必须避免无意义地重嵌 1402 条：加载旧 `entries.jsonl` 时，若旧记录没有 `embed_hash`，从旧记录已保存的 `embed_text` 现场计算旧 fingerprint。这样理论上仅 `9.26`～`9.28` 的 embedding 输入改变。

建议 helper：

```python
def embedding_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
```

保留现有 `hash` 作为 raw-content hash，不混淆两种语义。

**Step 6: 验证局部与真实语料**

```bash
python3 -m unittest tests.test_parse -v
./fault-rag reindex
./fault-rag lookup 9.26
./fault-rag lookup 9.27
./fault-rag lookup 9.28
./fault-rag search 'RoPE dynamic scaling wrong angles' \
  --repo inference --category cat9 --topk 10 --mode bm25
./tests/smoke.sh
```

Expected:

- 三条 category 均为 `cat9`；
- `9.27`、`9.28` 不再被 `--category cat9` 排除；
- 首次迁移只重嵌 embedding 输入真实变化的条目（目标为 3，而非全量 1402）；
- 随后再次 `./fault-rag reindex` 显示 `n_reembedded: 0`。

---

### Task 2: 修复 BM25 零命中和 RRF 顺序偏置

**Objective:** BM25 只给正得分文档分配 rank；零命中时返回空集，hybrid 模式不再让零分 BM25 文档获得语料顺序红利。

**Files:**
- Modify: `fault_rag/search.py:31-70`
- Create: `tests/test_search.py`
- Modify: `tests/smoke.py:51-93`

**Step 1: 写失败测试**

在 `tests/test_search.py` 用 stdlib `unittest` 直接测试 ranking helper，避免加载 bge-m3：

```python
scores = np.array([0.0, 0.0, 0.0])
self.assertEqual(_positive_ranks(scores, [0, 1, 2]), {})

scores = np.array([0.0, 2.0, 1.0])
self.assertEqual(_positive_ranks(scores, [0, 1, 2]), {1: 0, 2: 1})
```

再加 integration smoke：不存在 token 的 `mode="bm25"` 应返回 `[]`。

**Step 2: 验证测试先失败**

```bash
python3 -m unittest tests.test_search -v
```

Expected: 当前 `_ranks` 会把零分文档全部排序，测试失败。

**Step 3: 最小实现**

- dense ranking 保持对全部 eligible 文档排序；余弦相似度可以为负，不应套 `>0` 阈值。
- BM25 使用单独 helper，只保留 `score > 0` 的 eligible rows。
- `fused` 候选集使用 `set(dense_r) | set(bm25_r)`，不要遍历所有 eligible 文档。
- `mode="bm25"` 且无正分时自然得到空结果。
- `mode="hybrid"` 时 dense 候选仍完整，BM25 零命中只是不贡献分数。

**Step 4: 增加顺序偏置回归测试**

构造两个 BM25 全零、dense rank 不同的合成案例，验证 hybrid 排序完全由 dense 决定，不受原始行号影响。

**Step 5: 验证**

```bash
python3 -m unittest tests.test_search -v
./fault-rag search 'zzzxqv_nonexistent_token_94731' --mode bm25 --topk 3 --no-neg
./tests/smoke.sh
```

Expected: 不存在 token 返回 `"results": []`；现有有效 query 仍通过。

---

### Task 3: 把 pointer/meta 条目与正式故障分离

**Objective:** 将 `10.4 chat_template_faults` 明确分类为 pointer，默认 search 不把它当故障返回，但 exact lookup 仍可访问并指向 `10.66`。

**Files:**
- Modify: `fault_rag/parse.py:73-102,116-181,184-288,367-378`
- Modify: `fault_rag/search.py:11-28,37-88,132-168`
- Modify: `fault_rag/cli.py:22-34,59-74`
- Modify: `tests/test_parse.py`
- Modify: `tests/test_search.py`
- Modify: `tests/smoke.py`
- Modify: `README.md:30-57,73-83`
- Modify: `AGENTS.md:15-47`

**Step 1: 写失败测试**

测试 `10.4`：

```python
entry = parse_or_lookup("10.4")
self.assertEqual(entry["entry_type"], "pointer")
self.assertFalse(entry["neg"])
self.assertIsNone(entry["confidence"])
```

并验证：

- 默认 `search("chat template faults")` 不返回 `10.4`；
- `lookup("10.4")` 仍返回该 pointer；
- `related("10.4")` 仍能看到 `10.66`；
- 正常 fault 和 negative chunks 仍可被 search。

**Step 2: 定义明确 schema**

每条记录增加：

```text
entry_type = fault | pointer | negative
```

保持 `neg` 字段兼容现有消费者：只有 `negative` 时 `neg=true`，pointer 不能伪装成 negative。

pointer 判定不要只看 `confidence is None`。使用严格组合条件：

- `coverage == "existing"`；
- `trigger == "n/a"`；
- confidence 不属于允许集合；
- raw/mechanism 含明确 pointer 声明（如“不是新故障”或 `Cited for completeness; not a new entry`）。

若普通 `fault` 缺 confidence，则完整性检查应失败，而不是静默接受。

**Step 3: 修改默认搜索语义**

- 默认 eligible 类型：`fault` + `negative`；
- 默认排除 `pointer`；
- exact `lookup` 不排除任何类型；
- 如确有全文检索 pointer 的需求，CLI 增加显式 `--include-pointers`，不要默认混入。

**Step 4: 扩展 stats，但保持兼容**

`stats()` 增加：

```json
"by_entry_type": {
  "fault": 1387,
  "pointer": 1,
  "negative": 14
}
```

现有 `total` 和源编目 grammar count 保持不变，避免把“源文件 header 数”和“可作为故障检索的条目数”混为一谈。

**Step 5: 文档化引用纪律**

README/AGENTS 明确：

- pointer 仅表示重定向/统计占位，不是独立故障；
- pointer 不得作为故障证据引用；
- exact lookup 可用于沿 links 找到真实条目。

**Step 6: 验证**

```bash
python3 -m unittest tests.test_parse tests.test_search -v
./fault-rag reindex
./fault-rag lookup 10.4
./fault-rag related 10.4
./fault-rag search 'chat template faults' --topk 10
./fault-rag stats
./tests/smoke.sh
```

Expected: `10.4` 为 pointer、默认 search 不出现、lookup/related 可访问，普通 fault 数为 1387、pointer 为 1、negative 为 14。

---

### Task 4: 显式处理重复 snake name

**Objective:** 不篡改权威源、不擅自合并不同 issue，但让名称歧义在 lookup/search/stats 中可见，防止 agent 把同名条目误认为唯一记录。

**Files:**
- Modify: `fault_rag/index.py:198-215`
- Modify: `fault_rag/search.py:71-88,91-103,132-168`
- Modify: `tests/test_search.py`
- Modify: `tests/smoke.py`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Step 1: 写失败测试**

对已知同名条目验证：

```text
vllm_prefix_cache_silent_noop_mamba_gdn_hybrid
→ inference:6.28
→ inference:6.33
```

期望 exact lookup 返回两条，且每条包含：

```json
"name_ambiguous": true,
"same_name_ids": ["6.28", "6.33"]
```

唯一名称应为：

```json
"name_ambiguous": false,
"same_name_ids": []
```

**Step 2: 建立 collision map**

Index 构建 `by_name` 后，生成仅包含多行名称的 `name_collisions`。key 必须包含 repo，避免跨仓库偶然同名被误判为同一命名空间：

```python
(repo, name) -> sorted(ids)
```

aliases 与 canonical names 分开审计，避免 alias 命中被错误标成 canonical name collision。

**Step 3: 输出歧义，不自动改名或合并**

- lookup 继续返回两条，保持事实完整；
- search result 与 `_full()` 增加歧义字段；
- `stats()` 增加 collision 计数和列表；
- 不在 RAG 仓库重写两个源条目的名称；上游是否合并由权威编目维护者决定。

**Step 4: 增加 agent 使用规则**

README/AGENTS 写明：若 `name_ambiguous=true`，引用必须使用 `repo + id + name + file:line`，不得只用 snake name。

**Step 5: 验证**

```bash
python3 -m unittest tests.test_search -v
./fault-rag lookup vllm_prefix_cache_silent_noop_mamba_gdn_hybrid
./fault-rag stats
./tests/smoke.sh
```

Expected: 精确返回 `6.28`、`6.33`，歧义字段正确；唯一名称不受影响。

---

### Task 5: 建立人工标注的检索质量 benchmark

**Objective:** 用可审计的 query→相关 ID 标注集量化 Recall@5、MRR@10、nDCG@10，并把检索退化从“主观感觉”变成可阻断的回归信号。

**Files:**
- Create: `eval/cases.jsonl`
- Create: `eval/run_eval.py`
- Create: `eval/README.md`
- Create: `eval/baseline.json`
- Create: `tests/test_eval_metrics.py`
- Modify: `tests/smoke.py`
- Modify: `README.md`

**Step 1: 先测试 metric 计算**

使用小型固定排名写 `tests/test_eval_metrics.py`，覆盖：

- 单相关文档命中 rank 1；
- 多相关文档只命中一部分；
- 完全未命中；
- duplicate ID 不得重复计分；
- negative entry 作为相关答案时正常计分。

用手算可核对的列表断言 Recall@K、MRR@K、DCG/nDCG。禁止 broad `try/except` 或自动忽略坏 case。

运行：

```bash
python3 -m unittest tests.test_eval_metrics -v
```

Expected: 在 metric 实现前失败，完成最小实现后通过。

**Step 2: 定义可审计 case schema**

每行至少包含：

```json
{
  "case_id": "training-grpo-adv-zero-001",
  "query": "GRPO advantage std zero 组内无对比",
  "mode": "hybrid",
  "filters": {"repo": "training"},
  "relevant": [{"repo": "training", "id": "RL-ADV.02"}],
  "tags": ["training", "zh-en", "rl"],
  "rationale": "query directly describes zero within-group advantage variance"
}
```

每个 relevant ID 必须先用 `lookup` 回读源条目核验；不能根据当前搜索输出反向造标签。

**Step 3: 建立首版平衡数据集**

至少 36 个 case，建议分层：

- 14 个 inference 正向故障；
- 10 个 training 正向故障；
- 4 个 negative/rejected/blindspot；
- 4 个中文或中英混排；
- 4 个 exact-ish code/config 症状；
- 其中若干 case 可交叉归入多个 tag，但总 case 数不得低于 36。

覆盖硬件 SDC、KV/prefix cache、RoPE、cudagraph、checkpoint、numerics、loss、RL、MoE、数据问题及负结果。不要只选当前 smoke 已经容易通过的 query。

**Step 4: 实现单进程 evaluator**

`eval/run_eval.py`：

- 模型只加载一次；
- 对每个 case 调用真实 `search()`；
- 输出 aggregate 及按 tag 分层的 Recall@5、MRR@10、nDCG@10；
- 输出失败 case 的 query、relevant IDs、top-10 IDs；
- stdout 为 JSON，诊断写 stderr；
- 非法 case、重复 `case_id`、不存在 relevant ID 必须直接失败。

**Step 5: 生成并人工审阅 baseline**

```bash
python3 eval/run_eval.py \
  --cases eval/cases.jsonl \
  --output eval/baseline.json
```

建议验收门槛（这是目标门槛，不是当前已测事实）：

- aggregate Recall@5 ≥ 0.85；
- aggregate MRR@10 ≥ 0.70；
- aggregate nDCG@10 ≥ 0.75；
- negative slice Recall@5 ≥ 0.75；
- 任一 future run 相对 checked-in baseline 的任一 aggregate 指标下降超过 0.02 时失败。

如果首次真实结果未达门槛，不得降低标签质量或删除难例；记录失败 case，单独优化检索后再建立 baseline。

**Step 6: smoke 与 full eval 分层**

- `tests/smoke.py` 只保留 6 个代表性 benchmark case，控制日常验证耗时；
- 完整 36+ case evaluator 用于发布前和检索算法改动后；
- README 写清两种命令和耗时预期。

**Step 7: 验证**

```bash
python3 -m unittest tests.test_eval_metrics -v
./tests/smoke.sh
python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json
```

Expected: metrics unit tests、smoke、完整质量门禁全部通过；输出包含 aggregate 与 slices，不依赖人工阅读日志判断成功。

---

### Task 6: 初始化 Git 并建立可追踪基线

**Objective:** 在前五项全部通过后初始化仓库，确保模型、索引、缓存、临时评测输出和秘密不会进入版本控制，并提交可复现的首个稳定基线。

**Files:**
- Modify: `.gitignore`
- Track: `fault-rag`, `fault_rag/`, `tests/`, `eval/`, `scripts/`, `README.md`, `AGENTS.md`, `requirements.txt`, `.hermes/plans/`
- Never track: `models/`, `data/`, Python caches、临时日志和临时 benchmark 输出

**Step 1: 扩充 `.gitignore`**

至少包含：

```gitignore
data/
models/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
*.log
/tmp/
eval/results/
```

保留 `eval/baseline.json` 可追踪；它是人工审阅后的质量基线，不是临时结果。

**Step 2: 初始化前做敏感信息与大文件预检**

```bash
cd /workspace/RAG/fault_review
find . -type f -size +20M -not -path './models/*' -not -path './data/*' -print
```

再检查常见 token/private-key 字样；任何命中必须人工确认，不得自动吞掉。确认 README/日志没有访问令牌、cookie 或私钥。

**Step 3: 初始化仓库**

```bash
git init -b main
```

**Step 4: dry-run 暂存检查**

```bash
git add -n .
git status --short --ignored
```

验收：

- `models/`、`data/` 显示 ignored；
- 没有 2.3 GB 模型或 `.npy/.pkl` 索引进入 staged 候选；
- 源编目仓库文件不在本仓库路径内；
- 只有项目源码、测试、评测标签、文档和计划会被追踪。

**Step 5: 最终全套验证**

```bash
python3 -m compileall -q fault_rag tests eval scripts
python3 -m unittest discover -s tests -p 'test_*.py' -v
./tests/smoke.sh
python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json
./fault-rag reindex
```

Expected:

- 全部 unit/smoke/eval 通过；
- 最后一次未改源的 reindex 为 `n_reembedded: 0`；
- 索引仍为 1402 行，矩阵仍为 `(1402, 1024)`，全部 finite，无零向量。

**Step 6: 创建 initial commit**

```bash
git add .
git status --short
git commit -m "feat: stabilize fault catalog RAG v1"
```

**Step 7: 回读验证 commit 内容**

```bash
git status --short --branch
git show --stat --oneline HEAD
git ls-files | grep -E '^(models|data)/' && exit 1 || true
```

Expected: working tree clean；`models/`、`data/` 无 tracked 文件；initial commit 包含源码、测试、benchmark、文档和本计划。

---

## 最终验收矩阵

| 要求 | 验收证据 |
|---|---|
| category 修复 | `9.26`～`9.28` 均为 `cat9`，带 `--category cat9` 可召回 |
| cache 正确失效 | category 改动只重嵌实际 `embed_text` 变化条目，下一次为 0 |
| BM25 零命中 | 不存在 token 的 BM25 查询返回空，不再返回 1.1/1.2/1.3 |
| hybrid 无顺序偏置 | BM25 全零时 hybrid 排序只由 dense 决定 |
| pointer 分离 | `10.4 entry_type=pointer`，默认 search 排除，lookup/related 保留 |
| 重名显式化 | 6.28/6.33 都返回且标记 `name_ambiguous=true` |
| 编目事实不被改写 | 四个源 Markdown 文件无写操作 |
| 质量可量化 | 36+ 人工标注 cases；Recall@5/MRR@10/nDCG@10 有 checked-in baseline |
| 回归门禁 | unit + smoke + full eval 全通过 |
| 可追踪 | Git `main` 有 initial commit，working tree clean |
| 大文件隔离 | `models/`、`data/` 完全 untracked/ignored |

## 风险与处理

1. **category 改变会改变 embedding 输入。** 必须先修 `embed_hash`，否则“索引字段已修、向量仍旧”会形成隐性不一致。
2. **pointer 不是 negative。** 不得把 `10.4` 塞入 `neg=true`；它是第三种 entry type。
3. **重复名称可能是上游有意保留的两个 issue。** RAG 只暴露歧义，不擅自合并、重命名或写回源编目。
4. **benchmark 容易自我验证。** relevant IDs 必须从源条目机制人工核验，不能按当前 top-k 结果生成标签。
5. **Git 排在最后。** 按用户指定顺序，前五步没有中间 commit；实施中应保留本计划和逐步测试输出，初始化后一次提交稳定基线。
6. **完整 eval 成本较高。** 单进程复用 bge-m3；smoke 与 full eval 分层，不为追求速度牺牲完整发布门禁。

## 明确不做

- 不修改两个源编目仓库。
- 不增加 cron。
- 不更换 embedding 模型。
- 不引入数据库、Web 服务或新框架。
- 不使用 broad `try/except`、自动 fallback 或静默跳过坏标签。
- 不把模型权重、embedding、BM25 pickle 提交 Git。
