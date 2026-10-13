# fault-rag

用命令行检索**推理故障**和**训练故障**编目，供人或交互式 Agent 通过 Bash 调用。输入症状、机制或已知编号，返回可追溯到快照原文的结果。默认结合 BM25 与 `BAAI/bge-m3` 向量检索，支持中英文混合查询。

## 快速开始

```bash
git clone https://github.com/RinshannKaihou/fault_review.git
cd fault_review

# 使用独立 Python 环境；已有满足 requirements.txt 的环境可跳过前两行
python3 -m venv "$HOME/fault-rag-venv"
source "$HOME/fault-rag-venv/bin/activate"
python -m pip install -r requirements.txt
python scripts/download_model.py       # 下载 bge-m3 到 models/bge-m3/
./fault-rag reindex --full              # 首次从 catalogs/ 构建本地索引
```

之后在仓库目录中直接查询；`fault-rag` 也可以用绝对路径从其他目录调用：

```bash
# 按症状/机制检索，默认混合 BM25 + 向量检索
./fault-rag search "loss spike 但无 NaN" --topk 5

# 限定训练库与分类
./fault-rag search "GRPO advantage std zero" --repo training --category RL训练回路   # 旧分类 RL-ADV 也可用

# 按编号或名称取完整条目
./fault-rag lookup INF-0044
./fault-rag lookup 1.44      # 旧编号自动对应到新编号 INF-0044
./fault-rag lookup HW.01     # → TRN-0001

# 查看编目里显式引用或提及的关联条目
./fault-rag related HW.05 --depth 1

# 查看当前本机索引统计与分类
./fault-rag stats
./fault-rag taxonomy
```

查询与统计命令返回 **JSON**；`reindex` 会另行打印构建进度与汇总。查询结果包含 `id`（新编号 `INF-xxxx` / `TRN-xxxx`）、`legacy_ids`（旧 Markdown 编号）、`name`、`title`、`repo`、`category`（原因子系统）、`legacy_category`（旧分类）、`confidence`、`status`、`entry_type`、`file` 等字段。引用时请打开结果中的 `file`（条目 YAML）核对原文，并同时给出库名和编号，不要只引用搜索摘要或条目名称。同名条目会带 `name_ambiguous` / `same_name_ids` 提示。

例如，当前快照运行 `./fault-rag search "topk single core ignores indices_tensor" --mode bm25 --repo inference --topk 2` 时，首条结果中的部分字段如下（**不是完整 JSON 输出**；排名可能随编目更新）：

```json
{
  "id": "INF-0044",
  "legacy_ids": ["1.44", "1.70"],
  "name": "ttmetal_topk_singlecore_ignores_indices_tensor",
  "repo": "inference",
  "category": "算子/内核",
  "legacy_category": "cat1",
  "entry_type": "fault",
  "neg": false
}
```

## 常用命令与选项

| 命令 | 用途 |
|---|---|
| `search "描述"` | 混合检索；`--topk 5` 控制条数，`--mode bm25\|dense\|hybrid` 选择检索方式 |
| `search ... --repo inference\|training` | 只检索指定编目；还可用 `--category`（原因子系统如 `KV缓存`，或旧分类如 `cat6`、`NUM`）、`--confidence`、`--trigger` 过滤 |
| `search ... --no-neg` | 排除拒录项、盲区与附录中的负结果；`--include-pointers` 显式包含指针条目；`--full` 返回完整字段 |
| `lookup <编号或名称>` | 精确查新编号、旧编号、名称或别名；名称支持部分匹配，**不存在的编号不会模糊匹配到其他编号** |
| `related <编号或名称> --depth 1` | 沿编目引用查关联条目；`link_kind` 显示关联类型，不代表已证实同一机制 |
| `stats` / `taxonomy` | 查看本机索引条数、类型和分类 |
| `reindex` / `reindex --full` | 从本仓库 `catalogs/` 重建本机索引；前者复用未变化的向量，后者全部重算 |

`neg: true` 是负结果，不是可以直接复用的故障案例；`entry_type: pointer` 是指向其他记录的说明，不是独立故障，默认不参加全文检索。精确 `lookup` 仍可读取指针。`related` 只是导航：其中的 `var` 是编目作者较强的“变体”断言，不能仅凭这条边断定两个故障机制相同。

旧编号都记在条目的 `legacy_ids` 里，`lookup` 会把旧编号对应到新编号，包括已合并的 `1.70`、`11.30`（分别到保留条目 `INF-0044`、`INF-1048`）；`lookup 11.70` 仍只命中它自己的条目。`status: pending_audit` 表示每日扫描刚写入、尚未经过次日自动审计的条目，引用时要注明。条目里的回访记录（`original.notes`）计入 BM25 检索。

## 数据从哪里来

| 本仓库文件 | 内容 |
|---|---|
| `catalogs/inference/entries/*.yaml`、`catalogs/training/entries/*.yaml` | 两个编目的结构化条目快照（每条一个 YAML，字段见上游 `catalog/SCHEMA.md`） |
| `catalogs/*/id_registry.tsv` | 新编号与旧 Markdown 编号的登记表 |
| `catalogs/inference/FAULT_MASTER_REFERENCE.zh.md` | 已冻结的推理 Markdown 编目，只用于附录 C/D 的负结果 |
| `catalogs/training/rejected.md`、`blindspots.md` | 拒录证据与盲区，作为负结果索引 |
| `catalogs/SOURCES.txt` | 本次快照对应的上游提交 |

编目真源分别位于 `/workspace/inference_error_review` 与 `/workspace/training_error_review` 的 `catalog/entries/`（2026-10-13 起由每日扫描通过 `tools/catalog_tool.py` 写入）。这里的 `catalogs/` 是**已提交的快照**，不一定与真源当天的内容相同；真源每日扫描或 Git 推送**不会自动同步 RAG**。本仓库的 `data/`（索引）与 `models/`（模型）不随 Git 提交。拉取新版快照后，需在本机运行 `./fault-rag reindex`，然后用 `./fault-rag stats` 检查当前索引；不要把 Git 更新等同于本机索引更新。

编目快照由维护者审查后手动发布，通常每周一次。`scripts/sync_catalogs.sh` 会**先复制快照、再重建索引、最后运行检索评测**，不是原子发布，也不会自行提交 Git；不要把它接在每日扫描后无人值守运行。详细审查与发布约定见 [AGENTS.md](AGENTS.md)。

## 检查与排错

```bash
python3 -m unittest discover -s tests -q
python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json
```

现有评测包含 36 条按来源核验的查询（指标含 Recall@5/@10、MRR@10、nDCG@10），标注里的旧编号会自动对应到新编号，用于发现既有检索退步；通过它**不代表**本次新增、修订或退役的条目都已得到语义验证。评测字段和门槛见 [eval/README.md](eval/README.md)。

- 提示 `index not built yet`：运行 `./fault-rag reindex`（读取本仓库 `catalogs/`）。
- 提示模型缺失：运行 `python scripts/download_model.py`；网络受限时可为 Hugging Face 下载配置可用镜像。
- 明知编号却查不到：先确认 `./fault-rag stats` 对应的快照已更新；再检查是否为已合并的旧编号。
