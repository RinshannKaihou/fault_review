# fault-rag

推理 / 训练两个故障编目的命令行检索工具，以及把这两个编目迁移成结构化 YAML 时留下的工具与记录。

- **`fault-rag`**（主体）：输入症状、机制或编号，从两个编目里找相似故障。BM25 与 `BAAI/bge-m3` 向量混合检索，中英混排都能查；输出 JSON，供人或 agent 通过 Bash 调用。
- **`normalize/`**：2026-10 编目结构化迁移的流水线、网关模型审计、每日扫描切换手册，以及后续批量任务（如 NPU 历史存量回扫）的方案与脚本。

检索结果**只供参考**，用于提出假设、寻找相似先例，不能直接当结论。引用前要打开条目原文核对（见[引用纪律](#引用纪律)）。

## 数据流

编目的真源不在本仓库。两个上游仓库每天由扫描任务写入新条目，本仓库只读取它们的快照：

```text
上游真源   inference_error_review/catalog/entries/INF-*.yaml
          training_error_review/catalog/entries/TRN-*.yaml
   │      每日扫描写入（status: pending_audit），次日 02:00 UTC 网关审计后定稿
   ▼
scripts/sync_catalogs.sh   维护者审查后手动运行：复制快照 → reindex → 两道评测门禁
   ▼
catalogs/                  已提交的快照；catalogs/SOURCES.txt 记录对应的上游提交与条目数
   ▼      ./fault-rag reindex
data/、models/bge-m3/      本机索引与模型，不入 Git
   ▼
./fault-rag search | lookup | related
```

因此：上游当天新写的条目，要等下一次手动同步后才查得到；`git pull` 拿到新快照后，还要在本机运行 `reindex`，索引才会更新。

## 快速开始

```bash
git clone https://github.com/RinshannKaihou/fault_review.git
cd fault_review

python3 -m venv "$HOME/fault-rag-venv"      # 已有满足 requirements.txt 的环境可跳过这两行
source "$HOME/fault-rag-venv/bin/activate"
python -m pip install -r requirements.txt
python scripts/download_model.py             # 下载 bge-m3（约 2.2 GB）到 models/bge-m3/
./fault-rag reindex --full                   # 首次从 catalogs/ 构建索引
./fault-rag stats                            # 确认索引条数
```

无法直连 PyPI 或 Hugging Face 时，可用 `pip install -i <镜像地址>` 安装依赖，下载模型前设置 `export HF_ENDPOINT=https://hf-mirror.com`。检索只需 CPU。

## 查询

`fault-rag` 会自己切换到仓库根目录，所以从任意目录用绝对路径调用都可以。

```bash
# 按症状/机制检索（默认 hybrid，返回 8 条）
./fault-rag search "loss spike 但无 NaN" --topk 5

# 限定编目与分类；--category 接受原因子系统，也接受旧 Markdown 分类
./fault-rag search "GRPO advantage std zero" --repo training --category RL训练回路
./fault-rag search "GRPO advantage std zero" --repo training --category RL-ADV

# 按编号或名称取完整条目；旧编号自动换算，已合并的旧编号指向保留条目
./fault-rag lookup INF-0044
./fault-rag lookup 1.44        # → INF-0044
./fault-rag lookup 1.70        # 已并入 → INF-0044
./fault-rag lookup HW.01       # → TRN-0001
./fault-rag lookup training_sdc_checkpoint_inherit   # 两库同名，返回 INF-0006 与 TRN-0005 两条

# 沿编目里的关联走一层
./fault-rag related HW.05 --depth 1

# 索引统计与分类
./fault-rag stats
./fault-rag taxonomy
```

| 命令 | 用途与选项 |
|---|---|
| `search "<描述>"` | `--topk N`（默认 8）；`--mode hybrid\|dense\|bm25`（默认 hybrid） |
| | 过滤：`--repo inference\|training`；`--category <子系统或旧分类>`（如 `KV缓存`、`cat6`、`NUM`、`neg_rejected`）；`--confidence verified\|documented\|speculative`；`--trigger yes\|partial\|no\|not_target`（推理编目的触发标签） |
| | `--no-neg` 排除负结果；`--include-pointers` 让指针条目参加检索；`--full` 附带条目全部字段 |
| `lookup <编号或名称>` | 依次精确匹配新编号、旧编号、名称或别名；不是编号形式的查询再按名称子串匹配（最多 10 条）。**编号形式的查询查不到时返回空，不会模糊命中别的编号** |
| `related <编号或名称> --depth N` | 返回关联条目（不含自身），每条带 `link_kind` |
| `stats` / `taxonomy` | 索引按库、分类、置信度、类型、状态的计数，以及同名冲突；`taxonomy` 只列分类计数 |
| `reindex [--full]` | 从 `catalogs/` 重建索引；默认只为文本变化的条目重算向量，`--full` 全部重算 |

查询命令的 stdout 一律是 JSON，错误信息在 stderr；`reindex` 另外打印构建进度。

## 结果怎么读

`search` 的单条结果如下（节选自 `./fault-rag search "topk single core ignores indices_tensor" --mode bm25 --repo inference`；排名会随编目更新而变化）：

```json
{
  "rank": 1,
  "dense_rank": null,
  "bm25_rank": 0,
  "id": "INF-0044",
  "name": "ttmetal_topk_singlecore_ignores_indices_tensor",
  "title": "ttnn.topk 单核路径忽略传入的 indices_tensor",
  "legacy_ids": ["1.44", "1.70"],
  "repo": "inference",
  "category": "算子/内核",
  "legacy_category": "cat1",
  "confidence": "documented",
  "status": "active",
  "neg": false,
  "entry_type": "fault",
  "file": "/…/catalogs/inference/entries/INF-0044.yaml",
  "links": []
}
```

- `id`：新编号 `INF-xxxx` / `TRN-xxxx`，只表示分配顺序，不含分类含义；旧 Markdown 编号在 `legacy_ids` 里。
- `category`：原因子系统（条目里第一个 `cause.objects[].subsystem`）；`legacy_category`：旧 Markdown 分类。
- `dense_rank` / `bm25_rank`：两路检索各自的名次（从 0 起；`null` 表示这一路没有召回该条）。
- `file`：条目 YAML。主要字段有 `title`、`kind`、`status`、`cause`（原因对象与行为）、`effect`（影响对象、详情、可见性）、`mitigation`（检测与修复，多数仍为 `待定`）、`labels`（置信度、阶段等）、`evidence`（来源链接）、`relations`（`var` / `cross`）、`original`（扫描写下的原始叙述）、`migration`（迁移与审计记录）。结构化字段是从 `original` 抽取的摘要，下结论前请读 `original`。

### 引用纪律

- 写全 **库 + 编号 + 名称 + file**，例如：`TRN-0005 training_sdc_checkpoint_inherit`（旧号 HW.05；`catalogs/training/entries/TRN-0005.yaml`）。只写名称不够：`name_ambiguous: true` 表示同一编目内有同名条目，`same_name_ids` 列出全部同名编号；跨库同名（如上面的 `INF-0006` / `TRN-0005`）不会被标记。
- 置信度照实写：`documented` 不等于已复现，`speculative` 不能当结论。
- `status: pending_audit`：每日扫描刚写入、还没经过次日审计的条目，引用时要注明。
- `neg: true`：负结果，来自训练编目的 `rejected.md`、`blindspots.md` 和推理 Markdown 编目的附录 C/D。它表示“这条路被否过”，不是可复用的故障案例。
- `entry_type: pointer`：指向其他记录的说明，不是独立故障。默认不参加 `search`，`lookup` 仍能读到。
- `related` 只用于导航。`link_kind` 分三种：`var`（编目作者断言的变体，最强）、`cross`（显式交叉引用，常见于训推两库之间）、`mentions`（正文提到了对方编号）。即使是 `var`，也不能仅凭这条关联断定两个故障机制相同。

## 检索方式

- **BM25**：中文用 jieba 分词，代码与英文用正则切词。索引文本包括新旧编号、名称、新旧分类、标题，以及叙述字段（机制、行为、签名、触发面、复现、来源、回访记录）。
- **向量**：`bge-m3` 只编码叙述原文（名称、旧分类、机制、行为、签名、触发面）。
- **融合**：hybrid 模式用 RRF 合并两路名次。向量路会给每个合格条目排名，所以 hybrid 总有结果；`--mode bm25` 时如果没有任何关键词命中，会返回空列表。空列表不代表编目里没有相关故障，可以换 hybrid 或换个说法再查。

`cause`、`effect` 等结构化字段刻意**没有**加进向量文本，也没有整批加进 BM25。在 63 条独立集上实测：加进向量文本后，向量偏向机制措辞，离用户的症状描述更远（R@10 0.984 → 0.968）；加进 BM25 后，主题相近的条目反超正确条目（R@5 0.952 → 0.937）。详见 [eval/ACCEPTANCE.md](eval/ACCEPTANCE.md)，以及 `fault_rag/parse.py:embed_text`、`fault_rag/index.py:bm25_text` 的注释。

## 快照与同步

| 路径 | 内容 |
|---|---|
| `catalogs/{inference,training}/entries/*.yaml` | 两个编目的结构化条目，每条一个 YAML（字段规范见上游 `catalog/SCHEMA.md`） |
| `catalogs/*/id_registry.tsv` | 新旧编号登记表 |
| `catalogs/inference/FAULT_MASTER_REFERENCE.zh.md` | 已冻结的推理 Markdown 编目，只取附录 C/D 作负结果 |
| `catalogs/training/rejected.md`、`blindspots.md` | 训练编目的拒录证据与盲区，作负结果 |
| `catalogs/SOURCES.txt` | 本次快照对应的上游提交与条目数 |

快照由维护者手动发布：

1. 先在隔离副本里运行 `scripts/sync_catalogs.sh`，并对本次新增、修订、退役的条目做定向检查（评测门禁只能发现已有查询的退步，覆盖不到新条目）。
2. 确认没有人在使用时，在正式仓库重跑，核对无误后提交并推送 `catalogs/`。

`sync_catalogs.sh` 依次复制快照、重建索引、运行两道评测门禁。它**不是原子操作**：中途失败时，`catalogs/` 和索引可能已经改了。它也不会自己提交 Git，所以不要挂在每日扫描后面无人值守运行。上游仓库换了位置时，用 `FAULT_RAG_SRC_INFERENCE_REPO` / `FAULT_RAG_SRC_TRAINING_REPO` 指定新的仓库根目录。

## 评测门禁与测试

```bash
python3 -m unittest discover -s tests -q     # 单元测试
bash tests/smoke.sh                          # CLI 冒烟测试
python3 eval/run_eval.py --cases eval/cases.jsonl      --check eval/baseline.json
python3 eval/run_eval.py --cases eval/indy_cases.jsonl --check eval/baseline_indy.json
```

两个评测集：`cases.jsonl` 是 36 条按来源核验的查询；`indy_cases.jsonl` 是 63 条独立盲测查询，由只读过源条目的独立生成者编写，2026-10-14 起纳入门禁。门禁规则：Recall@5 / Recall@10 不得低于基线，MRR@10 / nDCG@10 最多下降 0.02，此外还有绝对下限和负结果分片门槛。标注里的旧编号会自动换算成新编号。

检索结果只供参考，用户关心的是正确条目能否出现在浏览窗口内，所以 Recall@5/@10 是硬指标，MRR 等排序指标只作参考。通过门禁**不代表**新增、修订或退役的条目语义正确。字段说明、门槛和基线更新流程见 [eval/README.md](eval/README.md)，历次验收记录见 [eval/ACCEPTANCE.md](eval/ACCEPTANCE.md)。

## normalize/：编目结构化工作区

2026-10，两个编目从自由格式 Markdown 改成每条一个 YAML 的 FMEA 式结构（原因对象 / 行为 → 影响对象 / 详情 → 改进措施），编号改为不含分类含义的新编号。迁移成品已部署到两个上游仓库，上游从 2026-10-13 起直接写 YAML。`normalize/` 现在保存这次迁移的工具与记录，以及后续批量任务的方案。

| 日期 | 里程碑 | 记录 |
|---|---|---|
| 10-12 | 1607 条 Markdown 条目转为 YAML：机械转换，Claude 抽取结构化字段，再由多个网关模型交叉审计（审计不用 Claude） | [normalize/README.md](normalize/README.md)、`run/REPORT.md`、`audit/REPORT.md` |
| 10-13 | 每日扫描写 YAML 的离线试点：glm-5.3 输出 JSON，脚本转成 YAML 并校验 | `scan_pilot/REPORT*.md` |
| 10-13 | 上游切换：Markdown 冻结，每日扫描通过 `tools/catalog_tool.py` 写 YAML，新增 02:00 UTC 每日审计 | [normalize/cutover/CUTOVER.md](normalize/cutover/CUTOVER.md)（含回滚材料） |
| 10-13 | 本仓库改读 YAML 快照，两道门禁的 Recall 均未下降 | [eval/ACCEPTANCE.md](eval/ACCEPTANCE.md) |

后续计划按顺序进行，截至 2026-10-15 尚未开工：修复悬空的 `var` 关联 → 模式/实例分层（合并近重复条目）→ NPU 历史存量回扫（[normalize/npu_backfill/PLAN.md](normalize/npu_backfill/PLAN.md)）→ 补全改进措施。

使用时注意：

- `normalize/SCHEMA.md`、`normalize/vocab.yaml` 是迁移时的版本，现行版本以上游的 `catalog/SCHEMA.md`、`catalog/vocab.yaml` 为准。
- `normalize/out/entries/` 是迁移当时的成品，之后不再更新；现行条目以上游和本仓库的 `catalogs/` 为准。
- `normalize/sync_upstream.py` 用于切换期间把上游 Markdown 里有、YAML 里还没有的条目补转到上游。
- `normalize/audit/votes/` 保存各模型的原始投票，作溯源材料。

## 目录结构

```text
fault-rag        CLI 入口（bash；切到仓库根目录后运行 python3 -m fault_rag）
fault_rag/       parse.py 读快照 → index.py 建 BM25 与向量索引 → search.py 检索；embed.py 封装 bge-m3
catalogs/        上游编目的已提交快照（只读，不手工编辑）
scripts/         sync_catalogs.sh（发布快照）、download_model.py（下载模型）
eval/            评测集、基线、run_eval.py、验收记录
tests/           单元测试与冒烟测试
normalize/       编目结构化迁移的工具与记录（见上）
AGENTS.md        给 agent 的调用与引用约定
data/  models/   本机索引与模型（已被 gitignore）
```

## 排错

- 提示 `index not built yet`：运行 `./fault-rag reindex`。
- 提示模型缺失：运行 `python scripts/download_model.py`；访问不了 Hugging Face 时先设置 `HF_ENDPOINT`。
- 明知编号却查不到：先对照 `catalogs/SOURCES.txt` 和 `./fault-rag stats`，确认快照和本机索引都是新的（上游当天的新条目要等下次同步）；再确认该编号是否已被合并（用 `lookup` 查旧编号会指向保留条目）。
- `git pull` 之后结果没变：本机索引还是旧的，运行 `./fault-rag reindex`。
