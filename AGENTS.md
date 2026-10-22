# AGENTS — fault-rag 使用约定

本仓库是**故障编目检索工具**（RAG），供任意 interactive agent CLI 通过 Bash 调用。
不是在维护编目本身——编目真源在 `/workspace/inference_error_review` 与
`/workspace/training_error_review`，本仓库对它们**严格只读**。检索与引用读的是
`catalogs/` 下的 vendored 快照：两个编目的结构化条目 `catalogs/<repo>/entries/*.yaml`（经人工审查后手动同步，可能滞后于真源；每日扫描不自动同步 RAG；快照对应的上游提交见 `catalogs/SOURCES.txt`）。

## 入口

```bash
/workspace/RAG/fault_review/fault-rag <cmd>     # 从任意 cwd 调用
```

stdout 一律是 JSON。出错信息在 stderr。

## 什么时候用哪个命令

| 你要做什么 | 命令 |
|---|---|
| 用症状/机制描述找相似故障模式 | `search "<描述>" --topk 5` |
| 已知编号或 snake_case 名，取完整条目 | `lookup INF-0044` / `lookup 1.44`（旧编号自动对应）/ `lookup sdc_multibit_warp_spray` |
| 顺覆盖关系/交叉指针找关联条目 | `related TRN-0005 --depth 1`（旧号 `HW.05` 亦可） |
| 看某个故障模式下的全部实例 | `search "<描述>" --pattern PAT-INF-0035`，或读模式文件 `catalog/<repo>/patterns/` |
| 人工确认发布已审查的编目更新 | 先在隔离副本运行 `scripts/sync_catalogs.sh` 与定向校验；无读者窗口在正式库重跑、核验后才提交 `catalogs/`。脚本非原子，失败不得发布 |

## search 要点

- 默认 hybrid：BM25 + bge-m3 稠密向量 RRF 融合，再用交叉编码器（bge-reranker-v2-m3）重排前 30 名，
  与混合名次再做一次 RRF。中英混排查询都支持。冷启动一次约 11 秒；`--no-rerank` 跳过重排（约 6 秒），
  只在需要快速粗查时用。`rerank_rank`（0 起）是重排名次，`null` 表示没经过重排（`--no-rerank`、
  `--mode bm25|dense`，或精排模型缺失——缺失时 stderr 会提示，输出的 `reranked` 为 false）。
- 过滤：`--repo inference|training`、`--category <原因子系统，如 KV缓存> | <旧分类 NUM|cat1|KV1>`、
  `--pattern PAT-INF-0035`（只看该故障模式这一族）、
  `--confidence verified|documented|speculative`、`--trigger yes|partial|no|not_target`、
  `--no-neg`（排除负结果块）、`--include-pointers`（显式包含指针行）、
  `--full`（返回条目全部字段）。
- pointer 默认不参与 search；精确 `lookup` 仍可读取并沿 links 找到真实条目。
- hybrid 的向量路会给每个合格条目排名，所以 hybrid **总有结果**，排在后面的不一定相关；
  `bm25_rank: null` 表示该条没有任何关键词命中、只靠向量相似进来，要读原文再判断。
  `--mode bm25` 下关键词零命中返回空列表，这不代表编目里没有相关故障，可换 hybrid 或换说法再查。
- 新编号 `INF-xxxx` / `TRN-xxxx` 不含分类语义；旧编号记在 `legacy_ids`，`lookup` 会对应到新编号，已合并的 `1.70`、`11.30` 也会对应到保留条目（`INF-0044`、`INF-1048`）。精确查不到的编号不会模糊命中别的编号。
- `status: pending_audit` 的条目是每日扫描刚写入、尚未经次日自动审计的，引用时注明。
- `pattern: PAT-INF-0035` 是条目所属的**故障模式**（同一底层问题的一族）。编号带编目命名空间，
  `PAT-INF-` 与 `PAT-TRN-` 不可混。模式的 `members` 由 `var` 关系派生，**不是**人工策划的分类；
  模式自身的 `title` / `shared_mechanism` 在上游 `catalog/patterns/<id>.yaml`，RAG 不投影它们。
  模式只表示「同族」，**不等于已核实同机制**——深度 >1 的树隐含 var 可传递的假设，
  且 199 个模式里 55 个的共享机制仍是 `待定`。引用时不要据模式声称两条机制相同。
- 每条结果带 `summary`（`现象：…｜原因：…｜修复：…`），先用它判断相关性再决定打开哪条——
  字面撞词的干扰项（如「贪心解码」撞上正则「贪心匹配」）看摘要就能排除。摘要只用于展示，不参与排序。
  `修复：未记录` 只表示**编目没写**，不表示上游没修：2026-10-19 抽样 80 条，来源 issue 有 34%
  上游已有合入的关联 PR 而条目未记录。需要修复信息时去条目 `evidence` 里的链接确认，不要据此断言「无修复」。
- 每条结果带 `file`（条目 YAML）：需要完整上下文时直接用 Read 打开它，
  `original` 下是扫描写的完整原始说明；`title` 和 `cause` / `effect` 等结构化字段是从它抽取的摘要，不要只凭摘要下结论。
- 结果里 `neg: true` 的是**负结果**（rejected/blindspots/附录C/D）——
  它告诉你「这条路被否过」，不要当成可复用故障模式引用。

## 回答用户时的引用纪律

- 引用条目必须带 **编号 + snake_case 名 + file**，例如
  `TRN-0005 training_sdc_checkpoint_inherit`（旧号 HW.05；catalogs/training/entries/TRN-0005.yaml）。
- 置信度照实标注（`verified` / `documented` / `speculative`），不要把
  `documented` 说成已复现。
- 跨仓库关联用 `related` 查，不要凭记忆猜指针。
- `related` 只表示存在编目关联，`link_kind: var` 是来源作者更强的变体断言，不能据此声称机制已核实相同；`INF-0647`（旧号 6.254）的该关系仍待复核。
- `entry_type: pointer` 不是独立故障，不得作为故障证据引用。
- `name_ambiguous: true` 时必须使用 repo + ID + name + file；不能只写
  snake name。当前同名的 `INF-0421` / `INF-0426`（旧号 6.28 / 6.33）是两条不同来源记录。
- `name_ambiguous` 只检查**同一编目内**的重名。跨库同名不会被标记，例如
  `training_sdc_checkpoint_inherit` 同时是 `INF-0006` 与 `TRN-0005`（同一机制的推理/训练两个视角），
  按名称 `lookup` 会返回两条。所以任何时候都要写明 repo + ID。

## 本仓库不做什么

- 不写进两个编目仓库（连 `data/` 也不放那边）；`catalogs/` 只是它们的只读快照，不在此手工编辑。
- 不缓存查询结果当知识——条目以 reindex 后的 `data/entries.jsonl` 为准。
- 不把 `speculative` 条目当检测结论；检索到负结果要如实转告。

## 发布前检索质量门禁

```bash
cd /workspace/RAG/fault_review
python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json
python3 eval/run_eval.py --cases eval/indy_cases.jsonl --check eval/baseline_indy.json --per-case
```

两条命令分别检查 36 条人工核验 query 和 63 条独立盲测 query 的 Recall@5、Recall@10、MRR@10、nDCG@10 及 negative
slice。**bench 用聚合口径**：R@5/R@10 不得低于基线，MRR/nDCG 容差 0.02。

**indy 用逐条口径**（`--per-case`）：基线通过的 query 必须继续通过 R@5 与 R@10，任何一条掉出即 block；
排名在通过集内上下浮动不 block，但会写进报告的 `churn` 字段，每次发布需人工看一遍 `slipped` 列表。
两种口径都不看"聚合少 0.016"，因为 63 条上一条就是 1.6pp，聚合零容差在批量扩容下结构性不可通过
（2026-10-21 推理库 +46% 语料时实测：仅 2 条真失守，聚合却报 4 项回退）。

`eval/fix_list.json` 是逐条门禁的豁免名单，每条必须写 `kind`（缺了评测直接报错），只有两种：

- `weak_gold`：**基线时**单路检索（纯 BM25 / 纯 dense，按该题自己的过滤条件测）都进不了前 20，
  它的基线排名是 RRF 的偶然。必须附单路诊断证据。
- `accepted_regression`：新语料下**真的变差**、经用户确认接受的退化。必须写明前后单路名次、原因和移出条件，
  发布说明里点名。

禁止把真实退化登记成 `weak_gold`。判定要在基线索引上测（按 `baseline_indy.json` 生成时的快照重建，
数值须与基线完全一致），不能只比两个新快照。

标签来自源条目（用旧编号书写，评测时自动对应到新编号），不得按当前 top-k 反向生成。
旧题通过不代表新增、退役或修订条目语义正确；每次发布须补本次变更的定向检查。


## 故障排查

- `index not built yet` → 先跑 `reindex`。
- 模型缺失 → `python3 scripts/download_model.py`（bge-m3 与精排模型，可只给一个名字）；访问不了 huggingface.co 时先
  `export HF_ENDPOINT=https://hf-mirror.com`。但若进程走了代理（如指向 mihomo 的 `HTTPS_PROXY=http://127.0.0.1:7890`），
  hf-mirror 会把请求 308 跳回 huggingface.co，hf 客户端随即报 "Local entry not found"。此时二选一（2026-10-22 均实测可用）：
  把 `hf-mirror.com` 加进 `NO_PROXY`/`no_proxy` 直连镜像；或去掉 `HF_ENDPOINT`，经代理直连 huggingface.co。
- `reindex` 报 `No module named 'yaml'` → `pip install -r requirements.txt`（读条目 YAML 需要 PyYAML）。
- 明知编号却查不到 → 对照 `catalogs/SOURCES.txt` 与 `stats`：上游当天新写的条目要等下次手动同步；`git pull` 后还要 `reindex`。
- 编目真源仓库搬家了 → 给 `scripts/sync_catalogs.sh` 设 `FAULT_RAG_SRC_INFERENCE_REPO` / `FAULT_RAG_SRC_TRAINING_REPO`（仓库根目录）环境变量（或改脚本默认值）后重跑。
