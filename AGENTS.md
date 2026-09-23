# AGENTS — fault-rag 使用约定

本仓库是**故障编目检索工具**（RAG），供任意 interactive agent CLI 通过 Bash 调用。
不是在维护编目本身——编目真源在 `/workspace/inference_error_review` 与
`/workspace/training_error_review`，本仓库对它们**严格只读**。检索与引用读的是
`catalogs/` 下的 vendored 快照（`scripts/sync_catalogs.sh` 定期同步，可能滞后于真源）。

## 入口

```bash
/workspace/RAG/fault_review/fault-rag <cmd>     # 从任意 cwd 调用
```

stdout 一律是 JSON。出错信息在 stderr。

## 什么时候用哪个命令

| 你要做什么 | 命令 |
|---|---|
| 用症状/机制描述找相似故障模式 | `search "<描述>" --topk 5` |
| 已知编号或 snake_case 名，取完整条目 | `lookup HW.01` / `lookup sdc_multibit_warp_spray` |
| 顺覆盖关系/交叉指针找关联条目 | `related HW.05 --depth 1` |
| 编目有更新（每日扫描合并后） | `scripts/sync_catalogs.sh`（拷快照 + reindex + 门禁），再 commit `catalogs/` |

## search 要点

- 默认 hybrid（BM25 + bge-m3 稠密向量，RRF 融合）。中英混排查询都支持。
- 过滤：`--repo inference|training`、`--category NUM|cat1|KV1`、
  `--confidence verified|documented|speculative`、`--trigger yes|partial|no`、
  `--no-neg`（排除负结果块）、`--include-pointers`（显式包含指针行）、
  `--full`（返回条目全部字段）。
- pointer 默认不参与 search；精确 `lookup` 仍可读取并沿 links 找到真实条目。
- BM25 完全零命中时返回空结果，不要把空结果解释成“目录前几条最相关”。
- 每条结果带 `file` 和 `line`：需要完整上下文时直接用 Read 打开源文件该行，
  不要只凭 snippet 下结论。
- 结果里 `neg: true` 的是**负结果**（rejected/blindspots/附录C/D）——
  它告诉你「这条路被否过」，不要当成可复用故障模式引用。

## 回答用户时的引用纪律

- 引用条目必须带 **编号 + snake_case 名 + file:line**，例如
  `HW.05 training_sdc_checkpoint_inherit`（catalogs/training/FAULT_MASTER_REFERENCE.zh.md:140）。
- 置信度照实标注（`verified` / `documented` / `speculative`），不要把
  `documented` 说成已复现。
- 跨仓库关联用 `related` 查，不要凭记忆猜指针。
- `entry_type: pointer` 不是独立故障，不得作为故障证据引用。
- `name_ambiguous: true` 时必须使用 repo + ID + name + file:line；不能只写
  snake name。当前同名的 `6.28` / `6.33` 是两条不同来源记录。

## 本仓库不做什么

- 不写进两个编目仓库（连 `data/` 也不放那边）；`catalogs/` 只是它们的只读快照，不在此手工编辑。
- 不缓存查询结果当知识——条目以 reindex 后的 `data/entries.jsonl` 为准。
- 不把 `speculative` 条目当检测结论；检索到负结果要如实转告。

## 发布前检索质量门禁

```bash
cd /workspace/RAG/fault_review
python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json
```

该命令检查 36 条人工核验 query 的 Recall@5、MRR@10、nDCG@10 及 negative
slice；标签来自源条目，不得按当前 top-k 反向生成。

## 故障排查

- `index not built yet` → 先跑 `reindex`。
- 模型缺失 → `https_proxy=http://127.0.0.1:7890 python3 scripts/download_model.py`。
- 编目真源仓库搬家了 → 给 `scripts/sync_catalogs.sh` 设 `FAULT_RAG_SRC_INFERENCE` / `FAULT_RAG_SRC_TRAINING` 环境变量（或改脚本默认值）后重跑。
