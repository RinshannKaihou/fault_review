# AGENTS — fault-rag 使用约定

本仓库是**故障编目检索工具**（RAG），供任意 interactive agent CLI 通过 Bash 调用。
不是在维护编目本身——编目真源在 `/workspace/inference_error_review` 与
`/workspace/training_error_review`，本仓库对它们**严格只读**。检索与引用读的是
`catalogs/` 下的 vendored 快照（经人工审查后通常每周手动同步，可能滞后于真源；每日扫描不自动同步 RAG）。

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
| 人工确认发布已审查的编目更新 | 先在隔离副本运行 `scripts/sync_catalogs.sh` 与定向校验；无读者窗口在正式库重跑、核验后才提交 `catalogs/`。脚本非原子，失败不得发布 |

## search 要点

- 默认 hybrid（BM25 + bge-m3 稠密向量，RRF 融合）。中英混排查询都支持。
- 过滤：`--repo inference|training`、`--category NUM|cat1|KV1`、
  `--confidence verified|documented|speculative`、`--trigger yes|partial|no`、
  `--no-neg`（排除负结果块）、`--include-pointers`（显式包含指针行）、
  `--full`（返回条目全部字段）。
- pointer 默认不参与 search；精确 `lookup` 仍可读取并沿 links 找到真实条目。
- BM25 完全零命中时返回空结果，不要把空结果解释成“目录前几条最相关”。
- 精确查不到的合法 ID 不会模糊命中别的 ID；已退役的 `1.70 → 1.44`、`11.30 → 11.24` 需按编目勘误手动查保留 ID，当前没有自动跳转。
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
- `related` 只表示存在编目关联，`link_kind: var` 是来源作者更强的变体断言，不能据此声称机制已核实相同；`6.254` 的该关系仍待复核。
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
旧题通过不代表新增、退役或修订条目语义正确；每次发布须补本次变更的定向检查。

## 故障排查

- `index not built yet` → 先跑 `reindex`。
- 模型缺失 → `python3 scripts/download_model.py`（需能访问 huggingface.co）。
- 编目真源仓库搬家了 → 给 `scripts/sync_catalogs.sh` 设 `FAULT_RAG_SRC_INFERENCE` / `FAULT_RAG_SRC_TRAINING` 环境变量（或改脚本默认值）后重跑。
