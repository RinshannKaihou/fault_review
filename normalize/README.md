# 故障编目结构化迁移（normalize/）

把推理 / 训练两个故障编目从自由格式 Markdown 迁移为每条一个 YAML 的结构化条目（FMEA 式：原因对象 / 原因行为 → 影响对象 / 影响详情 → 改进措施）。结构规范见 [SCHEMA.md](SCHEMA.md)，受控词表见 [vocab.yaml](vocab.yaml)（v1.2）。

**现状：迁移已完成并上线。** 2026-10-13 两个上游仓库切换为结构化真源（推理 `5ab1f1f`、训练 `df1115f`），Markdown 总参考随之冻结；此后每日扫描直接写 `catalog/entries/*.yaml`，次日 02:00 UTC 由网关审计定稿。切换过程与回滚材料见 [cutover/CUTOVER.md](cutover/CUTOVER.md)。

本目录现在只用于溯源和后续批量任务：

- `out/entries/*.yaml`（1607 条：`INF-*` 1342 条、`TRN-*` 265 条）和 `id_registry.tsv` 是 10-12 迁移时的成品，之后**不再更新**。现行条目和编号登记表在上游的 `catalog/entries/`、`catalog/id_registry.tsv`，RAG 读的是它们的快照 `../catalogs/`。
- `SCHEMA.md`、`vocab.yaml` 是迁移时的版本；如与上游 `catalog/SCHEMA.md`、`catalog/vocab.yaml` 有出入，以上游为准。
- 下表第 1–7 步读取的是上游已冻结的 Markdown，并会改写 `out/`；迁移已完成，不需要再重跑。

## 流水线

| 步骤 | 命令 | 产出 |
|---|---|---|
| 1. 机械转换 | `python3 convert.py` | 编号、标签、来源、关系、原文；编号由 `id_registry.tsv` 冻结 |
| 2. 试点 | `pilot/`（45 条，三路抽取） | 词表 v0→v1，`pilot/REPORT.md` |
| 3. 全量抽取（Claude） | `make_batches.py` + workflow → `run/out`、`run/judged` | `run/REPORT.md` |
| 4. 合并 | `python3 finalize.py` | 把抽取结果写入 `out/entries`（会覆盖第 6 步结果，需重跑第 6 步） |
| 5. 校验 | `python3 validate.py` | 枚举、编号、关系检查 |
| 6. 网关审计（不用 Claude） | `audit/`：`build_items.py` → `run_task.sh` / `audit_merge.py` → `finish_audit.sh` | `audit/REPORT.md`；`audit_apply.py` 从冻结基线 `audit/baseline_*.tar.gz` 只增量应用，可重复执行 |
| 7. 补来源 | `python3 enrich_evidence.py`（在第 6 步之后运行） | 无链接条目：“项目 #编号”→ 经 GitHub API 核实的链接（缓存在 `evidence_cache.json`），否则写文字引用 |
| 8. 每日扫描写入试点 | `scan_pilot/run_pilot.py`（build → run → repair → judge-items → evaluate） | `scan_pilot/REPORT*.md`：glm-5.3 输出 JSON + 一轮校验反馈后格式达标；分类接近 Claude，文本忠实性偏弱 → 切换时用辅助脚本校验回路 + 每日网关审计 |
| 9. 补转 | `python3 sync_upstream.py --repo inference\|training [--dry-run]` | 迁移后到切换前，每日扫描又往 Markdown 写了新条目；本步把这些条目经上游 `catalog_tool.py new` 写入（status `pending_audit`），再由上游 `tools/daily_audit.sh` 审计 |
| 10. 切换 | `cutover/`（手册、新说明文件与 cron 提示词、回滚材料） | 2026-10-13 执行，记录见 `cutover/CUTOVER.md` 末尾 |

## 后续批量任务

截至 2026-10-15 均未开工，按顺序进行：修复悬空的 `var` 关联 → 模式/实例分层（合并近重复条目）→ NPU 历史存量回扫（方案与预筛工具见 [npu_backfill/PLAN.md](npu_backfill/PLAN.md)）→ 补全改进措施。这些任务都写上游真源，新条目和改动同样走 `catalog_tool.py` 和每日审计。

## 审计协议

第 6 步的审计沿用 llm-judge-labeling 协议：
- 三个便宜网关模型投票，未达成一致的字段交给 4 个模型组成的 7 票评审团，至少 2/3 多数才算定论；
- 弱结论保留基线值，并在条目的 `migration.audit.uncertain` 中列出；
- 编排者不读原文和判定内容。

`audit/votes/` 保存每个模型的原始投票，作为溯源材料。
