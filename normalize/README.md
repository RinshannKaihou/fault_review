# 故障编目结构化迁移（normalize/）

把推理 / 训练两个故障编目从自由格式 Markdown 迁移为每条一个 YAML 的结构化条目（FMEA 式：原因对象 / 原因行为 → 影响对象 / 影响详情 → 改进措施）。结构规范见 [SCHEMA.md](SCHEMA.md)，受控词表见 [vocab.yaml](vocab.yaml)（v1.2）。

**成品**：`out/entries/*.yaml`，共 1607 条（`INF-*` 1342 条、`TRN-*` 265 条）。新旧编号对照见 `id_registry.tsv`。目前上游编目仓库尚未切换到这份数据。

## 流水线

| 步骤 | 命令 | 产出 |
|---|---|---|
| 1. 机械转换 | `python3 convert.py` | 编号、标签、来源、关系、原文；编号由 `id_registry.tsv` 冻结 |
| 2. 试点 | `pilot/`（45 条，三路抽取） | 词表 v0→v1，`pilot/REPORT.md` |
| 3. 全量抽取（Claude） | `make_batches.py` + workflow → `run/out`、`run/judged` | `run/REPORT.md` |
| 4. 合并 | `python3 finalize.py` | 把抽取结果写入 `out/entries`（会覆盖第 6 步结果，需重跑第 6 步） |
| 5. 校验 | `python3 validate.py` | 枚举、编号、关系检查 |
| 6. 网关审计（不用 Claude） | `audit/`：`build_items.py` → `run_task.sh` / `audit_merge.py` → `finish_audit.sh` | `audit/REPORT.md`；`audit_apply.py` 从冻结基线 `audit/baseline_*.tar.gz` 只增量应用，可重复执行 |

审计协议沿用 llm-judge-labeling：
- 三个便宜网关模型投票，未达成一致的字段交给 4 个模型组成的 7 票评审团，至少 2/3 多数才算定论；
- 弱结论保留基线值，并在条目的 `migration.audit.uncertain` 中列出；
- 编排者不读原文和判定内容。

`audit/votes/` 保存每个模型的原始投票，作为溯源材料。
