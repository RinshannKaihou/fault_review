# AGENTS

任意 interactive agent CLI 的入口。先读本文件，再动编目。

## 角色

你在维护一份**训练故障编目**，不是在写综述散文，也不是在实现训练代码。

当前阶段是**持续编目与每日扫描**：种子编目已建立，每日扫描已启用；只把可核实的训练期故障收成可检索条目。

## 必读顺序

1. `taxonomy.md` — 收什么、不收什么、编号与分类怎么理解
2. `catalog/SCHEMA.md` 和 `catalog/vocab.yaml` — 条目字段、写法、受控词表与判定规则
3. `catalog/rejected.md`、`catalog/blindspots.md` — 避免重复调研
4. `python3 tools/catalog_tool.py stats` / `find` — 当前条数与去重（不要整个编目塞进上下文）
5. 任务相关的 `foundation/notes/*.md`

## 真源

- **编目真源**是 `catalog/entries/*.yaml`（每条一个文件），只能用 `tools/catalog_tool.py` 写入和修改；字段与写法见 `catalog/SCHEMA.md`，词表见 `catalog/vocab.yaml`。
- `views/`（CATALOG.zh.md、fault_catalog.csv/.xlsx）由 `catalog_tool.py build` 生成，禁止手改。
- `catalog/FAULT_MASTER_REFERENCE.zh.md`、`exports/`、`catalog/index/` 是切换前的历史档案，不再更新。
- 禁止在根目录写 `daily_scan_*.md`、`.bak`、按日复制的查询脚本。

## 写入与合并纪律

1. 先 `python3 tools/catalog_tool.py find <owner/repo#号 | URL | arXiv ID | slug | 编号>` 查重。
2. 已有同机制 → `update <编号> --add-evidence <URL>` 只补来源，不占新号。
3. 同一机制、不同仓库 → 新条目写 `coverage: var`，`relations` 指向已有条目编号。
4. 新条目：写 JSON（格式见 `catalog/SCHEMA.md`）→ `python3 tools/catalog_tool.py new --file /tmp/<名>.json`；按返回的 `errors` 改到通过为止。**不要手写 YAML**（`#` 会被当成注释截断内容）。
5. 扫描时看到修复 PR / 规避方法，写进 `mitigation.fix`。
6. 写完运行 `catalog_tool.py build` 和 `catalog_tool.py validate`；validate 失败不得提交。

不要给未做实验的条目编造检测器预测。本地已跑过的注入/监控结论写进 `original.mechanism` 或 `update --note`。

## 收录边界（详见 taxonomy.md）

收：静默或延迟可见的正确性 / 稳定性故障。

不收进主编目：纯 OOM、NCCL timeout、启动期 assert、CI、文档、性能数字。这些可写在每日报告的「已扫描未编目」。

观测性骗过（job 绿、ckpt 已坏、监控空文件）进 `OBS` 或 `catalog/blindspots.md`。

## 禁止

- 把推理库条目整段复制过来；交叉条目只写指针
- 在本仓库实现 Megatron 注入 / 监控
- 用「感觉会这样」当来源
