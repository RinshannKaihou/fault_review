# AGENTS

任意 interactive agent CLI 的入口。先读本文件，再动编目。

## 角色

你在维护一份**训练故障编目**，不是在写综述散文，也不是在实现训练代码。

当前阶段是**持续编目与每日扫描**：种子编目已建立，每日扫描已启用；只把可核实的训练期故障收成可检索条目。

## 必读顺序

1. `taxonomy.md` — 收什么、不收什么、ID 怎么编
2. `catalog/FAULT_MASTER_REFERENCE.zh.md` 顶部滚动汇总 + 目录（不要整文件塞进上下文）
3. `catalog/rejected.md`、`catalog/blindspots.md` — 避免重复调研
4. `catalog/index/known_issue_ids.txt` 等 — 去重
5. 任务相关的 `foundation/notes/*.md`

## 真源

- **编目真源**是 `catalog/FAULT_MASTER_REFERENCE.zh.md`。
- `exports/*` 是单向导出。改 md，再跑 `python3 tools/export_faults.py`。
- 禁止手改 csv/xlsx 当主数据。
- 禁止在根目录写 `daily_scan_*.md`、`.bak`、按日复制的查询脚本。

## 条目最小字段

```
### <PREFIX>.<NN> `<snake_case_name>`

stage: `<pretrain|sft|rl|shared>`（可多值，用 `/`） · Cov: `<NEW|var:已有名>` · 置信度 `<verified|documented|speculative>`

- **机制**：技术层发生了什么
- **来源**：可点开的 URL / issue / 论文
- **行为效应**：训练侧看见什么（loss / grad / ckpt / reward），有没有 crash
- **发现来源**：YYYY-MM-DD 基础调研 | 每日扫描 | 本地实验
```

不要给未做实验的条目编造「族」或「检测器一定看得见」。本地已跑过的注入/监控可以写**观测信号**字段。

## 合并纪律

1. 先在 `catalog/index/` 和主文档里搜 issue 号、arxiv ID、snake_case 名。
2. 已有编号 → 只补来源，不占新号。
3. 同一机制、不同仓库 → `var:已有名`，挂在最近类目末尾。
4. 新条目追加到对应 `## PREFIX` 节末尾，编号 = 该类当前最大 + 1。不要插到中间。
5. 合并后跑 `python3 tools/extract_known_ids.py` 和 `python3 tools/export_faults.py`。
6. 滚动汇总按实际 `### PREFIX.` 标题重算，禁止硬编码旧数字。
7. 用编辑工具改主文档时，`old_string` 必须包含相邻 `###` 标题整行，防止吞标题。

## 收录边界（详见 taxonomy.md）

收：静默或延迟可见的正确性 / 稳定性故障。

不收进主编目：纯 OOM、NCCL timeout、启动期 assert、CI、文档、性能数字。这些可写在每日报告的「已扫描未编目」。

观测性骗过（job 绿、ckpt 已坏、监控空文件）进 `OBS` 或 `catalog/blindspots.md`。

## 禁止

- 把推理库条目整段复制过来；交叉条目只写指针
- 在本仓库实现 Megatron 注入 / 监控
- 用「感觉会这样」当来源
