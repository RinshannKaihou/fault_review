# 切换手册：每日扫描改写结构化编目

前置：第 1 步（两个上游仓库已有 `catalog/entries`、`tools/catalog_tool.py`、`catalog_audit.py`、`daily_audit.sh`）与第 2 步（沙盒验证）已完成。

## 时间窗

在训练扫描（00:30 UTC，约 01:00 完成）之后、推理扫描（23:00 UTC）之前完成全部操作；两个仓库在操作前后都必须是干净且与 origin/main 一致。切换后第一次真实运行：当晚 23:00 推理扫描 → 次日 00:30 训练扫描 → 02:00 每日审计。

## 步骤

1. **补转**：`python3 normalize/sync_upstream.py --repo inference|training`（Markdown 里有、编目里还没有的条目 → pending_audit），再在各仓库运行 `bash tools/daily_audit.sh` 审计并提交。
2. **冻结 Markdown**：在两个 `FAULT_MASTER_REFERENCE.zh.md` 顶部加冻结声明；旧导出脚本与 `exports/`、根目录 `fault_catalog.*` 保留为历史档案，不再运行。
3. **替换说明文件**（草稿在本目录）：
   - 推理：`daily_scan.prompt.md` ← `inference/daily_scan.prompt.md`
   - 训练：`hermes/daily_scan.prompt.md`、`AGENTS.md`、`taxonomy.md` ← `training/`
   - 两个仓库各提交一次并推送，核对远程 SHA。
4. **改 cron 提示词**（文本见 `cron_prompts.md`）：
   - `hermes -p third cron edit 8645356b40ae --prompt "<推理新提示词>"`
   - `hermes -p third cron edit 1b8481fa4e2f --prompt "<训练新提示词>"`
   - 回读 `jobs.json` 核对文本已更新。
5. **新建每日审计任务**：
   - 安装 `hermes_scripts/fault_catalog_daily_audit.sh` → `~/.hermes/profiles/third/scripts/`（profile 自己的脚本目录，不是 `~/.hermes/scripts/`）
   - `hermes -p third cron create '0 2 * * *' --name '故障编目每日审计' --script fault_catalog_daily_audit.sh --no-agent --deliver feishu:<扫描任务同一会话>`
6. **监控**：当晚推理扫描、次日训练扫描与 02:00 审计后各检查一次：新条目都经 `catalog_tool new` 写入、validate 通过、推送成功、飞书收到完整摘要、审计已把 pending_audit 定稿。

## 回滚

旧版文件与旧 cron 提示词都在 `rollback/`。

- 说明文件：`git revert` 第 3 步的提交（或把 `rollback/` 下的文件拷回），推送。
- cron：`hermes -p third cron edit <id> --prompt "$(cat rollback/<id>.prompt.txt)"`；`hermes -p third cron pause <审计任务 id>`。
- 回滚后扫描恢复写 Markdown；切换期间写入的 YAML 条目保留，下次切换时不需要再补转。

## 已知差异与风险

- 沙盒用火山网关的 `zhipu/glm-5.3`（high）；正式扫描是 Z.ai 直连、任务级 `reasoning_effort: max`。行为应接近，头几晚需要盯。
- Z.ai 额度按周/月计，测试扫描不再使用 Z.ai。
- RAG 仓库 `catalogs/` 仍是 10-12 的 Markdown 快照；RAG 改读 YAML 是下一阶段的工作，在此之前 RAG 检索不会包含切换后的新条目。

## 执行记录（2026-10-13）

- 10:2x UTC 推理仓库 `5ab1f1f`、训练仓库 `df1115f`：冻结 Markdown、替换说明文件。
- 两个扫描任务的提示词已用 `hermes cron edit` 更新并回读核对；时间、模型（glm-5.3 / zai）、`reasoning_effort: max`、飞书投递未变。
- 新建审计任务 `1fc648dfb5a0`（`0 2 * * *`，no-agent，投递同一飞书会话）。
- 第一次试运行失败：网关下的 `python3` 是 hermes 自带的 3.14，缺 PyYAML / requests。`catalog_tool.py` 与 `catalog_audit.py` 已改为缺依赖时自动用 `/usr/bin/python3` 重新执行（推理 `6366fef`、训练 `12797e2`），第二次试运行通过。这个修复同样覆盖扫描 agent 调用 `catalog_tool` 的路径。
- 首次真实运行：10-13 23:00 推理扫描 → 10-14 00:30 训练扫描 → 02:00 审计。
