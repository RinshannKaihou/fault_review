# 切换后的 cron 任务（hermes profile `third`）

具体步骤维护在各仓库的说明文件里；cron 提示词作为薄入口保留工作目录、Git 前置止损、认证与 RAG 查重工具约定、源头拓扑卫生与最终交付契约。

## 推理扫描 `8645356b40ae`（0 23 * * * UTC，deepseek-v4.1-flash / opencode-go）

```
工作目录必须是 /workspace/inference_error_review。先读 daily_scan.prompt.md 与 NPU_SCAN.md。开始前严格核验 Git 前置条件（前置脚本 fault_scan_npu_inference.sh 已先执行，若报 BLOCKED 或当前非干净 main / fetch 失败 / HEAD 与 origin/main 不一致，立即停止并在中文摘要准确报告阻塞，不自动处理存量）。通过后严格按说明文件完成扫描、写日报、写入编目、生成视图、校验、提交与推送：
1. GitHub API 请求优先调用 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/gh_api.py`（自动读取 profile .env 中的 GITHUB_TOKEN 且永不回显密钥；认证 Search 间隔 ≥2 秒，用 `--max-pages` + `--out "$TMPDIR/..."` 补齐超 30 条的分页尾页与关键评论核验）。
2. NPU 来源读取前置脚本 JSON 摘要中的 `delta_candidates` 与 `scope_window`（若部分 API 失败，如实记录缺口，不宣称完整扫描或无新发现）。
3. 入库前查重与归属：先跑 `tools/catalog_tool.py find '<规范URL>'` 查字面重复；拟新建条目前必须再执行 `/workspace/RAG/fault_review/fault-rag search "<核心失效机制 + 可观测症状>" --no-rerank --topk 5`，排查同库异源同机制（改用 update 补证据或标 var）、训练库配对条目（标 cross，配合训练库 `catalog_tool.py find` 补查最新未同步项）及已否决负结果（`neg: true`）。纯训练反向算子（`*_bw` / `*Grad`）、优化器更新核与纯训练 loss 算子归口训练库，在日报注明归训练域，不在本库新建 `INF-*`。
4. 编目只能经 `tools/catalog_tool.py` 读写，禁止手改 YAML、`views/` 或 `FAULT_MASTER_REFERENCE.zh.md`。`effect.objects` 必须按物理因果直接度从前到后排列（最直接受损的状态载体如 `数值结果`/`隐状态`/`KV状态`/`模型权重` 在前，终端表象如 `输出内容`/`测试判定` 在后），禁止数组内重复枚举。回访阶段一并运行 `/usr/bin/python3 /workspace/RAG/fault_review/scripts/rag_auto_sync.py list-todos --scope inference` 检查待处理返工单（若有：核对原 issue 后经 `tools/catalog_tool.py` 修正因果逆序、补全可观测表象词或补 `var`/`cross` 边，完成后调 `done-todo --id <todo_id>` 核销；若核实原文不成立则加 `--dismiss --note '<原因>'` 核销）。
5. 等待限流只在前台分段 sleep（单次 ≤60 秒，累计上限 5 分钟；到点改用已有索引/正文完成判定并在日报写明未覆盖范围，绝不放弃写入、build/validate、提交与推送；禁止后台进程）。本轮 `git push` 与 `git ls-remote` 核验一致后，执行 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/fault_scan_sources/collect.py mark-consumed --scope inference --until <scope_window.until>` 推进消费水位。最后一条回复必须是完整中文扫描摘要。
```

```bash
hermes -p third cron edit 8645356b40ae --prompt "$(cat <<'EOF'
<上面的文本>
EOF
)"
```

## 训练扫描 `1b8481fa4e2f`（30 0 * * * UTC，deepseek-v4.1-flash / opencode-go）

```
工作目录必须是 /workspace/training_error_review。先读 AGENTS.md、taxonomy.md、hermes/daily_scan.prompt.md 与 NPU_SCAN.md。开始前严格核验 Git 前置条件（前置脚本 fault_scan_npu_training.sh 已先执行，若报 BLOCKED 或当前非干净 main / fetch 失败 / HEAD 与 origin/main 不一致，立即停止并在飞书摘要准确报告阻塞，不自动处理存量）。通过后严格按说明文件完成扫描、写日报、写入编目、生成视图、校验、提交与推送：
1. GitHub API 请求优先调用 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/gh_api.py`（自动读取 profile .env 中的 GITHUB_TOKEN 且永不回显密钥；认证 Search 间隔 ≥2 秒，严格按 `tools/scan/repos.yaml` 构造查询并使用合法限定符如 `is:issue`/`is:pr`/`is:open`，禁止无效限定符 `issue:open`；用 `--max-pages` + `--out "$TMPDIR/..."` 补齐超 30 条的分页尾页与关键评论核验）。
2. NPU 来源读取前置脚本 JSON 摘要中的 `delta_candidates` 与 `scope_window`（独立于推理任务的消费水位；若部分 API 失败，如实记录缺口，不宣称完整扫描或无新发现）。
3. 入库前查重：先跑 `tools/catalog_tool.py find '<规范URL>'` 并查 `catalog/rejected.md`、`catalog/blindspots.md`；拟新建条目前必须再执行 `/workspace/RAG/fault_review/fault-rag search "<核心失效机制 + 可观测症状>" --no-rerank --topk 5`，排查同库异源同机制（改用 update 补证据或标 var）、推理库配对条目（标 cross，配合推理库 `catalog_tool.py find` 补查最新未同步项）及已否决负结果（`neg: true`）。
4. 编目只能经 `tools/catalog_tool.py` 读写，禁止手改 YAML、`views/` 或 `catalog/FAULT_MASTER_REFERENCE.zh.md`。`effect.objects` 必须按物理因果直接度从前到后排列（最直接受损载体如 `数值结果`/`隐状态`/`训练样本流`/`训练信号`/`梯度`/`优化器状态`/`模型权重` 在前，宏观终端后果如 `训练轨迹`/`测试判定` 在后），禁止数组内重复枚举。回访阶段一并运行 `/usr/bin/python3 /workspace/RAG/fault_review/scripts/rag_auto_sync.py list-todos --scope training` 检查待处理返工单（若有：核对原 issue 后经 `tools/catalog_tool.py` 修正因果逆序、补全可观测表象词或补 `var`/`cross` 边，完成后调 `done-todo --id <todo_id>` 核销；若核实原文不成立则加 `--dismiss --note '<原因>'` 核销）。
5. 等待限流只在前台分段 sleep（单次 ≤60 秒，累计上限 5 分钟；到点改用已有索引/正文完成判定并在日报写明未覆盖范围，绝不放弃写入、build/validate、提交与推送；禁止后台进程）。本轮 `git push` 与 `git ls-remote` 核验一致后，执行 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/fault_scan_sources/collect.py mark-consumed --scope training --until <scope_window.until>` 推进消费水位。最后一条回复必须是完整飞书扫描摘要。
```

```bash
hermes -p third cron edit 1b8481fa4e2f --prompt "$(cat <<'EOF'
<上面的文本>
EOF
)"
```

## 每日审计 `1fc648dfb5a0`（02:00 UTC，脚本模式，`no-agent`）

两次扫描之后运行；先推理后训练，由 `~/.hermes/profiles/third/scripts/fault_catalog_daily_audit.sh` 安全解析 `~/.bashrc`（及 `/root/.bashrc` 回退）中的 `JUDGE_API_KEY` 与 `JUDGE_API_URL`，依次调用两仓库的 `tools/daily_audit.sh` 完成 Git 前置检查、`cross-sync`、`pattern-sync`、`catalog_audit.py run`、生成视图、校验与推送。

## RAG 每日同步与质检（03:30 UTC，脚本模式，`no-agent`）

每日审计之后运行；由 `~/.hermes/profiles/third/scripts/fault_rag_daily_sync.sh` 调用 `/workspace/RAG/fault_review/scripts/rag_auto_sync.py sync`：
- 在隔离 `git worktree` 中同步已通过审计的 `active` 条目（自动跳过新建 `pending_audit` 与已隔离冲突项），复用向量缓存执行 `reindex`，并依次运行 CLI 门禁（bench 聚合 + indy 逐条）、Web 同构门禁与因果拓扑校验；若单条新条目挤占老题触发门禁回退，自动执行单条隔离（Quarantine）保其余健康条目照常发布并推送远端。
- 按需发起只读薄层 LLM 质检（新条目口语症状盲测 + 未连边近邻筛查）：低风险瑕疵（因果逆序、口语盲测漏召回、疑似漏标 `var`/`cross`）写入 `/root/.hermes/profiles/third/cache/rag_feedback/{inference,training}_todo.jsonl` 交次日扫描任务回访自动闭环（单签名最多派发 1 次防死循环）；疑似实质重复（需 `merge`）、滞留 ≥2 天 `pending_audit` 与门禁隔离项落盘至 `/workspace/RAG/fault_review/data/pending_human.json` 供异步人工裁决。

---

## 变更记录

### 2026-11-26：新增 RAG 每日自动同步与跨日返工闭环

1. **新增第 4 个定时任务（`03:30 UTC`，`fault_rag_daily_sync.sh`）**：隔离 `worktree` + 四道硬门禁 + 单条冲突自动隔离 + 内嵌只读 LLM 口语盲测与漏边筛查，原子更新本地 `data/` 向量索引并推送 `catalogs/` 触发 Pages 构建。
2. **三档闭环路由**：第一档低风险编目瑕疵自动派发至 `rag_feedback/*_todo.jsonl`，两个每日扫描任务在回访阶段通过 `rag_auto_sync.py list-todos / done-todo` 自动核销闭环；第二、三档需人工拍板项落盘 `data/pending_human.json` 并推送飞书摘要。

### 2026-10-16：限流等待加上限（两个扫描任务）

- 单次命令调用 sleep ≤ 60 秒，累计等待上限 5 分钟；到点改用已有数据判完候选并在日报记录缺口，不放弃本轮交付。

### 2026-11-20：适配脱敏网关、打通 `GITHUB_TOKEN`、分 Scope 水位与接入 `fault-rag` 语义查重

1. **审计网关脱敏适配**：部署新版 `fault_catalog_daily_audit.sh`（用 Python `shlex` 取代 `eval`，同时校验 `JUDGE_API_KEY` 与 `JUDGE_API_URL`）。
2. **GitHub 认证打通**：`fault_scan_npu_entry.py` 自动向 `collect.py` 注入 `third/.env` 中的 `GITHUB_TOKEN`；新增 `gh_api.py` 封装 Search（30/min）与 Core（5000/h）请求及自动翻页，解决未认证空等与前 30 条截断问题。
3. **NPU 分 Scope 消费水位**：`collect.py` 引入 `scope_cursors.{inference,training}` 与 `*_delta.jsonl`，仅在 `git push` 核验后通过 `mark-consumed` 推进各自水位，彻底消除错峰重跑（`>3h`）引起的漏扫窗口。
4. **接入 `fault-rag` 与源头拓扑卫生**：`fault-rag` 补齐 `/usr/bin/python3` 自动回退守卫；两个扫描任务在 `new` 前跑 `fault-rag search --no-rerank --topk 5` 做跨表述/跨库查重，并前置强制 `effect.objects` 物理因果正序、无重复枚举及纯训练反向算子归属边界。
