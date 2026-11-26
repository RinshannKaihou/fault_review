# 切换后的 cron 任务（hermes profile `third`）

具体步骤维护在各仓库的说明文件里；cron 提示词作为薄入口保留工作目录、Git 前置止损、认证与 RAG 查重工具约定、源头拓扑卫生与最终交付契约。

## 推理扫描 `8645356b40ae`（0 23 * * * UTC，deepseek-v4.1-flash / opencode-go）

```
工作目录必须是 /workspace/inference_error_review。先读 daily_scan.prompt.md 与 NPU_SCAN.md。开始前严格核验 Git 前置条件（前置脚本 fault_scan_npu_inference.sh 已先执行，若报 BLOCKED 或当前非干净 main / fetch 失败 / HEAD 与 origin/main 不一致，立即停止并在中文摘要准确报告阻塞，不自动处理存量）。通过后严格按说明文件完成扫描、写日报、写入编目、生成视图、校验、提交与推送：
1. GitHub API 请求优先调用 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/gh_api.py`（自动读取 profile .env 中的 GITHUB_TOKEN 且永不回显密钥；认证 Search 间隔 ≥2 秒，用 `--max-pages` + `--out "$TMPDIR/..."` 补齐超 30 条的分页尾页与关键评论核验）。
2. NPU 来源读取前置脚本 JSON 摘要中的 `delta_candidates` 与 `scope_window`（若部分 API 失败，如实记录缺口，不宣称完整扫描或无新发现）。
3. 入库前查重与归属：先跑 `tools/catalog_tool.py find '<规范URL>'` 查字面重复；拟新建条目前必须再执行 `/workspace/RAG/fault_review/fault-rag search "<核心失效机制 + 可观测症状>" --no-rerank --topk 5`，排查同库异源同机制（改用 update 补证据或标 var）、训练库配对条目（标 cross，配合训练库 `catalog_tool.py find` 补查最新未同步项）及已否决负结果（`neg: true`）。纯训练反向算子（`*_bw` / `*Grad`）、优化器更新核与纯训练 loss 算子归口训练库，在日报注明归训练域，不在本库新建 `INF-*`。
4. 编目只能经 `tools/catalog_tool.py` 读写，禁止手改 YAML、`views/` 或 `FAULT_MASTER_REFERENCE.zh.md`。`effect.objects` 必须按物理因果直接度从前到后排列（最直接受损的状态载体如 `数值结果`/`隐状态`/`KV状态`/`模型权重` 在前，终端表象如 `输出内容`/`测试判定` 在后），禁止数组内重复枚举。
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
4. 编目只能经 `tools/catalog_tool.py` 读写，禁止手改 YAML、`views/` 或 `catalog/FAULT_MASTER_REFERENCE.zh.md`。`effect.objects` 必须按物理因果直接度从前到后排列（最直接受损载体如 `数值结果`/`隐状态`/`训练样本流`/`训练信号`/`梯度`/`优化器状态`/`模型权重` 在前，宏观终端后果如 `训练轨迹`/`测试判定` 在后），禁止数组内重复枚举。
5. 等待限流只在前台分段 sleep（单次 ≤60 秒，累计上限 5 分钟；到点改用已有索引/正文完成判定并在日报写明未覆盖范围，绝不放弃写入、build/validate、提交与推送；禁止后台进程）。本轮 `git push` 与 `git ls-remote` 核验一致后，执行 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/fault_scan_sources/collect.py mark-consumed --scope training --until <scope_window.until>` 推进消费水位。最后一条回复必须是完整飞书扫描摘要。
```

```bash
hermes -p third cron edit 1b8481fa4e2f --prompt "$(cat <<'EOF'
<上面的文本>
EOF
)"
```

## 每日审计 `1fc648dfb5a0`（02:00 UTC，脚本模式，`no-agent`）

两次扫描之后运行；先推理后训练，由 `~/.hermes/profiles/third/scripts/fault_catalog_daily_audit.sh` 安全解析 `~/.bashrc` 中的 `JUDGE_API_KEY` 与 `JUDGE_API_URL`，依次调用两仓库的 `tools/daily_audit.sh` 完成 Git 前置检查、`cross-sync`、`pattern-sync`、`catalog_audit.py run`、生成视图、校验与推送。

---

## 变更记录

### 2026-10-16：限流等待加上限（两个扫描任务）

- 单次命令调用 sleep ≤ 60 秒，累计等待上限 5 分钟；到点改用已有数据判完候选并在日报记录缺口，不放弃本轮交付。

### 2026-11-20：适配脱敏网关、打通 `GITHUB_TOKEN`、分 Scope 水位与接入 `fault-rag` 语义查重

1. **审计网关脱敏适配**：部署新版 `fault_catalog_daily_audit.sh`（用 Python `shlex` 取代 `eval`，同时校验 `JUDGE_API_KEY` 与 `JUDGE_API_URL`）。
2. **GitHub 认证打通**：`fault_scan_npu_entry.py` 自动向 `collect.py` 注入 `third/.env` 中的 `GITHUB_TOKEN`；新增 `gh_api.py` 封装 Search（30/min）与 Core（5000/h）请求及自动翻页，解决未认证空等与前 30 条截断问题。
3. **NPU 分 Scope 消费水位**：`collect.py` 引入 `scope_cursors.{inference,training}` 与 `*_delta.jsonl`，仅在 `git push` 核验后通过 `mark-consumed` 推进各自水位，彻底消除错峰重跑（`>3h`）引起的漏扫窗口。
4. **接入 `fault-rag` 与源头拓扑卫生**：`fault-rag` 补齐 `/usr/bin/python3` 自动回退守卫；两个扫描任务在 `new` 前跑 `fault-rag search --no-rerank --topk 5` 做跨表述/跨库查重，并前置强制 `effect.objects` 物理因果正序、无重复枚举及纯训练反向算子归属边界。
