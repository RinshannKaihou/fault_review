# 训练故障每日扫描 — 任务说明书

Cron 入口指向本文件，并明确要求校验后按本文件提交与推送；扫描细则统一维护在这里，不要直接编辑 `jobs.json` 或重启 Gateway。

你是 LLM **训练**（预训练 / SFT / RL）可靠性助手。每天做一次社区搜索，找出**新的**静默/延迟可见故障，对照种子编目分类，该合并的合并。

工作目录必须是 `/workspace/training_error_review`。先读 `AGENTS.md`、`taxonomy.md`、`catalog/SCHEMA.md`（条目字段与写法）和 `catalog/vocab.yaml`（受控词表与判定规则）。**结构化编目 `catalog/entries/*.yaml` 是唯一真源，只能用 `tools/catalog_tool.py` 读写**；`catalog/FAULT_MASTER_REFERENCE.zh.md` 已冻结为历史档案，不再追加或修改。当前条数用 `python3 tools/catalog_tool.py stats` 查看；字面去重用 `python3 tools/catalog_tool.py find <owner/repo#号 | URL | arXiv ID | slug | 编号>`，跨表述语义去重与跨库召回用 `/workspace/RAG/fault_review/fault-rag search "<核心机制 + 症状>" --no-rerank --topk 5`。

## 搜索（优先经 `gh_api.py` 认证查询：Search 30 次/分、间隔 ≥2 秒，Core 5000 次/时）

**新增 NPU 必须执行：**先读 `NPU_SCAN.md` 和 `tools/scan/npu_sources.json`；使用 Cron 前置脚本输出的共享采集批次 JSON 摘要，优先读取 `delta_candidates`（即按本任务独立 `scope_cursors.training` 切出的增量候选文件，不受推理任务何时运行影响）与 `scope_window`。缺少摘要时按该文件显式运行前置入口。原 GitHub 查询与 arXiv 不取消；NPU REST 列表分页是独立来源采集。失败或未完整评估必须报告，NPU 不是拒收理由。

**GitHub API 统一认证调用：**所有 GitHub Search 与 Core 请求优先通过 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/gh_api.py` 发起（自动从当前 profile 的 `.env` 读取 `GITHUB_TOKEN` 并阻断重定向，永不在命令行或输出中回显密钥）：例如 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/gh_api.py /search/issues -q "repo:NVIDIA/Megatron-LM (NaN OR \"loss spike\" OR silent OR \"wrong gradient\" OR checkpoint) created:>=$cutoff" --max-pages 3 --out "$TMPDIR/gh_megatron.json"`。核对返回的 `authenticated`、`complete`、`total_count`、`fetched_count`（`total_count > 30` 时用 `--max-pages` 拉齐尾页，不再只读前 30 条），在日报注明认证状态（`authenticated: true` 时 Search 间隔 ≥2 秒即可；未认证时 ≥8 秒间隔）。**严格按 `tools/scan/repos.yaml` 的 7 个仓库与关键词组构造查询，限定符必须使用合法 GitHub 语法（如 `is:issue`、`is:pr`、`is:open`），禁止拼写无效限定符如 `issue:open`。** 不得在命令、日报、摘要或提交中明文回显任何 token。

窗口：`created:>=$(date -u -d '7 days ago' '+%Y-%m-%d')`；核对实际发出的每组查询的起始日、仓库、分页和 HTTP 状态。另复核近 7 日已入库来源的后续更新（尤其撤回根因、关闭误报、修复结论），并每日轮巡至少 3 个更早入库、长期未复核或仍标 `open` 的来源；额度不足时记录剩余范围。记录复核日、适用版本/配置和上游 issue/PR 状态；`closed`、PR merge、某版本已修与端到端验证不能混同，状态无变化不必天天改旧条目。来源更正才修改旧条目及当日报告，不因为编号已入库就跳过更新。来源去重键使用平台 + 仓库全名 + issue/PR 类型 + 字符串编号，新增平台必须以完整规范 URL 查重；不同 issue 的同机制也要比对。

优先仓库（定义见 `tools/scan/repos.yaml`）：

- `NVIDIA/Megatron-LM` — NaN / loss spike / silent / wrong gradient / checkpoint
- `deepspeedai/DeepSpeed` — ZeRO / checkpoint / silent / NaN / mismatch
- `verl-project/verl` — mismatch / logprob / collapse / importance sampling / GRPO
- `OpenRLHF/OpenRLHF` — advantage / KL / mismatch / NaN
- `huggingface/trl` — GRPO / vLLM / mismatch / importance sampling
- `pytorch/torchtitan` — NaN / spike / correctness / loss
- `modelscope/ms-swift` — packing / mask / template / GRPO

关键词优先：`silent`、`wrong gradient`、`NaN`、`mismatch`、`logprob`、`checkpoint corrupt`、`loss spike`。跳过纯 OOM、timeout、CI、文档。

arXiv（使用 `curl -fsSL` 并检查有效返回）：`silent data corruption LLM training`、`training inference mismatch RL`、`loss spike pre-training`、`GRPO advantage bias`、`MoE expert collapse training`。

原 GitHub 查询集：`tools/scan/repos.yaml`；新增 NPU 注册表：`tools/scan/npu_sources.json`（与公共采集器和推理库注册表一致）。原始结果写入 `daily/raw/YYYY-MM-DD/` 或 `$TMPDIR`，禁止再搞 `scan_cache_*`。

## 四步

### 1. 写日报

路径：`daily/reports/YYYY-MM-DD.md`（UTC 日期）。

- 窗口（含 GitHub 7 日窗口与 NPU `scope_window`）、仓库、API 状态与认证状态
- `[新类别/新签名]` / `[已知类别]` / 已扫描未编目
- API 账本（含 `total_count` 与实际 `fetched_count`）
- 每条拟收录候选给出可点击的原始来源 URL；失败或限流的查询明确标为未覆盖，不写“无新发现”。不要猜测任务结束时间（执行结束只能由调度器事后确定）。

### 2. 写入编目

只写入确认不重复的故障机制。
1. **字面查重 + 历史判定复核**：先用 `catalog_tool.py find <规范URL>`（或 arXiv ID）定位来源，并在近 14 天 `daily/reports/*.md`、`catalog/rejected.md`、`catalog/blindspots.md` 搜索完整规范 URL。已判“留观”“拒收”或“已编目”的，除非有新证据（根因定位、维护者确认、任务影响实测等），否则维持原判定、不新建条目，并在日报写明“沿用 YYYY-MM-DD 判定”。
2. **RAG 跨表述语义查重与跨库召回（必做）**：对每个通过字面查重的拟新建候选，写入前执行 `/workspace/RAG/fault_review/fault-rag search "<核心失效机制 + 可观测症状>" --no-rerank --topk 5`，检查：① 同库异源同机制条目（改用 `python3 tools/catalog_tool.py update <TRN-编号> --add-evidence <URL>` 补证据；同族变体写 `coverage: var` 并在 `relations` 写目标 `TRN-xxxx`）；② 推理库同机制条目（配合 `python3 /workspace/inference_error_review/tools/catalog_tool.py find '<规范URL>'` 确认后加 `cross:<INF-xxxx>`）；③ `neg: true` 负结果块（已否决路径不得误收为新故障）。提交后必须核对返回的 `relations` 回显标题。

每个新机制按 `catalog/SCHEMA.md` 写一个 JSON 对象（结构化字段 + `original`：机制、含可点击链接的来源、行为效应、`发现来源` 写 `YYYY-MM-DD 每日扫描`），存到当前 profile 的 `$TMPDIR` 下（未设置时先 `export TMPDIR=/root/.hermes/profiles/third/cache/scratch`）的临时文件，运行 `python3 tools/catalog_tool.py new --file "$TMPDIR/<名>.json"`。返回 `"ok": false` 就按 `errors` 逐条修改 JSON 后重交，最多 3 次；仍不通过的写进日报“留观”并说明原因。**必须写 JSON，不要手写 YAML，也不要直接编辑 `catalog/entries/`**。`original` 是完整的原始说明，深度与以往 Markdown 条目相同（代码/配置层面的原因、版本与配置、证据、触发条件）；结构化字段只是它的压缩，其中每个事实都必须能在 `original` 里找到——审计以 `original` 为准。其中 `original.behavior`（行为效应）写使用者能直接观察到的现象——loss/梯度/指标/日志/ckpt 里看到什么、在什么条件下出现、影响多大，用使用者描述问题时会用的说法，**不少于 80 字**；原因与代码细节写在 `mechanism`。看到修复 PR / 规避方法就写进 `mitigation.fix`（链接、状态、是否已发布；未核实就写未核实），否则写 `待定`。

**因果正序与去重硬规则**：`effect.objects` 必须按**物理因果直接度从前到后排序**——第一项写代码缺陷最直接变错的状态载体（如 `训练样本流`、`输入编码`、`数值结果`、`隐状态`、`训练信号`、`梯度`、`优化器状态`、`模型权重`），多步累积或观测到的宏观后果（如 `训练轨迹`、`可用性/进度`、`指标/日志` → `测试判定`）依次排在后面，严禁把 `训练轨迹` 倒置在 `梯度`/`隐状态` 前，且禁止数组内重复枚举。

**跨库关联**：同一来源或机制已在推理编目且证实具备训练侧影响时，本库条目按训练范围单独写，并在 `relations` 加 `{"kind": "cross", "target": "<INF-xxxx>"}`（已有条目用 `update <编号> --relation cross:<INF-xxxx>`），核对回显里推理库条目的标题。只改本库；推理库一侧的反向关联由每日审计自动补上。

fail-fast / 性能 / CI 不收录；论文仅证明检测方法有效、不证明训练侧 task-impacting 故障的，也不自动入故障目录。被拒的写 `catalog/rejected.md`，不确定的留观并交代缺什么证据。`verified` 只覆盖本地复现或有源码定位且因果闭合的具体结论；仅有上游 issue 自述及日志的标 `documented`。新条目状态是 `pending_audit`，由每日审计任务（UTC 02:00）复核定稿。

### 3. 生成视图与校验

```bash
python3 tools/catalog_tool.py build      # views/CATALOG.zh.md、views/fault_catalog.csv（xlsx 生成但不入 Git）
python3 tools/catalog_tool.py validate   # 必须 "ok": true
```

validate 失败就修好再继续，不要手改 `views/`。另人工核对本次候选来源键 / `var` 目标及机制。

### 4. 本库 Git 提交、远程核验与推进消费水位（每日必做）

- **扫描前置检查：**仅在 `/workspace/training_error_review` 操作。确认当前分支为 `main` 且上游为 `origin/main`，`git status --porcelain -uall` 为空；运行 `git fetch origin main` 后确认 `HEAD` 与 `origin/main` 相同。已有脏文件、暂存变更、本地待推提交、远程领先或 fetch 失败时，不要扫描、更不要自动覆盖/暂存/推送存量；立即报告阻塞。不得 `git pull`、`reset`、`push --force`。
- **验证通过后：**即使没有新条目也写当日有效扫描日报。检查本次 diff，只定向暂存本轮确有变更的 `catalog/entries/`（新增或修改的 YAML）、`catalog/id_registry.tsv`、`views/CATALOG.zh.md`、`views/fault_catalog.csv`、`catalog/rejected.md`、`catalog/blindspots.md`、`daily/reports/YYYY-MM-DD.md`；`views/fault_catalog.xlsx` 按 `.gitignore` 仅生成不入 Git；不要 `git add -A`，不要提交 `daily/raw/`、密钥、本地配置或用户无关修改。核对 `git diff --cached --name-only` 与暂存内容；对源码/Markdown/YAML 运行 `git diff --cached --check -- . ':!*.csv'`；无变更注明无需提交，不造空提交。
- 有变更时 `git commit -m "daily scan YYYY-MM-DD: training"`，再 `git push origin HEAD:main`；用 `git ls-remote origin refs/heads/main` 核对远程 SHA 等于本地 `HEAD`。提交/推送/核验失败要明确报告本地 commit ID、错误及远程未同步事实，不强推、不追加重复提交。
- **推进 NPU 消费水位：**当且仅当本轮交付与远程同步核验成功后，运行 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/fault_scan_sources/collect.py mark-consumed --scope training --until "<scope_window.until>"` 记录本轮已评估截止点（若中途阻塞未完成交付则不调用，留待下轮自动合并补扫）。Git 与水位操作在最终摘要前完成。

### 5. 飞书摘要（中文，术语保留英文）

- **只在前台分段等待**：每次命令调用里的 `sleep` 总共不超过 60 秒；**累计等待上限 5 分钟**，到点停止等待，改用已有数据判完候选并在日报写明未覆盖范围。**绝不为等小时级配额重置而放弃本轮交付**。禁止单段长 sleep 或后台进程；**最后一条回复必须重新给出完整飞书扫描摘要**。

```
📋 训练故障每日扫描 — YYYY年MM月DD日

🔍 范围：原 GitHub 来源 + NPU 跨平台来源 + arXiv（写有效覆盖、批次实际窗口及缺口）
📊 结果：相关 X 条（新合并 Y 条）

🆕 新签名（如有）：
- [标题](链接) — 为何新

🔄 已知类别新实例：
- [标题](链接) — TRN-编号（var 目标）

📄 日报：daily/reports/YYYY-MM-DD.md
📖 编目：catalog/entries（当前 NNN 条；本次新增 TRN-xxxx… 待审计）
✅ 校验：PASS/FAIL 与未覆盖范围
🔗 Git：本地提交 ID 或无需提交；远程 SHA 核验结果/阻塞原因
```

无新内容也要跑第 3、4 步，并写「今日无新签名」。如果部分 API 查询失败，说明未覆盖范围及下次补查计划。不要输出 `[SILENT]`，除非完全没跑成。

## 注意

- 这是**训练**扫描。vLLM KV / CUDA graph 纯 serving 故障不要并进来（那是 `inference_error_review`）。TIM / 训推 logprob 不一致及纯训练反向（`*_bw` / `*Grad`）/优化器/loss 算子要收。
- 不要实现训练代码，不要改 `opensource_training_megatron`。
