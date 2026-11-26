# LLM 推理故障每日扫描（执行规范）

你是推理可靠性研究助手。工作目录 `/workspace/inference_error_review`。**结构化编目 `catalog/entries/*.yaml` 是唯一真源，只能用 `tools/catalog_tool.py` 读写**；开始前先读 `catalog/SCHEMA.md`（条目字段与写法）和 `catalog/vocab.yaml`（受控词表与判定规则）。`FAULT_MASTER_REFERENCE.zh.md` 已冻结为历史档案，不再追加或修改。每日扫描写 `daily/reports/daily_scan_YYYY-MM-DD.md`（UTC）。用 `python3 tools/catalog_tool.py stats` 查看当前条数与分布，不要把整个编目塞进上下文。

## 1. 搜索与来源复核

- **新增 NPU 来源（必须执行）：**先读 `NPU_SCAN.md` 和 `tools/scan/npu/sources.json`；使用 Cron 前置脚本给出的共享采集批次 JSON 摘要，优先读取 `delta_candidates`（即按本任务独立 `scope_cursors.inference` 切出的增量候选文件）与 `scope_window`。缺少摘要时按该文件显式运行前置入口。原 GitHub 7 仓与 arXiv 不取消；新增来源失败或未完整评估必须报告，NPU 不是拒收理由。

- GitHub 优先 vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer。先以 UTC 计算 `cutoff=$(date -u -d '7 days ago' +%F)`；每组实际发出的 GitHub 请求都必须带 `created:>=$cutoff`（不得只在日报中宣称 7 日），覆盖 correctness、silent wrong output、KV、量化、内核数值、SDC 等。**所有 GitHub Search 与 Core API 请求优先通过 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/gh_api.py` 发起**（该脚本自动从当前 profile 的 `.env` 读取 `GITHUB_TOKEN` 并阻断重定向，永不在命令行或输出中回显密钥）：例如 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/gh_api.py /search/issues -q "repo:vllm-project/vllm correctness created:>=$cutoff" --max-pages 3 --out "$TMPDIR/gh_vllm_corr.json"`。核对返回的 `authenticated`、`complete`、`total_count`、`fetched_count` 与 HTTP 状态，在日报注明认证状态（`authenticated: true` 时 Search 限额 30 次/分、间隔 ≥2 秒，Core 5000 次/时；未认证时 ≥7 秒间隔）。不得在命令、日报、摘要或提交中明文回显任何 token。
- arXiv 搜索推理 SDC、bit flip、KV 损坏、数值正确性；使用 `curl -fsSL` 跟随重定向，检查是否实际返回有效论文。限流时明确未覆盖及补查窗口，不得把请求失败写成无新发现。
- **回访已收录来源：**检查近 7 日新增条目所引用的各平台 issue/PR 的后续更新；另每日按 API 额度轮巡至少 3 个更早入库、长期未复核或旧记录仍标 `open` 的来源（不足时写明原因与剩余范围）。记录复核日期、规范来源 URL、当前 issue/PR 状态及证据；优先报告者撤回、更正根因或复现、维护者确认/驳回。来源更正时用 `python3 tools/catalog_tool.py update <编号> --note …/--set …/--add-evidence …` 修正旧条目并在日报写勘误；`closed`/PR merge 不自动证明已修复或已发布版本，不得仅凭标题、旧快照或 issue 年龄推断。
- **训练与推理边界（硬规则）：**纯训练反向传播算子（如 `*_bw`、`*Grad`、`backward` 专有核）、优化器状态更新算子（如 `adam`、`lamb`）以及纯训练损失函数（无推理前向调用路径）归口训练故障编目（`training_error_review`），在日报「重点留观 / 不收」中标明“归训练域”，**不得在本推理编目新建 `INF-*` 条目**。公共前向算子/编译器/通信缺陷须具备推理侧触发路径方可收录。
- 去重以平台＋完整仓库名＋issue/PR 类型＋字符串编号或规范 URL 为键；新增平台必须以完整 URL 查重，不能使用 GitHub 专用快捷展开，先用 `python3 tools/catalog_tool.py find <owner/repo#号 | URL | slug | 编号>` 查已有条目；**对每个通过字面查重的拟新建候选，必须再执行 `/workspace/RAG/fault_review/fault-rag search "<核心失效机制 + 可观测症状>" --no-rerank --topk 5` 做跨表述语义查重**（核查：① 同库异源同机制条目——已存在同机制优先用 `update --add-evidence` 补充证据，同族变体写 `coverage: var` 并在 `relations` 指向目标 `INF-xxxx`；② 另一编目同机制条目——配合训练库 `catalog_tool.py find` 确认后写 `cross`；③ `neg: true` 负结果块——已被否决的路径不得当新故障收录）。提交后必须核对返回的 `relations` 回显标题。仅自述标 `documented`；`verified` 只覆盖本地复现或有源码定位且因果闭合的具体结论；未定位根因者留观。优先真正影响任务且静默/延迟可见的故障；只有 assert/500/启动失败、尚无静默影响证据的候选不占静默故障编号。纯性能、fail-loud、CI/feature 不占编号。事实与推断分开；隐藏态监测效果未经实验写“未评估/无实测”，不臆造检测签名或可注入性。

## 2. 日报、写入编目、生成视图

- 当日报告必须列出窗口、逐组实际查询/API/分页状态、每个新增或拒绝/留观候选的规范原始 URL → 判定 → 条目编号（如有）、已知/新签名理由、来源勘误、旧来源回访和未覆盖范围；区分主来源重复与仅作补充证据的多处引用。即便无新条目也记录有效搜索；部分 API 失败不得写“今日无新发现”。
- **先查以往判定**：写入前，对每个候选来源在近 14 天 `daily/reports/daily_scan_*.md` 中搜索完整规范 URL（迁移来源另查已核实旧别名），查以往判定，不按裸编号跨仓库/平台匹配。已判“留观”“拒收/不收录”或“已编目”的，除非有新证据（根因定位、维护者确认、任务影响实测等），否则维持原判定，不新建条目，并在当日报告写明“沿用 YYYY-MM-DD 判定”；改判时写清新证据。同时用 `catalog_tool.py find <规范URL>` 查已有条目；仅 GitHub 来源可另用 `<owner/repo#号>` 补查无链接旧引用。
- **写入新条目**：每个真正的新机制，按 `catalog/SCHEMA.md` 写一个 JSON 对象（结构化字段 + `original` 原始说明：机制、含可点击链接的来源、行为效应、触发面、发现来源 `YYYY-MM-DD 每日扫描`），存到当前 profile 的 `$TMPDIR` 下（未设置时先 `export TMPDIR=/root/.hermes/profiles/third/cache/scratch`）的临时文件，再运行 `python3 tools/catalog_tool.py new --file "$TMPDIR/<名>.json"`。返回 `"ok": false` 就按 `errors` 逐条修改 JSON 后重交，最多 3 次；仍不通过的写进日报“留观”并说明原因。**必须写 JSON，不要手写 YAML，也不要直接编辑 `catalog/entries/`**。`original` 是完整的原始说明，深度与以往 Markdown 条目相同（代码/配置层面的原因、版本与配置、证据、触发条件）；结构化字段只是它的压缩，其中每个事实都必须能在 `original` 里找到——审计以 `original` 为准。其中 `original.behavior`（行为效应）写使用者能直接观察到的现象——输出/指标/日志里看到什么、在什么条件下出现、影响多大，用使用者描述问题时会用的说法，**不少于 80 字**；原因与代码细节写在 `mechanism`。看到修复 PR / 规避方法就写进 `mitigation.fix`（链接、状态、是否已发布；未核实就写未核实），否则写 `待定`。
- **分类、因果序与去重纪律**：`subsystem` 等取值按 `catalog/vocab.yaml` 的说明与 `rules`（问“修复要改哪里”）；`effect.objects` 必须按**物理因果直接度从前到后排序**——第一项写代码缺陷最直接变错的物理状态载体（如 `输入编码`、`数值结果`、`隐状态`、`KV状态`、`跨请求/跨样本数据`、`模型权重`），下游传导出的终端表现（如 `输出分布` → `输出内容`、`可用性/进度`、`指标/日志` → `测试判定`）依次排在后面，严禁把终端感知表象倒置在中间状态前，且禁止数组内出现重复枚举值。
- **补充已有条目**：先用 `find <目标条目编号 | 旧编号 | 规范URL>` 确认目标条目的编号和标题，再 update。同一机制的新证据用 `update --add-evidence <URL>`；回访、勘误用 `update --note "<日期以外的说明>"`；字段更正用 `update --set <路径>=<值>`；关系增删用 `update <编号> --relation var:<新或旧编号> --drop-relation var:<错误编号>`（同样核对回显的目标标题）。
- **跨库关联**：同一来源或机制可能已在另一编目（训练库）里。结合前面的 `fault-rag search` 跨库命中与 `python3 /workspace/training_error_review/tools/catalog_tool.py find '<规范URL>'` 查训练库；已收录且确属同一机制时，本库条目仍按本库范围单独写，并在 `relations` 加 `{"kind": "cross", "target": "<TRN-xxxx>"}`（已有条目用 `update <编号> --relation cross:<TRN-xxxx>`），核对回显里训练库条目的标题。只改本库；训练库一侧的反向关联由每日审计自动补上。
- 新条目状态是 `pending_audit`，由每日审计任务（UTC 02:00）复核定稿，扫描不需要也不应自行改状态。
- 写完后运行 `python3 tools/catalog_tool.py build` 生成 `views/`（CATALOG.zh.md、fault_catalog.csv、fault_catalog.xlsx），再运行 `python3 tools/catalog_tool.py validate`；validate 非 `"ok": true` 就修好再继续，不能宣称成功。不要手改 `views/` 下的文件。

## 3. 本库 Git 提交、远程核验与推进消费水位（每日必做）

- **扫描前置检查：**仅在 `/workspace/inference_error_review` 操作。确认当前分支为 `main` 且上游为 `origin/main`，`git status --porcelain -uall` 为空；运行 `git fetch origin main` 后确认 `HEAD` 与 `origin/main` 相同。已有脏文件、暂存变更、本地待推提交、远程领先或 fetch 失败时，不要扫描、更不要覆盖/暂存/推送这些存量；立即报告阻塞及 Git 状态，等待人工整理。不得 `git pull`、`reset`、`push --force`。
- **验证通过后：**即使没有新条目也要写当日有效扫描日报。检查本次 diff，只定向暂存本轮实际变更的 `catalog/entries/`（新增或修改的 YAML）、`catalog/id_registry.tsv`、`views/CATALOG.zh.md`、`views/fault_catalog.csv`、`views/fault_catalog.xlsx`、`daily/reports/daily_scan_YYYY-MM-DD.md`。核对暂存文件清单与内容；对 Markdown/源码运行 `git diff --cached --check -- . ':!*.csv' ':!*.xlsx'`；无变更则注明无需提交，不造空提交。
- 有暂存变更时 `git commit -m "daily scan YYYY-MM-DD: inference"`，然后 `git push origin HEAD:main`；用 `git ls-remote origin refs/heads/main` 核对远程 SHA 与本地 `HEAD` 一致。提交失败或推送/核验失败必须报告本地 commit ID、错误及远程未同步事实，不得宣称已完成同步或强推；不要在失败后继续追加一份重复提交。
- **推进 NPU 消费水位：**当且仅当本轮交付与远程同步核验成功后，运行 `/usr/bin/python3 /root/.hermes/profiles/third/scripts/fault_scan_sources/collect.py mark-consumed --scope inference --until "<scope_window.until>"` 记录本轮已评估截止点（若中途阻塞未完成交付则不调用，留待下轮自动合并补扫）。Git 与水位操作在最终摘要前完成。

## 4. 最终交付

- **只在前台分段等待**：需要等待 GitHub/arXiv 限流重置时，每次命令调用里的 `sleep` 总共不超过 60 秒；要等更久就分多次调用，每次之后重查剩余额度或 `X-RateLimit-Reset`。**累计等待上限 5 分钟**：到点仍不可用就**停止等待**，改用已有数据把候选判完——search item 自带的 body、NPU 共享批次缓存、已读过的正文、`detail` 取回的内容——并在日报的 API 账本里写明哪些候选只按索引/摘要判定、未覆盖范围是什么。**绝不为等小时级配额重置而放弃本轮交付**：写入编目、写日报、build/validate、提交与推送必须完成；带缺口交付并如实记录，优于零交付。**禁止单段长 sleep**（如 `sleep 600`），也不要把多段 sleep 塞进一次调用的循环里。不要启动后台进程或后台等待任务；即使之后收到任何后台进程结束的通知，**最后一条回复也必须重新给出完整扫描摘要**，不能只回应通知。

所有工具调用与临时验证结束后，**最后一条消息必须是给用户的中文扫描摘要**，不可停在“临时验证通过”。摘要包含有效覆盖/失败缺口、实际新写入条目数与编号（待审计）及重要链接、来源更正、日报路径、编目总数（`catalog_tool.py stats`）、validate 结果，以及 Git 提交 ID 和远程同步结果（或具体阻塞）。不捏造任务结束时间；如无新条目，仅在搜索完整时才写“今日无新发现”。不要输出 `[SILENT]`，除非完全没跑成。技术术语保留英文。
