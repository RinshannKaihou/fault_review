# NPU 历史存量回扫 — 方案

状态：**已定范围与顺序，未开工。** 排在悬空链接（A）与模式分层（B）之后、改进措施（C）之前。

## 为什么必须单做一轮

每日扫描的 NPU 采集器（`profile third` 的 `fault_scan_sources/collect.py`）按 `updated_at`
增量取数，水位单调前进。2026-10-15 的 `state.json` 里 22 个源全部停在
`2026-10-14T23:00:52Z`，取数用 `updated_after=水位-1h`。

推论：**一个在 2026-10-07 前最后更新、之后再没有人碰过的 issue，日常扫描永远不会
把它放进候选索引。** 不是优先级低，是不可见。只有它被重新评论/改状态（`updated_at`
前移）才会重新出现。因此存量只能靠一次性回扫补齐。

首轮（10-14 夜推理、10-15 训练）实际看到的窗口是 `updated_at ∈ [2026-10-07, 10-14]`，
8317 条候选，产出 27 条编目条目。

## 现状基线（2026-10-15，推理 902f0ff / 训练 26315b3）

按"证据 URL 落在 gitcode/atomgit/gitee 或 3 个 GitHub NPU 专仓"统计：

| | 条目总数 | NPU 来源条目 | 占比 |
|---|---:|---:|---:|
| 推理 | 1389 | 14 | 1.0% |
| 训练 | 282 | 13 | 4.6% |
| 合计 | 1671 | **27** | 1.6% |

来源仓分布：`cann/ops-nn` 9、`Ascend/pytorch` 6、`triton-ascend` 5、
`cann/ops-transformer` 4、`vllm-ascend` 3、`cann/hcomm` 3、`cann/hccl` 2、
`cann/ops-math` 2、`cann/runtime` 2、`op-plugin` 1、`mindformers` 1。
**19 个 GitCode 仓里 8 个零产出**：torchair、ge、msprobe、mssanitizer、
MindSpeed/-LLM/-RL、MindIE-LLM、msmodelslim。

另有 17 条旧条目（INF-0040/0389/0860/1056、TRN-0056/0176/0227 等）正文提到
ascend/npu，但来源是 vllm/sglang 主仓——GPU 栈视角的旁证，不计入 NPU 来源条目。

3 个训推共有来源（triton-ascend#2531、Ascend/pytorch#5296、cann/ops-nn#6482）
已双向 cross，`cross-sync` 生效。

## 存量规模（2026-10-15 实测，非估算）

issue 全量列表经 `prefilter.py list` 清点，每仓 `fetched` 与 API `total_count` 一致：

| | 历史 issue（updated < 10-07） | 全量 issue | 全量 PR |
|---|---:|---:|---:|
| GitCode/AtomGit 19 仓 | **37,479** | 44,182 | ~130,000 |
| GitHub 3 NPU 仓 | **4,691** | 4,970 | 17,280 |
| 旧 Gitee 镜像 4 仓 | — | 30,951 | ~122,000 |

`cann/*` 多数仓 2025-09 才建、存量集中在 2026 年内；`Ascend/pytorch`（2022 起）、
`mindspore/mindformers`（2022 起）、`mindspore/mindspore`（2020 起，10,773 条）是长尾。
旧 Gitee 按 NPU_SCAN.md 只作历史回访，不进本轮范围。

## 过滤器校准（关键结论）

用首晚 27 个真实入库的 issue 当标注集，反测标题级正则的召回：

| 口径 | 召回 |
|---|---|
| NPU_SCAN.md 第 2 条那套严格关键词（精度/静默/错值/乱码…） | **18/27 = 67%** |
| `prefilter.py` 的宽口径 `KEEP`（加 丢失/溢出/偏移/越界/漏/miscompile/不对/异常/脏数据…） | **26/27 = 96%** |

严格口径漏掉的都是真实收录项：`AllGather Recursive HD 接收偏移错误`（TRN-0277）、
`npu_tensor.cpu() 数据丢失（0 值）`（TRN-0272）、
`silent miscompile — (vec_bool != scalar_true) folds to all-true`（INF-1372）、
`IndexPutV2 偏移值溢出 uint32`（TRN-0271）。

宽口径唯一漏项是 `cann/hccl#825`（TRN-0270 的来源），因为它以"需求建议"形式提交——
所以 `DROP` 里**不得**排除 `需求建议`。

结论：回扫必须是 **宽口径正则 → 廉价模型判读标题 → 选读正文** 三级漏斗。
直接套 NPU_SCAN.md 的严格关键词会丢三分之一。

## 本轮范围：T1 + GitHub 3 仓

| 档 | 仓 | 历史 issue | 宽口径候选 |
|---|---|---:|---:|
| **T1** | pytorch, op-plugin, torchair, ops-transformer, hccl, hcomm, MindSpeed, MindSpeed-LLM, MindSpeed-RL, MindIE-LLM, msmodelslim | 13,965 | **1,680** |
| **GitHub** | vllm-ascend, triton-ascend, sgl-kernel-npu | 4,691 | ~270 |
| （不做）T2 | ops-nn, ops-math, ge, runtime, msprobe, mssanitizer, mindformers | 12,741 | 1,837 |
| （不做）T3 | mindspore/mindspore | 10,773 | 1,287（945 条创建于 2024 前） |

T1 候选按仓：ops-transformer 717、pytorch 266、hccl 161、hcomm 153、
MindSpeed-LLM 95、torchair 93、msmodelslim 76、MindIE-LLM 58、op-plugin 39、
MindSpeed 13、MindSpeed-RL 9。

合计约 **1,950 候选 → 深读 200–300 条正文 → 预计新增 40–70 条**，
NPU 覆盖从 27 条到 70–100 条。

外推依据：首晚 2,547 条近更新 issue 产出 14 条收录（0.55%）；历史存量密度应更低
（大量已 closed/已修、已过时），取 0.2–0.4% × 18,656 ≈ 37–75 条。

T2 的 1,837 候选留作后续决定——首晚 ops-nn/ops-math 产出密度不低（11 条），
值得在本轮量出真实收录率后重新评估。

## 开工前提

A（悬空链接）与 B（模式分层）先完成。B 若后做，本轮灌入的 40–70 条会抬高
近重复对数量（当前 cos≥0.80 约 103 对），等于让 B 重做一遍。

GitHub 侧**不需要** `GITHUB_TOKEN`：环境里 `gh` 已登录（core 5000/h、search
30/min），`prefilter.py` 的 GitHub 路径走 `gh api`，认证留在 gh 自己的凭据存储里。
那份凭据带 `repo` scope（对全部仓库含私有的读写权），不该为了读公开 issue 把它导出
成环境变量传给采集器和扫描 agent——所以这里不导出，也不写进 `.env`。

（日常扫描要不要配 token 是**另一个独立决定**：扫描 prompt 里的 7 仓 Search 查询用
`curl` 读 `$GITHUB_TOKEN`，未设就按匿名 10/min 跑并 `sleep 7`。若要提速，建议新建一个
**无任何 scope** 的 PAT 写进 `/root/.hermes/profiles/third/.env`，不要复用 gh 那份
带 `repo` 写权的 token——扫描 agent 要读远端不可信的 issue 正文，给它写权没有必要。）

## 执行步骤

1. **清点**（已完成 GitCode 侧）：`prefilter.py list`，19 仓 37,479 条标题，约 400 个
   列表页、13 分钟、匿名无限额。GitHub 侧走 `gh api`，按**创建月份**切片
   （年切片会撞 Search 的 1000 条上限：vllm-ascend 2026 年 2,724 条、2025 年 1,601 条）。
2. **试点 150 条**：从 T1 候选池分层随机抽 150，走完整漏斗，量出真实
   候选→深读→收录两级转化率，再决定是否放量到全部 1,950。与此前每个阶段
   "先跑 20 条试点"的做法一致。
3. **判读与写入**：按 NPU_SCAN.md 现有纪律（查重用完整 URL、近 14 天日报、
   `catalog_tool.py find`；跨库同机制写 `cross`；范围变更要注明且不因此升置信度）。
   新条目一律 `pending_audit`。
4. **审计分批**：10-15 的审计处理 32 条用了 70.5 分钟。脚本超时现为 10800s，
   但为安全起见**每天写入不超过 20 条**，由既有 02:00 审计任务吸收；
   不要一次灌 70 条。批量判读只在 03:00–15:00 UTC 跑（网关额度约 16:00 UTC 重置），
   避开 23:00 推理扫描、00:30 训练扫描、02:00 审计。
5. **RAG 同步**：全部落盘后同步 `catalogs/`，过 36 条基准集与 63 条独立集两道门槛
   （Recall@5/@10 不得下降，MRR/nDCG 容差 0.02），并补本次变更的定向检查。

## 成本

| 项 | 量 |
|---|---|
| GitCode 列表清点 | ~400 请求（已完成） |
| GitHub 列表清点 | ~140 请求（`gh api`，月切片，约 8 分钟） |
| 标题判读 | ~100 网关调用（1,950 条，批 20/次） |
| 正文拉取 | ~600 HTTP 请求（200–300 条 × 1–3） |
| 条目写入与校验回路 | ~150–210 网关调用 |
| 新条目审计 | ~400 网关调用（沿用 daily_audit.sh） |
| **合计网关调用** | **~700** |

量级介于 A（~200）与 B（~4,400）之间，远低于 C（~9,000）。
主要成本是 200–300 份正文的阅读 token，不是网关调用数。

## 工具

- `prefilter.py list` — 全量拉取 issue 标题（不取正文），缓存 JSONL；
  `--only owner/repo` 试跑单仓，`--platform` 分平台。GitCode 走匿名 urllib，
  GitHub 走 `gh api`。两条路径都校验声明总数，GitHub 侧还按编号去重并报告重复数
  ——这个安全网抓到过一个真 bug：两个 `created:` 限定词会被 GitHub 静默忽略，
  导致每个月切片都返回全量（sgl-kernel-npu 重复 4,275 行），必须用 `created:A..B`。
- `prefilter.py filter --tier N --out <path>` — 应用校准过的宽口径正则，
  按仓输出候选数与候选清单。

两者都不写编目，也不碰 `collect.py` 的缓存与水位。缓存默认在 `/tmp/npu_backfill`，
不入 Git（37k 条上游标题不适合版本化）。
