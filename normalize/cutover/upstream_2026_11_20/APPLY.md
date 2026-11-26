# 上游编目仓库（`inference_error_review` / `training_error_review`）手动落地与恢复指南

> 本目录由 `fault_review` 生成。按 `fault_review/AGENTS.md` 只读纪律，本会话未向两个上游编目仓库写入任何文件；Hermes `third` 运行时脚本、环境变量与 Cron 提示词已在本机就位（三个定时任务当前保持 `paused`）。请按以下 4 步完成上游对齐与恢复调度。

---

## 第 1 步：先快进两个上游仓库对齐远端最新提交（必做）

当前本机两个上游仓库落后于远端（推理远端 `5568770`，训练远端 `30b173b`）。如果不先快进，Hermes 前置检查会因 `HEAD != origin/main` 直接报 `BLOCKED`。

```bash
git -C /workspace/inference_error_review pull --ff-only origin main
git -C /workspace/training_error_review pull --ff-only origin main

# 核对工作区干净且与远端一致
git -C /workspace/inference_error_review status --short
git -C /workspace/training_error_review status --short
```

---

## 第 2 步：（推荐）应用上游校验、审计去同源与说明书同步

本目录提供了幂等更新脚本 `apply_upstream_updates.py`（对应源码级审阅参考见 `upstream_improvements.patch`），一并完成：
1. **`tools/catalog_tool.py`**：`check_entry` 增加 `effect.objects` 与单组件 `cause.objects.subsystem` 去重检查，并内嵌 `causal_topology.json` 的 17 个状态载体拓扑秩与 5 条受控回边校验（`new` 新建条目时硬拦截因果逆序提示调换顺序；存量 `validate` 仅记 warning 不破坏历史兼容）。
2. **`tools/catalog_audit.py`**：确保 `API` 读取 `JUDGE_API_URL` 环境变量；同时将盲审 `BASE` 法官中的 `deepseek-v4.1-flash`（与当前两个扫描任务的执笔模型同源）替换为网关已标定的高一致率异源模型 `doubao-seed-2.1-pro`（96% 一致率，`reasoning_effort: low`），消除“自己写、次日自己参审”的同源偏置。
3. **`catalog/vocab.yaml`（升至 `v1.4`）**：增补 `torchair`、`msprobe`、`mssanitizer`、`msmodelslim`、`mindie-llm`、`vllm-ascend`、`sgl-kernel-npu`、`triton-ascend` 8 个 NPU 常用栈名，消除 `validate` 常驻的 70+ 条半受控 `stack` 告警。
4. **同步扫描说明书与 NPU 采集脚本真源**：将更新后的 `inference_daily_scan.prompt.md`、`training_daily_scan.prompt.md` 以及支持分 Scope 消费水位/`GITHUB_TOKEN` 注入的 `fault_scan_npu_entry.py`、`collect.py`、`test_fault_scan_npu_entry.py` 同步进仓库版本控制。

先做 `--dry-run` 预览，确认无误后执行 `--apply`：

```bash
python3 /workspace/RAG/fault_review/normalize/cutover/upstream_2026_11_20/apply_upstream_updates.py --dry-run
python3 /workspace/RAG/fault_review/normalize/cutover/upstream_2026_11_20/apply_upstream_updates.py --apply
```

随后在两个上游仓库分别运行单测、重建视图、校验并提交推送：

```bash
for r in /workspace/inference_error_review /workspace/training_error_review; do
  echo "=== $r ==="
  (cd "$r" && \
    python3 -m unittest discover -s tools -p 'test_*.py' -v && \
    python3 tools/catalog_tool.py build && \
    python3 tools/catalog_tool.py validate && \
    git status --short)
done
```

核对 diff 无误后分别提交推送（推完确保 `HEAD == origin/main`）：

```bash
cd /workspace/inference_error_review
git add catalog/vocab.yaml daily_scan.prompt.md tools/ views/
git commit -m "scan/audit: enforce causal effect order, dedup enums, decouple base judge, sync NPU cursor"
git push origin HEAD:main

cd /workspace/training_error_review
git add catalog/vocab.yaml hermes/daily_scan.prompt.md tools/ views/
git commit -m "scan/audit: enforce causal effect order, dedup enums, decouple base judge, sync prompt"
git push origin HEAD:main
```

---

## 第 3 步：恢复前干跑核验（无副作用预检）

确认两个仓库与 `origin/main` 完全一致后，测试一次前置入口与每日审计空跑：

```bash
bash /root/.hermes/profiles/third/scripts/fault_scan_npu_inference.sh | head -c 500 && echo
bash /root/.hermes/profiles/third/scripts/fault_scan_npu_training.sh | head -c 500 && echo
bash /root/.hermes/profiles/third/scripts/fault_catalog_daily_audit.sh
```

预期：前两条输出包含 `"complete":true`、`"delta_candidates"` 与 `"scope_window"` 的紧凑 JSON；第三条输出 `nothing pending; no commit`（`✅ inference` / `✅ training`）。

---

## 第 4 步：恢复三个 Hermes 定时任务

```bash
hermes -p third cron resume 8645356b40ae
hermes -p third cron resume 1b8481fa4e2f
hermes -p third cron resume 1fc648dfb5a0
hermes -p third cron list
```

次日早间检查一次：
- `23:00 UTC` 推理扫描与 `00:30 UTC` 训练扫描日报是否显示 `authenticated: true`（Search 30/min）、是否记录 `fault-rag` 查重结论及推进 `scope_cursors`；
- `02:00 UTC` 每日审计是否正常将新 `pending_audit` 条目定稿为 `active` 并推送。
