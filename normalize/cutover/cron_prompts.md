# 切换后的 cron 任务（hermes profile `third`）

具体步骤只写在各仓库的说明文件里；cron 提示词只保留入口、Git 前置检查、照文件执行和交付要求，以后改规则只改文件。

## 推理扫描 `8645356b40ae`（0 23 * * * UTC，glm-5.3 / zai）

```
工作目录必须是 /workspace/inference_error_review。先读取 daily_scan.prompt.md，开始扫描前严格执行该文件第 3 节 Git 前置检查：当前 main 必须干净、成功 fetch 且 HEAD 与 origin/main 一致；不满足时停止扫描并在中文摘要里准确报告阻塞，不自动处理存量。通过后严格按该文件完成扫描、写日报、写入编目、生成视图、校验、提交与推送。编目条目只能通过 tools/catalog_tool.py 写入或修改，不得手写 YAML 或手改 views/ 与 FAULT_MASTER_REFERENCE.zh.md。提交、推送、远程核验任一步失败必须明确报告未同步，禁止强推或混入旧文件。等待限流只在前台分段 sleep：每次命令调用最多 sleep 60 秒、循环重查额度，禁止单段长 sleep，不要启动后台进程；最终回复必须是完整中文扫描摘要，即使之后收到后台进程通知也要重新给出完整摘要。
```

```bash
hermes -p third cron edit 8645356b40ae --prompt "$(cat <<'EOF'
<上面的文本>
EOF
)"
```

## 训练扫描 `1b8481fa4e2f`（30 0 * * * UTC，glm-5.3 / zai）

```
工作目录必须是 /workspace/training_error_review。先读 AGENTS.md、taxonomy.md 和 hermes/daily_scan.prompt.md；开始扫描前严格执行该文件第 4 节 Git 前置检查：当前 main 必须干净、成功 fetch 且 HEAD 与 origin/main 一致；不满足时停止扫描并在中文摘要中准确报告阻塞，不自动处理存量。通过后严格按该文件完成扫描、写日报、写入编目、生成视图、校验、提交与推送。编目条目只能通过 tools/catalog_tool.py 写入或修改，不得手写 YAML 或手改 views/ 与 catalog/FAULT_MASTER_REFERENCE.zh.md。提交、推送、远程核验任一步失败必须明确报告未同步，禁止强推或混入旧文件。等待限流只在前台分段 sleep：每次命令调用最多 sleep 60 秒、循环重查额度，禁止单段长 sleep，不要启动后台进程；最终回复必须是飞书扫描摘要，即使之后收到后台进程通知也要重新给出完整摘要。
```

## 新增：每日审计（02:00 UTC，脚本模式，不需要 agent）

两次扫描之后运行；先推理后训练，各自做 Git 前置检查、审计 `pending_audit` 条目、生成视图、校验、提交推送。

```bash
hermes -p third cron create '0 2 * * *' --name '故障编目每日审计' --no-agent \
  --script 'bash /workspace/inference_error_review/tools/daily_audit.sh; bash /workspace/training_error_review/tools/daily_audit.sh'
```

（`--script` 的具体语法以 `hermes cron create --help` 为准；需要 `JUDGE_API_KEY` 在该任务环境中可见。）
