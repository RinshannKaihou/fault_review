# LLM 生命周期二分因果拓扑（`topology/`）

将 `vocab.yaml`（v1.3）中的 **26 个原因子系统（`subsystem`）** 与 **17 个状态/观测载体（`effect_object`）** 形式化为跨 6 个生命周期阶段的双平面二分因果图，并在零第三方依赖下对全库快照（`data/entries.jsonl`）执行拓扑映射与数据质量体检。

## 文件说明

| 文件 | 说明 |
|---|---|
| `causal_topology.json` | 拓扑规范 v1.0：6 个宏观阶段、17 个带拓扑秩（`rank`）与二分探针（`probe`）的状态载体、50 条前向骨干 DAG 边、7 对同层紧耦合对、5 条跨步/跨阶段显式反馈环（`LOOP-1..5`）、26 个子系统的平面归属与直连/跨层注入边界 |
| `build_graph_data.py` | 投影构建器：不改上游 YAML，自动去重、按拓扑秩归一化 233 条因果倒置序列、将 74 条推理库纯训练算子动态挂载到训练阶段，输出 `web/dist/data/causal_graph.json` |
| `audit_topology.py` | 纯 Python 标准库全库映射体检工具（输出 JSON 指标，支持 `--check` 门禁校验词表对齐） |
| `test_topology.py` | 拓扑规范不变量测试（前向骨干严格拓扑秩单调递增、逆序自动归一化、动态阶段挂载与词表 1:1 对齐） |
| `REPORT.md` | 2026-11-17 快照（`2,508 INF + 764 TRN = 3,272` 条）拓扑定稿与七维映射体检完整报告 |

## 运行方式

```bash
# 1. 运行拓扑规范不变量与快照守恒单元测试（纯标准库，约 0.1s）
python3 -m unittest discover -s topology -t . -v

# 2. 执行全库映射体检并校验 vocab.yaml 枚举对齐门禁
python3 topology/audit_topology.py --check --write-json /tmp/topology_health.json
```

> 本模块严格遵守主仓库只读纪律：只读 `data/entries.jsonl` 与受控词表，不修改 `catalogs/` 或上游两个编目仓库。
