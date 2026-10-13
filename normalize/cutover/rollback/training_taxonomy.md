# 训练故障 taxonomy

本文件冻结**类目、编号、收录边界**。基础调研可以微调类目说明，但不要再发明第二套编号。

## 设计选择（已定）

| 选择 | 取值 |
|---|---|
| 编目拓扑 | **一份** `FAULT_MASTER` + `stage` 字段，不按阶段拆三份 |
| 栈范围 | 实验室主栈（Megatron）+ 社区主流（DeepSpeed / torchtitan / TRL / veRL / OpenRLHF / ms-swift） |
| 收录 | 静默或延迟可见的正确性 / 稳定性。fail-fast 默认不进主编目 |

编号是「领域前缀 + 两位序号」，例如 `NUM.01`、`CKPT.03`、`RL-RO.02`。  
**不要**再使用推理侧的 A / 1–15 / K 三层编号。

`stage` 取值：`pretrain` · `sft` · `rl` · `shared`。一条可以是 `pretrain/sft`。

## 类目

| 前缀 | 领域 | 默认 stage | 收什么 |
|---|---|---|---|
| `HW` | 硬件 / 训练期 SDC / ECC / Xid | shared | 算对形状、结果错、无 Xid；带毒权重进入 ckpt |
| `NUM` | 数值 / 混精 / loss scale / FP8 / 非确定性 | shared | 有限值静默错、scaler 骗过、batch 归约序 |
| `PAR` | 并行正确性与集体通信 | shared | TP/PP/DP/CP/EP/SP 静默不一致、集体通信错值 |
| `CKPT` | 检查点 / resume / optimizer / ZeRO | shared | 加载成功但权重/矩错位、reshard 换专家 |
| `DATA` | 数据管线 / shuffle / packing / mask / tokenizer | shared | 各 rank 看到不同序、mask 把 prompt 当 label |
| `OPT` | 优化器 / lr / clip / 矩 | shared | Adam 矩与权重错配、clip 把 SDC 吃掉 |
| `ACT` | 重计算 / offload / 激活检查点 | shared | recompute 图与前向分叉 → 静默错梯度 |
| `KER` | 训练 kernel（FA bwd、TE、Triton、fused CE） | shared | 合法配置下算出错值 |
| `MOE` | 路由塌缩 / EP / dropless | pretrain | 专家死掉、router 与权重错位 |
| `LOSS` | loss spike / 发散 / 长跑静默退化 | pretrain | 曲线还能跑、质量已毁 |
| `SFT` | 模板 / packing 串文档 / 标签泄漏 | sft | 下一条文档当本条 completion |
| `RL-RWD` | reward / 格式黑客 / RM 漂移 | rl | 奖励涨、能力掉 |
| `RL-ADV` | advantage / GAE / GRPO 组统计 | rl | std=0、难度偏置、组内无对比 |
| `RL-KL` | KL / entropy 塌缩 | rl | k1/k3 看不出、策略已死 |
| `RL-RO` | rollout ↔ train 不一致（TIM） | rl | 同权重不同 logprob |
| `OBS` | 观测性骗过 | shared | job 绿、监控 0 字节、健康检查假阳性 |

## 收录边界

**进主编目**

- 无 crash / 无 assert，但 loss、grad、ckpt、reward、rollout 已错。
- crash 发生时坏 ckpt 已经写出，或恢复后静默接上坏锚点。
- 健康检查或监控显示正常，实际训练步已偏离。

**不进主编目（可记「已扫描未编目」）**

- 纯 OOM、NCCL timeout、启动期 `ValueError`、缺依赖。
- 纯性能、文档、CI flake。
- 「超参没调好」而没有机制级来源。

**进 `rejected.md`**

- 引用是假的、与已有条目完全重复、或文献反证了该机制。

**进 `blindspots.md`**

- 没有训练步 / 没有前向可供分析的失败（编排层假活、队列吞 job）。

## 与推理目录交叉

| 现象 | 写哪里 |
|---|---|
| 训练期 SDC 污染权重 | 本仓库 `HW` |
| 带毒 ckpt 被 vLLM 加载 | 推理目录已有 `1.6 training_sdc_checkpoint_inherit`；本仓库只留指针 |
| FA / Triton 内核静默错值 | 训练路径 → 本仓库 `KER`；纯 serving 路径 → 推理目录 |
| 训推 logprob 不一致 | 本仓库 `RL-RO`（这是 RL 系统故障，不是推理 serving 故障） |

指针格式：`交叉：inference_error_review 1.6 \`training_sdc_checkpoint_inherit\``。

## 置信度

| 值 | 含义 |
|---|---|
| `verified` | 机制已对过代码或本实验室复现 |
| `documented` | 症状与修复理由有公开出处，未逐行核代码 |
| `speculative` | 合理假设，未确证；种子编目尽量不用 |

## 覆盖（Cov）

- `NEW`：新机制或新的门控变量
- `var:snake_case`：已有机制的另一实例 / 另一栈

## 计数

只数主文档里匹配 `^### [A-Z][A-Z0-9-]*\.\d+ \`` 的标题。  
`rejected.md` / `blindspots.md` / META 说明不计条。
