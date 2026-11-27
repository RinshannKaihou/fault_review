# 推理编目被拒 / 退役候选记录（2026-10-13 结构化切换后）

本文件记录 2026-10-13 冻结 `FAULT_MASTER_REFERENCE.zh.md` 之后在每日扫描中被拒收、证伪撤回或因越界退役的候选（带完整规范 URL，供每日扫描查重与下游 `fault-rag` 作为负结果检索）。

## R.1 上游已证伪 / 报告者撤回的假阳性

- [cann/ops-transformer #6625](https://gitcode.com/cann/ops-transformer/issues/6625)（原 `INF-2167` / `TRN-0414`）：报告者复核确认 C++ 探针脚本未将 `suVal` 与 `su=1` 开关联动导致恒传 `nullptr`，算子 `sequsedQOptional` 分核与越界校验实际工作正常，标题已改 `[已撤回]`。
- [Ascend/msmodelslim !1164](https://gitcode.com/Ascend/msmodelslim/merge_requests/1164)（原 `INF-2491`）：纯 README 文档中 `transformers` 示例版本号笔误修复，无任何推理代码变更。

## R.2 纯性能退化 / 硬件能力回退（输出无损）

- 910B/910C 无原生 FP8 GEMM 时按文档将 FP8 权重反量化为 BF16 运行（[vllm-ascend #21157](https://github.com/vllm-project/vllm-ascend/issues/21157)，原 `INF-2438`）、`max_num_seqs` 非 8 对齐部分档位退回 eager（原 `INF-2018`）、异步 EPLB 负载重排停摆但计算无损（[vllm #66144](https://github.com/vllm-project/vllm/issues/66144)，原 `INF-2458`）等：输出分布与数值无损，不占静默正确性故障编号。（注：投机解码中因 draft 权重覆盖/未初始化/位置错配导致的底层计算错误，虽被 target 验证兜底外显为接受率暴跌，仍保留在主编目。）

## R.3 显式崩溃 / 超时退出（Fail-Loud）与纯训练反向缺陷

- 无静默错误输出、直接触发设备异常或集体通信超时退出的报告（如 [triton-ascend #3043](https://github.com/triton-lang/triton-ascend/issues/3043) `507035`、[cann/ops-nn #8642](https://gitcode.com/cann/ops-nn/issues/8642) UB 超限直崩、[cann/hcomm #1278](https://gitcode.com/cann/hcomm/issues/1278) `AllToAllV` 600s 超时）：不收进静默故障编目。
- 纯训练期反向梯度算子（`*Grad` / `*Backward`）、优化器更新及 RL rollout↔trainer 重要性采样偏差（TIM）：归属 `training_error_review`，不在推理编目重复占号（原 `INF` 误收的 27 条纯训练条目已于 2026-11-17 迁入或归并至训练库）。

## R.4 C++ `aclnn` 契约外负向测试（GIGO）与非大模型算子

- 绕过 PyTorch/推理引擎前端直接向 C++ 两段式 `aclnn*` 传入契约外非法参数（如 `eps < 0`、`RoPE mode=2^32`、`slotMapping=[-1]`）以及脱离大模型推理路径的传统 CV/3D/RNN 算子（如 `MaxPool3D`、`Winograd`、`Histc`、`LstmCell`）：不纳入 LLM 推理静默故障主编目。
- [cann/ops-math #5305](https://gitcode.com/cann/ops-math/issues/5305)（2026-11-26）：`col2_im` 在 910B2 上 48 个输出中 10 个正确、32 个错值、6 个 NaN，属 `conv_transpose` 类传统 CV 重建算子、与 LLM 推理路径无关，按本条范围规则不占 `INF-*` 编号。
