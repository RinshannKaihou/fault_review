# 推理编目观测与评测盲区（2026-10-13 结构化切换后）

本文件记录推理侧测试资产假绿、离线比对工具失真与运行时假活盲区（补充冻结档案 `FAULT_MASTER_REFERENCE.zh.md` 附录 C/D，供每日扫描查重与 `fault-rag` 负结果检索）。

## IB.1 算子仓内部单测与比对脚本假绿（非推理算子缺陷）

- [cann/ops-math #4777](https://gitcode.com/cann/ops-math/issues/4777) / `#4778`（原 `INF-2565`）：仓库内部单测脚本 `compare_data.py` dtype 错配且空目录时直接报绿灯通过。
- [cann/hccl #1001](https://gitcode.com/cann/hccl/issues/1001)（2026-11-19）：自定义 AllGather / ReduceScatter 样例的 `VerifyResult` 只用 `abs(error) > 1e-5` 判失败，NaN 与任何数比较都为 false，因此单点 NaN 或全 NaN 输出仍打印 `Passed` 并返回 0；修复 PR [!4154](https://gitcode.com/cann/hccl/merge_requests/4154) 建议在误差判断前加 `std::isfinite`。属样例校验器假绿，不代表集合通信算子已损坏。
- [flashinfer-ai/flashinfer #9829](https://github.com/flashinfer-ai/flashinfer/issues/9829)（2026-11-21）：`fused_add_rmsnorm` 的 Trace 参考实现漏加更新后的 residual，参考值本身错；按参考对拍会给出错误结论，属评测/参考实现缺陷。
- 同类算子单测参考脚本或 `zip` 截断漏检问题（原 `INF-1708`、`INF-1761`、`INF-1908`、`INF-1989`、`INF-2003`）：均为测试基础设施缺陷，不代表线上推理算子损坏，统一记入本盲区（原条目已标 `retired`）。
- 2026-11-27 批次集中暴露的 UT 假绿：[cann/ops-math #5322](https://gitcode.com/cann/ops-math/issues/5322) / [#5323](https://gitcode.com/cann/ops-math/issues/5323) / [#5324](https://gitcode.com/cann/ops-math/issues/5324) / [#5325](https://gitcode.com/cann/ops-math/issues/5325) / [#5326](https://gitcode.com/cann/ops-math/issues/5326) / [#5327](https://gitcode.com/cann/ops-math/issues/5327) / [#5328](https://gitcode.com/cann/ops-math/issues/5328)、[cann/ops-nn #10343](https://gitcode.com/cann/ops-nn/issues/10343) / [#10383](https://gitcode.com/cann/ops-nn/issues/10383) / [#10386](https://gitcode.com/cann/ops-nn/issues/10386) / [#10388](https://gitcode.com/cann/ops-nn/issues/10388) / [#10391](https://gitcode.com/cann/ops-nn/issues/10391)、[cann/ops-transformer #8897](https://gitcode.com/cann/ops-transformer/issues/8897) / [#8898](https://gitcode.com/cann/ops-transformer/issues/8898) / [#8904](https://gitcode.com/cann/ops-transformer/issues/8904) / [#8908](https://gitcode.com/cann/ops-transformer/issues/8908) / [#8924](https://gitcode.com/cann/ops-transformer/issues/8924)：内核 UT 写出输出但不与 golden 比较、或忽略 `compare_data.py` 的失败退出码；#10343 的 GeluMul UT 把 golden 与自身比较，错误输出仍 PASSED；[#8955](https://gitcode.com/cann/ops-transformer/issues/8955) / [#8956](https://gitcode.com/cann/ops-transformer/issues/8956) 对整型 `expanded_row_idx` 用相对容差，off-by-one 被漏检。这些只说明测试判定不可信，不能据此推断算子已损坏。

## IB.2 运行时守护进程假活与后台线程死锁

- [cann/runtime #1357](https://gitcode.com/cann/runtime/issues/1357)（原 `INF-2496`）：`aicpu_sched` 工作线程初始化异常退出后进程状态仍存活，主机侧同步等待永久挂起（与附录 C.6 健康检查假阳性同属活性盲区）。
