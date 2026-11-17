# 推理编目观测与评测盲区（2026-10-13 结构化切换后）

本文件记录推理侧测试资产假绿、离线比对工具失真与运行时假活盲区（补充冻结档案 `FAULT_MASTER_REFERENCE.zh.md` 附录 C/D，供每日扫描查重与 `fault-rag` 负结果检索）。

## IB.1 算子仓内部单测与比对脚本假绿（非推理算子缺陷）

- [cann/ops-math #4777](https://gitcode.com/cann/ops-math/issues/4777) / `#4778`（原 `INF-2565`）：仓库内部单测脚本 `compare_data.py` dtype 错配且空目录时直接报绿灯通过。
- 同类算子单测参考脚本或 `zip` 截断漏检问题（原 `INF-1708`、`INF-1761`、`INF-1908`、`INF-1989`、`INF-2003`）：均为测试基础设施缺陷，不代表线上推理算子损坏，统一记入本盲区（原条目已标 `retired`）。

## IB.2 运行时守护进程假活与后台线程死锁

- [cann/runtime #1357](https://gitcode.com/cann/runtime/issues/1357)（原 `INF-2496`）：`aicpu_sched` 工作线程初始化异常退出后进程状态仍存活，主机侧同步等待永久挂起（与附录 C.6 健康检查假阳性同属活性盲区）。
