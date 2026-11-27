# 框架外盲区

这里的失败**没有可分析的训练步或前向**，或检测器原理上全盲。不要把它们硬塞进 `HW`/`LOSS` 充条数。

## B.1 编排层假活

调度器显示 Running、GPU util > 0，step 不涨（卡在数据加载或集体通信）。见 `OBS.03`。隐藏态 / loss 探针没有新样本。

## B.2 单模型自参照看不见的带毒 ckpt

`HW.05`：发布权重已经是被 SDC 烤过的自洽流形。没有干净参照 ckpt，geometric / 谱 / loss 都说「正常」。这是训练编目对推理监控的主要输入。

## B.3 监控文件存在但为空

`OBS.01` / `OBS.02`。任何基于 JSONL 的检测器都没有输入。

## B.4 RL 里 KL 探针的符号抵消

`RL-KL.01`：K1 有符号，TIM 已经在扭曲目标时探针仍可贴零。不是「没故障」，是选错了观测。

## B.5 与推理目录的职责切分

| 盲区 | 不写进训练主编目的原因 |
|---|---|
| PD 分离健康检查绿、控制路径死 | 推理 serving，见推理附录 C.6 |
| 采样层 / 解码循环 post-forward | 推理结构性盲点 |
| 纯推理 KV 串扰 | 推理 `K*` / §8 |

训推 logprob 不一致（TIM）**要写在训练目录** `RL-RO`：它改的是 RL 目标，不是某个用户请求的 serving 输出。

## B.6 msprobe 精度对拍把反号 Inf 归一化成相同值

`Ascend/msprobe #655` + 修复 [!1123](https://gitcode.com/Ascend/msprobe/merge_requests/1123)（open）：`handle_inf_nan()` 只检查 Inf/NaN 位置，+Inf 与 -Inf 位置匹配时双双替换为 0——**正负无穷符号相反本应判不一致，对拍却显示一致**（CPU 复现：actual=[1,inf] vs expected=[1,-inf] 返回 `(False,'')` 且两输入都被改成 [1,0]）。精度对拍工具把「符号翻转级」的数值分歧归一化成相等，训练侧 NPU/CPU 对比的绿灯可能来自工具缺陷而非模型一致。观测/评测工具假一致，按口径记盲区不占故障条目；修复 PR 要求在归一化前先拒绝符号不匹配的 Inf。

2026-10-27 回访：上游给出第二版修复 [!1125](https://gitcode.com/Ascend/msprobe/merge_requests/1125)（open，2026-10-26）——改为分别用 isposinf/isneginf 校验正负无穷位置、全部通过后才置 0，任一不匹配返回 'Nan' 由上层报错，并同步更新报错文案；与 !1123 并行未合入。

## B.7 mssanitizer UB 检测以物理上限 248KB 为界，漏检 SIMT DCache 越界

`Ascend/mssanitizer #239`（open，2026-10-27，A5 + CANN 9.3.0 weekly）：SIMT VF 内 UB 越界场景（官方示例 cann/asc-devkit examples/05_simd_simt_hybrid/01_trouble_shooting/simd_simt_ub_usage_limit）实测工具未能检出。工具助手分析：A5 的 UB 检测方案以物理内存上限 248KB（256KB−8KB 预留）为有效范围；但混合场景下 SIMT 需要用到 DCache（GM 的缓存），越界到 DCache 属非法行为、被踩踏后造成难以定位的精度问题——检测边界按物理 UB 上限划，覆盖不了 DCache 区。观测/检测工具漏检面按口径记盲区不占故障条目；同工具同日新增 `--error-exitcode` 选项（!434，open）支持按检出结果返回退出码。

## B.8 msprobe nan_check 落盘 tensor 命名错位（调用次数与参数序号）

`Ascend/msprobe #676` / `#677`（open，2026-11-07/08，msprobe 26.0.0 e7bc732，vllm 0.30.0+empty + vllm_ascend 0.19.1rc2）：nan_check 的 is_nan 模式在 api 输出含 NaN 时落盘 tensor 文件，但命名与真实调用错位——#676：落盘文件名中的 `{API调用次数}` 从 0 重新计数，`Torch.vllm.all_reduce.2.forward` 的 dump data_name 指向的 pt 文件看起来属于 `all_reduce.0`；#677：图模式命名格式变成 `{api_type}.{api_name}.{input/output}.{参数序号}{API调用次数}`（与 tensor 场景字段序不一致），且 output 命名少一位 `{参数序号}`（如 `Torch.vllm.all_reduce.output.0.pt` 无调用序号）。NaN 取证时按文件名对回 API/轮次会定位到错误的调用点，污染「哪一层先出 NaN」的结论。观测工具落盘命名缺陷按口径记盲区不占故障条目。

## B.9 mssanitizer 检测流程缺陷：MC2 保护区初始化/IPC 边界与 fixpipe quantPre 误报

`Ascend/mssanitizer #251`（closed，2026-11-09）：MC2 检测的保护区初始化、IPC 资源边界及 ELF 加载处理流程存在问题（正文未展开机制细节）。`#252`（closed，2026-11-09，A5）：fixpipe quantPre 位宽解码漏配导致**误报**——检测器把合法的 fixpipe 量化前置操作报为越界/异常。两类都是检测工具自身缺陷：#252 假阳性会侵蚀告警可信度（狼来了效应），#251 流程缺陷可能漏检；正文均为标题级、修复已闭环（closed），按口径记盲区跟踪，不转述为模型计算已损坏。

## B.10 CANN runtime mstx 打点通道静默全失效（版本组合回归）

`cann/runtime #1359`（2026-11-10，open）：合入 PR !6030（msproftx 资源命名）后的 runtime 配未适配的旧 mstx 客户端时，`mstxRangeStartA` 全部返回 0（MSTX_INVALID_RANGE_ID）、`host/data/` 不落 tx 原始数据、runtime 侧无任何报错日志。代码级根因：mstx 客户端对 `InitInjectionMstx` 结果全有或全无——失败即把 mark/range 指针全部置 0 并钉 FAILED；profapi 侧 `GetModuleTableFunc` 按对端枚举上限遍历、未知模块（CORE_MEM）走 default 判 MSTX_FAIL，触发全清空。修复 !6213（merged，按本仓实现清单遍历注册表）。profiling 打点整体无数据且无告警属观测通道静默失明，按口径记盲区不占故障条目。

## B.11 torchair eager-override 对融合 kernel 喂零，精度对拍失去意义

`Ascend/torchair !3823`（open，2026-10-09，基于 #3816）：AscendC 后端复用 inductor lowering 后，融合 kernel 的 aten 子图只能在 codegen 期用启发式从 `IR.origins` + `read_writes` 事后重建；`aggregate_origins` 跨 kernel 累积未 realize 输入、`origin_node` 多为 None 时退化为 `origins[0]` 猜测、matmul→epilogue 的寄存器内传递对 `read_writes` 不可见。`TORCHINDUCTOR_ASCENDC_EAGER_OVERRIDE` 还原不出的 fragment 输入会 `fabricate`（喂零），`eager_fn` 拿零当真实张量重放——精度对拍显示一致，但参考语义已错。PR 用 `traced_graph` 在 lowering 期记录精确 aten 子图，把“事后启发式”换成“编译期记录”。观测/验证工具假一致，按口径记盲区不占故障条目。

## B.12 算子仓内部单测参考脚本（golden / UT）与 msprobe 对拍溢出假绿

`cann/ops-nn #8064`（`tests/assets/golden.py` 标量 `IndexError` 与广播除数错，原 `TRN-0613`）、`foreach log` 族 UT 漏校验（原 `TRN-0600`）以及 `Ascend/msprobe` Cosine 比对大数溢出/`nan_check` 反向落盘方向错（原 `TRN-0576`、`TRN-0628`）：均为测试资产或离线诊断工具自身缺陷，模型训练前反向计算本身未受损，统一归入盲区记录（原条目已 `retired`）。另 `cann/runtime #1357`（`aicpu_sched` 死 worker 致同步挂死，原 `TRN-0713`）并入 `B.1` 编排层假活。

## B.13 mssanitizer memcheck 跨 kernel GM 影子残留与分步单元素写致 out of bounds 误报

`Ascend/mssanitizer #272`（open，2026-11，910B2 / CANN 9.3.0）：memcheck 的 GM per-core 归属影子在 kernel 结束后未归还，下一个 kernel 复用同一段 GM 时把合法写报成 `out of bounds`。最小场景单次 `aclrtMalloc` 4096B：kernel1 由 8 个 block 各写 512B，kernel2 由 1 个 block 重写整段，主机侧 1024 个 float 全为 1.0（VERIFY_OK），kernel2 仍报 `out of bounds of size 3584`（=4096−512，即 kernel1 其它 7 个切片）；PyTorch 内存池复用场景第 17 次分配报 16 条（2×3584 + 14×3072），数值全部正确。`PYTORCH_NO_NPU_MEMORY_CACHING=1` 仅规避复用，`--log-level=error` 会关掉整个越界检查，都不是修复。修复 MR [!498](https://gitcode.com/Ascend/mssanitizer/merge_requests/498) / [!499](https://gitcode.com/Ascend/mssanitizer/merge_requests/499)（open，2026-11）针对影子归属残留。

`#273`（open，2026-11，A3 910_93 / CANN 9.3.0.P009 / mssanitizer 26.3.0）：单个 kernel、单次 `aclrtMemcpy` 分配、无内存池复用，kernel 以每 36B 一跳做分步单元素写（合法索引 [0,tokensNum)）时，8 条报错地址全部落在 256B 输出张量内部（0x15000–0x150fc），前后各 4KB 守卫逐字节零改写；同进程 InplaceIndexAdd 再报 8 条 size 68。真越界阳性对照仍能报出，属地址归属判定误报。两类都是检测工具假阳性：告警会侵蚀越界检查可信度，按口径记盲区，不转述为模型计算已损坏。
