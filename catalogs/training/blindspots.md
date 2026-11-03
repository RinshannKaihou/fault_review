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
