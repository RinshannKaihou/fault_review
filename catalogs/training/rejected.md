# 被拒 / 降级的候选

负结果。每日扫描和后续调研先查这里，避免再走一遍。

| 候选 | 拒绝理由 |
|---|---|
| 把 OOM 当故障模式 | 显存耗尽是容量/配置，不是静默正确性。记「已扫描未编目」。 |
| 把 NCCL timeout 当故障模式 | 活性失败，有明确信号。除非 timeout 之后 ckpt 已带毒且被当成成功 resume。 |
| 把「学习率太大」当独立故障 | 没有机制级出处时只是超参。有出处的尖峰机制见 `LOSS.01` / `LOSS.02`。 |
| 把 Apex+RMSNorm 崩当作静默故障 | 本实验室已核实：装 Apex 会让 mcore 强制 `FusedLayerNorm` 并对 RMSNorm **assert**。这是 fail-fast，写在 `foundation/notes/local_megatron.md`，不进主编目。 |
| 把 CE logits OOM 当静默故障 | `vocab_parallel_cross_entropy` 的 `.float()` 一次性分配，TP=1 时硬 OOM。容量问题。 |
| 推理 572 条整库搬迁 | 训练目录只收训练期机制。交叉条目写指针。 |
| `finite_value_sdc_dominance` 当可注入故障 | 这是统计 META（~99% SDC 是有限值）。写在 `NUM.04` 的来源里，不单独占检测目标编号。 |
| ms-swift #10078（2026-09-16）：GKD `--beta` 广义 JSD 在端点跳变、β 近端点退化为「信号强度旋钮」 | 维护者反证 + 文献反证：实现严格匹配 GKD 论文（arXiv:2306.13649）Eq.(1) 与 TRL 参考实现，论文本身注明 JSD 梯度只在**方向**上随 β→0/1 逼近 FKL/RKL、幅度按 ~β 缩放；β=0/1 端点是补充定义。「文档说插值」与「内公式极限为 0」的张力是文档表述问题，非静默错值故障。报告者的 27B β=0.9 实测（loss ~1/13）与该文档化语义一致。 |
| [DeepSpeed #8586](https://github.com/deepspeedai/DeepSpeed/issues/8586)（远程回填曾误列 OPT.16：重复清梯度导致静默漏更新） | 原 issue 明写 **“Numerics are unchanged”**：ZeRO-3 每个 sub-group 重复遍历全部参数并清理梯度引用，实测差异是优化器 step 耗时（GLM-4.5-Air 约 400→202 ms），不是已证实的梯度错值或漏更新；不得据远程速记推断静默训练错误。2026-09-24 跨分支合并时拒绝入主编目。 |
| [DeepSpeed #8608](https://github.com/deepspeedai/DeepSpeed/issues/8608)（universal ckpt 转换只认 Adam state 键，Muon `momentum_buffer` 转换即 `KeyError: 'exp_avg'` 崩） | fail-fast：转换在写出任何文件**之前**抛 KeyError（`ds_to_universal.py:149`/`:191`），无静默错状态产出；即使担心错缩放，报告者的实现选择是「拒绝转换」而非静默 carry。功能缺口（其他优化器 ckpt 无法转 universal）真实但不属静默正确性故障。2026-09-26 每日扫描拒收。 |
| [TRL #8352](https://github.com/huggingface/trl/issues/8352)（SDPO `--scale_rewards false` 字符串落入 group scaling）与 [#8344](https://github.com/huggingface/trl/issues/8344)（SDPO IterableDataset 不重复 `num_generations`） | 维护者 2026-11-14 明确以「批量代码扫描、没有真实运行命中」关闭（not_planned）并拒绝该类报告；#8344 的默认路径还是 TypeError fail-loud，只有 `dispatch_batches=False` 才静默不分组。2026-11-16 扫描拒收，不新建条目。 |
| [TRL #8356](https://github.com/huggingface/trl/issues/8356)（DPO/KTO `ref_model=None` 时按 config `_name_or_path` 重新加载参考模型，丢掉内存中对模型实例的改动）与 [#8359](https://github.com/huggingface/trl/issues/8359)（GRPO 带 `tools` 时 completion 被拆出 `reasoning_content`，`think_format_reward`/`reasoning_accuracy_reward` 恒返回 0） | 同一维护者 2026-11-14 以同一条理由关闭（not_planned）：「批量代码扫描开 issue、不是真实运行命中」，与 #8352/#8344 同批。沿用 2026-11-16 判定口径，不新建条目；若将来出现真实 run 复现或维护者自认，再按新证据重评。 |

| [cann/ops-transformer #6625](https://gitcode.com/cann/ops-transformer/issues/6625)（原 `TRN-0414` / `INF-2167`：SparseFlashMlaMetadata 丢弃 sequsedQ） | 上游报告者 2026-11-15 改标 `[已撤回]`：自建 C++ 探针脚本未将 `suVal` 与 `su=1` 开关联动，全部传入 `nullptr`；修正探针后分核与越界校验均正常。已将 `TRN-0414` 与 `INF-2167` 同步退役。 |
| 纯编译融合未命中或回退慢速备用核（如 [cann/ops-nn #8308](https://gitcode.com/cann/ops-nn/issues/8308)、[Ascend/pytorch #5855](https://gitcode.com/Ascend/pytorch/issues/5855)） | 计算精度与梯度完全无损，仅耗时或融合覆盖率退化，按纯性能问题退役（`TRN-0647`、`TRN-0627`），不占静默正确性故障编号。 |
| C++ `aclnn` 两段式直调契约外负向用例（如 `eps=-1`、`groupList` 负数、非大模型传统 CV/ML 算子） | 正常 PyTorch / LLM 训练路径在前端或契约层不可达的 GIGO 负向输入及脱离大模型训练路径的传统算子，2026-11-17 批量退役并不再收录。 |

种子期没有「引用被证伪后又恢复」的条目。出现时按日期追加（须附完整规范 URL）。
