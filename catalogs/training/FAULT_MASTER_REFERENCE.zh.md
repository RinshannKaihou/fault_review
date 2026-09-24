# LLM 训练故障模式总参考（Master Fault Reference）

> **本文档是什么**：预训练 / SFT / RL 已编目故障的**单一自包含参考**。对照仓库是 `../inference_error_review`（推理 serving）。本文档只收**训练期**机制；与推理交叉的条目写指针，不复制正文。
>
> **当前阶段**：种子编目始于 2026-08-17，现已进入持续每日扫描。条数以正文 `### PREFIX.NN` 标题为准，禁止硬编码。

---

## 如何使用

| 你的问题 | 去哪里 |
|---|---|
| 编过这条吗？ | `catalog/index/known_names.txt` / `known_issue_ids.txt` / `known_arxiv_ids.txt` |
| 类目怎么分？ | [`taxonomy.md`](../taxonomy.md) |
| 以前否过吗？ | [`rejected.md`](rejected.md) |
| 框架外盲区？ | [`blindspots.md`](blindspots.md) |
| 历史摸底怎么挖的？ | [`../foundation/REPORT.md`](../foundation/REPORT.md) |

---

## 滚动汇总

> 2026-08-17：种子编目初版。矿源：训练 SDC / loss-spike / TIM / GRPO 文献 + DeepSpeed / Megatron / OpenRLHF 已核实 issue + 本地 Megatron 实验。**64 条**。
> 2026-08-18：每日扫描首跑（GitHub 7 仓库 + arXiv 4 查询）。新增 CKPT.06 / DATA.06 / OBS.04，补 RL-RO.01 / RL-RWD.03 来源。**67 条**。
> 2026-08-19：每日扫描。新增 NUM.05 / NUM.06 / PAR.04 / LOSS.05 / CKPT.07 / CKPT.08 / RL-RO.07，补 RL-RO.01 来源（MIPI/MIPU）。**74 条**。
> 2026-08-20：每日扫描。新增 PAR.05 / CKPT.09 / OPT.04 / SFT.05 / RL-KL.04（Megatron FSDP 梯度 2^N 翻倍、MXFP8 ckpt round-trip、clip 洗坏梯度、左 pad 打分位置错、变体 trainer loss 副本漂移），补 RL-RO.01（2602.01826）/ RL-RO.05（verl #7463）/ KER.03（Megatron PR #6654）/ CKPT.07（verl #7462/#7463/#7476）来源。**79 条**。
> 2026-08-21：每日扫描。新增 PAR.06 / CKPT.10 / ACT.03 / SFT.06 / RL-RO.08（verl `fsdp_size=1` mesh 形状/rank 数混淆致梯度同步静默关闭、trl SDPO EMA teacher 在 ZeRO-3 分片后惰性初始化读空 shard、Megatron mHC attention CUDA-graph split 静默改默认 capture range、OpenRLHF RewardDataset 截断内生造 identical pair、verl disaggregated 权重同步漏传 LoRA adapter），补 RL-RO.01（2606.09821）/ RL-ADV.05（OpenRLHF #1310）/ CKPT.07（trl #6791）来源。**84 条**。
> 2026-08-22：每日扫描。新增 OBS.05 / RL-RWD.04（verl SFT 验证 loader 复用训练 batch 几何+双 drop_last → 0 batch → `val/loss=NaN` 且 exit 0、verl 工具 schema 窄 Pydantic 载体 round-trip 整键丢弃 JSON Schema 约束），补 RL-RO.01 来源（FP8-RL 2601.18150、DVP 2512.23087）。**86 条**。
> 2026-08-23：每日扫描。新增 PAR.07 / KER.04 / RL-ADV.06 / OBS.06（DeepSpeed Ulysses 进程级 `_ulysses_num_kv_heads` 锁死第一模型头数、ROCm 7.14 wheel 换 7.2 即 loss parity 静默失败、trl GRPOWithReplayBuffer 1-D/2-D 形状契约脱节致 replay 静默退化、trl RL trainer 指标路径不掩 padding 致 KL/entropy/reward 诊断系统性错值）。**90 条**。
> 2026-08-24：每日扫描。新增 SFT.07 / SFT.08（trl 官方示例 LoRA target_modules 对 NemotronH 混合架构静默欠匹配仅挂 4/56 层、DPO ipo/sigmoid_norm 长度归一化分子分母掩码错位致 loss/梯度静默错值），补 OBS.06 来源（trl #6803 Online DPO 实例独立 issue）。**92 条**。
> 2026-08-26：每日扫描（08-25 未运行，窗口与 08-24 报告部分重叠，已按 issue 号去重）。新增 KER.05（verl OPD 多教师蒸馏 v0.9.0→HEAD 同脚本同种子 distillation loss 自 step 1 起静默分叉，B300/寒武纪 MLU 双栈复现；OPD 目录内仅启动脚本变更，数值漂移藏在共有路径）。**93 条**。
> 2026-08-27：每日扫描。**无新签名**（93 条不变）。窗口内新候选 3 条均判边界外：DeepSpeed #8321（ZeRO-3 rollout 恒满长解码，#8264 已保 EOS 语义，纯性能）、#8285（AutoTP 改写未分片 vision tower 属性，首前向即 LayerNorm shape 崩，fail-fast）、verl #7546（fully-async `full_determinism` 路由塌缩单 replica，纯吞吐且官方不支持该组合，closed）。arXiv 窗口内 0 篇新命中。
> 2026-08-28：每日扫描。新增 KER.06（Megatron `clamped_swiglu` 经 torch.compile 把 float kwargs `clamp_value` 冻结为编译期常量，后续调用静默复用旧值；per-layer clamp 配置真实存在即中招，上游 pytorch/pytorch#194976），补 RL-RO.07 来源（trl #6945 AsyncGRPO 无 IS 修正却被迫用 processed_logprobs 当 PPO 分母，同策略下 ratio=幸存质量 S）。**94 条**。
> 2026-08-29：每日扫描。新增 RL-RWD.05（trl `think_format_reward` 与模板预填 `<think>` 脱节：GRPO 只把生成 token 当 completion，开标签留在 prompt 侧，锚定正则永不匹配 → 格式 reward 恒 0.0 静默死区，trl #6966）。窗口内其余候选均判边界外（见当日日报）。**95 条**。
> 2026-08-30：每日扫描。新增 DATA.07 / MOE.05 / OPT.05 / CKPT.11 / SFT.09 / SFT.10 / SFT.11 / RL-RWD.06 / RL-ADV.07 / RL-RO.09（DeepSpeed 变长 packing off-by-one 每期静默丢尾样本、AutoEP 替换静默丢 e_score_correction_bias（20/22 模型族）、AutoEP 模块替换后自建优化器漏挂全部专家参数→静默冻结、Megatron resume 重置 is_first_microbatch 致 MTP 跨 depth 梯度被覆写丢弃、trl extract_prompt 对前缀包含对 off-by-one 把末轮消息复制进 chosen/rejected、verl diff==0 探针切片把 system prompt 切成空、trl 自动 processor 不跟 model revision 造成权重/模板 revision 分叉、trl XPO/Nash-MD 拿 policy token ids 直喂 RM、verl rloo_vectorized 单样本组 advantage 恒 0、Megatron colocated serving 捕获的 fused expert 视图冻结在捕获时刻→rollout 用陈旧专家权重），补 SFT.01（ms-swift #9968）/ KER.05 + CKPT.07（verl #7597 嵌套 fsdp strategy 静默丢弃）/ RL-RO.01（arXiv:2511.17826）来源。**105 条**。
> 2026-08-31：每日扫描。新增 PAR.08 / CKPT.12 / DATA.08（DeepSpeed AutoTP compile pass 不下降进 activation-checkpointing 子图→TP collective 静默缺失、错前向错梯度 exit 0，Llama 实测 0/14 插入；verl FSDP merger 把复制 buffer 当分片条目 torch.cat 拼成 world_size 倍；Megatron GPTDataset loss_mask 缓存存引用被首样本就地掩码穿透污染，并静默破坏 bitwise resume），补 MOE.05（修复 #8369）/ SFT.11（修复 #6978）来源；08-30 待复查的 #8357 本轮凭独立机制细节收录（PAR.08）。**108 条**。
> 2026-09-01：每日扫描。新增 LOSS.06（DeepSeek-V4 §4.2.3：MoE 路由与主干同步更新互相放大 outlier 构成恶性循环→反复 loss spike 需 rollback；缓解为 Anticipatory Routing 用 θ_{t−Δt} 缓存路由 index 解耦，经 torchtitan RFC #4349 挖出），补 KER.06 来源（同节 SwiGLU Clamping 证实 per-layer clamp 已是生产级训练基础设施依赖）。**109 条**。
> 2026-09-02：每日扫描。新增 HW.08 / MOE.06（TrainSDC：前向 SDC 易损性由 Q/K 路径位置支配、反向由梯度指数分布支配，ABFT 类均匀设防覆盖差；torchtitan Qwen3 registry 无条件注册 expert-bias 更新 hook，升级后同 ckpt 同 seed 从 step 2 起静默分叉、消融 bitwise 闭合因果），补 CKPT.07 来源（verl #7662 `use_inference_chat_template`/`tokenization_sanity_check_mode` 唯一消费者在测试专属类，agent-loop 活跃路径不读、设置后无警告无行为）。**111 条**。
> 2026-09-03：每日扫描。新增 DATA.09 / SFT.12 / RL-RO.10（Megatron `FusedScaleMaskSoftmax` SWA fallback 无条件覆写 caller attention mask——MiMo-V2.5 39/48 层命中、padding/packing mask 静默失效；ms-swift 多轮轨迹拆分默认 `loss_scale` 使前几轮被重复计权训练、维护者确认须显式 `last_round`；DeepSpeed Hybrid Engine ZeRO-3 QKV 布局互转缺 `modifier_rank` 不持久化——rollout 用布局损坏权重采样、首 token 187→39318、修复后 max_abs=0），补 RL-KL.04（trl #7009 GRPOTrainer 本体 Liger 路径静默丢 MoE aux loss，coef 默认 0.001 零配置即中招）/ RL-RWD.06（母 issue trl #6951）来源。Megatron #7049（full recompute 串 microbatch）报告者自撤不收。**114 条**。
> 2026-09-04：每日扫描。新增 NUM.07 / PAR.09 / CKPT.13 / DATA.10 / OPT.06 / RL-RO.11 / OBS.07（Megatron Triton autotune 计时非确定致逐 rank tiling 分叉——8-rank 13/22 winner 不一致、deterministic-mode 无共享 cache 则节点级分叉；DeepSpeed CPU 类型串分桶永不匹配→梯度 allreduce 静默跳过、mean 下不可见 sum 下错 world_size 倍；curriculum sampler 把全局 numpy RNG 状态当自己的存→resume 采样流从头重放；trl GRPO 只认单一 EOS id、多 EOS 模型 clipped_ratio≈0.97 且 mask_truncated 开启即零学习信号；DeepSpeed fp16+ZeRO-0 backward 未乘 scale 而 step 恒除→有效更新缩小 ~32768 倍训练停滞；ms-swift agentic rollout decode/re-encode 非双射→训练序列≠采样序列；Megatron eval/no_grad 仍重放训练态捕获的 CUDA graph→验证指标混入 router token-count 等训练态 buffer 突变），补 RL-RO.10（修复 #8392）/ RL-KL.04（修复 #7037/#7038 fail-fast 化）/ SFT.11（verl #7658 revision 不 pin 随 main 漂移）来源。ms-swift #10003 连续第四轮待复查（无新评论）。**121 条**。
> 2026-09-05：每日扫描。新增 NUM.08 / NUM.09 / NUM.10 / CKPT.14 / OPT.07 / OPT.08 / ACT.04 / MOE.07 / RL-ADV.08 / OBS.08（Megatron expert token 计数注册 float buffer 被 Float16Module 转 BF16→~40K 计数间距 256、bias 更新方向合法运行间翻转且 eager-vs-eager 亦复现；DeepSpeed 混精 cast 按 `is_floating_point()` 一刀切→FP8/MX/NVFP4 冻结量化权重被 bf16 化、MX e8m0 指数 scale 不 round-trip；DeepSpeed bf16 静态 scaler 检出溢出即零日志跳 step→bf16 RL run 全部 optimizer step 被静默跳过、唯一症状为下游权重逐字节不变；Megatron GPT sharded_state_dict 无条件 pop output_layer._extra_state→量化 lm_head load 后静默恢复未量化；MFSDP v2 zero_grad(set_to_none) 解绑 Python 侧 .grad 绑定而 graph replay 不重放绑定→每参数静默跳过、grad norm 恒 0.0；Muon 对融合 gated fc1 单矩阵正交化——gate/up 独立投影被联合正交化；persistent ckpt worker 异常不发布 completion→单 rank 挂死全部 peer、外部 watchdog resume 半成品 ckpt 风险；SGLang process_weights_after_loading 把量化专家参数 republish 为新对象丢 weight_loader 属性、MXFP4 训练↔rollout 布局契约破坏；GRPO 猜中答案与推理答到拿同量级 advantage→guess-like 行为被强化（arXiv:2609.04063）；trl loss-path 指标按 micro-batch 平摊平均被尾窗 remainder 加权 + capture 时点随 loss_type 分叉双因素错值），补 OPT.05 来源（DeepSpeed #8377 HF Trainer/Accelerate 自建优化器路径同机制）。**131 条**。
> 2026-09-06：每日扫描。新增 CKPT.15 / CKPT.16 / CKPT.17 / DATA.11 / OPT.09 / KER.07 / MOE.08 / MOE.09 / LOSS.07 / RL-KL.05 / RL-RO.12 / RL-RO.13 / OBS.09（ms-swift 并行 ckpt reader 吞 worker 异常无条件返回 success→tensor 停在 sentinel；异步 RL resume 契约漏在途 rollout 状态——OpenRLHF oversampling buffer 不入 ckpt 已生成 rollout 永久丢失 / verl V1 async resume 重复 warmup 跳 prompt；torchtitan gpt_oss HF adapter 旧名映射使 export 静默整组省略 routed experts；trl MiniLLM advantage mask 判 input_ids 而非 labels 恒全 1+pad 位折扣混入；Megatron clip 判据按优化器类型而非参数 owner→MFSDP v2 + precision-aware 下 clip 完全 no-op（213.6 vs clip 1.0、与 clip=0 bit 级相同）；Megatron activation offload 静默绕过 absorbed MLA；Megatron hash MoE aux loss 优化从未 dispatch 的 learned 路由；torchtitan 负载均衡 hook 只扫 model.layers 漏 mtp_layers→MTP 专家偏置冻结；Muse Glimmer final norm 零初始化→首步输出与梯度全零；OpenRLHF entropy 在 logits 原地除 T 之前计算→bonus 与策略温度分裂；verl Mooncake 权重同步 completion slot 不 drain→旧版本张量在 success 语义下滚入下一版本；verl 多教师 OPD 经 sparse Hydra 继承 load_format=dummy 且 teacher 无后续同步→整段训练用随机教师；trl gather_for_metrics 对已聚合标量无法去重 padding→六 trainer 11 处指标尾批偏置），补 OPT.08 / LOSS.06 / RL-KL.04 / SFT.10 / RL-RWD.04 来源（#7085 per-head QKV NS 端口、RFC #4495 anticipatory routing 实现草案、#7062 vendored Liger 同族、#7756 SFT.10 修复、#7766 schema 修复）。**144 条**。
> 2026-09-07：每日扫描。新增 CKPT.18 / CKPT.19 / OPT.10 / KER.08 / RL-KL.06 / RL-RO.14（ms-swift resume 重建 sampler 丢 epoch 种子→epoch≥1 恢复用 epoch-0 排列、81% 样本重复且 15% 未训；DeepSpeed AutoTP 类名白名单 219/233 norm 类不在名单→权重静默留 meta；DeepSpeed ZeRO-3 Muon 更新挂 IPG 路径逐 micro-batch 执行→gas=n 时有效 momentum=βⁿ、NS 见部分梯度、gas=1 vs 4 权重差 1e-01；DeepSpeed fp16+Muon 溢出梯度在溢出检查前折进 momentum→永久毒化、失败自持至 loss-scale 耗尽、测试断言被 fp16 cast 洗白恒绿；trl RLOO 用高方差 k1 而 GRPO 用 k3→跨 trainer KL 惩罚强度不可比；agent rollout 上下文压缩使训练条件=树而非序列——time-travel leakage / 条件序 TIM（arXiv:2609.00865）），补 OPT.08 来源（DeepSpeed #8437 AutoTP shard 整矩阵 NS→update 随 TP 度变 3.9e-01）。**150 条**。
> 2026-09-08：每日扫描。新增 KER.09 / OPT.11 / OBS.10（Megatron fused_mla_yarn_rope_apply 四个 Triton kernel 掩码用块内坐标比全局 head 数→非 2 幂 head_num 下幻影 lane 静默越界**写**、bwd 同掩码 store、autotune 首调即执行不安全 tiling；DeepSpeed Muon 在 ZeRO stage 0（默认）从不执行 Newton-Schulz→最朴素配置静默训成 SGD，作者自纠后修复 PR #8442 落地、09-07 暂缓项凭新证据转正；trl num_tokens 在 TP 下按进程求和→每 token 计 tp_size 次、tokens/s 与 token 预算核算系统性虚高），补 OPT.10 家族注记（#8439 精确律 |Δw|=clip×const——clip 即 Muon 的有效学习率，仍判文档化语义非 bug，跟踪在案）。**153 条**。
> 2026-09-09：每日扫描。新增 NUM.11 / CKPT.20 / KER.10 / OBS.11（PyTorch FSDP2 CPUOffload 下 DTensor RNG tracker 设备错配→CPU 随机操作每次从同一 RNG 状态出发产生相同值、offload 后张量同种子初始化；verl V1 trainer ESI 到期 force-save 未接线→`esi_redundant_time` 静默 no-op、抢占丢窗口；DSA cuDNN packed-THD+CP real kernel 在 GB200/SM100 数值错（top-k 8/8 全错、融合 cosine 0.365）且自合入起被 `flaky` 标记掩盖；verl V1 `_balance_batch` 合成 padding 零有效 token 行进 rollout-correction 均值→训练-推理 gap 指标被系统性稀释），补 NUM.10 来源（DeepSpeed #8461 `BF16_Optimizer` flat 分片恒 1-D→Muon 全走「ZeRO 已正交化」分支而该包装从不调用 Muon——静默退化第五配置、#8442 refusal 的出处）。**157 条**。
> 2026-09-10：每日扫描。新增 OPT.12 / LOSS.08 / SFT.13 / RL-RWD.07（Megatron `--override-opt-param-scheduler` 加载后用全局实参覆写参数组 LR bounds——per-group max_lr=5.0 被改成全局 1.0、resume 后 MoE 冻结/LoRA 分组按错 5×界训练；ms-swift Megatron GKD mean-of-means 归约→token 梯度权 1/(M·N_mb) 随 micro-batch 长度漂移、短回复权重高 100×；verl MultiTurnSFTDataset 逐消息 tokenization 丢上下文——Qwen3 reasoning_content 的 `<think>` 块被丢、`ignore_input_ids_mismatch=True` 一键压掉 sanity check 后静默训练、训后模型把最终答案写进 reasoning 通道；GRPO 组内 verifier 错误相关 ρ=0.530→8 样本组有效样本量 1.70、advantage 符号在 0.83% 组翻转），补 RL-ADV.07 来源（verl #7793 过滤致 singleton 时 scalar/vectorized RLOO 分歧的独立 issue：向量化 `*(c>1)` 清零 vs 标量透传，同 GRPO singleton 约定）。**161 条**。
> 2026-09-12：每日扫描（09-11 中途未产出报告，本轮并合 09-11 原始数据 + 09-12 新搜索，窗口 `created:>=2026-09-05`、按 issue 号去重）。新增 PAR.10 / PAR.11 / CKPT.21 / CKPT.22 / DATA.12 / OPT.13 / OPT.14 / OPT.15 / ACT.05 / KER.11 / NUM.12 / RL-KL.07 / RL-RO.15 / RL-RO.16 / OBS.12（DeepSpeed Ulysses 交换被 torch.compile inline 后 tracer 不建模「empty 张量+写出参 collective」→q/k/v 梯度恒零、forward 数值正确；Megatron GTP expert 梯度 ×GTP/EGTP 静默放大（#7160/#7165 256×GB300 grad-norm 偏 8.4×）；DeepSpeed pipeline ckpt 文件名两位 pad 溢出+字典序→TP≥100 时 rank 11 静默加载 rank 100 分片；torchtitan TorchFT 缓存 optimizer state 标量 LR 陈旧+scheduler 恢复不回填→joining replica 错 LR 权重分叉；DeepSpeed AutoSP 序列分片切断 shift-by-one labels 边界；DeepSpeed AutoEP FP32 clip 把不相交专家范数取平均→clip_coef 随专家放置漂移 0.20；FusedAdam 每 dtype 单 step 计数器在间歇梯度参数间漂移→bias correction 错；Muon 谱缩放把 GTP 对齐 padding 行计入逻辑形状（~27% 偏差）；DeepSpeed partition_activations recompute 参数表每非张量参后插 None、全右移；Megatron op-fuser grouped-MLP 只读全局 autocast→per-layer precision override 永不生效（纯 TE 静默量化、GTP 才炸）；OpenRLHF BF16 RM 组基线被舍入 [1,1,1,1.0078]→advantage [0,0,0,2]；OpenRLHF k3 clamp 前向有限而 backward 经 0×inf 出 NaN 梯度污染参数；verl FSDP2 `to_empty()` 断 tied embedding 别名→静默 untied 训练；ms-swift GRPO response_prefix 生成侧属 prompt、训练侧属 completion→策略按未采样 token 计分、ppl_ratio 爆炸；DeepSpeed ZenFlow 自管 dataloader 时 select_interval=0→首步选列训全程、全部既有测试跑在错路径上恒绿），补 CKPT.20（verl #7820 nccl_timeout 死键）/ CKPT.11（Megatron #7186 MFSDP v2 缺 no_sync 只剩末 microbatch 梯度）/ CKPT.13（修复 PR #8460）/ SFT.13（verl #7844 SGLang 多模态 pad id clip）/ RL-RO.05（verl #7805 R2 replay revert-regression 史）来源。**176 条**。
> 2026-09-13：每日扫描。新增 CKPT.23 / KER.12 / MOE.10 / LOSS.09 / SFT.14 / SFT.15（ms-swift on-policy resume 把 rollout 行数 offset 原样喂 query 行 sampler→n 倍跳过+data_seed 死键；trl #7181 维护者 root-cause 的 cuDNN SDPA backward kernel bug——显式 mask+`L%128==64`+logsumexp<−88.7 时 pad tile 打 0 分→dQ NaN、有限窗口静默错梯度；ms-swift Megatron colocate 路径 int device ordinal 使 vLLM post-load 处理从未执行且被宽 except 吞掉→rollout 用未 pack/未 fusion 权重采样；ms-swift DAPO/CISPO/FIPO 全局 token 归一后被 Trainer 再除 G→有效学习率随梯度累积漂移；ms-swift padding-free LD-DPO 用物理段长定共享前缀→prompt 位挤进 completion 折扣；ms-swift 显式 loss_type 路径丢 token weights→配置的 loss_scale 全部无效、零权重位泄漏），补 CKPT.20（verl #7815 独立 issue）/ RL-RO.15（verl #7834 关联 issue 正文：untied 路径 grad norm 0.3548→0.2271 实测）来源。09-12 报告的「下轮首查」两条（ms-swift #10100/#10072）本轮收录为 SFT.15/SFT.14。Megatron #7201（base-image 26.08 num-zeros 漂移）连续第二周无评论，继续跟踪。**182 条**。
> 2026-09-14：每日扫描。新增 MOE.11 / CKPT.24 / RL-RO.17（Megatron global_aux_loss 累积 expert 计数配当前 microbatch 瞬时分母→router 梯度相对 L2 偏 0.25、物理等长 padding 掩盖触发；DeepSpeed ZeRO-3 tag 加载 FP32 重建分支不透传 tag 重新解析 `latest`→显式请求 earlier 实际进模型 later、日志反指正确版本；trl vLLM sync_weights 只清 KV prefix cache 漏清多模态 encoder cache→视觉 encoder 训练中图像复现命中旧权重 embedding、扰动权重后 logprob 不变）。**arXiv export API 连续第二天 Rate exceeded**（https/http 双端点 + 退避重试无效），4 查询未完成。**185 条**。
> 2026-09-15：每日扫描。新增 KER.13（ms-swift 序列并行 ChunkedCrossEntropyLoss backward 把梯度 chunk 就地写进 saved logits 并把输入当梯度返回——叶子/非叶子/version-counter 形状 fail-fast，但单一 logits 消费者的 SP 训练路径静默跑通、backward 后调用方 logits 已被梯度值覆写），补 CKPT.20 来源（verl #7815 随 commit 2083eeb 修复关闭）。arXiv export API **连续第三天** Rate exceeded；Semantic Scholar 替代源未认证池同样大面积 429（仅 1 查询成功，唯一候选判 RL-ADV 已知机制的 VLA 域应用、边界外，未编号未入库）。**186 条**。
> 2026-09-16：每日扫描。新增 ACT.06 / OPT.16 / RL-ADV.09 / RL-KL.08（torchtitan SAR pass 只重放 clone 不重放 schema 正确的 in-place 写→backward 拿变异前值、前向正确梯度静默错；Megatron 混精 clip 把 bf16/fp32 梯度塞进同一 TE multi_tensor launch、bf16 在前时 fp32 梯度被按 2 字节读出垃圾 norm/clip 不 crash；GRPO 动态采样按 shaped 分过滤时 all-fail 组内 shaping 离差被 std 归一提升为满幅 phantom advantage、只换 filter metric 即 EM 0.160 vs 0.763；trl GRPO KL bias correction 把 sequence-mean IS 权重 (B,1) 广播到 per-token KL、多出 spurious 梯度项且默认配置即中招），补 KER.13（#10129 无评论关闭）/ OPT.13（#8478 修复 commit `338ed51` + PP 下零专家参数 rank 场景由 all-reduced flag 解决）来源。arXiv export API 恢复（5/5 成功，此前三天 429）。**190 条**。
> 2026-09-17：每日扫描。新增 PAR.12 / PAR.13 / ACT.07 / NUM.13（Megatron MFSDP v2 + Flex/HybridEP + 专家 outer optimizer sharding 间歇坏梯度/NaN——eager/无图/无裁剪三重排除仍存、alltoall 对照 3/3 健康而 HybridEP 复现 spike 3254；trl DPO/KTO 在 `use_liger_kernel=True` + plain DDP 下 unwrap 跑 loss 绕过 `DDP.forward`→reducer hooks 早退从不 all-reduce、2×H100 25/25 参数梯度跨 rank 不同；Megatron 细粒度激活 offload `tensor_pop` inline reload 对 D2H 无排序边→warmup 迭代梯度静默算在陈旧 staging buffer 上、B300 19/20 损坏而加边 0/20、文档承诺 compute stream 等待但 inline 路径不等；ms-swift GKD `jsd_loss` 接受 student FP32/teacher BF16 混精 logits→JSD 恒非负性破、符号翻转 −0.000170 vs FP32 +0.000509），补 OPT.16（#7338 归属裁决留在 Megatron-LM + 修复 PR #7279）/ RL-RO.01（QUADS 2607.15810：NVFP4 RL 不稳定主因是激活误差而非权重误差）来源。**194 条**。
> 2026-09-18：每日扫描。新增 NUM.14 / PAR.14 / PAR.15 / CKPT.25 / OPT.17 / OPT.18 / KER.14 / KER.15 / MOE.12 / LOSS.10 / LOSS.11 / RL-RO.18 共 **12 条**（10 条来自 Megatron-LM #7452——ezyang 用 Astra+spmd_types 对 mcore `7a9e59de7` 的并行正确性审计，13 finding 逐条 CPU/Gloo repro + Appendix B：trl QLoRA fp16 下转型打到 QDoRA magnitude vector→更新量小于 BF16 精度静默 no-op；显式 ProcessGroupCollection 五路径静默回落全局 MPU 组→重算/SP gather/eval 混入别的 replica 数据；Bridge Wan decoder SP gather 默认 grad 模式与「每 rank 全 loss」不匹配→梯度恒 ×TP；MFSDP ckpt 按 stage 局部层号做 key→PP resize 后 strict load 成功但各 stage 恢复第一阶段层；GPT-OSS learnable sink 参数未标 TP 标志→grad norm 只剩 rank 0 的 head 梯度；GDN out_norm 不在 finalize 白名单→TP>1+SP off 梯度漏 all-reduce 副本确定性分叉；vocab-parallel CE label smoothing 尾按本地词表平均→per-rank loss 不等；gathered logits 被喂 vocab-parallel CE→loss 恒偏 +log(TP)、target 梯度项消失；MoE aux loss 就地归约覆写 token 计数→aux 梯度恒 ×TP/CP；CP-local loss mean-of-means→梯度方向畸变；MTP per-token loss 本地 token 比率→梯度随 rank token 分布漂移；另 verl #7904 colocated 默认 `free_cache_engine=true` 的 vLLM sleep/resume 周期静默损坏 rollout 权重→多语言乱码 reward 全零、表象伪装成 GRPO reward collapse），补 PAR.12（#7400 根因突破：独立 CUDA C++ repro 直指 cublasLt+NCCL 并发 retained-output、NV bug 6798870）/ PAR.13（trl 维护者测量更正：DPO 需先迁 #7243 chunked 路径）/ ACT.07（#7399 评审中）来源。**206 条**。
> 2026-09-19：每日扫描。新增 HW.09 / PAR.16 / PAR.17 / CKPT.26 / KER.16 / MOE.13 共 **6 条**（FP-Sketch 事后 GEMM SDC 验证器、#7452 Appendix C 两条常数尺度误差转正：global_aux_loss 梯度经 DDP AVG 缩小 DP× / MFSDP v1 `average_in_collective=True` 忽略 scaling_factor 梯度恒 ×DP；trl AsyncGRPO stale-dropped group 把 resume 游标钉死→147/150 行重训；DeepSpeed Triton int32 偏移 2^31 回绕越界读写；over-dispersed routing 下 router 概率重要性信号塌缩、ppl 与下游能力反转），补 MOE.12（0z5a main `d564dd01d` 真实 NCCL 复现 + 隔离实验 aux 梯度精确线性 ×global/local + 修复 PR #7481）/ OPT.16（#7279 DeepSeek-V4-Pro 32 节点生产验证）/ RL-KL.04（trl #7270 GMPO+Liger 完整更新级复现、关闭为 #6808 duplicate、修复走向结构性移除分叉）/ NUM.14（修复认领）来源；KER.14 补昨日漏写的「发现来源」行。**212 条**。
> 2026-09-20：每日扫描。新增 DATA.13 / RL-RO.19 共 **2 条**（verl FSDP value-model 引擎按外层模块显式签名探测 packed 边界能力——TRL value-head wrapper 只有 `**kwargs` → `use_remove_padding=True` 下 critic 静默丢 `cu_seqlens`、Qwen3.5 linear-attention 状态跨样本泄漏、SP=1 即中招；ms-swift Ray Megatron IPC 权重同步 ACK 在 TP barrier 之前、发送端即刻复用共享 bucket——慢 TP rank 读到半新半旧权重、rollout 静默损坏且无同步异常），补 DATA.11（trl #6626：MiniLLM `_compute_advantage` gamma_pow 按绝对位置索引——`length_normalization=False` 时 advantage 恒多乘 γ^t、`=True` 时 float32 下溢出 0/0=NaN，修复 #6635 open 自 8 月；#7295 维护者提案移除该 trainer 把它列为论据）/ OPT.08（torchtitan #4692 RFC 全量融合矩阵×逻辑矩阵清单、K3 生产侧逐逻辑矩阵 NS + aspect-ratio lr）来源。**214 条**。
> 2026-09-21：每日扫描。新增 CKPT.27 / DATA.14 共 **2 条**（verl v0/v1 trainer 在非 Megatron 策略下 `async_save=True` 跳写 `latest_checkpointed_iteration.txt` 而 FSDPCheckpointManager 从不补写——ckpt 完好落盘但 `resume_mode=auto` 判无 ckpt、静默从 step 0 重训并覆写；trl GRPO/RLOO metric flush 按本地 key 集合调 collective——rank 间 key 集合分叉时指标混列 + mean-of-means 错权、reward/KL 诊断静默失准），补 RL-RO.19（修复 PR ms-swift #10207 已于 09-20 merge）、KER.16（修复 PR DeepSpeed #8591 已于 09-19 merge）、OPT.17（#7452 finding 8 被 dundysm 认领）、DATA.13（#7931 维护者确认 bug 在 veRL 能力探测侧）来源。**216 条**。
> 2026-09-22：每日扫描。新增 NUM.15（trl #7320 维护者 root-cause：transformers<5 `TrainingArguments` 把 mixed precision 发布为进程级环境变量并读回作默认值——`bf16=False` 无法清除先前 trainer 钉下的 bf16，同进程后续 fp32 阶段/bare `Accelerator` 静默继承 bf16 autocast；trl 默认 `bf16=True` 使第一个 trainer 即钉死进程；修复 PR #7321 已 merge），补 KER.14（修复 PR #7466 draft）/ LOSS.10（修复 PR #7513）/ OPT.17（修复 PR #7530 已开）/ OPT.18（vipulsarode 认领）/ MOE.12（Connor-XY 请 zhongbozhu 评审 #7481）/ KER.10（#7463 同 packed+CP 路径 256K crash 面补充：oncall 分配 hxbai triage）来源。**217 条**。
> 2026-09-23：每日扫描。新增 CKPT.28 / DATA.15 / MOE.14 / RL-KL.09 共 **4 条**（Megatron 原生路径加载 contiguous SwiGLU ckpt 缺 gate/up interleave 转换——SwiGLU 配错通道、initial loss 0.2→2.2 无 crash、Bridge 已转换而 main 缺步；torchtitan DSV4 重写 `get_attention_masks()` 丢 positions/文档边界元数据——sparse attention 窗口跨 packed 文档边界、position 重置契约被 override 破坏；trl `router_aux_loss_coef` 用可选字段 `output_router_logits` 推断 MoE——64 个 MoE config 中 30 个不声明（DeepSeek/GLM4-MoE/Kimi-Linear 全中）、近半架构 aux loss 静默 no-op；全管线 FP8 RL 复合量化噪声把负 advantage token 的 ratio 推出 trust region——惩罚梯度被错误归零、entropy surge + 乱码、Calibrated Clipping 缓解），补 RL-RO.01 来源（2609.22870 TIM 残差经非线性 clip 放大）；#7310 报告者给出 20 行最小 repro 坐实 DATA.14 死锁面；verl #7987（#7978 修复）1×B200 验证通过。**221 条**。
> 2026-09-24：每日扫描。新增 RL-RWD.08（trl OpenReward `env.reward` 哨兵 0.0 与合法 0 分坍缩——从未打分的 rollout 与真 0 分 bit 级同值、`unscorable_mask` 失效、放弃/超限被当真实负信号进 advantage；先行修复 PR #6430 曾获维护者正评后被关未合、缺陷在 HEAD 仍活），补 KER.14（Megatron #7464 第三方独立复现：TP1 vs TP2 全梯度相对 L2 0.0515、三机复现）/ OPT.18（vipulsarode scoped 修复 PR #7613 已开：GDN/GDN2 × RMSNorm/LayerNorm 全覆盖 + finalizer 恰好一次求和单测）/ CKPT.26（修复 PR trl #7294 cross-ref 落定，仍 open）来源。窗口内其余候选均判边界外（trl #7362 fail-fast + 纯成本、#7354 启动期 KeyError、#7353 文档；DeepSpeed #8639 初始化 fail-fast、#8572 启动期校验放宽边界；Megatron #7464 即 KER.14 独立复现；verl #7990 closed completed）。**222 条**。

> 2026-09-24：跨分支回填复核：远程 15 条中 12 条为已收录的同机制，新增 CKPT.29 / RL-ADV.10；DeepSpeed #8586 仅属性能开销，拒绝入库。**224 条**。

| 轴 | 分布 |
|---|---|
| **stage** | `shared` 为主，另有 `pretrain` / `sft` / `rl`（见各类节头） |
| **置信度** | `verified`（本地或逐行核实）· `documented`（公开出处） |
| **覆盖** | 种子期几乎全是 `NEW`；变体标 `var:x` |

---

## 目录

| 前缀 | 主题 | 条数 |
|---|---|---|
| HW | 硬件 / 训练期 SDC | 9 |
| NUM | 数值 / 混精 / 非确定性 | 15 |
| PAR | 并行 / 梯度归约 | 17 |
| CKPT | 检查点 / resume | 29 |
| DATA | 数据管线 | 15 |
| OPT | 优化器 | 18 |
| ACT | 重计算 / 卸载 | 7 |
| KER | 训练 kernel | 16 |
| MOE | MoE 路由 / EP | 14 |
| LOSS | loss spike / 长跑退化 | 11 |
| SFT | 模板 / packing / 标签 | 15 |
| RL-RWD | reward | 8 |
| RL-ADV | advantage / GRPO 统计 | 10 |
| RL-KL | KL / entropy | 9 |
| RL-RO | rollout ↔ train（TIM） | 19 |
| OBS | 观测性骗过 | 12 |

---

## 图例

- **stage**：`pretrain` · `sft` · `rl` · `shared`（可多值，`/` 分隔）
- **Cov**：`NEW` / `var:snake_case`
- **置信度**：`verified` · `documented` · `speculative`
- 新条目只要求 **机制 / 来源 / 行为效应 / 发现来源**。未做实验的条目不写检测器预测。

---

# HW · 硬件 / 训练期 SDC

## HW

### HW.01 `training_sdc_unhealthy_node_optima_shift`

stage: `pretrain/shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：生产集群里被自动化清扫出的「不健康节点」仍能跑完 step。用 XLA 确定性执行 + 同步机制，把 SDC 隔离到子模块计算、单次 optimizer step、一段训练窗口三个尺度。多数节点上子模块/梯度扰动相对小，但足以把模型推进**另一个局部最优**（权重轨迹分叉）；部分节点会打出 loss spike。
- **来源**：Ma, Pei, Lausen, Karypis, "Understanding Silent Data Corruption in LLM Training", ACL 2025 / arXiv:2502.12340。https://arxiv.org/abs/2502.12340
- **行为效应**：无 Xid、无 NaN 也能训下去。最终 ckpt 与健康节点同配置不同权重；微调阶段可见不同幅度的 loss spike。ABFT 对 GEMM 的覆盖在该研究中大多失败——故障往往不在矩阵乘单元。
- **发现来源**：2026-08-17 基础调研

### HW.02 `llama3_sdc_job_interrupt_fraction`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：16K GPU Llama 3 预训练 54 天快照：466 次 job 中断，419 次意外。其中 GPU SDC 被单独记了 6 次（占意外中断 **1.4%**）。约 78% 意外中断归确认/疑似硬件。这是「训练期 SDC 不是轶事」的生产计数，不是单次 bit-flip 实验。
- **来源**：Grattafiori et al., "The Llama 3 Herd of Models", arXiv:2407.21783 §3 中断表。https://arxiv.org/abs/2407.21783
- **行为效应**：集群层表现为未计划中断；未被抓住的同类事件会以坏梯度/坏权重形式活进后续 step（见 HW.01 / HW.05）。
- **发现来源**：2026-08-17 基础调研

### HW.03 `fleet_sdc_cpu_base_rate`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：超大规模 CPU fleet 上静默计算错误的实测基率远高于宇宙线软错误：约 **1/1000 器件**量级。这是训练集群「偶发坏节点」的先验，不是 LLM 特有机制。
- **来源**：Dixit et al., "Silent Data Corruptions at Scale", arXiv:2102.11245；Hochschild et al., "Cores that don't count", HotOS 2021。https://arxiv.org/abs/2102.11245
- **行为效应**：单 step 几乎看不见；长跑中以极低频率污染一次更新。检测必须在 fleet / 设备尺度，而不是 per-step 语义探针。
- **发现来源**：2026-08-17 基础调研

### HW.04 `gemini_weekly_sdc_cadence`

stage: `pretrain` · Cov: `var:llama3_sdc_job_interrupt_fraction` · 置信度 `documented`

- **机制**：公开转述：Gemini 训练中 SDC 大约每 1–2 周一次。与 Llama 3 的 54 天 6 次同量级，用来标定「大训练必遇」而不是「实验室 curiosum」。
- **来源**：Ma et al. arXiv:2502.12340 对 Gemini 训练 SDC 频率的引用。https://arxiv.org/abs/2502.12340
- **行为效应**：长跑中周期性出现无法短时复现的轨迹异常或 NaN；停机诊断窗口往往已经错过故障器件。
- **发现来源**：2026-08-17 基础调研

### HW.05 `training_sdc_checkpoint_inherit`

stage: `pretrain/shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：训练期一次静默坏更新进入发布 ckpt。之后任何 serving 前向都是干净的，权重却已经不在「真实」流形上。
- **来源**：同 HW.01；Altenberd et al. arXiv:2604.00726。交叉：`inference_error_review` **1.6** `training_sdc_checkpoint_inherit`（serving 视角的同一机制）。https://arxiv.org/abs/2502.12340
- **行为效应**：发布模型 perplexity 略差、偶发更糟答案。单模型自参照监控原理上看不见，需要干净参照 ckpt 做流形对比。
- **发现来源**：2026-08-17 基础调研（从推理编目回链）

### HW.06 `llm_prism_permanent_gpu_fault`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：把 GPU **永久性**故障（不是单次 SEU）用 RTL 级仿真 × 随机注入打进 LLM 训练，刻画哪些层/位/算子会把永久故障变成 SDC。
- **来源**：LLM-PRISM, arXiv:2604.10390。https://arxiv.org/abs/2604.10390
- **行为效应**：故障器件上每个 step 都在吃同一类错计算；比瞬态 SDC 更容易把权重推离，也更容易被重复模式抓到。
- **发现来源**：2026-08-17 基础调研

### HW.07 `scout_pretrain_consensus_outlier`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：预训练中用对称共识/离群检测在数据并行副本之间定位故障 rank，而不是等 loss 炸掉。
- **来源**：SCOUT, arXiv:2608.11034。https://arxiv.org/abs/2608.11034
- **行为效应**：被污染的 DP 副本在共识统计上离群；若没有这类对照，故障会均进 AllReduce 后的「共识错误梯度」。
- **发现来源**：2026-08-17 基础调研

### HW.08 `sdc_forward_qk_backward_grad_exponent`

stage: `pretrain/shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：首个覆盖 Transformer **前向+反向**主要计算接口的系统 SDC 易损性刻画，揭示两条不对称传播机制：前向易损性强依赖**位置**——落在 Q/K 路径的故障产生**持续性**训练偏差（而非局部噪声）；反向易损性主要由**梯度指数分布**（grad exponent 尾部）支配、与计算位置基本无关。据此提出 TrainSDC 防护框架：Q/K 路径重算 + residual-gain 监控 + exponent-aware 梯度缩放，Llama-3.2-1B / Qwen3-0.6B 上以 1.65%–6.76% 开销维持接近无故障轨迹（sparse/dense 注入双验证）。
- **来源**：Xia, Xu, Yun, Lin, Liu, Li, "TrainSDC: Characterizing and Mitigating Silent Data Corruption in Large Language Model Training", arXiv:2608.30769（2026-08-31，窗口内）。https://arxiv.org/abs/2608.30769
- **行为效应**：单次瞬态 SDC 是否演化为轨迹级偏差由落点决定——Q/K 路径、大梯度指数位的故障无 Xid、无 NaN 地把训练推离原轨迹；对 Transformer 计算接口均匀设防的方案（ABFT 类）对这两类热点覆盖差。与 HW.01（节点级 optima-shift 现象学）、HW.06（永久故障的层/位 RTL 刻画）互补：这条给出**前向位置 × 反向梯度指数**的可操作易损性地图，也解释了 HW.01 中「ABFT 覆盖大多失败」的原因。
- **发现来源**：2026-09-02 每日扫描

### HW.09 `fp_sketch_post_hoc_gemm_sdc_localization`

stage: `pretrain/shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：半精度（fp16/bf16）操作数、FP32 累加交付的 tensor core GEMM 上的 SDC **事后验证器**能力边界刻画：ABFT 须 fused 进 kernel 或编码操作数、且每次校验和至多定位一个错误；FP-Sketch 在**未修改**的 GEMM 之后跑——sum sketch 每次调用检出损坏、hashed first-moment sketch + 独立重算确认可定位多个损坏条目（构造上无假阳性），每个故障给出坐标+幅度供 fleet 诊断；浮点下限制因素是 sketch 噪声而非桶碰撞，其常数依赖 BLAS 与操作数格式。
- **来源**："Sketching the Error, Not the Product: Post Hoc Fault Recovery for Half Precision GPU Matrix Multiplication"（FP-Sketch），arXiv:2609.19758（2026-09-17，窗口内）。https://arxiv.org/abs/2609.19758
- **行为效应**：这条是检测侧证据而非注入面：部署它的 run 可以在坏 GEMM 输出进梯度**之前**截获（对比 HW.01：未设防时 SDC 把训练推入另一局部最优）；「坐标+幅度」输出可直接对接 HW.03 的 fleet 级基率统计。与 HW.07（共识离群定位故障 rank）互补：那边 DP 副本间对称对照、这边单 kernel 输出域校验；与 HW.08（易损性地**图**）正交：那边告诉你哪里值得设防、这边提供不必改 kernel 的设防手段。
- **发现来源**：2026-09-19 每日扫描

---

# NUM · 数值 / 混精

## NUM

### NUM.01 `fp8_bias_invariant_martingale_divergence`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Softmax-CE / LayerNorm 输入沿全 1 向量的梯度在精确算术下抵消（𝟙ᵀg = 0）。低精度乘法噪声打破抵消，行均值 mₜ = (1/d) Wₜᵀ𝟙 变成无漂移鞅；累积能量 Aₜ→∞ 时参数几乎必然发散。这是 FP8 训练「看起来在跑、谱已塌」的理论核。
- **来源**：Zheng, Huang, Liang, "Mean-Square Divergence of Low-Precision Gradient Descent"（华为）。本地 `martingale_monitor` 实现 Ēₜ / ‖𝟙ᵀW‖ / srank。
- **行为效应**：loss 尚未分叉时 Ēₜ 已比 BF16 高一个数量级；有效秩向个位数塌缩；weight decay 挡不住。本实验室在 gpt-size E4（bitshift O4）上看到 `realized_var` 提前抬升。
- **发现来源**：2026-08-17 基础调研（文献 + 本地 monitor）

### NUM.02 `bf16_vs_fp16_rl_engine_mismatch`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：RL 的 rollout 引擎与 trainer 常一个 BF16、一个更激进的核。Qi et al. 指出把两边都放到 FP16 能压训推概率差。BF16 的大动态范围并不自动等于「两边算得一样」。
- **来源**：Qi et al., "Defeating the Training-Inference Mismatch via FP16", arXiv:2510.26788；VeXact 文 arXiv:2605.14220 引用。https://arxiv.org/abs/2510.26788
- **行为效应**：同权重同序列的 token logprob 出现 δₜ；极端 token |δₜ| 可到 O(1)，足以掀翻 GRPO。
- **发现来源**：2026-08-17 基础调研

### NUM.03 `fp_reduction_order_training_nondeterminism`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：非结合浮点 + 随 batch/tile 变化的归约树，使同输入两次 backward 得不到 bitwise 梯度。训练里这通常被当成噪声；一旦要做 A/B、重算 oracle、或 RL 的 π_old，它就变成正确性故障。
- **来源**：Thinking Machines, "Defeating Nondeterminism in LLM Inference"（机制同样作用于训练 GEMM）；Megatron-LM #6521（vLLM fused-MoE batch-invariant generation）。https://github.com/NVIDIA/Megatron-LM/pull/6521
- **行为效应**：同 seed 重跑曲线漂；RL 的 trainer 重算 logprob 与 rollout 对不齐（见 RL-RO.01）。
- **发现来源**：2026-08-17 基础调研

### NUM.04 `loss_scaler_swallows_overflow`

stage: `pretrain/shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：动态 loss scaling 把溢出 step 丢掉并缩小 scale。偶发 SDC/内核尖峰若表现为 inf，会被 scaler **合法跳过**，看起来像一次普通 skip，没有坏节点信号。
- **来源**：混精训练常规行为；与 HW.01「多数 SDC 是有限值」合看——scaler 只挡住那 ~1% 的 NaN/Inf，挡不住有限值坏梯度。Tung et al. arXiv:2605.04213（有限值 SDC 占优）。https://arxiv.org/abs/2605.04213
- **行为效应**：日志里只有 `overflow/skipped`；权重该更新的时候没更新，或不该跳过的坏有限值被吃进 Adam。
- **发现来源**：2026-08-17 基础调研

### NUM.05 `gtp_wgrad_dtype_breaks_fp32_reduce_contract`

stage: `pretrain/shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：GTP 关闭 grad-accumulation fusion 后 wgrad 由朴素 matmul 回退发出，dtype 跟随计算精度而非 `main_grad`。开了 `--accumulate-allreduce-grads-in-fp32` 的用户期望全链 FP32，实际 reduce-scatter 在 BF16 里跨 rank 求和、之后才进 FP32 累加器——RS 先跨 rank 舍入，累加器从未见过真值。
- **来源**：Megatron-LM PR #6624（修复 PR，附 before/after dtype 日志；截至 2026-08-19 open）。https://github.com/NVIDIA/Megatron-LM/pull/6624
- **行为效应**：无 crash、无 NaN；梯度精度静默劣化，多卡与单卡结果系统性偏差。修法是让 wgrad 以 `main_grad` 的 dtype 发出。
- **发现来源**：2026-08-19 每日扫描

### NUM.06 `low_precision_absorption_slingshot`

stage: `pretrain/shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：训练进入高置信阶段后，正确类 logit 与其余 logit 之差超出浮点吸收误差阈值 → 正确类梯度被舍入为恰好 0、错误类梯度仍非零 → 破坏类间梯度零和约束 → 分类器均值与特征均值指数增长的正反馈（Numerical Feature Inflation）→ Slingshot 式 loss spike。
- **来源**："Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes", arXiv:2605.06152。https://arxiv.org/abs/2605.06152
- **行为效应**：完整吸收 → 周期性尖峰；**部分吸收可以完全无可见 spike 但参数范数仍在疯长**（静默面）。与 NUM.03（归约序非确定）同为有限值静默错，根在表示精度而非通信。
- **发现来源**：2026-08-19 每日扫描

### NUM.07 `triton_autotune_rank_divergent_tiling`

stage: `shared` · Cov: `var:fp_reduction_order_training_nondeterminism` · 置信度 `documented`

- **机制**：Triton autotune 靠**现场计时**选 kernel config，winner 属于「那一刻那台机器」——计时噪声使不同 rank 对同一 (kernel, shape) 选出不同 `BLOCK_SIZE`/`num_warps`/`num_stages`，即每个 rank 用**不同的归约 tiling** 计算「同一个」算子，浮点累加序逐 rank 分叉。叠加面：`--deterministic-mode` 开 `TRITON_CACHE_AUTOTUNING` 但 cache 目录未共享时，每个节点各自独立 autotune，静默产生节点级分叉。实测 8-rank 混合配置：22 个 kernel/shape 条目中 **13 个** rank 间 winner 不一致；`l2norm_fwd_kernel` **全部 8 个 rank** 在两次运行间各选了不同配置。修复 = deterministic-mode 下强制共享 cache 目录（拒绝而非猜测）+ `TRITON_PRINT_AUTOTUNING` 跨 rank 对账；cache 完整性故意不校验（cache 按 shape/build/GPU/env 键控，异形状预热后的目录「看起来完整」但不够用，校验会误杀合法 run）。
- **来源**：Megatron-LM PR #6966。https://github.com/NVIDIA/Megatron-LM/pull/6966
- **行为效应**：无 NaN、无 crash、loss 正常——每个 rank 的梯度都「对」，但 DP allreduce 平均的是**不同累加序的结果**，run-to-run 与 rank-to-rank bit 级不复现；A/B 对照、bitwise resume、π_old 重算（RL）全部失效。与 NUM.03（batch/tile 归约树非确定）同根因的**调优器放大版**：不是归约树随机，是调优器把不同的树钉进了每个 rank/进程。
- **发现来源**：2026-09-04 每日扫描

### NUM.08 `bf16_token_count_rounding_flips_bias_sign`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：aux-loss-free router 的 `local_tokens_per_expert` 注册为**浮点 buffer**，`Float16Module` 随即把它转成模型精度（BF16）——token 计数这一**整数语义**在 BF16 里累积、跨 rank all-reduce。计数到 ~40K 量级时 BF16 spacing=256，专家间小差异被舍入抹掉甚至翻转：bias 更新方向 `sign(avg − count)` 在两次合法运行间得到相反符号（实测 W64 BF16：64 个 local 值 608–676、精确和 40,840；BF16 all-reduce 后 eager=41,216 / partial-CG=40,704 / 行均值=40,960 → 方向 −1 vs +1，bias update ∓0.0009994507）。BF16 加法非结合 + NCCL 归约树/通道序差异使 **eager-vs-eager** 也复现（pre-allreduce buffer 逐位相同、post-allreduce 64 rank 全不同）。修复 = 注册 int64 buffer（阻断精度转换）+ 用整数等价式 `total_tokens − tokens_per_expert · num_experts` 比较方向，只有最终方向转成 bias dtype。
- **来源**：Megatron-LM PR #7081（W64 BF16 trace + 相邻计数回归测试）。https://github.com/NVIDIA/Megatron-LM/pull/7081
- **行为效应**：无 NaN、无 crash、loss 正常；router bias 沿错误方向更新、训练轨迹分叉，且分叉随归约树/拓扑变化**不可复现**。与 NUM.03/NUM.07（归约序非确定）同根因，但被破坏的是整数计数语义而非连续梯度；与 MOE.06（bias 更新语义被隐式换掉）同一受体（expert bias），本条错在**数值载体**。
- **发现来源**：2026-09-05 每日扫描

### NUM.09 `mixed_precision_cast_sweeps_quantized_params`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed engine 的混精转换按 `p.is_floating_point()` 一刀切——FP8/MX/NVFP4 窄精度 dtype 对该谓词全为 True，**冻结的量化权重**（从不收梯度、只被量化 GEMM 消费）被整体 cast 成 `param_dtype`（bf16）。三种面：FP8 权重 1B→2B、量化表示丢失；MX 的 `float8_e8m0fnu` block-scale 存的是**指数**不是值，cast 到 bf16 再回来不 round-trip → scale 语义毁；NVFP4（`float4_e2m1fn_x2` 两个 4-bit 打包）无 `copy_` → `deepspeed.initialize` 直接 NotImplementedError。LoRA-on-quantized-checkpoint 常态场景（frozen 量化基座 + trainable adapter）一步 initialize 即中招。修复 = cast 跳过窄精度 dtype。
- **来源**：DeepSpeed #8414（repro：4096×4096 fp8 frozen param + bf16 + ZeRO-2，16 MiB→32 MiB bfloat16）+ 修复 PR #8416（skip narrow dtypes）。https://github.com/deepspeedai/DeepSpeed/issues/8414
- **行为效应**：FP8/MX 面**完全静默**——训练照常，量化 GEMM 之后拿到的是被 bf16 化的权重，LoRA 产物对量化部署不可用；NVFP4 面 fail-fast。与 CKPT.09（MXFP8 ckpt round-trip 丢 scale）互补：那边序列化破坏编码，这边 engine 初始化破坏。
- **发现来源**：2026-09-05 每日扫描

### NUM.10 `bf16_static_scaler_silent_skip_all_steps`

stage: `rl/sft` · Cov: `var:loss_scaler_swallows_overflow` · 置信度 `documented`

- **机制**：ZeRO-1/2 `step()` 检出非有限梯度时 `zero_grad` + early return，**无任何日志**。fp16 下还有 loss scale 变化作间接信号；bf16 拿到的是静态 `LossScaler`（`CreateLossScaler` 只给 fp16+dynamic 返回会打日志的 `DynamicLossScaler`）——什么都不打。`engine.skipped_steps` 只经 `_report_progress` 暴露，gated on `steps_per_print`，**默认 None → 永不触发**。bf16 + `check_grad_overflow=True` 的 RL run 每个 optimizer step 都被静默跳过：`engine.step()` 正常返回、lr scheduler 照走、指标照发，权重从未更新。
- **来源**：DeepSpeed #8415（bf16 RL run 实例：唯一可见症状是下游同步的权重逐字节相同；loss/reward 曲线全程"合理"，reward 噪声掩盖冻结策略）+ 修复 PR #8417（log skipped step）。https://github.com/deepspeedai/DeepSpeed/issues/8415 ；DeepSpeed #8461（`bf16`+`grad_accum_dtype=fp32`+ZeRO stage 1 选中的 `BF16_Optimizer` 是第五种包装：参数组被换成 flat fp32 分片后 `MuonWithAuxAdam.step` 按 `dim()<2` 全走「ZeRO 已正交化」分支，而该包装**从不**调用 Muon——step 退化为带 momentum 的 SGD；#8442 的 refusal 正是从这条洞拆出）。https://github.com/deepspeedai/DeepSpeed/issues/8461
- **行为效应**：run 表面完全健康、参数一步未动。与 NUM.04（scaler 吞溢出）同族观测面，差异：bf16 无 scale 变化连间接信号都没有；与 OPT.06（有效更新缩小 ~32768 倍）同为「训练在跑、参数没在学」——这条缩小、那条归零。同一开关的反向陷阱：`check_grad_overflow=False`（bf16 默认）时 Inf 梯度直接进优化器把 NaN 写进所有 trainable tensor，两个方向都不可见。
- **发现来源**：2026-09-05 每日扫描

### NUM.11 `fsdp_cpu_offload_rng_tracker_device_mismatch`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：PyTorch DTensor 的 RNG tracker 是进程级单例，其 `_distribute_region()` 用 `torch.random.fork_rng(devices=[cuda_device], ...)`。FSDP2 `CPUOffloadPolicy` 下 DTensor 的 mesh 仍是 CUDA 类型、而局部存储已驻留 CPU——CPU 侧随机操作被路由进 CUDA 配置的 tracker；`fork_rng` 退出时**无条件恢复 CPU 默认生成器状态**（与传入哪些设备无关）：每个被错误路由的 CPU 随机操作都从同一 RNG 状态出发，产生**完全相同的值**。torchtitan 侧最初表现为 `trunc_normal_` 拒绝采样重采循环永不收敛（lm_head 初始化挂死：`normal_` 反复产生相同越界值）；上游 PR 明示更宽的静默面——offload 之后初始化的张量可能收到**相同更新**。
- **来源**：pytorch/pytorch#196072（根因分析 + `fork_rng` 语义定位；影响面未广泛测试，作者自标）。https://github.com/pytorch/pytorch/pull/196072 ；torchtitan #4511（LM head `trunc_normal_` std=0.0625 初始化 hang，debug 配置 CPU offload 复现，维护者已 cc PyTorch 团队）。https://github.com/pytorch/torchtitan/issues/4511
- **行为效应**：两副面孔：初始化期可表现为挂死（可见），训练期相同值注入是**静默**的——无 crash、无 NaN，破坏的是随机性契约本身（offload 后初始化/dropout 等随机操作去相关失效）。与 CKPT.13 / CKPT.18（RNG 状态面）同主题——那边错在 **resume 路径**的随机性契约（存错对象/丢 epoch 种子），这边错在**运行时路由**：单进程内设备面变化使 tracker 与实际执行设备脱钩；与 NUM 家族的确定性/复制作业错值面相邻但机制独立（不是归约序、不是 cast，是 RNG tracker 设备谓词失效）。
- **发现来源**：2026-09-09 每日扫描

### NUM.12 `bf16_reward_rounds_away_group_baseline`

stage: `rl` · Cov: `var:bf16_vs_fp16_rl_engine_mismatch` · 置信度 `verified`

- **机制**：OpenRLHF grouped advantage（GRPO/PPO 组统计）直接继承 reward 张量的 dtype：**BF16 reward model** 的输出先于组中心化被舍入——`[1, 1, 1, 1.0078125]` 的组归一化 advantage 得到 ≈`[0, 0, 0, 2]` 而非 `[-0.5, -0.5, -0.5, 1.5]`（组内差 < BF16 ULP，基线被舍入掉），PPO 梯度随之改变；integer reward 在组统计/分数惩罚下直接失败；FP16 常数组可产生 NaN。修复 = 组统计前把 reward 提升到 ≥FP32（保留 FP64 输入），复用既有 logged-reward 同步。验证：92 测试全过（六种估计器、多种 dtype 含 FP64、常数组、分数惩罚、参照 PPO 梯度）+ CPU 上以真实 OpenRLHF reward head + 随机初始化 tiny BF16 Llama 对 float64 算术参照吻合。
- **来源**：OpenRLHF PR #1334（2026-09-11；BF16 RM 的组基线舍入实例 + 修复）。https://github.com/OpenRLHF/OpenRLHF/pull/1334
- **行为效应**：无 crash——BF16 RM 下组内小幅 reward 差被舍入抹平：GRPO 的组相对信号**劣化为阶跃**（要么全零要么放大），细粒度偏好学习静默失效；混合 dtype 管线（RM bf16、trainer fp32）最脆弱。与 NUM.02（bf16/fp16 RL 引擎 mismatch）同族：**RL 管线 dtype 边界的精度契约**，这边打击点在 reward→advantage 的组统计入口；与 RL-ADV.01（组 std=0 零学习）症状相似但根因不同：那边统计上真无差、这边**有差被舍入掉**。
- **发现来源**：2026-09-12 每日扫描

### NUM.13 `gkd_jsd_mixed_precision_sign_flip`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：ms-swift Megatron GKD 的 `jsd_loss` 接受 **student FP32 / teacher BF16 混精 logits**，温度 scaling 前不统一提升精度：BF16 teacher logits 进 log-softmax/log-sum-exp 时精度不足，使数学上恒非负的 JSD = 0.5·KL(P_s‖M) + 0.5·KL(P_t‖M) 算出**负值**。同输入、同 mask、同 beta/temperature，仅把 loss 计算精度提到 FP32 → +0.000509（与 FP64 参照 +0.000509 吻合），原混精路径 −0.000170——符号翻转、幅度错约 3×。16 个捕获案例（2 步 × 8 rank）CUDA replay 与原 loss 全部吻合 1e-8（排除捕获错误）；最小合成 repro 直接调用安装版 `swift/rlhf_trainers/gkd_loss.py` 的 `jsd_loss`，无需模型/数据/通信。top-64 稀疏化（`gkd_logits_topk=64`）进一步放大敏感性。
- **来源**：ms-swift #10156（2026-09-16；ms-swift 4.5.3 + Megatron-Core 0.18.0 + mcore-bridge 1.6.3，Qwen3.5-9B teacher / Qwen3.5-0.8B student，TP1/DP8 H100）。https://github.com/modelscope/ms-swift/issues/10156 ；修复草案 PR #10102（`SWIFT_GKD_JSD_FP32=1` opt-in，默认关、未合）。https://github.com/modelscope/ms-swift/pull/10102
- **行为效应**：无 crash、无 NaN——**JSD loss 符号翻转**：loss 值与梯度相对高精度参照出现符号级偏差，训练日志只显示一个「小负数 loss」，极易当数值噪声忽略。与 NUM.12（BF16 reward 舍入掉组基线）同族：**RL/蒸馏管线的 dtype 边界精度契约**，这边打击点在散度计算的 teacher/student logits 混精入口；与 LOSS.08（GKD mean-of-means 归约错权）同文件域不同机制——那边归约几何错、这边输入精度错。
- **发现来源**：2026-09-17 每日扫描

### NUM.14 `qdora_magnitude_vector_downcast_silent_noop`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl `SFTTrainer` 的 QLoRA 低精度训练路径在 trainer setup 时把参数**统一下转型**，连带把 DoRA/QDoRA 的 `lora_magnitude_vector` 从 FP32 降到 BF16——该向量的每步优化器更新量小于 BF16 可表示精度，`param += update` 被舍入吞掉：**参数保持 trainable、optimizer 正常 step、存储值永不变化**。Colab 三臂对照：DoRA（magnitude 保持 FP32）正常更新；QDoRA（被降到 BF16）训练全程 magnitude 逐位不变；QDoRA + trainer init 后把 magnitude cast 回 FP32 → 更新恢复。DoRA/QDoRA 参考实现均用 FP32 持有该向量但未成文契约。TRL 1.13.0 + PEFT 0.20.0 + bitsandbytes 0.50.2。
- **来源**：trl #7268（2026-09-18；Colab repro + 三臂对照表）。https://github.com/huggingface/trl/issues/7268 ；2026-09-18 arcusbuilds 认领修复（评论区）。
- **行为效应**：无 crash、无 warning——magnitude vector 静默冻结，DoRA 的方向/幅度解耦更新只剩方向分量，loss 可能照常下降（Base+A·B 仍在更新），质量偏差延迟可见；「trainable 参数零变化」这一事实只能靠显式 diff 前后值发现。与 NUM.09（混精 cast 扫过量化参数）/NUM.08（BF16 计数间距 256）同族「dtype 边界的更新量小于表示精度」，差异在这边是 **optimizer 更新被舍入完全吞掉**（不是偏差是 no-op）；与 OPT.06（有效更新缩小 32768 倍训练停滞）症状面同为「名义在训、实际不更新」，那边是 scale 除错、这边是精度不够；「setup 期统一下转型打到特殊参数」与 NUM.09 的 Float16Module 打法同型。
- **发现来源**：2026-09-18 每日扫描

### NUM.15 `training_args_env_leak_sticky_bf16_precision`

stage: `sft/rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：transformers <5 的 `TrainingArguments.__post_init__` 把 mixed precision 作为**进程级环境变量**（`ACCELERATE_MIXED_PRECISION`）发布，并在下一次实例化时把**同一变量读回作默认值**：`bf16=False` 与「未设置」在该路径不可区分——一个 bf16 值**从此无法被后续配置清除**（只能被显式 `fp16=True`/`bf16=True` 覆盖）。TRL 配置默认 `bf16=True`（`base_config.py:105`），于是同进程里**第一个** trainer 把整个 worker 进程钉在 bf16；之后任何 `bf16=False` 的 trainer / bare `Accelerator` 在 transformers<5 + accelerate 旧版（不显式传 `mixed_precision`、回落读环境变量）组合下**静默继承 bf16 autocast**。trl CI 实测（维护者 root-cause）：chunked logps 路径自管 autocast 跑 bf16、full-logits 路径（未 `train()`、model 未被 prepare、forward 无 autocast 包装）跑 fp32——同测试两路径精度分叉，1709/2200 元素 mismatch、最大绝对差 8.1e-4（bf16 投影舍入、超 fp32 容差两个量级）、逐位确定（非顺序敏感）。transformers v5 显式传参构造 `Accelerator`、不再写环境变量。
- **来源**：trl #7320（closed；albertvillanova 2026-09-21 root-cause：transformers 4.56.2 `training_args.py` L1867-1873 的发布/读回 + accelerate 1.4.0 `state.py` L866-874 的环境变量回落；明确「测试外真实影响面」= 同进程第二配置或 bare Accelerator：notebook 重跑、bf16 训练后接 fp32 阶段、in-process sweep——单 run 脚本拿到自己配置的精度）。https://github.com/huggingface/trl/issues/7320 ；修复 PR trl #7321（`Restore ACCELERATE_MIXED_PRECISION between tests`，已 merge——修复形状是测试侧 autouse fixture 清理进程态，transformers v5 上游已结构性移除该泄漏）。https://github.com/huggingface/trl/pull/7321
- **行为效应**：无 crash、无 NaN——**显式请求 fp32 的阶段被静默换成 bf16 autocast**：评测/打分/第二阶段拿到的是低精度前向，误差 1e-4 量级、被普遍当作数值噪声。生产暴露面（维护者划定）：同进程多配置工作流（bf16 训练 → fp32 评测/sweep/notebook cell 重跑）；RL 管线里 policy/reference 模型若经不同构建路径（一方被 prepare、一方没有）也可能踩到同型 autocast 不对称。与 NUM.02（bf16/fp16 RL 引擎 mismatch）同族「精度选择的进程级粘性」：那边是两引擎显式配置不同、这边是**环境变量把一个配置的决定泄漏给后续全部配置**；与 KER.06（torch.compile 冻结 float kwarg）同为「首次调用的状态钉死后续语义」，断点一个在编译期常量、一个在进程环境；与 NUM.09（混精 cast 扫过量化参数）同为「dtype 决策越过了配置声明的边界」。
- **发现来源**：2026-09-22 每日扫描

---

# PAR · 并行 / 梯度归约

## PAR

### PAR.01 `zero2_bf16_multi_subgraph_incomplete_reduce`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed 0.18.3 起（#7665 PyTorch-compatible backward）用「参与参数计数」推断 backward 结束并触发 ZeRO-2 跨 rank 归约。同一 `backward()` 里一个子模块被用两次（主损失 + 辅助损失）时，hook 触发次数与预测计数不一致，**归约在该参数梯度尚未攒完时开火**。单卡 bitwise 正确；多卡静默丢一半子图。
- **来源**：DeepSpeed #8224（2026-08-06，0.18.3–0.19.3 仍在），回归自 #7665。https://github.com/deepspeedai/DeepSpeed/issues/8224
- **行为效应**：无 warning。LR=0 时 loss 与好版本一致（前向没问题）；LR 一起，主损失约翻倍、辅助项塌向退化解。单卡对照才能发现。
- **发现来源**：2026-08-17 基础调研

### PAR.02 `allreduce_silent_wrong_value`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：集体通信完成、无 timeout，但 payload 已错（坏 NIC、静默传输损坏、错误的 process group）。NCCL watchdog 只看活性，不看数值。
- **来源**：Llama 3 报告将 NCCL watchdog timeout 与 SDC 分列；数值错误集体通信是 SDC 的通信面。https://arxiv.org/abs/2407.21783
- **行为效应**：所有 rank 同步地拿到同一份错梯度，loss 仍平滑。比「一个坏 GPU」更难查。
- **发现来源**：2026-08-17 基础调研

### PAR.03 `dp_consensus_washes_out_bad_rank`

stage: `pretrain` · Cov: `var:scout_pretrain_consensus_outlier` · 置信度 `documented`

- **机制**：DP AllReduce 把一个坏 rank 的梯度平均进全局更新。没有副本对照时，错误被「共识」合法化。
- **来源**：SCOUT arXiv:2608.11034；Ma et al. 2502.12340 的多节点对照设定。https://arxiv.org/abs/2608.11034
- **行为效应**：全局 grad norm 略偏，没有 rank-local 报警。
- **发现来源**：2026-08-17 基础调研

### PAR.04 `stale_placeholder_wgrad_poisons_main_grad`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：GTP 对无梯度参数返回 placeholder wgrad（共享 dummy buffer 的视图）。当 `zero_out_wgrad` 置位（MTP 共享 embedding 场景），DDP backward post-hook 会执行 `main_grad.add_(param.grad)`；placeholder 未清零 → **上一次 backward 留在共享 buffer 里的陈旧值**被加进本 step 的 main_grad。
- **来源**：Megatron-LM PR #6565（fixes #6078；单测复现：把 dummy buffer 灌 12345.0 后要求梯度与灌 0.0 bitwise 一致）。https://github.com/NVIDIA/Megatron-LM/pull/6565
- **行为效应**：**先静默错梯度，dummy buffer 脏到一定程度才 NaN**。Nemotron-next mxfp8 收敛 A/B 已见差异。
- **发现来源**：2026-08-19 每日扫描

### PAR.05 `fsdp_degenerate_dp_group_grad_doubling`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron-FSDP（`data_parallel_sharding_strategy='optim_grads_params'`）下每-microbatch reduce 路径对**退化为单 rank 的 DP 组**（EP=DP 时 expert-DP 组、或 DP=1 时全部参数）也执行归约：本应「累加 g_k」变成「乘 2 再累加」，`buf_N ≈ 2^N·g`。梯度方向不变、幅值按 2^N 指数放大。
- **来源**：Megatron-LM #6660（2 GPU EP=2 复现；逐 microbatch 读 grad buffer 实测 absmax 严格 ×2.0；定位 `megatron_fsdp.py:668-678` 的 `grad_reduce_every_bprop` 触发）。https://github.com/NVIDIA/Megatron-LM/issues/6660
- **行为效应**：双重静默：N≤~64 时无溢出，方向正确但巨幅的梯度被 clip 归一，训练「看起来正常」而 expert 更新已错；N=128 时 fp32 L2 范数平方溢出 → `grad norm=inf` → clip 把**所有**梯度乘 ~0，run 以零梯度「训练」且 `number of skipped iterations = 0`。关 clip 甚至表面收敛（2·buf+g 保方向，Adam 逐元素归一）。与 PAR.01（hook 计数错开火时机）不同：这是**退化 DP 组 × 每-microbatch reduce** 的算术级放大。观测面（clip 如何洗掉它）见 OPT.04。
- **发现来源**：2026-08-20 每日扫描

### PAR.06 `fsdp_size1_mesh_shape_rank_confound`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：verl FSDP1 后端下 `actor.fsdp_config.fsdp_size=1`（「数据并行、不分片」的自然写法）被 `create_device_mesh` 表达成 2-D mesh，sharding 策略从 mesh 的 **rank 数**而非**形状**推导：FSDP 组 size=1 → 判定「无需分片」→ 也就**从不建立任何梯度同步**（FSDP1 的梯度同步寄生在分片 allreduce 里）。多 GPU trainer 上每个 rank 静默训练自己的独立副本。
- **来源**：verl #7493（生产 2 个月未察觉：3 trainer rank 每次更新只用 48 条采样序列中的 16 条；当日已修复关闭）。https://github.com/verl-project/verl/issues/7493
- **行为效应**：run 正常跑完、loss 下降、grad_norm/entropy/KL 全正常；ckpt 的 rank-0 副本实际只见过 1/world_size 的数据。与 PAR.05 同为「退化组」家族但是对偶面：PAR.05 是归约仍在跑但算术放大，本条是归约根本不发生——数据并行静默退化成各 rank 独立训练。
- **发现来源**：2026-08-21 每日扫描

### PAR.07 `sp_kv_head_global_locks_first_model`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed Ulysses 序列并行在 `deepspeed/sequence/layer.py` 维护**进程级**全局 `_ulysses_num_kv_heads`，首次走 uneven-head all-to-all 时按第一个模型的头数 memoize（scatter 方向可从 tensor 形状恢复头数，gather 方向与 `_SeqAllToAll.backward` 恢复不了，故做成了全局）。此后同进程**任何**头数不同的模型都按第一个模型的值切分 all-to-all 布局——第二模型的 SP 通信静默错排。与 AutoTP 进程级 kv-head 全局（#8231，#8241 以 per-model `AutoTPMeta` 修复）同族；#8241 修 AutoTP 时在 `layer.py` 留 TODO 明知未修本处。
- **来源**：DeepSpeed #8291（代码级分析 + 前后 commit 对照：#8241 前后行为一致；受影响场景点名 teacher+student 蒸馏 / OPD 与 RL actor + reference policy 开 SP）。https://github.com/deepspeedai/DeepSpeed/issues/8291
- **行为效应**：合法配置、无报错；第二模型（典型是 RL 参照策略或蒸馏 teacher）的激活/梯度在 SP 边界被错误布局的 all-to-all 静默污染。actor+ref 双 SP 场景下参照 logprob 本身已坏，KL 探针与 TIM 校正从根上失真（放大 RL-KL.01 / RL-RO.01 的观测盲区）。
- **发现来源**：2026-08-23 每日扫描

### PAR.08 `autotp_compile_drops_tp_collectives`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed AutoTP compile pass 先给每个注入的 `TensorParallel_Layer` 关掉模块级 collective（`defer_collectives_to_compiler=True`），承诺由图 pass 补 graph 级替换；但 `pass_insert_tp_collectives` 只走 `gm.graph.nodes` 顶层，**不下降进** activation checkpointing 被 Dynamo 提升出的子 `GraphModule`——检查点区域内既没有模块级也没有图级 collective，row-parallel matmul 输出保持 partial sum。`fullgraph=True` 不保护：高阶 op 本身就是全图捕获，无 graph break 也无警告。此前的双层防护均不存在：①「关掉的集合」与「插入的集合」是否一致无人校验；②`TORCH_COMPILE_DISABLE=1` / dynamo 禁用时 deferred 层**永久 eager 执行且 collective 已关**。
- **来源**：DeepSpeed #8357（2×H200 实测：MLP+ckpt forward 2.7e-01 / worst grad 9.2e-01；HF Llama + `gradient_checkpointing_enable()` 3.1e-02 / 2.6e-01，**0/14** collective 插入，全部 exit 0；HF 默认 `use_reentrant=False` 恰是入子图的变体，故「开梯度检查点」即中招）；修复 #8355（`iter_graphs` 递归遍历 + 覆盖校验 fail-closed + deferred 层 eager 执行断言，把「静默错值」默认改成「大声失败」）。https://github.com/deepspeedai/DeepSpeed/issues/8357
- **行为效应**：`model.gradient_checkpointing_enable()` + `"passes":["autotp"]` 组合下静默错前向、错梯度、exit 0；插入还可能是**部分**的（检查点块外的层仍有 collective），图检查起来像被 instrument 过。与 PAR.02（链路层 allreduce 错值）不同层：这是**编译改写覆盖缺口**，单卡 reference 对比才能抓。
- **发现来源**：2026-08-31 每日扫描

### PAR.09 `cpu_typestring_bucket_skips_allreduce`

stage: `shared` · Cov: `NEW` · 置信度 `verified`

- **机制**：DeepSpeed ZeRO-1/2 非连续梯度归约按 legacy 类型串 `torch.{device}.FloatTensor` 分桶（`split_half_float_double()`）。CPU 张量的合法类型串是 `torch.FloatTensor` **不带设备前缀**，与构造出的模式永不匹配 → bucket 列表恒空 → **梯度 allreduce 整体静默跳过**，每个 rank 用自己的本地梯度继续更新。`gradient_allreduce_op=mean` 时完全不可见（各 rank 梯度相同，未归约的本地值恰等于 mean）；`sum` 时结果错 `world_size` 倍。GPU 上 `torch.cuda.FloatTensor` 恰好匹配，18 个 GPU CI 变体全绿——CPU 多进程路径是首个踩到的执行环境。修复 = 按 `t.dtype` 分桶（设备无关）。
- **来源**：DeepSpeed PR #8382（CPU/gloo world 2–3 实测：`TestGradientAllreduceOp` 2 failed→18/18，`TestUnmanagedGradientAccumulationOffload` 数值不匹配→修复；GPU 行为按构造不变）。https://github.com/deepspeedai/DeepSpeed/pull/8382
- **行为效应**：无 crash、无警告；`mean` 语义下数值恰好对（最静默），`sum` 语义下静默放大 world_size 倍——梯度累积 + `sum` 的 CPU 参考实现/测试会把错值当基线。与 PAR.01（bf16 多子图不完整归约）同症状（归约缺失），但根因是**设备名类型串构造**而非子图遍历；与 PAR.02（allreduce 静默错值）区分：这边 collective 根本没发。
- **发现来源**：2026-09-04 每日扫描

### PAR.10 `traced_all_to_all_zero_backward`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed Ulysses SP 的序列/头交换曾是 `torch.autograd.Function`：forward 分配 `torch.empty_like(input)` 并用 `all_to_all_single`（**写入输出参数**、不返回值）填充。torch.compile 不把 autograd function 当黑盒——inline 展开后 tracer 无法把「empty 张量 + 非产出型 collective 变异」建模为数据依赖：图中交换输出对交换输入的依赖消失，**自动微分出的导数恰为零**，手写 backward（转置交换）永不运行。执行时 collective 仍在搬字节，所以 forward 数值完全正确。修复 = 换 `_dim_zero_all_to_all` 调 `torch.distributed.nn.functional.all_to_all_single`（返回值的可微 collective），依赖进图、生成 backward 即转置交换。
- **来源**：DeepSpeed PR #8491（2026-09-11；「编译后的 Ulysses job 对 q/k/v 投影零梯度：一步更新后张量与 optimizer moments=0 bit 一致，输出投影/MLP/残差流照常更新」）。https://github.com/deepspeedai/DeepSpeed/pull/8491
- **行为效应**：无 crash、loss 有限、无警告——attention 的 q/k/v **静默停止学习**，残差其余部分正常前进；若不开 compile 同代码正确，归因极易指向超参/数据。与 PAR.08（编译 pass 漏插 collective）同是「torch.compile × 集合通信」改写缺口：那边 forward 错值（partial sum 不归约），这边 forward 正确、**backward 断流**；与 KER.06（compile 冻结 float kwarg）同族——tracer 对副作用型原语（collective 写出参）的建模缺口，受害面是梯度而非配置。
- **发现来源**：2026-09-12 每日扫描

### PAR.11 `gtp_expert_grad_overscaled_by_gtp_egtp`

stage: `pretrain` · Cov: `var:fsdp_degenerate_dp_group_grad_doubling` · 置信度 `documented`

- **机制**：Megatron GTP（梯度张量并行，rank 从 DP 轴刻出）下 MoE expert 梯度带一个 `GTP/EGTP` 的伪因子：dense 参数能从 `_allreduce_replicated_grads_over_gtp_remat_group` 的 AVG 补回 DDP `1/DP` 缩水（DP 随 GTP 增大而缩小），expert 参数只见过 EGTP 轴规模的补偿——`GTP/EGTP` 因子永不消除，默认 `EGTP=1` 时每个 expert 更新被放大 `GTP` 倍。修复 = `_rescale_expert_grads_for_gtp_remat()` 在非 TP all-reduce 后乘 `EGTP/GTP`（标量乘而非 AVG：同一 gtp_remat 组内各 rank 持不同专家，无可对齐副本）。同族第二处：`expert_gradient_scaling_factor` 复用被 GTP 缩小的 `dp_cp_group.size()` 当 prescale（#7165，A3B_30B GTP64 EP64 256×GB300 实测 grad-norm 偏 8.4×、params-norm 8.7×，修复后双双落回 A/A 带内）。`calculate_per_token_loss` 路径免疫（gtp 轴 SUM-reduce、除数已计 gtp peer token）。
- **来源**：Megatron-LM PR #7160（2026-09-09，closed merged；「失败是静默的——无 error 无 NaN，只表现为收敛差距」）。https://github.com/NVIDIA/Megatron-LM/pull/7160 ；Megatron-LM PR #7165。https://github.com/NVIDIA/Megatron-LM/pull/7165
- **行为效应**：无 NaN、无 crash；expert 梯度恒放大 GTP 倍（或 prescale 错），方向不变——全局 clip 开启时被归一掩盖（连坐 OPT.04），MoE 与 dense 的学习率平衡被系统性打破，收敛 gap 无日志可查。与 PAR.05（退化 DP 组 2^N 翻倍）同型：并行轴重切后**梯度归一因子与实际副本数失配**，dense 路径自愈、expert 路径漏掉；与 OPT.05（expert 被优化器遗忘）互补：那边专家不动、这边专家被过度驱动。
- **发现来源**：2026-09-12 每日扫描

### PAR.12 `hybridep_intermittent_bad_gradients`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron MFSDP v2 + **Flex/HybridEP dispatcher**（`NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`）+ 专家 outer optimizer sharding（`expert_outer_dp_sharding_strategy=optim`）下，全局梯度范数**间歇性**异常放大或 NaN。已排除的必要条件：CUDA graphs（训练/优化器图全关，eager 复现）、梯度裁剪（`clip_grad=0.0` 下复现）、rank-capacity 模式（unset 时同样 spike 到 3,254）、FP32 主梯度存储（FP16 主梯度 2/3 异常）。隔离对照：同模型 `alltoall` dispatcher 3/3 健康十步跑、轨迹贴合；换回 Flex/HybridEP 即复现 spike/NaN（65.97 @step3 / 3254.10 @step2）。根因尚未定位（issue 提供约化 repro，疑点在 flex dispatcher 的专家梯度归约路径）；敏感于插桩、健康跑也会出现，两节点 8×GB200 DeepSeek-V3 proxy（MLA + 16 专家 top-8，EP=2，4 个分布式优化器实例）。
- **来源**：Megatron-LM #7400（2026-09-16；PRE_OPT 仪器读数 6.2→107.7→84.6→NaN 的原始序列 + 四臂隔离表；MCore `7a9e9c6` + #7277、DeepEP hybrid-EP `de0dd11`、TE 2.18.0）。https://github.com/NVIDIA/Megatron-LM/issues/7400 ；根因突破（2026-09-17/18 补）：报告者先在双 GB200 上以纯 Python 脚本（TE grouped GEMM + NCCL all-gather/reduce-scatter 并发异流、无 HybridEP/FSDP/autograd/优化器/图）复现，再以**独立 CUDA C++ reproducer**（直接 `cublasLtMatmul` + `ncclAllGather`/`ncclReduceScatter`，单进程每 GPU 一 host 线程、无 PyTorch/TE 依赖）复现同一「retained output」症状——8×BF16 GEMM M=N=2048/K=4096、FP32 累加、beta=0 本应覆写每个输出元素，检查却见 `2**100` 预填值残留；已提交 NVIDIA 内部 bug 6798870。https://nvbugspro.nvidia.com/bug/6798870 ；关联线索 #7232（HybridDeviceOptimizer 被 distributed optimizer 双重实例化，同域 c=0 未定）。https://github.com/NVIDIA/Megatron-LM/issues/7232
- **行为效应**：无 crash：标准训练日志在 clip 关闭时打印 `grad norm: 0`（不代表梯度为零），异常只能靠独立 PRE_OPT 仪器看见；间歇性 + 对插桩敏感使复现困难，健康跑与坏跑交替。与 PAR.06（verl `fsdp_size=1` 静默关同步）区分：那边是配置面同步被关、数值恒错；这边数值**间歇**坏且 eager/无图/无裁剪三重排除后仍存，指向 dispatcher/归约路径本身的竞争或错值。clip 开启时会被裁剪吃掉变成「间歇性梯度被错误压制」（连坐 OPT 家族裁剪面）。
- **发现来源**：2026-09-17 每日扫描

### PAR.13 `liger_gate_skips_ddp_reducer_hooks`

stage: `sft` · Cov: `NEW` · 置信度 `verified`

- **机制**：trl `DPOTrainer`/`KTOTrainer` 的 `compute_loss` 在 **unwrapped model** 上跑 loss，只对 ZeRO-3/FSDP 走 `_forward_redirection`——plain DDP 下两个条件都不成立，`DistributedDataParallel.forward()` 从不执行 → `Reducer::prepare_for_backward()` 不被调用 → `expect_autograd_hooks_` 保持 false，reducer 的 autograd hooks 全部早退、**从不 all-reduce**，每个 rank 保留本地梯度。触发门是 `use_liger_kernel=True`（依赖 liger-kernel 安装，恰好是 unwrap 的动机）；由修复 ZeRO-3 的 #6372 引入（此前传 wrapped model、DDP 正确），v1.10.0–v1.12.0 在内。2×H100 实测：liger 开 → 25/25 参数跨 rank 梯度不同（max_abs_diff 1.265e-01）；liger 关 → 0/25（Qwen3-0.6B 310/310 同结论）。分布式测试 `test_dpo` 参数化 `ddp` 但从不传 `--use_liger_kernel`，且只断言 exit 0——静默失同步恰好骗过。
- **来源**：trl #7245（2026-09-16；最小 repro + GradProbe all-gather 对比；修复方向 = 复用 `kto_trainer.py:1322`/#7077 已有的 DDP-aware 条件 `is_fsdp_enabled or model is not unwrapped_model`）。https://github.com/huggingface/trl/issues/7245 ；维护者测量更正（2026-09-16/18 补，qgallouedec 评论）：KTO 本体走 `_ChunkedLogProbFunction` 吃 redirect 无碍；DPO 在 main 上加同 guard 会让每 rank 抛 `NotImplementedError`（`FusedLinearDPOFunction` 内部 `torch.func.grad_and_value` 不能在 `DistributedDataParallel.forward()` 里跑）——**DPO 需先等 #7243 迁到 chunked 路径**，迁移后同 guard 达 0/310 参数跨 rank 差异；KTO 侧 chunked logprob 本身不 all-reduce 是既有独立缺口
- **行为效应**：无 error、无 warning——N-GPU DDP 跑退化成 N 个独立单卡跑（各 `per_device_train_batch_size`），保存的 ckpt 只是 rank 0 的；收敛表象正常、有效数据量与多样本语义全部失效。与 PAR.09（CPU 类型串分桶跳过 allreduce）同症状面「梯度归约静默缺失」：那边是 ZeRO 分桶谓词失配、这边是 DDP reducer 生命周期被 unwrap 路径绕过——又是「fused/快捷路径绕过包装层」家族（cf. RL-KL.04/#7009 Liger 提前 return 丢 aux loss、#7062 vendored Liger 凿穿配置面）的第 6 个实例。
- **发现来源**：2026-09-17 每日扫描

### PAR.14 `explicit_pg_collection_group_fallback`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron-Core 传入与全局 MPU 组不同的显式 `ProcessGroupCollection`（Bridge decentralized-PG / 训推不同并行度评估真实使用）时，五条路径静默回落到**全局 MPU 组**执行集体通信：① MTP full-recompute 的 TE checkpoint 路径与非 TE `tensor_parallel/random.py` split/gather 都取 `get_tensor_model_parallel_group()` 而非 `self.tp_group`（#7403 只修了相邻的 MTP 输入投影）；② GPT MoE `padding_mask` scatter 漏传 `group=`（gpt_model.py:369，同函数上方 hidden-state scatter 传了）；③ `CrossAttention` 的 `linear_q/linear_kv` 构造不传 `tp_group/pg_collection`（stock T5 decoder spec 可达）；④ `LLaVAModel` embedding scatter 无 group；⑤ attention forward 把 `pg_collection.cp` **覆写回 build-time CP 组**且变异调用方共享的 collection（df4996f9f 引入）。后果都是「同形状、混数据」：4-rank repro 中 MTP 重算恢复 `[0,1,12,13]` 而非 `[0,1,2,3]`；LLaVA 样本 A 变 `[0,1,0,1]`；eval batch 2/3 输出误差 0.51/0.29 而 batch 1 吻合 1.2e-7——首 batch 掩盖。
- **来源**：Megatron-LM #7452（ezyang 2026-09-17；Astra + spmd_types 对 `7a9e59de7` 的审计，findings 4/5/6/7/13，每条附 CPU/Gloo 2–4 rank 自包含 repro 与 checker 拒绝见证）。https://github.com/NVIDIA/Megatron-LM/issues/7452
- **行为效应**：无 crash、无 NaN——重算激活 / SP all-gather / eval 分片拿到**别的 replica 的数据**，错误随 batch 与路径组合静默滚动。与 DATA.09（caller mask 被库内默认值顶掉）同打击「调用方运行时上下文被全局默认覆盖」；与 PAR.06（`fsdp_size=1` mesh 混淆）同族「显式进程组语义与全局 MPU 状态分叉」，这边断在**集体通信的组选择**；⑤的 eval-CP 回退与 OBS.07（eval 混入训练态）表现面相邻但机制独立。
- **发现来源**：2026-09-18 每日扫描

### PAR.15 `bridge_wan_gather_default_grad_sums_tp_copies`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron-Bridge 的 Wan 扩散模型 decoder（wan_model.py:267，Bridge `34d8d4b8`）对 `gather_from_sequence_parallel_region(x)` 用**默认** `tensor_parallel_output_grad=True`——backward 变 reduce-scatter。但 Wan 在每个 TP rank 上计算**同一个完整 loss**，于是 backward 把 TP 份相同梯度**求和**：TP=2 时 decoder 权重梯度范数 7.8876387 而非 3.9438193（恰好 ×TP），forward 输出各 rank 一致（错值只在梯度）。H100 Wan perf recipe 用 TP=2、CP=4、SP——默认即中招。修复 = 该 gather 传 `tensor_parallel_output_grad=False`，并与模型逐参数 TP 梯度平均标志一起复核。报告时点尚未对 Bridge 单独开 issue（审计报告注明 to be filed）。
- **来源**：Megatron-LM #7452 Appendix B（2026-09-17；ezyang 审计；数值 + 修复 + recipe 暴露面）。https://github.com/NVIDIA/Megatron-LM/issues/7452
- **行为效应**：无 crash、无 NaN——**decoder 参数的有效学习率恒 ×TP**，forward 全对、loss 正常下降，唯一表象是 Wan decoder 收敛动态随 TP 度不可比。与 PAR.11（GTP 梯度 ×GTP/EGTP）/MOE.12（aux 梯度 ×TP/CP）同病「梯度恒乘并行组大小」的第 3 个实例，这边断点是 **SP gather 算子的 grad 模式默认值与「每 rank 全 loss」的模型写法不匹配」；与 LOSS.09（G 缩放漂移）同为「并行度旋钮静默改有效学习率」。
- **发现来源**：2026-09-18 每日扫描

### PAR.16 `ddp_normalization_shrinks_aux_loss_grad_dp_factor`

stage: `pretrain` · Cov: `var:aux_loss_normalization_domain` · 置信度 `documented`

- **机制**：Megatron `--global-aux-loss` 的 aux 梯度路径把全局（跨 microbatch 累积、已对 aux loss 归一化的）aux loss 梯度交给 DDP 归约——DDP all-reduce 带 `1/DP` 平均，而该梯度的语义是**每步一份的标量损失梯度**（每个 DP rank 对同一 aux loss 贡献同一份、不是各持 partial）：AVG 之后 aux 梯度被**缩小 DP×**（audit 实测 DP=2 时 norm 0.3794 vs 正确 0.7588）。#7452 Appendix C 列为「常数尺度误差、无数值 dataflow 见证不计数」；09-19 裁决收录——**「应 SUM 的语义量被 AVG」本身就是分布类型错**，与 MOE.11（分母时序域）同一 aux-loss 归一化语义域、不同断点（归约算子选择），报表者注明 Qwen3-Next Bridge recipe 选中该路径。审核方（0z5a，#7481 作者）已声明其修复**不覆盖**本项（`global aux-loss normalization` 是独立在跟踪的工作）。
- **来源**：Megatron-LM #7452 Appendix C（2026-09-17；ezyang Astra+spmd_types 审计；DP=2 数值对照）。https://github.com/NVIDIA/Megatron-LM/issues/7452
- **行为效应**：无 crash——**aux loss 的有效系数随 DP 度静默缩小 DP×**：负载均衡压力不足，专家分布漂向不均衡；DP 扩容（常用吞吐手段）即改变训练目标语义，且 loss 曲线不显任何异常。与 PAR.11 / PAR.15 / MOE.12 同族「并行度旋钮 × 恒定因子改梯度」，方向相反（缩小而非放大）；与 MOE.11 互补成对：同一 aux loss 的分子域（累积计数）与归约域（SUM/AVG）各有一个静默断点；与 PAR.05（退化 DP 组 2^N）都打击「对什么求平均」的假设，这边是**语义整体量被当可平均量**。
- **发现来源**：2026-09-19 每日扫描

### PAR.17 `gradient_reduce_preprocessing_ignores_scaling_factor`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron-FSDP v1 `gradient_reduce_preprocessing` 在 `average_in_collective=True` 时**忽略 `scaling_factor`**：该模式把梯度平均折进集合通信算子执行，预处理阶段本应配平的缩放不再应用——expert 梯度 `[6, 10]` 而非 `[3, 5]`（**恒 ×2**，DP=2 repro）。#7452 Appendix C 第二条；09-19 裁决收录：与 PAR.05「应累加的梯度被乘 2」互为镜像（**应缩放的梯度没缩放**），且经 Qwen3.5-35B-A3B MFSDP recipe（EP=2）真实可达。修复 = preprocessing 阶段对 `average_in_collective` 分支同样施加 scaling（或显式拒绝该组合）。
- **来源**：Megatron-LM #7452 Appendix C（2026-09-17；audit 数值对照）。https://github.com/NVIDIA/Megatron-LM/issues/7452
- **行为效应**：无 crash——MFSDP v1 + `average_in_collective=True` 的 run 里**有效学习率恒 ×DP**，loss 照常下降、收敛动态随 DP 度不可比。与 PAR.05 同族「FSDP 梯度缩放算术」第 2 个实例（那边 2^N 翻倍、这边 DP× 不缩）；与 PAR.16 同日收录且方向相反（16 缩小 / 17 放大）——同一 audit 揭示 FSDP/DDP 缩放语义的两个对偶断点；与 PAR.13（liger 绕过 reducer）不同：那边梯度从不归约、这边归约了但系数错。
- **发现来源**：2026-09-19 每日扫描

---

# CKPT · 检查点

## CKPT

### CKPT.01 `expert_reshard_id_permutation`

stage: `pretrain/shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：用与保存时不同的 EP degree 加载，或 `num_experts % ep_size != 0`，或 Megatron↔HF 专家排序约定不同，router 的 expert-id 指到另一组权重。
- **来源**：DeepSpeed #5794、#6891；Megatron-LM #1754。交叉：推理编目 12.x 专家加载。https://github.com/deepspeedai/DeepSpeed/issues/5794
- **行为效应**：加载成功。MoE 模型流畅但能力像换了一批专家；router 熵可能仍看起来健康。
- **发现来源**：2026-08-17 基础调研

### CKPT.02 `zero_optimizer_state_partition_mismatch`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：ZeRO-1/2/3 把 Adam 矩按 rank 切开。换 world size / 换 ZeRO stage / 换 TP 后，权重对上了、矩没对上，或 universal checkpoint 的几何分片图与当前并行布局不一致。
- **来源**：DeepSpeed #8230（universal checkpoint 几何分片 RFC）；#8198 AutoEP ZeRO-1/2 universal conversion。https://github.com/deepspeedai/DeepSpeed/issues/8230
- **行为效应**：resume 后 loss 先跳一下再「继续降」，其实是矩错位后的另一次优化；或静默用零矩当新 Adam。
- **发现来源**：2026-08-17 基础调研

### CKPT.03 `from_pretrained_silent_random_init`

stage: `sft/shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：键名对不上时加载器不报 missing key，未命中的参数保持随机初始化。训练侧表现为「从错误的起点微调」。
- **来源**：交叉：`inference_error_review` **12.2** `from_pretrained_silent_random_init`。训练加载路径同一漏洞。
- **行为效应**：SFT/RL 曲线能降，降的是随机子模块上的拟合；评测远低于同数据应有水平。
- **发现来源**：2026-08-17 基础调研（从推理编目回链）

### CKPT.04 `te_grouped_empty_extra_state_misdecode`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：无状态 FP8 recipe 把「没有 extra state」存成空 byte tensor。grouped-linear 的 ckpt 适配器在 FP8 开启时把它当序列化元数据解码，得到空状态再索引失败或丢掉 per-expert 键布局。
- **来源**：Megatron-LM #6509 / 已合入的 #5997。https://github.com/NVIDIA/Megatron-LM/pull/6509
- **行为效应**：保存/恢复 grouped TE 专家时 FP8 元数据错位；128 卡 DeepSeek V4 上表现为必须修完才能稳定 resume。硬失败或静默丢 extra state 取决于路径。
- **发现来源**：2026-08-17 基础调研

### CKPT.05 `dist_ckpt_strictness_assume_ok`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron 分布式 ckpt 的 `dist-ckpt-strictness` 默认可取 `assume_ok_unexpected`：多出来的键被当成无事。并行布局演进后，本该报错的 unexpected key 被吞掉。
- **来源**：Megatron-Core 分布式 checkpoint 文档（`dist_ckpt_strictness`）。https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html
- **行为效应**：加载「成功」，部分新模块或旧布局残留参数被忽略/错配。
- **发现来源**：2026-08-17 基础调研

### CKPT.06 `config_default_swallows_cli_override`

stage: `pretrain/sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：模型构造函数参数带默认值，入口脚本转发了兄弟参数却漏掉这一个，CLI 值被静默丢弃。实例：Megatron `pretrain_vlm.py` 不把 `--rotary-base` 传给 `LLaVAModel`，语言模型恒用 dataclass 默认 10000；而 rotary base 是预训练 ckpt 的固有属性（Llama 3.x=500000）。同一构造在 `examples/multimodal/model.py` 是传的（证明是遗漏非有意），且基础 `TransformerConfig` 无 `rotary_base` 字段可兜底——kwargs 是唯一通道。
- **来源**：Megatron-LM #6585（2026-08-17，含 call-site 级复现命令）。https://github.com/NVIDIA/Megatron-LM/issues/6585
- **行为效应**：无 crash、无日志。LM 的 `RotaryEmbedding` 以错误 base 训练，长上下文质量静默受损；视觉侧不受影响，归因更难。配置值即「ckpt 属性」，错在保存之前，任何加载侧检查都救不回。
- **发现来源**：2026-08-18 每日扫描

### CKPT.07 `dead_or_shadowed_arg_silent_noop`

stage: `pretrain/rl` · Cov: `var:config_default_swallows_cli_override` · 置信度 `documented`

- **机制**：CKPT.06 的同族变体——配置面上的参数**声明了但从未被消费**（死参数），或被内部硬编码默认值**覆盖**（shadowed），两种都静默无效。实例一：ms-swift `--mtp_shared_weights` 全库只有定义行、无任何 consumer，用户以为在训部署对齐的单层重复 MTP drafter，实际得到 N 层独立权重（serving 引擎从不执行这些层）。实例二：verl `critic.model.model_type` 在 resolved config 里正确，`RayPPOTrainer.init_workers()` 构造 TrainingWorkerConfig 时用硬编码 `value_model` 覆盖，`EngineRegistry.new()` 静默选回 stock 引擎。
- **来源**：ms-swift #9936（pip download + grep 全库复现）。https://github.com/modelscope/ms-swift/issues/9936 ；verl #7460（sentinel model_type 复现）。https://github.com/verl-project/verl/issues/7460 ；同族新实例（2026-08-20）：verl #7463 `actor.router_replay.mode` 是死 key，worker 只读 `actor.{megatron,veomni}.router_replay.mode`，官方 Ascend 文档指向死 key，官方路径设 R3 也静默 `disabled`，`experimental/separation` 还从死 key 读 R2/R3 冲突分支。https://github.com/verl-project/verl/issues/7463 ；verl #7462 critic 批处理键从未复制进 TrainingWorkerConfig（`use_dynamic_bsz`/`ppo_micro_batch_size_per_gpu`），v1 trainer 另用错误的赋值目标覆盖 token budget，文档键 `forward_max_token_len_per_gpu` 从未被读。https://github.com/verl-project/verl/issues/7462 ；verl #7476 `ulysses_sequence_parallel_size` 对 Qwen3.5 静默不生效（per-rank 激活仍全长），allocator snapshot 才发现。https://github.com/verl-project/verl/issues/7476 ；trl #6791（2026-08-21 补）：dict 型 config 字段（如 `--dataset_kwargs '{"skip_prepare_dataset": true}'`）无法从 CLI 设置——`HfArgumentParser._parse_dataclass_field` 只特判 list，dict 落入通用分支被当成标量，CLI 上游限制让「跳过数据集预处理」这类开关只能进 `model_init_kwargs`，误用即静默不生效。https://github.com/huggingface/trl/issues/6791 ；verl #7597（2026-08-30 补）：嵌套 `actor.fsdp_config.strategy` 冲突值在 main 上静默丢弃——真源是顶层 `actor.strategy`，嵌套写法（#7540 报告命令即用）看起来设了 FSDP2、实际跑 FSDP1，19 个声称 FSDP2 的官方 launcher 同样踩中。https://github.com/verl-project/verl/pull/7597 ；verl #7662（2026-09-02 补）：`multi_turn.use_inference_chat_template`（文档推介为「rollout 对齐生产推理」的开关）与默认 `strict` 的 `multi_turn.tokenization_sanity_check_mode` 唯一运行时消费者都在 `AsyncRolloutRequest`——该类在 main 上仅被测试引用，活跃 agent-loop 路径不读这两个键，设置后无行为变化也无警告；文档推荐的 sanity check 同样实际不跑。https://github.com/verl-project/verl/issues/7662
- **行为效应**：无报错、无日志差。ms-swift 侧训出的 MTP 结构与部署结构错位；verl 侧自定义 critic 引擎静默退化。同 CKPT.06：错在构建期，加载侧检查救不回。#7476 实例：SP 无效被误读成显存容量问题，浪费整簇排查。
- **发现来源**：2026-08-19 每日扫描（实例三/四/五 2026-08-20 补）

### CKPT.08 `epoch_boundary_resume_silent_skip`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：多 epoch SFT 在 epoch 边界存 ckpt 后 resume，dataloader/sampler 状态恢复路径错过「进入下一 epoch」的迁移，**静默跳过整个下一个 epoch**；run 以 status 0 正常退出，请求数量的更新从未发生。
- **来源**：verl #7401（生产 `CheckpointHandler` + `StatefulDataLoader` 的确定性复现 + 4×3090 GPU A/B）。https://github.com/verl-project/verl/issues/7401
- **行为效应**：欠训模型被当作完成 run，操作员/调度器看到的是绿色。观测面与 OBS.04 同型（exit 0 即真相缺失），根在 resume 契约。
- **发现来源**：2026-08-19 每日扫描

### CKPT.09 `mxfp8_roundtrip_requant_scale_drop`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：`--fp8-param-gather` 路径的 ckpt 只存 BF16 反量化权重、不存 MXFP8 block scale；load 侧必须重量化，而 `MXFP8 → BF16 → MXFP8` **不是幂等的**。机制（逐 block 实测）：反量化后 block max ≤ 224·s（半 ceiling）时 scale 会掉一个 E8M0 指数，该 block 全部 32 个 code 翻倍——每个元素恰差 -1 code、单向、有界。
- **来源**：Megatron-LM PR #6666（附独立 repro 仓；132 个 weight key 中含 'scale' 的为 0；GB300 单 tensor 1664/32768 code、52/1024 scale 差异；scale 减半当且仅当 dequant max ≤ 224·s，1024/1024 无例外；4×GB300 真 save/load 实测多参数量级差异）。https://github.com/NVIDIA/Megatron-LM/pull/6666
- **行为效应**：**值无损的编码损坏**：`dequant(W)` 与 `dequant(W2)` 逐元素 0/32768 差异，GEMM forward/dgrad 0 差异 → 训练继续且 bit 级等价，但零容差 ckpt parity check 失败、resume 后的量化编码与 fp32 master 的派生关系断裂。量化损失本身发生在训练期（ckpt 之前），round-trip 不增加误差。收录角度：合法配置下「加载成功但 ckpt parity 检查被骗过」的静默面；修复是 load 后从 FP32 master 重导量化参数。
- **发现来源**：2026-08-20 每日扫描

### CKPT.10 `zero3_lazy_teacher_init_reads_partitioned_state`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl `PEFTAdapterEMACallback` 在 `on_train_begin()` 才惰性创建 SDPO teacher adapter——此时 student 已被 DeepSpeed ZeRO-3 分片。两个静默面：(1) `get_peft_model_state_dict()` 直接读未 gather 的分区参数，读到 `numel()==0` 的空 shard 视图，teacher 与 EMA shadow 从**空张量**初始化；(2) 分片后插入的新参数脱离 ZeRO instrumentation，首个 forward 崩（`'dict' object has no attribute '_in_forward'`）。
- **来源**：trl #6828（实测：分区态 28 个 adapter tensor `numel()==0`；修复方向 = 在 DeepSpeed prepare 之前建结构 + gather 后取 detach clone）。https://github.com/huggingface/trl/issues/6828
- **行为效应**：face (1) 静默——teacher/EMA 初始化源错但训练照常，蒸馏目标从错误 anchor 出发；face (2) fail-fast。与 CKPT.02（ZeRO 分片图与加载布局错位）不同：这里分片布局是对的，错的是**读取时机在 gather 窗口之外**——合法配置下生命周期序错，非几何错。
- **发现来源**：2026-08-21 每日扫描

### CKPT.11 `resume_resets_first_microbatch_flag`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `TEGroupedLinear.is_first_microbatch` 是进程局部 flag，**不在 ckpt 状态里**：resume 后重置为 True。MTP 层跨 depth 复用（同一 layer 每步前向多次）时，TE 用该 flag 决定 main_grad 是 accumulate 还是 overwrite（beta=0）——resume 后首次共享层调用变成 overwrite，**丢掉一个 depth 的梯度**，resume 训练与不中断训练从此分叉。
- **来源**：Megatron-LM #6965（三段对照：不中断=grad(d2)+grad(d1)；resume 后=只剩 grad(d1)；修复后 2 节点 iter_10 全部 7 个 state shard **逐字节一致**、64 节点 255/256；关 grad-accum fusion 分叉即消失，与机制预测一致；修复为 `is_first_microbatch_tracked(config)` 不维护时给 TE 传 None）。https://github.com/NVIDIA/Megatron-LM/pull/6965 ；Megatron-LM PR #7186（2026-09-12 补：MFSDP v2 hybrid 数据并行同族——HSDP/HFSDP 的 DP-outer 轴跨 micro-batch 保持 Partial、最后一个 backward 才归约，MFSDP v2 适配器缺 `no_sync`：每个 backward 都 finalize 该轴并标记累积 buffer stale → 下个 microbatch 清零，**只有最后一个 microbatch 的梯度到达优化器**；修复 = adapter 实现 `no_sync` 上下文 `microbatch(ctx, is_last=False)`）。https://github.com/NVIDIA/Megatron-LM/pull/7186
- **行为效应**：**run-to-run A/A 复查抓不住**——两个 fresh 进程犯同样的错且互相一致；只有「训了 N 步的进程」对「刚 resume 的进程」三进程对照才可见。表象是 resume 后一步 grad-norm 分叉而 loss 暂时还跟得上。与 CKPT.08（resume 静默跳 epoch）同族：状态不完整的 resume，但丢的是**梯度累积语义**而非数据游标。
- **发现来源**：2026-08-30 每日扫描

### CKPT.12 `fsdp_merger_concatenates_replicated_buffers`

stage: `sft/rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：FSDP 分片 ckpt 中**复制**条目（persistent buffer 与未被 wrap 的参数）以普通 tensor 而非 `DTensor` 存储——每个 rank 各存一份完整拷贝。verl `FSDPModelMerger._load_and_merge_state_dicts` 对无 placement 信息的条目落入 `torch.cat(..., dim=0)` 通用分支，把 N 份相同拷贝**拼接**成 world_size 倍大小的 tensor（`_merge_by_placement` 明知 replicate 应取 `tensors[0]` 去重，此分支漏用该规则）。多模态/复合模型重灾区：8-rank Gemma4 ckpt 中 `layer_scalar [1]→[8]`、vision tower `std_scale/std_bias [1152]→[9216]`，实测非 DTensor 条目集合恰等于模型 persistent buffer 全集（32 项同名）。
- **来源**：verl #7612（CPU 单测在 main 上 `assert torch.Size([4]) == torch.Size([1])` 失败、修复后通过；真实 ckpt 逐条验证）。https://github.com/verl-project/verl/pull/7612
- **行为效应**：merged ckpt 形状错误 → 加载侧通常 shape mismatch 报错；但**形状碰巧兼容时**（如 `[1]→[8]` 类标量 buffer 沿新维度复制、或下游 reshape 吞掉多余维度）会静默把 buffer 数值排布改写。与 CKPT.01（专家 reshard 排列错位）同族的 merge 侧错位，但根因是**复制条目被当分片条目拼接**，非 id 排列。
- **发现来源**：2026-08-31 每日扫描

### CKPT.13 `wrong_rng_object_checkpointed`

stage: `sft/pretrain` · Cov: `var:rank_shuffle_seed_desync` · 置信度 `verified`

- **机制**：组件持有自己的 RNG（`np.random.default_rng(seed)`）供采样，`state_dict()`/`load_state_dict()` 却存取**另一个 RNG 对象**的状态——DeepSpeed curriculum `DeepSpeedDataSampler` 存的是全局 legacy `np.random.get_state()`。三个静默面：(1) 存下的值与该组件的消耗进度**无关**（采样 0 批与 1000 批存出同一状态）；(2) resume 时组件 RNG 被 `__init__` 重新播种、load 从不触碰它 → **采样流从头部重放**（同 cluster mix、同 shuffle 序列）；(3) 恢复全局 RNG 还会移动进程内**其他**消费 `np.random` 的组件。修复 = 存 `self.np_rng` 自身状态。
- **来源**：DeepSpeed #8405（最小复现：3 次采样前后 `np_rng_state` 不变；resume 后 draw 1 == 原始 draw 1 `[0, 3, 3, 2]`；代码审计发现，非训练 run 报告；修复 PR #8406）。https://github.com/deepspeedai/DeepSpeed/issues/8405 ；修复 PR #8460（2026-09-12 补：`state_dict` 改存 `self.np_rng.bit_generator.state`（sampler 实际抽取流）+ 兼容旧全局元组 ckpt 的 `_load_np_rng_state`——旧档无 sampler 流记录，明示降级为 fresh seed 而非加载失败）。https://github.com/deepspeedai/DeepSpeed/pull/8460
- **行为效应**：resume 后 curriculum 难度调度/cluster shuffle 从头再来——后半程重复前半程见过的样本序，训练看似正常推进，数据课程语义已静默重置；全局 RNG 被牵连则任何依赖 numpy 全局流的组件同时漂移。与 DATA.01（rank 间 shuffle seed 失同步）对偶：那边是**同一时刻各 rank 看到不同序**，这边是**同一 rank 跨 resume 看到重复序**；与 CKPT.08（resume 静默跳 epoch）同为「恢复的状态不是运行时真正用的状态」。
- **发现来源**：2026-09-04 每日扫描

### CKPT.14 `quantized_extra_state_silently_dropped_on_load`

stage: `sft/pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `GPTModel.sharded_state_dict` 无条件 pop `output_layer._extra_state` 并 assert 为空——对「旧 GPT ckpt 只存 output 权重」的历史语义是对的，但量化框架把 per-module 量化 state 存在 `_extra_state` 里：**save 侧**直接崩（多元素 byte tensor 上 `and` 触发 ambiguous bool），**load 侧更糟**——同一方法也是 load plan 的依据，key 被 pop 掉后 saved state 被忽略，量化 output layer 被静默恢复成**未量化**权重。`MambaModel` 无此规则，因此 Mamba 系模型早已能 ship 量化 `lm_head`（Nemotron-3.5-Lightning-30B-A3B-NVFP4）而 GPT 系不能。修复 = 仅当占位符为空时才 drop，携带数据时保留。
- **来源**：Megatron-LM PR #7086（Qwen3.6-35B-A3B：`lm_head` 509M params ≈ 35% per-token 权重读取，量化不可用即损失大头 decode 吞吐）。https://github.com/NVIDIA/Megatron-LM/pull/7086
- **行为效应**：量化 `lm_head` 的 ckpt「加载成功」但量化语义已丢——训练/推理权重路径分叉，无 crash（save 面才崩）；与 CKPT.09（MXFP8 round-trip 丢 scale）同为量化 state 在存取路径上的静默丢失，本条丢的是**整个量化元数据条目**；与 CKPT.05（strictness 吞 unexpected key）同面：load plan 主动 pop 掉本应恢复的 key。
- **发现来源**：2026-09-05 每日扫描

### CKPT.15 `parallel_ckpt_reader_swallows_shard_errors`

stage: `shared` · Cov: `var:dist_ckpt_strictness_assume_ok` · 置信度 `documented`

- **机制**：ms-swift 的 Megatron 并行 ckpt reader patch 里 `concurrent.futures.wait()` 只等 worker 不取结果——读 shard 的异常（缺数据文件、shard 损坏）被吞掉，reader 无条件返回 successful `torch.futures.Future`：`dcp.load()` 报成功，未加载的 tensor 停在 sentinel 值。修复 = wait 后逐 worker 取 result，把异常以 `CheckpointException` 传给 caller。
- **来源**：ms-swift PR #10045（已合并；CPU DCP save/load + `MCORE_READER_MAX_WORKERS=2` 复现：缺第二个 tensor 数据文件时旧代码 load 成功、tensor 保持 sentinel；注入 `RuntimeError('corrupted shard')` 同样被吞）。https://github.com/modelscope/ms-swift/pull/10045
- **行为效应**：resume「成功」但部分权重从未从盘上读入——训练从 sentinel/旧值出发静默跑偏，无 crash 无日志。与 CKPT.05（strictness 吞 unexpected key）同为加载器「自报成功」家族：那边吞键、这边吞异常；与 CKPT.03（missing key 随机初始化）下游同型。
- **发现来源**：2026-09-06 每日扫描

### CKPT.16 `async_rollout_buffer_state_lost_on_resume`

stage: `rl` · Cov: `var:epoch_boundary_resume_silent_skip` · 置信度 `documented`

- **机制**：异步 RL 的 resume 契约只覆盖 dataloader/模型/优化器，不覆盖**在途 rollout 状态**，三个独立实现同型踩坑：(1) OpenRLHF oversampling（`vllm_generate_batch_size > batch_size`，README 文档化配置）把多生成的 `Experience` 留在 `_sample_buffer`，ckpt 只存 `prompts_dataloader.state_dict()`——resume 后 dataloader 跳过已生成 prompt、buffer 从空开始，**已生成未训练的 rollout 永久丢失**；且无 dataloader-exhausted 位，恰好在 generation-batch 边界结束的 episode 会整个重放。(2) 同实现里 loader 耗尽后的下一次调用把「iterator is None」误读为「新 episode」，清空仍非空的 buffer、从头重启同一 episode，可无限循环。(3) verl V1 `colocate_async/separate_async` resume 时 `fit()` 已 reissue 在途 prompt，两个 async trainer 又无条件补交 `num_warmup_batches` 批——相对未抢占 run **静默跳过 prompt** 并超填 in-flight 窗口。
- **来源**：OpenRLHF #1322（最小复现：loader 0..7 / batch 2 / gen 4，首块 [0,1]、buffer 存 2,3，resume 后 2,3 消失）。https://github.com/OpenRLHF/OpenRLHF/issues/1322 修复 #1323。https://github.com/OpenRLHF/OpenRLHF/pull/1323 ；OpenRLHF #1320（尾块丢弃 + episode 重启复现）。https://github.com/OpenRLHF/OpenRLHF/issues/1320 修复 #1321。https://github.com/OpenRLHF/OpenRLHF/pull/1321 ；verl #7759。https://github.com/verl-project/verl/issues/7759 修复 PR #7762（记录 reissue 计数、有恢复即跳过 warmup）。https://github.com/verl-project/verl/pull/7762
- **行为效应**：ckpt 正常、指标正常、run 继续——丢的是数据流本身（prompt 被跳过/重复，episode 可能永不终止）。与 CKPT.08（resume 静默跳 epoch）同族：恢复的状态不是运行时真正消费的全部状态；异步化把「rollout 缓冲」变成了新的未 checkpoint 状态面。
- **发现来源**：2026-09-06 每日扫描

### CKPT.17 `converter_stale_mapping_omits_experts`

stage: `pretrain/sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：torchtitan gpt_oss 的 grouped expert 参数改名加 shape 后缀后，HF state-dict adapter 的四条映射仍指旧名：HF export 侧对不存在的目标**静默跳过**——导出的 HF ckpt 缺全部 routed expert 权重；import 侧指向不存在的参数。改名（而非语义）驱动的映射漂移，转换器既不报错也不计数。
- **来源**：torchtitan PR #4475（已合并，supersedes #3555；修复四条 expert mapping，HF 侧无 `expert_bias_E` 对应物时 import 置零并保留分布式布局）。https://github.com/pytorch/torchtitan/pull/4475
- **行为效应**：导出的「HF 格式」ckpt 静默无专家——下游任何宽松加载（CKPT.03 面）直接随机初始化整组专家；严格加载才崩。与 CKPT.01（converter 专家 id 错排）同为转换层 MoE 权重完整性故障，本条是**整组省略**而非错排；触发器是上游参数改名，无版本对照的用户无从发现。
- **发现来源**：2026-09-06 每日扫描

### CKPT.18 `resume_rebuilt_sampler_loses_epoch_seed`

stage: `sft/pretrain` · Cov: `var:wrong_rng_object_checkpointed` · 置信度 `documented`

- **机制**：resume 路径重建 dataloader 时丢失 **epoch 维度**的采样种子。HF Trainer resume 顺序是先 `epoch_dataloader.set_epoch(epoch)` 再 `skip_first_batches(N)`；ms-swift 的 `_patch_skip_first_batches` 把后者替换成 `get_train_dataloader(skip_batches=...)`——新建 `BatchSamplerShard` 的 `curr_seed` 重置回 `base_seed`（= `data_seed`），没有任何代码对它调 `set_epoch()`：恢复的 dataloader 迭代的是 **epoch-0 排列**再跳前 N 批，而不是进行中 epoch 的排列。epoch 0 内 resume 时 `data_seed+0==data_seed` 天然无差，epoch≥1 才暴露。同库 #9685/#9902 已为其他路径修过同类 `set_epoch` 传播语义，patched 路径绕开了两个修复。
- **来源**：ms-swift #10050（真实 run：Qwen3-8B 全参 SFT、zero3、4 epochs、两次 crash 后 resume——恢复后 86 步与原 pass 同权重同全局步的 per-step loss Pearson = **−0.23**、均值 0.1637→0.0791（重训已见样本）；模拟 81% 样本重复、15% 语料在该 epoch 从未训练；第二次数值 64%/24%）。https://github.com/modelscope/ms-swift/issues/10050
- **行为效应**：resume「成功」、loss 正常下降（更低——重复样本再次拟合）；已训样本重复计权、该 epoch 尾部样本从未见过。与 CKPT.13（存错 RNG 对象→采样流从头重放）同族：那边 state_dict 存了与消耗进度无关的状态，这边**重建路径根本不传播 epoch**；与 CKPT.08（resume 静默跳 epoch）同为恢复的数据流语义不完整。
- **发现来源**：2026-09-07 每日扫描

### CKPT.19 `class_name_allowlist_leaves_weights_on_meta`

stage: `sft/pretrain` · Cov: `var:from_pretrained_silent_random_init` · 置信度 `documented`

- **机制**：DeepSpeed AutoTP 用**硬编码类名白名单**（26 项）决定是否从 ckpt 加载模块权重（`Loading.is_load_module`）。norm 类无子模块、递归探不到——不在名单上就永不加载，meta-device 路径上参数留在 meta。transformers 5.16.1 的 233 个 norm 类中 **219 个不在名单**（全部 Gemma / GLM / Granite / Olmo / MiniMax / Nemotron / Zamba 代系）；名单只能手工随模型家族增长（#8306 给 Qwen3.5 手补四个），跟不上时失败是静默的。最小复现：结构完全相同的两个 norm 类、state_dict 都给 7.0——名单内加载成功、名单外 `.is_meta` 为 True。`GlmMoeDsaRMSNorm`（DeepSpeed 自测用的 GLM-5.2 模型的 norm）即中招。
- **来源**：DeepSpeed #8447。https://github.com/deepspeedai/DeepSpeed/issues/8447
- **行为效应**：替换完成无报错、无缺失计数——未被加载的权重留在 meta（后续 ZeRO init / 首前向的处置决定是崩还是随机值；issue 验证到 gate 行为与 meta 参数幸存于 replacement 之后，未跑 end-to-end 错误生成）。与 CKPT.03（missing key 静默随机初始化）同下游（参数没有从 ckpt 来），机制在**按类名的加载白名单**而非键名匹配；与 CKPT.17（映射漂移整组省略）同为「转换/加载层对不认识的模块静默跳过」。
- **发现来源**：2026-09-07 每日扫描

### CKPT.20 `v1_trainer_esi_force_save_unwired`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：verl V1 trainer（`trainer.use_v1=true`，`ppo_trainer.yaml` 默认）的保存条件只有 last-step / `save_freq` 整数倍，从不 import `should_save_ckpt_esi`、不跟踪 `max_steps_duration`——文档化的 `trainer.esi_redundant_time`（Volcengine MLP / SageMaker 容量块到期前 force-save 的缓冲窗口）在默认 trainer 上是**静默 no-op**。V0、Ray SFT、experimental separation、fully-async 四条路径都实现了 ESI force-save，V1 迁移时丢了。容量块在两个 `save_freq` tick 之间到期时，preemption 丢掉的不只是「进行中的 step」而是整个未落盘窗口。
- **来源**：verl #7757（代码对照：V0 `ray_trainer.py` 的 `esi_close_to_expiration` 条件 vs V1 `trainer_base.py` 的两条件 save；配置 `save_freq=100 / esi_redundant_time=30 / total_training_steps=3` + 到期时间戳落在非 tick 步，实测无 force-save 日志、无 `global_step_*` 写出）。https://github.com/verl-project/verl/issues/7757 ；verl #7820（2026-09-12 补：同族 worker 进程组配置——`actor_rollout_ref.nccl_timeout` / `critic.nccl_timeout` 自 #6067 起被 `engine_workers.py:88` 的硬编码 `timeout_second=None` 短路，7 个官方示例脚本设置死键、`megatron_critic.yaml` 文档还主动教用户调大；grep 证实只剩 dataclass 字段与 docstring，无任何读取）。https://github.com/verl-project/verl/issues/7820 ；verl #7815（2026-09-13 补：#7820 的独立 issue 正文——dead key 从 #6067（2026-05 合入）起存在，7 个官方示例脚本设死键；修复 PR #7820 于 2026-09-10 开出，把超时线程进 worker 进程组构造）。https://github.com/verl-project/verl/issues/7815 ；2026-09-15 补：#7815 已随 commit 2083eeb 修复关闭（closed completed 2026-09-14）。
- **行为效应**：无报错；配置项被接受、resolve 正常、run 正常推进——被抢占后才发现 ckpt 停在最近 tick，租约成本换来的冗余窗口没有兑现任何保护。与 CKPT.07（死配置参数静默 no-op）同形：**契约在配置面存在、在执行面缺席**，差异是这条的代价按真实训练进度计（窗口内全部 step 作废）；与 CKPT.15（异步 ckpt worker 异常被吞）互补：那边是保存动作失败被瞒，这边是保存决策根本不发生。
- **发现来源**：2026-09-09 每日扫描

### CKPT.21 `shard_filename_lexical_sort_loads_wrong_rank`

stage: `shared` · Cov: `var:expert_reshard_id_permutation` · 置信度 `verified`

- **机制**：DeepSpeed pipeline checkpoint 的分片文件名把 rank 字段填充到**两位**（`get_rank_repr`）；并行度到 100 时 pad 溢出，而两个读侧调用点（`PipelineModule.ckpt_layer_path_list`、`DeepSpeedEngine._get_all_ckpt_names`）把 glob 结果按**字符串序**排序后按 model-parallel rank **位置索引**（`MegatronSDLoader` 位置式消费、`get_merge_state_dicts`/`get_split_state_dict` 切连续 rank 区间）——`model_100` 字典序落在 `model_10` 与 `model_11` 之间：**rank 11 加载 rank 100 的分片**。形状仍匹配、无异常，权重就是错的。修复 = 读侧 `natural_keys` 数值排序（不 rename 磁盘格式，两位以内顺序不变）。写侧单测 `assert 100 == 11` 之前 fail 证实。
- **来源**：DeepSpeed PR #8452（2026-09-08；「这只在 TP/MP ≥100 时咬人，罕见所以 2021 年潜伏至今；但失败是静默的，而修复只是一个排序键」；「the failure this issue predicted」——上游 issue 早有预言）。https://github.com/deepspeedai/DeepSpeed/pull/8452
- **行为效应**：无 crash、无 shape mismatch、无日志——≥100 路并行的 pipeline ckpt resume/load 加载**另一 rank 的权重**，等价于一次未声明的 TP 重排；下游若 finetune 则从错权重出发，行为不可预期。与 CKPT.01（expert reshard 置换）同族：**rank↔分片的映射在格式转换/读取层被静默置换**；差异在这边根因是文件名字典序 vs 数值序（纯 IO 契约），且门槛明确（并行度 ≥100）。
- **发现来源**：2026-09-12 每日扫描

### CKPT.22 `cached_optimizer_state_stale_scalar_lr`

stage: `shared` · Cov: `NEW` · 置信度 `verified`

- **机制**：torchtitan TorchFT 的 optimizer state dict 被**缓存**：张量状态靠共享引用保持最新，但**标量 LR 是值拷贝**——scheduler 更新 optimizer 后缓存里的 LR 是陈旧值；同时 restore scheduler state 只更新 scheduler 元数据**不回写 optimizer param groups 的 LR**：scheduler `get_lr()` 报告的值正确、optimizer 实际用的是另一个。两个缺陷叠加：**joining replica 以错误 LR resume**——梯度 all-reduce 完全一致、更新后权重分叉，破坏同步数据并行「同参数+同优化器状态+同步梯度 ⇒ 同更新」的基本不变量。修复 = 导出前刷新缓存里的 LR + 两个状态都加载完后从 scheduler 回填 optimizer 实际 LR。
- **来源**：torchtitan PR #4598（2026-09-11；作者在 TorchTitan+TorchFT 零偏差数值对齐验证中发现，replicas 发散 root-cause 图 + 修复回归测试）。https://github.com/pytorch/torchtitan/pull/4598
- **行为效应**：无 crash、无 NaN——TorchFT 拓扑变化/replica 加入后各副本以不同 LR 更新，权重静默分叉，后续梯度平均把分叉抹成「貌似一致实则偏离任意副本应有轨迹」；零偏差验证（对齐 uninterrupted run）是唯一可靠探针。与 OPT.12（scheduler override 抹平组界）同族：**scheduler↔optimizer 的 LR 传导契约断裂**，那边是 resume 加载覆写、这边是缓存值拷贝 + 恢复不回填；与 CKPT.02（矩错位）互补：那边优化器**状态**错、这边优化器**标量控制量**错。
- **发现来源**：2026-09-12 每日扫描

### CKPT.23 `onpolicy_resume_offset_counted_in_rollout_rows`

stage: `rl/sft` · Cov: `var:resume_rebuilt_sampler_loses_epoch_seed` · 置信度 `documented`

- **机制**：ms-swift Megatron on-policy（GRPO/GKD）训练里，ckpt 的 `state.consumed_train_samples` 计的是**已优化的 rollout 行数**（queries × num_generations），而 on-policy dataloader 索引的是**未重复的 query 行**；`_prepare_dataloader` 把持久化 offset 原样喂给 `MegatronPretrainingRandomSampler`——n 路 generation 下 resume 跳过 **n 倍太远**，训练静默接在数据集错误位置继续。同一报告的第二断点：该 sampler 用 `epoch` 单独作 epoch 排列种子，配置的 `data_seed` 对 on-policy 数据序**毫无效果**。修复 = 构造 dataloader 时 `//= num_generations` 换算回 query 行（不持久化计数本身、不可整除即 `RuntimeError` fail-closed）+ sampler 接受 `seed`、两个分片分支统一 `seed + epoch`。
- **来源**：ms-swift #10099（双 bug 报告：resume offset ×n 跳过 + sampler seed 不生效）、#10098；修复 PR #10113（同 PR 覆盖 #10098 的 strict 化）。https://github.com/modelscope/ms-swift/issues/10099 https://github.com/modelscope/ms-swift/pull/10113
- **行为效应**：resume 无任何报错，采样流从错误 offset 继续：部分 query 从未训练、部分重复训练——梯度与 loss 曲线正常，数据覆盖静默偏移（与 CKPT.18 的 81% 重复/15% 缺训同后果面）。与 CKPT.18（epoch 种子不传播）同族：**resume 时数据流位置的语义换算缺失**，那边丢 epoch 维、这边丢 generation 重复维；`data_seed` 死配置面与 CKPT.07/CKPT.20（死键）同型。
- **发现来源**：2026-09-13 每日扫描

### CKPT.24 `zero3_tag_load_reads_latest_weights`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed ZeRO-3（非 MoE/非流水线）`load_checkpoint(root, tag="earlier", load_optimizer_states=False)` 时，`DeepSpeedEngine._load_checkpoint()` 的 FP32 权重重建分支调 `get_fp32_state_dict_from_zero_checkpoint(load_dir)` **不透传 tag**——helper 内部重新解析 `root/latest` 指针。`latest` 指向 `later` 时，返回的加载路径与 `client_state` 都指 `earlier`，**实际进模型的权重是 `later`**。同一缺陷对称分支：请求的 ckpt 存在但目录无 `latest` 文件时反而 `ValueError`。修复 = 透传 engine 已解析的 tag（`get_fp32_state_dict_from_zero_checkpoint(load_dir, tag=tag)`）。
- **来源**：DeepSpeed #8499（真实训练 + 双 ckpt + 参数逐比对 repro，作者建议 severity: high；master 71d316d）。https://github.com/deepspeedai/DeepSpeed/issues/8499
- **行为效应**：无报错、无警告——rollback、eval 或新微调 run 静默用错模型，且唯一可见线索（返回路径/client_state）指向「正确」那份，日志无法自证。与 CKPT.01/CKPT.21（reshard 错位/文件名字典序错 rank）同族：**load 语义成功 + 内容错误**；差异在这边不是分片错位，而是**版本选择被 `latest` 文件劫持**——元数据来源与权重来源分离，显式 tag 的契约在重建分支断掉。
- **发现来源**：2026-09-14 每日扫描

### CKPT.25 `mfsdp_ckpt_stage_local_layer_keys_pp_resize`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `TransformerBlock` 在本地 `ModuleList` 里按 **stage 局部索引**（`decoder.layers.0..n-1`）注册层，只有 `sharded_state_dict`（transformer_block.py:782，offset 在 837）加上全局层号偏移；MFSDP adapter 把 `state_dict_for_save_checkpoint` 别名到普通 `state_dict`（mcore_fsdp_adapter.py:304），其 DTensor 元数据只描述 TP/DP 分片、不含层位。于是 MFSDP `fsdp_dtensor` ckpt 在 **PP=1 存、PP=2 加载**（Bridge MFSDP 文档明确支持该 resize）时：strict DCP load **成功**，但 stage 1 恢复的是 layers 0..n-1（第一阶段的层）而不是自己的 3–4——每个后续 stage 被第一阶段的层静默填充。4-MLP repro：输出 `[1.4989,1.6989,1.8989,2.0989]` 变 `[0.4661,0.6661,0.8661,1.0661]`（max err 1.0328）；换全局命名 exact 恢复。修复 = MFSDP state-dict key 加 pipeline 层偏移（或走 `sharded_state_dict`）、修复前拒绝 PP 不匹配加载。
- **来源**：Megatron-LM #7452（2026-09-17；finding 12，231 行 CPU/Gloo repro 含/不含真实 MFSDP buffer manager 双验证；Llama3-70B B300 recipe 用 MFSDP PP=1 暴露面确认）。https://github.com/NVIDIA/Megatron-LM/issues/7452
- **行为效应**：load 无报错、strict 校验绿——**每个 PP stage 训练第一阶段层的复制**，权重系统性错位但训练完全正常地跑下去；loss 不一定立刻异常（各 stage 内部一致）。与 CKPT.03（reshard 错位）/ CKPT.21（两位 pad 字典序错 rank）同族「load 成功 + 内容错位」，差异在这边是**key 语义域错误**（局部层号当全局层号）而非分片几何；「adapter 用普通 state_dict 别名绕开 sharded 契约」与 CKPT.05（非 sharded 路径丢 reshard 元数据）同型。
- **发现来源**：2026-09-18 每日扫描

### CKPT.26 `async_resume_cursor_pinned_by_stale_dropped_group`

stage: `rl` · Cov: `var:async_rollout_buffer_state_lost_on_resume` · 置信度 `documented`

- **机制**：trl AsyncGRPO `_save_checkpoint` 用「不在 `_trained_groups` 中的最小 group id」推 resume 位置——但 `RolloutQueueDataset.__iter__` 的 staleness 检查**丢弃**的 group（如慢 group 超 `max_staleness=4`）永远到不了 collator、永远不会进 `_trained_groups`：一个 stale-dropped group 把保存的 resume 游标**钉死**在它的位置。ckpt 正常写出、无任何错误。无 GPU 12 秒 repro（150 组、1 慢组）：run 1 实际训完 150 组、saved `prompt_index=2`（被 group 2 钉住）；resume 后从行 2 重启——**147/150 行被训练两次**、只有 3 行是新的。`AsyncDistillationTrainer._save_checkpoint` 的 `_trained_prompts` 同代码同病。修复 = resume 游标对 `_trained_groups ∪ _dropped_groups` 取（在途组仍照旧压住游标）。
- **来源**：trl #7274（2026-09-18；纯 CPU repro gist + 双 run 对照）。https://github.com/huggingface/trl/issues/7274 ；修复 PR trl #7294（cross-ref 于 2026-09-18，仍 open）——修法即「`_trained_groups ∪ _dropped_groups`」并加在途组照旧压游标，与本条记载的修复方向一致。https://github.com/huggingface/trl/pull/7294
- **行为效应**：无 crash、ckpt 可加载、指标正常——**resume 后近全量数据被重复训练**（数据分布被旧样本主导），有效样本量与预算核算系统性失真；「保存的数字看起来合理」（在合法范围内）使其不被怀疑。与 CKPT.16（异步 RL resume 丢在途 rollout 状态）同族**异步 RL 的 resume 契约缺状态面**第 4 个实例：那边丢 buffer 内容、这边游标被死键钉住，方向相反（丢 vs 重）；与 CKPT.18（sampler 重建丢 epoch 种子 → 81% 重复）行为效应同型「resume 后大比例样本重复」，这边重复是**staleness 丢弃副作用**而非种子丢失。
- **发现来源**：2026-09-19 每日扫描

### CKPT.27 `async_save_skips_tracker_write_non_megatron_scratch_restart`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：verl v0/v1 trainer 的 `_save_checkpoint` 都有一个无条件分支：`actor.checkpoint.async_save=True` 时只 `logger.info("skip write latest_checkpointed_iteration.txt ...")` 后 `return`——**不检查 `actor.strategy`**。该跳过假设「checkpoint manager 稍后会写 tracker」，但只有 `MegatronCheckpointManager` 自己写 tracker（megatron_checkpoint_manager.py:1136-1146，`async_save && rank==0` 守卫）；backing `fsdp`/`fsdp2`/`veomni`/`torchtitan` 全部引擎的 `FSDPCheckpointManager` 里 `async_save` **零次出现**——同步存分片、从不碰 tracker。于是非 Megatron 策略下 tracker 被两边同时跳过：trainer 跳（等 manager）、manager 无实现。`config.py:37` 注释明写「Only implemented for Megatron as of now」但无校验，`async_save: False` 存在于**每个** backend 的 actor.yaml:142——FSDP 上是可达旋钮。
- **来源**：verl #7952（2026-09-20；驱动真实 `RayPPOTrainer._save_checkpoint` 的对照 repro：async_save=False → tracker 写出、`find_latest_ckpt_path` 返回 global_step_3；async_save=True → 只有 `global_step_3/`、tracker 不存在、返回 None；`grep -c async_save` megatron=10 / fsdp=0；upstream main `23a341e3`）。https://github.com/verl-project/verl/issues/7952 ；同日补：修复 PR verl #7955（方向 a——`ActorConfig.__post_init__` 拒绝 `async_save` + 非 Megatron strategy 组合 + CPU 单测）已开出。
- **行为效应**：ckpt 数据完整落盘、run 正常推进、无 error 无 warning（唯一日志是一条 INFO）——重启时 `find_latest_ckpt_path` 返回 None → `resume_mode=auto` 判「无 ckpt」→ **静默从 step 0 重训并覆写盘上全部完好 ckpt**（长 run 的整段进度作废）。SFT 不受影响（CheckpointHandler 无条件写 tracker）。与 CKPT.20（V1 ESI force-save 不接线）同形「契约在配置面、执行面缺席」、同是 verl trainer 的保存决策面，差异：那边该存的时刻没存、这边存了但**指针（可达性）没写**；与 CKPT.15（ckpt worker 异常被吞返回 success）互补：那边内容坏、这边内容完好但不可达；「跳过方与善后方分属两层、互不知情」的断裂与 CKPT.16（异步 RL resume 契约漏在途状态）同族——异步保存契约没有单一 owner。
- **发现来源**：2026-09-21 每日扫描

### CKPT.28 `swiglu_interleave_layout_unconverted_on_load`

stage: `pretrain/sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron-LM `--moe-mlp-glu-interleave-size 32` 改变激活侧对 FC1 输出的解释（gate/up 按 32 通道交织），但**不重排 ckpt 权重**——原生 Megatron-LM 加载 contiguous SwiGLU ckpt（`[all gate | all up]`）时不做 `[gate 0:32 | up 0:32 | gate 32:64 | ...]` 的 interleave 转换，SwiGLU 配对错误通道；grouped-weight load hook 只 stack/split 专家张量、不做该置换。Megatron-Bridge 已在 load 时显式 interleave / save 时 de-interleave（Megatron-Bridge #2841），原生路径（含 main）缺这步。
- **来源**：Megatron-LM #7576（2026-09-22；Qwen3.5-35B-A3B text SFT、16×GB200、TP1/PP1/CP8/EP16、DCP、seqlen 131072；三配置对照 + Bridge 同 MXFP8/CuTeDSL/GroupedTensor 配置正常训练的反证——排除 single-weight / FP8 param gather 因素；launcher 已设 interleave flag）。https://github.com/NVIDIA/Megatron-LM/issues/7576
- **行为效应**：无 crash、无 warning——initial LM loss ~0.2（BF16+GroupedTensor 正确路径）→ ~2.2（MXFP8+CuTeDSL 两配置同样中招），训练正常推进但激活-权重通道系统错配（报告者明确：layout mismatch 而非 FP8 rounding）。与 CKPT.09（MXFP8 round-trip 丢 requant scale）同族「ckpt 边界的布局/量化契约断裂」，差异在这边 **BF16 同样中招的纯置换缺失**、不经量化放大；与 MOE.07 / RL-RO.09（量化专家布局 republish 破坏训推同步契约）共享「容器/引擎对权重布局假设不一致」病根——这边打击静态 load、那边打击在线 sync；「flag 改变解释约定但转换不在其职责内」的契约空洞与 CKPT.06（config 默认吞 CLI override）同型。
- **发现来源**：2026-09-23 每日扫描

### CKPT.29 `hf_rotary_inv_freq_uninitialized_after_to_empty`

stage: `pretrain/sft/rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：torchtitan 的 Transformers modeling backend 先在 `meta` 设备建模、再 `to_empty()`，使 rotary 模块的非持久化缓冲区 `inv_freq` 未初始化。旧兼容补丁依赖实例属性 `rope_init_fn` 重新计算，但 Transformers 5.x 已将它移入 `__init__` 局部变量；backend 同时覆写 `_init_weights` 且漏掉 rotary 分支。因此模型初始化与 checkpoint load 都不会恢复正确的 RoPE 频率。
- **来源**：torchtitan #4783（2026-09-16；针对当前 5.x 路径的 CPU-only 复现将 `to_empty()` 后缓冲区填为 NaN，执行初始化后仍为 NaN，且与正常构造的 `inv_freq` 不相等；截至 2026-09-24 issue 为 open）。https://github.com/pytorch/torchtitan/issues/4783
- **行为效应**：CPU 复现确认该路径的旋转频率保持错误且无异常；在使用该 backend 的训练中，错误频率会污染位置编码与前向。来源提及旧问题 #3775 的 loss 差异，**不是本条当前版本的端到端训练效果实测**；未评估检测器效果。与 CKPT.25 的 PP 层键错位不同，此处失效的是 `meta` 实例化后的非持久化缓冲区初始化契约。
- **发现来源**：2026-09-16 远程扫描回填；2026-09-24 合并复核

---

# DATA · 数据管线

## DATA

### DATA.01 `rank_shuffle_seed_desync`

stage: `pretrain/sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：各 DP rank 用了不同的 shuffle 种子或有的 rank 忘了设，导致同一 global batch 的切片不是同一文档的不同分片，而是不同语料。梯度在语义上在平均两件无关的事。
- **来源**：Megatron/DeepSpeed 数据加载常见事故（本实验室 `verify_dataset.py` / MMIDIDX 校验动机之一）。
- **行为效应**：loss 仍降，token/s 正常；评测方差异常大。无 crash。
- **发现来源**：2026-08-17 基础调研

### DATA.02 `packing_cross_document_attention`

stage: `sft/pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：把多条样本 pack 进同一序列以提高利用率，但 attention mask / position id 没按文档边界切断。后一条的 token 能看见前一条的内容。
- **来源**：SFT packing 的标准正确性陷阱；与 SFT.02 成对（一条是 attn，一条是 label）。
- **行为效应**：模型学会「抄下一条的答案」；验证集单条样本分数虚高，真实对话掉点。
- **发现来源**：2026-08-17 基础调研

### DATA.03 `loss_mask_includes_prompt`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：completion-only 训练却把 prompt token 的 CE 也算进去，或 chat template 的 special token 没 mask。模型被训练成复述用户话。
- **来源**：TRL / ms-swift / llama-factory 反复出现的 mask 配置错误。
- **行为效应**：SFT loss 看起来健康且更低（prompt 更好预测）；指令遵循变差。
- **发现来源**：2026-08-17 基础调研

### DATA.04 `tokenizer_train_infer_id_mismatch`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：训练用 tokenizer A、保存时只存权重；后续 SFT/RL/评测换了 B，或 `padded-vocab-size` 与真实 vocab 不一致导致 embedding 行错位。
- **来源**：本实验室 Qwen3 必须 `--padded-vocab-size 151936`（自然长度 151680 会对不齐）。交叉：推理编目分词器类。
- **行为效应**：一切能跑。生成/loss 像「近邻词被换了」。
- **发现来源**：2026-08-17 基础调研（本地 Megatron）

### DATA.05 `mmididx_wrong_magic_silent_garbage`

stage: `pretrain` · Cov: `NEW` · 置信度 `verified`

- **机制**：把 nanoGPT 的 `.bin/.idx` 或未完成的分片当成 Megatron MMIDIDX。阅读器不总是立刻崩，可能吐出合理范围的垃圾 token id。
- **来源**：`opensource_training_megatron` CLAUDE.md / `verify_dataset.py`。
- **行为效应**：训练起步 loss 就在随机基线附近打转，或「能降但降不动」；无格式错误。
- **发现来源**：2026-08-17 基础调研（本地）

### DATA.06 `padding_free_boundary_merges_sequences`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：padding_free 打包把多条序列 flatten 成一条 token 流，还原侧靠 `position_ids` 的下降沿推断序列边界。前一序列长度为 1 时下降沿不存在（`0 | 0 1 2` 中 `0 < 0` 为 False），两条独立序列被静默合并。ms-swift `revert_padding_free()` 实测：单 batch 的 sentence embedding 从 516 条错减为 260 条，labels 数量不变 → 样本与标签错位。
- **来源**：ms-swift #9903（含根因代码分析 + debug 日志，已修复关闭）。https://github.com/modelscope/ms-swift/issues/9903
- **行为效应**：轻度：训练不报错但 embedding-label 已错位，InfoNCE 学错配对；重度：`InfoNCE zero negative` 崩。与 DATA.02 同族但机制不同：DATA.02 丢的是打包侧 attention 边界，这里打包没错、**边界恢复推断**错，故单独编号。
- **发现来源**：2026-08-18 每日扫描

### DATA.07 `seqlen_packer_off_by_one_drops_last_sample`

stage: `pretrain/sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed `batch_by_seqlens`（variable batch size 管线）按 token 上限贪心扩切片 packing，`range(batch_init + m, len(metrics), m)` 的独占上界使 `batch_end` 永远取不到 `len(metrics)`——能被尝试的最宽切片是 `metrics[batch_init:len-1]`，**数据集最后一个样本永远进不了任何 microbatch**；尾样本使 `batch_init == valid_batch_end`，被循环注释当「没数据了」正常 break。
- **来源**：DeepSpeed #8347（CPU 复现：`seqlens=[1,2,3,4,5], max_tokens=100` 只出样本 `[0,1,2,3]`；修复后新增单测 5 failed→5 passed；24 组模式 sweep 无重复 id、覆盖完整）。https://github.com/deepspeedai/DeepSpeed/pull/8347
- **行为效应**：无报错；`sequence_picking_order=dataloader/seqlen` 时同一尾样本每个 epoch 都被静默丢弃，`random` 时每 epoch 丢不同样本——数据覆盖缺口随 epoch 漂移。训练曲线完全正常。与 DATA.06（padding_free 边界恢复错）同为 packing 侧静默数据丢失，但本条在**批次构造算术**（off-by-one 切片边界），非边界推断。
- **发现来源**：2026-08-30 每日扫描

### DATA.08 `loss_mask_cache_aliasing_leaks_first_sample`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `GPTDataset.__getitem__` 缓存 `loss_mask` 后就地掩码：FILL 路径 `self.cached_loss_mask = loss_mask`（存**引用**），随后 `loss_mask[labels == _pad_token_id] = 0.0` 就地写穿透缓存槽位。缓存单槽且 FILL 每数据集只跑一次（`masks_and_position_ids_are_cached` 置 True 后不复位），**只有第一个被服务的样本**能写入；若它含 pad 位（短于 `sequence_length` 的真 padding，仅 validation 可产生；或 pad id 作为普通数据出现——token 流中任何该 id 都中招，内容依赖位置），其零掩码永久污染缓存，之后每个 HIT 样本 clone 出的都是脏 base。训练 split 因 `drop_last_partial_sequence=True` 硬编码不产生真 padding，但 pad id 数据位触发不罕见（`--mock-data` 的 `(arange+1)%vocab` ramp 几乎每样本命中）。`MockGPTDataset`/`GPTFIMDataset` 继承此 `__getitem__`；`SFTDataset` 自定义不受影响。`cached_attention_mask`/`cached_position_ids` 同样存引用但当前无就地写。
- **来源**：Megatron-LM #6981（完整别名分析 + FILL/HIT 时序图 + 单测 `test_mask_cache_does_not_leak_padding`：修复前 AssertionError、修复后通过）。https://github.com/NVIDIA/Megatron-LM/pull/6981
- **行为效应**：静默——loss 按 mask sum 归一化故数值保持 plausible。**破坏 bitwise resume**：哪个样本跑 FILL 是进程属性而非数据属性——从头启动的 run 与 resume 的 run 对同一样本服务不同 mask，ckpt 状态/`tokens`/`labels`/`position_ids` 全部 bitwise 一致的情况下梯度从 resume 后第一步就分叉。与 DATA.03（mask 范围错）同族但根因是**缓存别名**：掩码逻辑本身正确，坏在共享可变状态。
- **发现来源**：2026-08-31 每日扫描

### DATA.09 `swa_branch_overwrites_caller_attention_mask`

stage: `sft/pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `FusedScaleMaskSoftmax.forward_torch_softmax` 的 sliding-window 分支**无条件覆写**调用方传入的 attention mask：`mask = get_sliding_window_causal_mask(sq, sk, self.window_size)`；隔壁 plain-causal 分支有 `mask is None` 守卫，SWA 分支没有。带 SWA 层的模型在 torch fallback 路径（`scaled_masked_softmax_fusion=False`）上，padding mask（`micro_batch_size>1`）或 packing 的块对角文档 mask 被 SWA 层静默丢弃——pad token / 跨文档 token 照常参与注意力；非 SWA 层行为正常，错位只出现在部分层，逐层排查才可见。
- **来源**：Megatron-LM #7039（MiMo-V2.5 全参 SFT 实际命中：48 层中 39 层 SWA 走该 fallback；`[2,1,8,8]` 零分 + 尾部 3 个 key 的 padding mask 复现——padded key 概率非零；修复 = 组合而非覆写 `mask = swa_mask if mask is None else (mask | swa_mask)`，`mask=None` 调用方行为不变）。https://github.com/NVIDIA/Megatron-LM/issues/7039
- **行为效应**：无异常、无警告，loss 被静默污染；issue 报告唯一安全配置退到 `micro_batch_size=1` + 关 packing（变长多模态数据的重大效率损失）。与 DATA.02（packing 不按文档边界切断注意力）同症状面——pad/跨文档注意力泄漏，但根因在**消费侧**：packer/调用方生成的 mask 是正确的，被融合算子 fallback 的条件分支吞掉。与 KER.03（kernel 同步语义静默错值）互补：这是 mask 语义在分支覆盖上的静默降级，非融合 kernel 本身算错。
- **发现来源**：2026-09-03 每日扫描

### DATA.10 `secondary_eos_mask_misclassifies_terminated`

stage: `rl` · Cov: `var:chat_template_train_serve_mismatch` · 置信度 `documented`

- **机制**：模型在 `generation_config.eos_token_id` 里声明**多个** EOS id（Phi-3.5 `[32007, 32001, 32000]`、Gemma-3 `[1, 106]`、Qwen3 `[151645, 151643]`），助手回合实际以**次级** id 结束（`<|end|>` / `<end_of_turn>`）。生成侧（transformers 与 vLLM `generation_config="auto"`）对全表 EOS 停止——正确；trainer 侧 GRPO `GRPOTrainer` 的 completion 边界却只认 `tokenizer.eos_token_id` 单个值，四个消费点全错位：EOS mask 只在主 id 后切（次级 EOS 之后的 padding/续采样 token **留在 loss**）、`clipped_ratio`/`mean_terminated_length` 把正常终止记为截断、`mask_truncated_completions=True` 时这些 completion 被整条掩掉、tool-suffix 前缀修剪找错 id。修复 = 收集 `tokenizer.eos_token_id + generation_config 全表` 进 `self.eos_token_ids` 并统一用于四个位点。
- **来源**：trl PR #7039（Phi-3.5 实测 `clipped_ratio≈0.97`@step1（max_completion_length=2048）而几乎所有 completion 实际终止于 `<|end|>`；修复后 <0.1、`mean_terminated_length` 非零；Gemma-3-4B-it GRPO vLLM colocate + transformers 双路径验证；含回归测试）。https://github.com/huggingface/trl/pull/7039
- **行为效应**：多 EOS 模型上三种静默面同时发生：loss 学到「终止之后的垃圾 token」、截断率指标失真（把好数据看成没学完）、`mask_truncated_completions` 开启时**几乎全部样本被掩掉 → RL 零学习信号**而 run 无报错。与 SFT.01（模板训服不一致）同根因族——控制 token 语义在训练/生成两侧定义不同；与 RL-RWD.05（预填 `<think>` 死区）互补：那边 reward 侧锚定失败，这边 completion 边界判定失败。
- **发现来源**：2026-09-04 每日扫描

### DATA.11 `minillm_advantage_mask_checks_wrong_tensor`

stage: `rl` · Cov: `var:loss_mask_includes_prompt` · 置信度 `documented`

- **机制**：trl MiniLLM trainer 构造 completion mask 时判 `input_ids[:, prompt_lengths:] != -100`——但 `-100` 填充写在 `labels`（几行前由 attention_mask 构建），`input_ids` 从不含 -100：**mask 恒全 1**，padding 位保留 reward、每个更早位置的 advantage 把 pad 位的 reward 折进折扣和。同函数第二缺陷：长度归一化把 masked 位替换成 1e-4 再除——短 completion 的分母混入每个 pad 位的 `1e-4·γ^(i-t)`（reward=1.0、γ=1.0 时单 token pad 到 512 得 0.9514 而非 1.0），**advantage 的大小取决于 batch padding 而非序列本身**。修复 = mask 改读 labels、长度只数未掩位。
- **来源**：trl PR #7044（fixes #7024；指正 issue 里「labels 同时当 gather 索引」的一行修法会把 -100 喂进 `torch.gather`）。https://github.com/huggingface/trl/pull/7044 ；2026-09-20 补：**该 trainer 家族仍带第二处未修复静默错值**——#6626（open since 2026-08）：`_compute_advantage` 的 `gamma_pow[i] = gamma**i` 用**绝对位置**索引而非 `i−t`，`length_normalization=False` 时每个 advantage 多乘一个 `gamma^t`（常 reward=1、T=512、γ=0.9 下 t=200 处实测 advantage 缩小 1.4e9 倍——后段 token 指数级压制、无任何异常）；`=True` 时 `gamma^t` 分子分母相消，但 `gamma**i` float32 下溢为 0.0 后 0/0=NaN（γ=0.5 T=512 首 NaN t=150，NaN 位与下溢位逐点重合）。修复 PR #6635 open 自 8 月。trl #7295（2026-09-19，维护者提案移除 MiniLLMTrainer）把 #6626/#6635 列为「两个 open bug 均住在序列级 reverse-KL + gamma 折扣路径」的移除论据——若移除落地，本条机制面随模块消亡。https://github.com/huggingface/trl/issues/6626 ；https://github.com/huggingface/trl/issues/7295
- **行为效应**：无 crash；MiniLLM 的 advantage 系统性错值（pad 越多偏得越大），策略更新方向随 batch 组成漂移。与 DATA.03（loss mask 含 prompt）同族——都在 mask 构造处检查了错误的张量；与 RL-ADV.03（token 长度偏置）症状互补：那边是设计内的长度偏置，这边是 padding 泄漏制造的外生长度偏置。
- **发现来源**：2026-09-06 每日扫描

### DATA.12 `sp_shard_cuts_causal_lm_labels`

stage: `pretrain/sft` · Cov: `NEW` · 置信度 `verified`

- **机制**：DeepSpeed AutoSP（序列并行编译路径）按序列维切分 `input_ids`，但 causal-LM 的 labels 是 `input_ids` 的 **shift-by-one**——序列分片边界处，本片的 label[t+1] 属于**下一片**的 token：朴素按同维度切 labels 使每个分片**丢失最后一个 token 的监督**并在边界制造跨片错位监督；且各分片独立 `cross_entropy(reduction='mean')` 再对分片等权平均，忽略各片有效 token 数（ignore_index 掩码量不同）。修复（PR #8457）= 定位 `labels` 的 shifted 依赖链、对 shifted label 节点按 `make_contiguous` 正确重切 + `aggregate_loss` 按 valid-token 计数全局归约；同时对不可表达配置（class-weighted / 非 mean reduction / sharded key dim 的 attention mask）从「静默错值」改为 fail-fast。PR 描述明言修复对象是「silently dropping tokens or entering invalid collectives」。
- **来源**：DeepSpeed PR #8457（2026-09-08；AutoSP 编译器正确性：跨 sequence-shard 边界保留 causal-LM 目标 + 按有效 token 数聚合分片 loss + 校验 mesh/头数布局）。https://github.com/deepspeedai/DeepSpeed/pull/8457
- **行为效应**：无 crash——每个 SP 边界丢监督 + 边界 token 监督错位 + 分片 loss 加权错值，SP>1 的预训练 loss 带结构性 bias，与 SP=1 基线不可比。与 DATA.07（变长 packing off-by-one 丢尾样本）同族：**数据切分边界 vs 目标对齐契约**的 off-by-one；与 LOSS.08（mean-of-means 归约）共享「分片/分批独立归一再等权平均」的病根；与 PAR.08（AutoTP 漏插 collective）互为 AutoSP/AutoTP 编译路径对偶的同类覆盖缺口。
- **发现来源**：2026-09-12 每日扫描

### DATA.13 `wrapper_signature_probe_drops_packed_boundaries`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：verl FSDP/FSDP2 value-model 引擎用 `"cu_seqlens" in signature(module.forward).parameters` 探测 packed 边界能力——只看**返回的外层模块**的显式签名。当 value head 走 TRL fallback（`AutoModelForCausalLMWithValueHead`）时，wrapper 的 forward 只有 `**kwargs`、无显式 `cu_seqlens` 参数（但会把 kwargs 透传给底座），探测返回 False → `use_remove_padding=True` 下 `prepare_model_inputs` 对 critic 静默省略 `cu_seqlens`/`cu_seqlens_cpu`；被 patch 的 Qwen3.5 linear-attention 分支只认显式 `cu_seqlens`、**不从 position_ids 重置恢复边界**，同一 microbatch 里独立打包的样本在 critic 的线性注意力状态上互相可见。SP=1 即中招（不需要 SP>1）；actor 路径 #6660 已修，TRL critic wrapper 是漏网实例。修复方向 = 对「已验证的透明 wrapper」解析能力或显式能力契约（把任意 `**kwargs` 当支持不安全）。source 级复现器（AST 抽取原始谓词/语句 + 记录 stub、纯标准库）：underlying=True / TRL wrapper=False、critic inputs 无边界键、显式供给时**同一** forward 体原样透传——已证边界参数缺失，数值影响未测（报告者明示不主张）。
- **来源**：verl #7931（2026-09-19；Qwen3.5 + TRL v0.27.0 value-head critic + FSDP 引擎，commit `3efe38c75`；关联 #6660/#6780/#7471）。https://github.com/verl-project/verl/issues/7931 ；2026-09-20 补：维护者 wuxibin89 确认遗漏发生在 **veRL 的能力探测**（早于 TRL forward 被调用）、非 TRL 版本回归——bug 定位在 verl 侧，等修复方向（bump trl 依赖 vs 修探测逻辑）落定。
- **行为效应**：无 crash——critic 对 packed 样本的隔离静默失效（线性注意力态跨样本泄漏面由机制成立；真实 ckpt 的 value/PPO 更新差未测量）。与 DATA.02（attention 边界丢失）/ DATA.06（边界恢复推断错）同族不同位：这里边界在**引擎输入组装层被能力探测错判丢弃**；「按显式签名探测能力」的病根与 CKPT.19（类名白名单漏 norm）同型——结构探测遇上透明包装即静默降级。
- **发现来源**：2026-09-20 每日扫描

### DATA.14 `metric_flush_key_set_rank_local_contamination`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl GRPO/RLOO trainer 的 metric flush 遍历**本进程本地**的 pending key 集合并对每个 key 调 collective（`gather_object(self._pending_extra_logs[column])`、`accelerator.gather(local_mean)`）——各 rank 的 key 集合来自**条件触发的用户 reward 函数**（如「只在触发解析器/证明验证时 `log_metric`」），天然因 rank 而异。三种交错：① 某 rank 有 key、另一 rank 无 → 单边进 collective → 分布式死锁（活性面）；② 两个 rank 各持有不同 key 集合但同 step 都进循环 → `gather` 在不匹配的 key 上配对——**两条无关指标互相混进对方的聚合历史**；③ 都有 key 但样本数不同 → `gather(mean)` 之后再 mean 是**未加权平均 of means**，全局指标被各 rank 触发频次加权，系统性偏离真值。
- **来源**：trl #7310（2026-09-20；DDP/Accelerate 路径 flush 逻辑逐行分析 + 修复方案先行自证：跨 rank 同步 key 并集、`[sum, count]` 成对 gather 做精确加权归约、未触发列 `[None]` padding）。https://github.com/huggingface/trl/issues/7310
- **行为效应**：①是 hang（有信号）；②③**无 crash 无 warning**——logged metric 值静默错（混列 + 错权），reward 曲线/KL/reward-hacking 诊断的可信度被破坏。修复侧备注：`local_sum/local_count` 成对 gather 的加权均值属正确形状、但 `tot_cnt==0` 时除零守卫返回 0.0（把「全 rank 都没触发」记成 0 而非 NaN/缺失——低频指标的分母语义仍需看护）。与 OBS.09（gather_for_metrics 对已聚合标量无法去重 padding → 尾批偏置）同族「**指标聚合的 rank 几何假设破裂**」不同断点：那边假设行对齐、这边假设 **key 集合对齐**；与 RL-KL.04/OBS.06（KL/entropy/reward 指标路径不掩 padding）同理是「诊断仪表先于训练目标坏掉」——训练损失本身不受影响，正是 RL-RO/TIM 类故障依赖的观测面失准。
- **发现来源**：2026-09-21 每日扫描

### DATA.15 `sparse_attn_window_crosses_packed_document_boundary`

stage: `pretrain/sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：torchtitan DeepSeek V4 [重写 `get_attention_masks()`](https://github.com/pytorch/torchtitan/blob/6c2dadbb3109998092d1164e990c6377b2b8fb50/torchtitan/models/deepseek_v4/model.py#L222-L232) 时**丢弃 positions、padding 与文档边界元数据**，sparse attention 窗口从**整条扁平 token 流的偏移**构建——position id 重置（文档边界信号）被无视，query 64（第二文档 position 0）允许 attend key 63（第一文档末 token）。共享 decoder 把 position 重置当文档边界处理，DSV4 override 破坏该契约；报告者指明只修 sliding window 不够（compressed-KV 与 indexer 行为同样未验）。
- **来源**：torchtitan #4801（2026-09-20；CPU 复现调真实 mask hook 构建实际 block mask：`DSV4 sparse mask: True` vs `shared packed-document mask: False` + 单文档对照 True；TorchTitan `6c2dadb`、PyTorch 2.12.0+cpu；未量化训练 loss 影响）。https://github.com/pytorch/torchtitan/issues/4801
- **行为效应**：无 crash——packing 文档隔离静默失效，跨文档注意力泄漏（DATA.02 的模型架构侧实例）；修复回归还应检查「改一条文档、另一条文档输出不变」。与 DATA.02（packing mask 未按文档边界切断）同族同现象，差异在断点位于**模型侧 mask hook override** 而非数据管线 mask 构造；与 DATA.06（padding-free 边界合并序列）互补：那边序列粘连、这边窗口越界；与 DATA.13（wrapper 签名探测丢 packed 边界能力 → cu_seqlens 丢失）共享「packed 几何元数据在某层被静默丢弃」病根。tianyu-l 09-21 已 cc 维护者（@drisspg @floatingtrees）。
- **发现来源**：2026-09-23 每日扫描

---

# OPT · 优化器

## OPT

### OPT.01 `adam_moment_not_resumed`

stage: `shared` · Cov: `var:zero_optimizer_state_partition_mismatch` · 置信度 `documented`

- **机制**：只加载 model，不加载 optimizer / 加载了但 key 对不上。Adam 用零矩接着一个已经训练很久的权重，等效于突然换优化器状态。
- **来源**：分布式 resume 常规事故；与 CKPT.02 同族，本条强调「权重对、矩空」。
- **行为效应**：resume 后前几百 step 像重新 warmup，随后可能 spike。
- **发现来源**：2026-08-17 基础调研

### OPT.02 `grad_clip_absorbs_sdc_spike`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：全局 grad clip 把一次 SDC 造成的大梯度压回阈值。范数日志「正常」，方向已经被那次尖峰污染过。
- **来源**：Ma et al. 2502.12340（SDC 可表现为单步梯度扰动）；训练实践中 clip 是默认吸收器。
- **行为效应**：没有 NaN、没有 scaler skip；权重被推了一小步到错误方向。
- **发现来源**：2026-08-17 基础调研

### OPT.03 `lr_zero_phase_hides_bad_gradients`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：warmup 把 LR 钉在 0 时，错梯度不算进权重，曲线与健康 run 重合。一离开 0，故障才显形。DeepSpeed #8224 用这个当对照。
- **来源**：https://github.com/deepspeedai/DeepSpeed/issues/8224
- **行为效应**：排障时「前 N step 完全一致」被误读成「实现没问题」。
- **发现来源**：2026-08-17 基础调研

### OPT.04 `global_clip_rescales_corrupted_grad_norm`

stage: `shared` · Cov: `var:grad_clip_absorbs_sdc_spike` · 置信度 `documented`

- **机制**：全局 clip_grad_norm 只看整体 L2 范数：任何把范数抬高（或抬翻溢出）的梯度损坏都会被统一的 `clip_coef` **静默重缩放回 clip 阈值**，方向不校验、逐参数幅值不校验、skipped-iteration 不触发。PAR.05（2^N 梯度翻倍）是该机制的精确放大器：N≤~64 时 `2^N` 倍损坏梯度被 clip 归一后照常 step——「方向对、幅值假」的更新持续打进权重；N=128 时范数溢出为 inf，clip_coef≈0，run 变成零梯度空转。
- **来源**：Megatron-LM #6660（实测 clip 把 ×2^16 的 expert 梯度统一重缩放掉，`number of skipped iterations = 0`；clip 关闭时同 run 表面收敛）。https://github.com/NVIDIA/Megatron-LM/issues/6660
- **行为效应**：clip 开着时梯度损坏的幅值证据被抹掉，只有范数溢出（N 大或损坏更猛）才可见；grad_norm 图「正常」不等于梯度没坏。与 OPT.02 同族（clip 吃掉尖峰），不同面：OPT.02 是单步尖峰被吃，本条是**持续性幅值损坏被常态化重缩放**，且放行与清零两个 regime 之间隔着一次 fp32 溢出。
- **发现来源**：2026-08-20 每日扫描

### OPT.05 `module_replacement_orphans_optimizer_params`

stage: `sft/rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed `_configure_expert_parallel` 在 `_configure_optimizer` **之前**替换全部 MoE 模块，中间无任何 param group 重映射。调用方自建优化器（HF Trainer / Accelerate 在 ds_config 无 optimizer 块时的默认路径）经 `torch.optim.Optimizer.__init__` 已把**旧 expert/router 张量**物化进 param groups；替换后的活 `GroupedExperts` 权重不属于任何 param group——**全部专家与 router 被静默冻结**。
- **来源**：DeepSpeed #8354（ZeRO-3 + autoep_size=2、一步 lr=0.1 实测：attention |Δ|=1.003e-1 moved，router/experts 全部 0.000e+0 FROZEN；优化器写进 ds_config 则全部 1.005e-1；无 zero.Init 的路径改显 AttributeError，zero3_init_flag:true 才是静默面）。https://github.com/deepspeedai/DeepSpeed/pull/8354 ；DeepSpeed PR #8377（2026-09-05 补，同一机制的 HF Trainer/Accelerate 路径：caller-built optimizer 惰性物化 param_groups → 替换后的专家参数无组可归 + 旧参数残留组内——HF 集成在 ds_config 无 optimizer 块时的默认分支即踩中，ZeRO-3 下同样全部专家冻结）。https://github.com/deepspeedai/DeepSpeed/pull/8377
- **行为效应**：`zero3_init_flag: true` 下**完全静默**——attention/shared expert/norm 正常训练、loss 正常下降，MoE 核心参数一动不动，ckpt 里专家权重=预训练原值。与 CKPT.03（加载侧 missing key 随机初始化）对偶：参数在模型里且值正确，但被优化器**遗忘**。SFT.07（LoRA 只挂 4/56 层）的「欠训练但曲线在动」同症状，机制在优化器绑定时机而非模块选择器。
- **发现来源**：2026-08-30 每日扫描

### OPT.06 `unscaled_grad_divided_by_loss_scale`

stage: `sft` · Cov: `var:loss_scaler_swallows_overflow` · 置信度 `verified`

- **机制**：DeepSpeed fp16 + ZeRO stage 0：`engine.backward()` 直落 `loss.backward()`，**没有**先乘 loss scale；而 `FP16_Optimizer.step()` 无条件把梯度除以 `cur_scale`（典型 32768）——从未放大过的梯度被照常缩小，有效参数更新比预期小约 `cur_scale` 倍。训练静默停滞在高位 loss，无 NaN、无 crash。该路径留有 `# TODO` 自认未完成；ZeRO>0 与 AMP 路径正确，只有 fp16 非 ZeRO 组合中招。修复 = 该组合路由进 `FP16_Optimizer.backward()`（先乘 scale 再 backward，与 step 的预期对齐）。
- **来源**：DeepSpeed PR #8393（CIFAR-10 MoE 基准（fp16、ZeRO-0、8 experts）实测：修复前 acc 22–28%、loss 停滞 ~1.8；修复后 acc 49%、loss 收敛 ~0.7）。https://github.com/deepspeedai/DeepSpeed/pull/8393
- **行为效应**：lr 看似正常、梯度有限、loss 微降后平台化——表象与「lr 太小 / 数据太难」无法区分，极易被当超参问题丢弃。与 NUM.04（scaler 吞溢出）同属 loss-scaling 契约断裂，但方向相反：那边该更新的 step 被跳过，这边全部 step 被缩小；与 OPT.03（lr=0 阶段藏坏梯度）共享「有效步长≈0 却无信号」的观测面。
- **发现来源**：2026-09-04 每日扫描

### OPT.07 `zerograd_set_to_none_unbinds_graph_bound_grads`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron MFSDP v2 的 `install_sharded_grads()` 在 backward 期间**从 Python 侧**把 `.grad` 绑到 persistent main_grad 视图；CUDA graph replay 只重放 GPU kernel，**不重放 Python 绑定动作**。`zero_grad()` 的默认 `set_to_none=True` 解除绑定后，replay 的 step 里 `.grad` 恒为 None——优化器跳过**每个参数**，run 静默停止训练且 **grad norm 恒 0.0**。修复绕不开「别解绑」：graph 模式下强制 `set_to_none=False` 原地清零，保住绑定。另一层坑：`ChainedOptimizer` 恒以位置参数转发自己的 True 默认值，caller 意图不可达，assert 会在每个 graphed step 上误触——只能 override 不能断言。此前该组合被 `MFSDP v2 不支持 CUDA graphs` 的 fail-fast 挡住（防御即掩盖：门控撤掉后 bug 立即暴露）。
- **来源**：Megatron-LM PR #7075（修复 + `hybrid_llama3_proxy_mfsdp_v2_cuda_graph` golden-value 测试（1×8 GPU，bit-exact determinism check 通过），Closes #6152）。https://github.com/NVIDIA/Megatron-LM/pull/7075
- **行为效应**：无 crash、无 NaN、无报错——grad norm 0.0 被当作「正常小的梯度」，训练看似推进实则权重冻结。与 OBS.07（eval 重放训练态 graph）同根（replay 只执行被捕获的 kernel，Python 语义不随行），但破坏的是**梯度绑定生命周期**而非 eval 语义；与 NUM.10（bf16 静默 skip 全部 step）同症状面（指标健康、参数不动），机制独立。
- **发现来源**：2026-09-05 每日扫描

### OPT.08 `fused_gated_fc1_single_matrix_orthogonalization`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：gated MLP（SwiGLU/GEGLU）的 `linear_fc1.weight` 是**两个逻辑矩阵的物理拼接**（`[gate_weight; up_weight]`，TP 下 partition_stride=2）。Muon/NorMuon 家族对参数做 Newton-Schulz 正交化时把融合矩阵当**单一矩阵**处理：gate 与 up 两个独立投影被当成一个几何对象联合正交化——耦合了本应独立的更新、用了错误的矩阵几何。修复 = 按 TP-local 切成 gate/up 两半各自 NS 后拼回（GTP-remat 参数需先 gather 去 padding 再切）；参数形状与 ckpt 格式不变。torchtitan 侧同期在演进 per-head NS（DeepSpeed #8384），说明 Muon 更新几何是当前活跃故障面。
- **来源**：Megatron-LM PR #6688（修复：检测 2D `linear_fc1` + `gated_linear_unit` 时挂 TP-local split metadata，gate/up 各自 NS 后拼回；TensorParallelAdaptiveMuon 同步处理）+ 追踪 issue #7077（closed as duplicate of #6688）。https://github.com/NVIDIA/Megatron-LM/pull/6688 ；dev 分支端口 PR #7085（2026-09-06 补：`--muon-split-qkv-per-head` 把 per-head NS 推广到 QKV 融合权重——TP>num_query_groups 时梯度跨 TP 维重组后按**逻辑布局**逐 head NS 再还原各 rank 本地 shard，并用 `QKVLayout` 注解把 split 从名字匹配泛化到 MLA/DSv4 hybrid 投影；`muon_tp_mode=distributed` 对不完整 query-group 的投影一次性警告后回退非 TP NS——「Muon 更新矩阵几何」仍是活跃故障面）。https://github.com/NVIDIA/Megatron-LM/pull/7085 ；DeepSpeed #8437（2026-09-07 补）：AutoTP 列并行 shard 上跑整矩阵 NS——正交化依赖**全部行**，shard 是行的真子集，shard 的正交化 ≠ 全矩阵正交化的对应行块（Llama tp=2 实测：梯度一致到 7e-7，update 相对差 **3.9e-01**、cosine 0.936）——同一参数的 Muon 更新随 TP 度变化，TP 度从并行几何变成隐形超参；#8420（Kimi-K3 KDA / GLM-5.2 DSA indexer 的头几何不在 tagger 读取的字段里、留在 full-matrix 路径——当前保守正确，跟踪在案）。https://github.com/deepspeedai/DeepSpeed/issues/8437 ；2026-09-20 补：torchtitan #4692（RFC，2026-09-13）给出**全量融合矩阵 × 逻辑矩阵清单**并确认 upstream main 的 K3 调试配方仍用 AdamW（无 per-head Muon 路径，#4272 P0.5）：`w13` 行交错存储 `[2F,D]`（#4676 改叠放 `[2,F,D]`）、routed `w1_EFD`/`w3_EFD` 两张 `[E,F,D]`、`wkv_b` 按 head 连续块交错 K/V、`wkv_a` latent+rope 行拼接、`wq_b` 每 head nope+rope、KDA `q/k/v_proj` 等 `[H*d,*]`——Kimi 生产侧对 MLA 各投影逐逻辑矩阵独立 NS + aspect-ratio lr（kexue.fm/archives/11126）。「融合权重必须拆逻辑矩阵再 NS」是跨栈共识，本条机制面的 RFC 级枚举。https://github.com/pytorch/torchtitan/issues/4692
- **行为效应**：无 crash、无 NaN——Muon 更新量级/方向系统性错，训练照常收敛但每步优化几何被污染；与 OPT.01（矩未 resume）同为优化器数学契约破坏，本条是**更新算子的矩阵语义**而非状态丢失。#8437 面叠加：换 TP 度重跑或对比实验时 Muon 更新悄然不同（与 NUM.07 的「调优器把不同树钉进各 rank」同属并行度外溢进优化语义）。
- **发现来源**：2026-09-05 每日扫描

### OPT.09 `clip_dispatch_ignores_fsdp_owner_grad_location`

stage: `pretrain` · Cov: `var:global_clip_rescales_corrupted_grad_norm` · 置信度 `documented`

- **机制**：Megatron `clip_grad_norm`/`count_zeros` 对每个参数按「precision-aware optimizer ⇒ `use_decoupled_grad=True`」判梯度存放位；MFSDP v2 从不写 `decoupled_grad`、恒 reduce 进 `.grad`——判据没有「哪个 FSDP 版本持有该参数」的概念，precision-aware + MFSDP v2 组合下 clip 读错张量：**梯度保持未裁剪范数**（实测 213.6 vs clip=1.0），`clip_grad` 完全失效。修复 = 对带 `_mfsdp_parameter_group` 属性的参数返回 False。8×H100 DeepSeek-V3 MLA/MoE EP=2 验证：修复前 `use_precision_aware_optimizer=True + clip_grad=1.0` 的 loss 曲线与 `clip_grad=0.0` **bit 级相同**。
- **来源**：Megatron-LM PR #7074。https://github.com/NVIDIA/Megatron-LM/pull/7074
- **行为效应**：clip 名义开启、实际 no-op——SDC 尖峰/异常梯度不再被 clip 截住，OPT.02（clip 吞 SDC）的正面防线静默缺失；训练继续、偶尔 spike 被归因于超参。与 OPT.04（global clip 重缩放坏梯度）对偶：那边 clip 在算但**参考范数已污染**，这边 clip **根本没作用在真梯度上**；判据写法（按优化器类型而非数据放置事实）是同族病根。
- **发现来源**：2026-09-06 每日扫描

### OPT.10 `muon_update_applied_per_microbatch`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed ZeRO-3 把 Muon 的 Newton-Schulz + momentum 更新挂进 **IPG bucket reduce 路径**——该路径每个 micro-batch 都跑，无 accumulation-boundary 守卫（ZeRO-1/2 的等价逻辑在 `ipg_epilogue` 里、只于 `is_gradient_accumulation_boundary()` 时执行）。`gradient_accumulation_steps=n` 时每步 momentum 被推进 n 次（有效保留率 βⁿ 而非 β：β=0.95、gas=4 → 0.81，gas=16 → 0.44）、NS 作用在**部分梯度**上（NS 非线性：各 micro-batch 正交化之和 ≠ 和的正交化；Muon 更新尺度不变，样本再少的 micro-batch 也贡献单位尺度更新）、NS 计算量 ×n。同数据同种子对照：stage 2 gas=1 vs gas=4 权重相对差 3.3e-04（半精 NS 噪声底，同一训练），stage 3 = **1.0e-01（三百倍于此，另一个训练）**。
- **来源**：DeepSpeed #8443（NS kernel 调用计数表：stage1/2 全部正确、stage3+gas=4 得 16 而非 4）。https://github.com/deepspeedai/DeepSpeed/issues/8443
- **行为效应**：无 crash、无 NaN、loss 正常——用户配置的 momentum 不是得到的 momentum，且偏差随 `gradient_accumulation_steps`（无关旋钮）移动；ZeRO-3 + 梯度累积正是需要 ZeRO-3 的模型的标准配置。与 OPT.08（Muon 矩阵几何错）同为 Muon 流水线数学契约破坏：那边错在单次更新的矩阵语义，这边错在**更新频率与 optimizer step 脱钩**；与 OBS.08（指标按 micro-batch 平摊）形似而质异——那边只错诊断，这边错权重本身。
- **发现来源**：2026-09-07 每日扫描

### OPT.11 `muon_zero_stage0_skips_newton_schulz`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed 的 Muon 实现只在 flat/ZeRO 分片参数路径上接 Newton-Schulz；**ZeRO stage 0（默认）** 下参数原样 2-D 交给优化器、不进该路径，NS 从不执行——最朴素配置 `Muon()` + ZeRO-0 声明的是 Muon、实际训练的是 SGD（动量化）+ 缩放。无「不支持」报错、无回退日志。09-07 曾以「未支持配置的静默回退（CKPT.06/07 族）」暂缓；作者随后自纠「形状无法区分已更新/未更新」的错误论断——`FP16_UnfusedOptimizer` 保留逐参数 fp32 克隆（ndims=2），flat ZeRO 分片恒 1-D，形状可判——修复 PR #8442 已在 non-flat 分支执行 NS（stage 0 fp32 与 stage 1 更新一致，`max|w−SGD|=9.772e-02`），确认为实现 bug 而非设计边界。
- **来源**：DeepSpeed #8441 + 作者自纠评论 + 修复 PR #8442。https://github.com/deepspeedai/DeepSpeed/issues/8441
- **行为效应**：无 crash；loss 照降（SGD 也能训），但用户以为在跑 Muon 的 run 自 step 1 起整个优化算法被替换——正交化带来的收益/动力学全部缺失，对照实验的「Muon vs Adam」结论实际是「SGD vs Adam」。与 OPT.10（更新频率错）/OPT.08（矩阵语义错）构成 Muon 契约破坏三面：这边错在**算法本体是否执行**；「配置声明 ≠ 执行行为」与 CKPT.06/07 家族同型。
- **发现来源**：2026-09-08 每日扫描

### OPT.12 `scheduler_override_rewrites_group_lr_bounds`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `--override-opt-param-scheduler` 的本义是「用运行时实参覆盖 ckpt 里的旧 scheduler 状态」，但 `checkpointing.py` 加载后的赋值段（L3044-L3054）把**全局** `args.max_lr/min_lr` 直接写进**每个参数组**的 LR bounds：显式配置 per-group `max_lr=5.0 / min_lr=0.5` 的组被全局 `1.0/0.1` 覆写。带参数组 LR 分界的训练（MoE 冻结层、LoRA 高 LR 组、per-layer LR decay 关闭后分组）resume 即静默换界——`max_lr=5.0` 的组按 1.0 训，warmup/decay 边界全部错位，无任何日志。修复思路（issue 给出）：加载前捕获运行时组界、加载后恢复；上游测试只断言全局界，组界断言缺失。
- **来源**：Megatron-LM #7173（8×H800 实测：仅加测试断言 upstream 红 `assert 1.0 == 5.0`，加修复后全绿；真实 legacy torch ckpt 写读复现）。https://github.com/NVIDIA/Megatron-LM/issues/7173
- **行为效应**：无 crash、无警告；resume 后各参数组按错误 LR 界训练，分组语义（哪些层快、哪些层慢、哪些冻结）在加载瞬间被全局值抹平——分组收益静默消失或方向反转。与 OPT.01（矩没 resume）同症状面（resume 后前几百 step 行为异常但可归因于 warmup）；与 CKPT.13（sampler 把全局 RNG 当自己的存）同族：**resume 契约把全局状态错当组级状态**。
- **发现来源**：2026-09-10 每日扫描

### OPT.13 `clip_averages_norms_across_disjoint_expert_owners`

stage: `pretrain` · Cov: `var:clip_dispatch_ignores_fsdp_owner_grad_location` · 置信度 `verified`

- **机制**：DeepSpeed FP32 梯度裁剪路径（`clip_fp32_gradients`）结尾把 per-rank 范数在 DP 组上 all-reduce **取平均**——前提是每个 rank 持有相同参数。AutoEP 下各 rank 拥有**不同专家**，rank 局部范数描述互不相交的参数集：`sqrt(20)` 与 `sqrt(100)` 平均得 7.24 而非「每专家计一次」的 `sqrt(120)=10.95`——同一组唯一梯度、仅改变专家放置就能改变 `clip_coef` 达 0.20。修复（PR #8476/#8478）= 专家梯度不进复制范数、经精确的 `get_norm_with_moe_layers`（bf16/fp16 路径早已正确调用；fp32 内联的近似版警告注释所在的具名拷贝**零调用者**，警告从未到达出事地点）按各自 moe_ep_group 归约后 `(Σ norm_i^p)^(1/p)` 合并。上游 #8469（closed as dup of #8475）+ #8475 独立复现同一机制（owner-sharded vs replicated 布局 clip 系数差 0.2000005，修复后为 0，第二台主机复核）。
- **来源**：DeepSpeed #8475（2026-09-10，2×CUDA rank / ZeRO-0 / L2 clip / mpu=None 布局对照实验）。https://github.com/deepspeedai/DeepSpeed/issues/8475 ；修复 PR #8476。https://github.com/deepspeedai/DeepSpeed/pull/8476 ；PR #8478（`sqrt(120)`=10.954 vs 平均 7.236、34% 差，2×H20 四专家独立复现；09-15 进展评论：#8478 内两缺陷已随 `338ed51` 修复——先在 2×H20 复现，且 #8469 发现的更深处场景「pipeline 并行下**完全不拥有任何专家参数**的 rank」改由单次 all-reduced flag + `_EXPERT_PARALLEL_GROUP` 组名解决，uneven 布局测试通过）。https://github.com/deepspeedai/DeepSpeed/pull/8478
- **行为效应**：无 crash；clip 活跃时每次更新的 `clip_coef` 取决于**专家落在哪些 rank**而非梯度本身——同一 MoE 模型换 EP 拓扑 = 换有效 clip 阈值，跨布局实验不可比；无 MoE 时行为不变（原代码路径）。与 OPT.09（clip 判据按优化器类型而非参数 owner）同根：**裁剪几何假设参数复制，EP 打破假设**；与 PAR.11（GTP/EGTP expert 梯度缩放错）互补：那边梯度本身错倍、这边梯度对但裁剪参考范数错。
- **发现来源**：2026-09-12 每日扫描

### OPT.14 `fused_adam_shared_step_counter_bias_correction`

stage: `shared` · Cov: `NEW` · 置信度 `verified`

- **机制**：DeepSpeed FusedAdam 每 dtype 只维护**一个** step 计数器，且只在遍历中**递增最后一个活跃参数**的计数器、随后把该计数器用于该 dtype 的**全部**张量——参数间歇性无梯度（LoRA 冻结/解冻、条件分支、MoE 冷专家）时，计数器在不同参数间**漂移**：bias correction（`1-β^t`）按错误的 t 计算；多 dtype 组还让同一计数器按 dtype 各递增一次。修复 = 逐参数计数器 + 按 (dtype, step) 分桶批处理（同步参数仍共享一次 fused kernel 调用，state_dict 格式不变）。H800 实测：原实现 9 个新测试全挂（Adam/AdamW、均匀与混合 FP32/FP16/BF16、间歇梯度、state-dict round-trip、间歇冻结 layer bias vs PyTorch AdamW 参照），修复后全过 + 2-GPU FP32 ZeRO-0/3 累积更新对全模型 AdamW 参照一致。
- **来源**：DeepSpeed PR #8471（2026-09-10；「间歇训练下原代码 fail 全部九个新用例」；注意：不重建旧版本已写坏的计数器）。https://github.com/deepspeedai/DeepSpeed/pull/8471
- **行为效应**：无 crash、无 NaN——bias correction 系数错值在 **moment 估计上产生平滑偏差**，有效 LR 偏移随参数的训练历史分叉；症状是收敛速率与预期的不可归因偏差，config/log 全部正常。与 OPT.01（矩没 resume）不同：矩的数值格式对、`step` 标量本身错；与 NUM 家族（cast/scaler）不同：这是**优化器簿记状态共享**错误，纯 fp32 下也发生。间歇梯度场景（LoRA/条件路由/MoE 冷专家）是主要暴露面。
- **发现来源**：2026-09-12 每日扫描

### OPT.15 `muon_scale_counts_alignment_padding_rows`

stage: `pretrain` · Cov: `var:muon_update_applied_per_microbatch` · 置信度 `verified`

- **机制**：Megatron GTP 沿 dim0 分片权重，行数不能等分对齐时**先补零行再切**（如 126 行 GTP8 补到 128 行；62 行补到 128 时部分 rank 全是 padding；TP×GTP 组合先切 TP 再逐片补齐）。这些零行是存储/通信需要的，但 Muon 的谱缩放因子按**填充后的本地形状**计算——把 padding 行计入逻辑形状：Newton–Schulz 仍用 padded 张量做 collective（正确），scale 却该用 `logical_rows = local_rows × gtp_size − pad_length`。实测谱缩放错值例：GTP2 `[8,4]`→逻辑 `[6,4]`：2.828→2.449；TP4×GTP2 `[32,8]`→`[20,8]`：5.657→4.472（~27% 偏差）。
- **来源**：Megatron-LM PR #7236（2026-09-11；三种 TP×GTP 布局的 spectral scale 对照 + 单测/功能测试）。https://github.com/NVIDIA/Megatron-LM/pull/7236
- **行为效应**：无 crash、无 NaN——凡 GTP 行数不对齐的矩阵，Muon 更新带一个 layout 相关的缩放偏差；同一模型在不同 GTP 度下有效更新幅度不同（跨并行度实验不可比）。与 OPT.10（每 micro-batch 重复 NS）/OPT.11（stage0 不跑 NS）/OPT.08（门控 fc1 联合正交化）同为 **Muon 契约破坏**家族新面：算法执行了、缩放的**形状语义**错——「通信填充不是逻辑参数」的契约在优化器层失守；与 PAR.11（GTP/EGTP）同属 GTP 引入的静默缩放面，那边打击梯度归约、这边打击更新缩放。
- **发现来源**：2026-09-12 每日扫描

### OPT.16 `mixed_dtype_grad_list_single_multi_tensor_launch`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `--use-precision-aware-optimizer` + bf16 梯度 buffer（`grad_reduce_in_fp32=False`、`main_grads_dtype=bf16`）下，bf16 参数暴露 bf16 `decoupled_grad`，而 `mark_keep_in_fp32` 的原生 fp32 参数（DeepSeek-V4 hyper-connection alpha/bias、`ape`、`attn_sink` 一类）持有 fp32 梯度。`get_grad_norm_fp32` / `clip_grad_by_total_norm_fp32` 把两种 dtype 塞进**同一个** TE `multi_tensor` launch——kernel 按**列表首张量**的 dtype 派发、按该元素大小读全部张量：fp32 在前时 bf16 张量越界读出 illegal memory access（crash 面）；**bf16 在前时 fp32 梯度被按 2 字节读——静默算出垃圾 norm/clip，不 crash**。`assert grad.dtype in (float32, bfloat16)` 放行混 dtype 但从不分组。修复 = 按 dtype 分组各跑一次 kernel、范数平方累加后单次 all-reduce（collective 数与同质列表语义完全一致）；实测分组 `{'float32': 7, 'bfloat16': 295}`，DeepSeek-V4-Pro SFT（TP1/PP32/EP8）32/64 节点正常训练。已知非 TE #2918（int32 numel 溢出）。
- **来源**：Megatron-LM #7338（crash 站点 `CUDA_LAUNCH_BLOCKING=1` 定位到 `clip_grads.py:114`；修复已验证；oncall 建议转 TE 仓库跟踪，2026-09-15）。https://github.com/NVIDIA/Megatron-LM/issues/7338 ；归属裁决 + 修复 PR（2026-09-17 补）：NVIDIA oncall 一度建议转 TE 仓库，报告者以「混 dtype 列表由 MCore `clip_grads.py` 自己拼装、TE kernel 按首张量 dtype 派发是其文档化契约」反驳；Connor-XY 09-16 裁定留在 Megatron-LM——norm 与 scalar/tensor 双裁剪路径均缺 dtype 分组，draft 修复 #7279 按 dtype 分组并保持 collective 顺序不变。https://github.com/NVIDIA/Megatron-LM/pull/7279 ；PR #7279 生产验证（2026-09-18 补）：linmuchuiyang 在 DeepSeek-V4-Pro（~1.6T MoE）full-param SFT、32 节点 ×8 H100、TP1/PP32/EP8、bf16 梯度 buffer + CPU-offload precision-aware optimizer 上验证 head `333ecc95`——原 crash 路径干净运行，grad norm 与其自研 dtype 分组 workaround 一致（#7338 评论区）。
- **行为效应**：两种截然不同的可见面由**列表顺序**决定：fp32 梯度排前 → 首 `get_grad_norm_fp32` 即 NCCL illegal memory access（fail-fast，易误诊为通信故障）；bf16 梯度排前 → 混 dtype 梯度范数/clip 系数静默错值，grad norm 曲线照常打印但数值无意义，clip 保护面失效。与 OPT.09（clip 判据按优化器类型）/OPT.13（clip 范数跨不相交专家取平均）同族：**裁剪路径的 dtype/所有权几何假设被打破**；与 NUM 家族 cast 类不同：那边是数值精度转换错、这边是 kernel dispatch 按 dtype 分组缺失；`grad_reduce_in_fp32=True` 可回避但梯度 buffer 翻倍（12.7B 专家桶 +23.6 GiB，H100 80GB PP32 下 OOM）。
- **发现来源**：2026-09-16 每日扫描

### OPT.17 `tp_sharded_param_unmarked_skipped_by_grad_norm_filter`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `DotProductAttention.softmax_offset`（`softmax_type="learnable"`，GPT-OSS learnable attention sink）按**每本地 head 一项**创建——本质是跨 TP head-sharded 参数，却**从不标 `tensor_model_parallel`**（TEDotProductAttention + TE 2.15 同样漏标）。`MegatronOptimizer` 的 `_filter_grads_for_norm`（optimizer.py:299）按 `param_is_not_tensor_parallel_duplicate` 只在 **TP rank 0** 保留该参数：其余 rank 的 sink 梯度被从 grad-norm 里**静默丢弃**。2-rank repro：sink 梯度 3 与 4 → norm 算出 3 而非 5；`clip_grad=4` 时 clip 被跳过（scale 1.0 而非 0.8）。真实小 attention 块：norm 1.22608447 vs 1.22774088，一步 clipped 更新后输出误差 2.4e-4——**该 norm 用于裁剪全部参数**。修复 = 给 `softmax_offset` 设 `tensor_model_parallel=True, partition_dim=0`。同类错误：#5916（expert grouped-linear 梯度被 clip norm 漏计，2026-07-25 已修）。
- **来源**：Megatron-LM #7452（2026-09-17；finding 8，166 行 CPU/Gloo repro 把 fused multi-tensor L2 kernel 换 CPU sum 后跑真实 `clip_grads.get_grad_norm_fp32` + Gloo collective；Bridge GPT-OSS GB200 recipe 用 TP=2 CP=4 暴露面确认）。https://github.com/NVIDIA/Megatron-LM/issues/7452 ；2026-09-20 补：dundysm 认领 finding 8（计划 `softmax_offset` 标 `tensor_model_parallel`，与条目修复方向一致），修复 PR 待开出。修复 PR #7530 已开（2026-09-22 补：`mark learnable softmax_offset as tensor parallel for clip norm`，自述「Fixes Finding 8 of #7452 only」），尚未 merge。https://github.com/NVIDIA/Megatron-LM/pull/7530
- **行为效应**：无 crash、无 NaN——grad norm 略偏小、clip 阈值行为错位，所有共享该 norm 的参数的裁剪强度漂移；误差量级小（1e-3 相对）且随 head 数分布变化，极难从曲线上归因。与 OPT.09（clip 读错张量）/OPT.13（clip 平均不相交专家范数）/OPT.16（混 dtype 列表）同族「clip 路径的参数归属/几何假设破裂」的第 4 个实例，这边断点是**参数未标的 TP 分片标志**；与 MOE.08（负载均衡 hook 漏扫 mtp_layers）同型「过滤器按名字/标志白名单、新参数不在名单」。
- **发现来源**：2026-09-18 每日扫描

### OPT.18 `gdn_shared_out_norm_missing_tp_grad_sum`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron GDN（GatedDeltaNet）的 `out_norm`（common.py:267）建在 `value_head_dim` 上——**一个向量施加到所有 head**，而 head 被 TP 切分。`finalize_model_grads.py:452` 只在 **SP 开启**或参数名含 `q_layernorm`/`k_layernorm` 时对该类共享向量做 TP 梯度求和；GDN + TP>1 + **SP 关**时 `out_norm.weight` 的梯度从不 all-reduce，两个 TP 副本各自持有部分和。TP=2 全 GDN 模块 repro：其余参数梯度与未分片参照吻合 7.5e-9，`out_norm` 梯度 `[0.1200,0.0400]` / `[0.0197,-0.0077]` vs 完整 `[0.1397,0.0323]`；20 步 Adam 后两副本差 0.0039，加 TP sum 后每步 2.4e-7。暴露面：Bridge Qwen3.5 full-SFT recipes 用 TP=2/4 **不开 SP** 跑 MCore GDN。
- **来源**：Megatron-LM #7452（2026-09-17；finding 11，全 GDN 模块 CPU repro + 20 步 Adam 漂移对比）。https://github.com/NVIDIA/Megatron-LM/issues/7452 ；2026-09-22 补：vipulsarode 09-21 认领 finding 11（计划按 `11_gdn_out_norm_missing_tp_sum.py` repro 走），oncall tracking map 要求认领时链接 scoped PR——PR 尚未开出；2026-09-24 补：scoped 修复 PR #7613 已开（2026-09-23）——在 `6a36609` 上用原 repro 复现，修复后两 TP rank 的 `out_norm` 梯度与单进程参照吻合，覆盖 GDN/GDN2 × RMSNorm/LayerNorm，附参数 tagging 与 finalizer 恰好一次求和（SP on/off）单测。https://github.com/NVIDIA/Megatron-LM/pull/7613
- **行为效应**：无 crash、无 NaN——TP 副本从第一步起**确定性分叉**（不是 SDC 式随机，是结构性的部分和），`out_norm` 权重跨 rank 漂移、且永远偏小（漏加其余 rank 贡献）；下游 loss 无立刻信号。与 PAR.13（DDP reducer hooks 早退）/PAR.09（分桶谓词跳过 allreduce）同族「梯度归约静默缺失」，差异在这边只打击**名字白名单外的共享向量参数**——选择性漏归约；与 SFT 侧 `q_layernorm` 特判（同函数）对照可见这是**白名单式 finalize 假设**的固有脆弱性（新架构新参数不在名单）。
- **发现来源**：2026-09-18 每日扫描

---

# ACT · 重计算

## ACT

### ACT.01 `moe_data_dependent_recompute_diverge`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：MoE dispatch 是数据依赖控制流。activation checkpoint 重算时若随机数/路由决策与前向不一致，重算图与保存的激活对不上，反向得到静默 NaN 或错梯度。
- **来源**：交叉：推理编目模型图类「数据依赖 expert dispatch 破坏 checkpoint 重算契约」。训练主路径。
- **行为效应**：Checkperr 硬失败，或无声 NaN 梯度。取决于框架是否校验。
- **发现来源**：2026-08-17 基础调研

### ACT.02 `selective_recompute_stale_rng`

stage: `shared` · Cov: `var:moe_data_dependent_recompute_diverge` · 置信度 `documented`

- **机制**：选择性重计算漏存 dropout / router RNG。重算走了另一条噪声实现。
- **来源**：Megatron 重计算 + dropout 的已知契约；#6519 dual RoPE full recompute 一类修复说明这条线仍在演进。https://github.com/NVIDIA/Megatron-LM/pull/6519
- **行为效应**：开启 recompute 后曲线与关 recompute 的基线分叉，无 shape 错误。
- **发现来源**：2026-08-17 基础调研

### ACT.03 `replay_graph_diverges_from_train_graph`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron #5841 给 mHC recompute 引入 attention-only CUDA graph split 时，**顺带改了默认 capture range**：`cuda_graph_modules == ['attn']` 原本 capture 整个 `_forward_attention`，被静默收窄为 input-layernorm + self-attention；为新路径加的 gate 又 keyed on `'attn'`，range 在脚下被换掉后 gate 开始误拒以前合法的配置。同时旧全注意力 replay 路径本身带三个潜伏缺陷（recompute manager 够不到 MLP tail、overlap tail 缺 `pre_mlp_layernorm` checkpoint、`padding_mask` 未穿到 router 的 replay），使 **replay 图与训练图结构性分叉**。
- **来源**：Megatron-LM PR #6661（修复：把 split 收进 `mhc_recompute_attn_cuda_graph_split` 开关、恢复默认全 range、修三个旧缺陷；GB200 验证）。https://github.com/NVIDIA/Megatron-LM/pull/6661
- **行为效应**：默认路径的图捕获范围与用户预期/历史行为不一致——CUDA graph replay 的是**另一段子计算**；无 NaN、无 crash，golden values 才暴露。与 ACT.01/ACT.02（重算侧 RNG/路由不一致）不同：错的是 **capture/replay 范围定义本身**，与数据依赖控制流无关。
- **发现来源**：2026-08-21 每日扫描

### ACT.04 `async_ckpt_worker_failure_strands_peers`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `--use-persistent-ckpt-worker` 下，persistent checkpoint worker 子进程的异常（Python exception / preload 异常 / `os._exit` 硬退出）终止子进程而不发布 completion record——父进程无限轮询一个永不到达的 completion，**单 rank 故障把全部分布式 peer 挂死**在 ckpt finalize。父进程只轮询 completion queue，从不把 worker traceback/exit code 转成分布式错误；failed worker state 不清理，后续 worker 也起不来。
- **来源**：Megatron-LM #6970（2/6 rank 受控注入：async_fn 异常、preload 异常、`os._exit(17)` 三场景；本地修复验证三场景全 rank 传播成功）。https://github.com/NVIDIA/Megatron-LM/issues/6970 ；修复 PR #6971。https://github.com/NVIDIA/Megatron-LM/pull/6971
- **行为效应**：故障面是活性（hang 而非静默错值），但命中「坏 ckpt 窗口」的编排陷阱：若 worker 在写出部分 ckpt 后死、peer 被挂住、外部 watchdog 重启整个 job 并 resume 到**半成品 ckpt**，坏锚点被静默接上（与 CKPT.08/HW.05 的 resume 坏锚点风险面合流）。观察信号：job 卡在 ckpt finalize、step 计数冻结、completion queue 无消费。
- **发现来源**：2026-09-05 每日扫描

### ACT.05 `partitioned_activation_recompute_mangles_args`

stage: `shared` · Cov: `NEW` · 置信度 `verified`

- **机制**：DeepSpeed `partition_activations` 下，`get_partitioned_activations_for_backward` 对**每个**参数保存 `(value, size)` 成对条目——`tensor_flags` 与 `non_tensor_objects` 因此各持有**每参数两条**。`merge_tensors`（recompute 时还原参数表）的折叠启发式「跳过 True 后面那条」只对张量参数有效（非张量参数的 value 与其 `None` size 都是非张量，啥也没折叠）：recompute 收到的参数表里**每个非张量参数后面多插一个 `None`**，其后所有参数右移。CPU 驱动真实 save/restore 链路实测：`(tensor, 2)` → `(tensor, 2, None)`；`(tensor, None, True, tensor)` → `(tensor, None, None, True, None, tensor)`。修复 = 每参数占偶数下标，`[::2]` 丢弃奇数条。既有 `TestCheckpointNonTensor` 恰好覆盖这些参数形状但**从不在 partition_activations 下跑**——全张量参数免疫，所以一直没被发现。
- **来源**：DeepSpeed PR #8455（2026-09-08；`pytest -k partitioned` master 6 failed → 修复后 6 passed；端到端需真 Stream，作者如实标注未跑）。https://github.com/deepspeedai/DeepSpeed/pull/8455
- **行为效应**：recompute 参数表错位——检查点函数带任何非张量参数（flag / 标量超参 / None）且开 partition_activations 时，重算前向在**错误的参数表**上执行：可能崩（形状错）、更可能静默产出与原前向不一致的激活 → 错梯度。与 ACT.01（MoE 数据依赖 recompute 分叉）同族：**重算图 ≠ 原前向**；差异在这边是参数表序列化/折叠 bug 而非 RNG/数据依赖；与 DATA 家族的错位面不同：只打击开 partitioned activation ckpt 的组合。
- **发现来源**：2026-09-12 每日扫描

### ACT.06 `sar_recompute_skips_inplace_writes`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：torchtitan GraphTrainer 的 `selective_activation_remat_pass`（SAR）沿 dataflow producer 边选重算点。对「functional 包装 + 内部 in-place kernel」的 `autograd.Function`（`y = x.clone(); op_(y); return y`，schema 正确、算子返回 void），写入没有任何 dataflow 输出——pass 只重放 `clone` 不重放写入，backward 拿到**变异前**的值。CPU 复现：`loss=(x+1)^2, x=2`，前向 loss=9 正确、梯度 6→**4**；`graph.lint()` 通过、执行无任何错误。已在 2026-09-15 main `f5c97dc0` 直接执行该 commit 的 pass 源码复核（pass 级，非完整训练运行）。起源是 functional partial-RoPE 包装 in-place kernel 的真实用例；报告者本地「重放缺失写」可把梯度恢复为 6，但通用修复还需处理 alias / 写序 / 其他写目标（支持范围待维护者定夺）。
- **来源**：torchtitan #4688（pass 级 CPU 复现脚本 + main 复核；`torch==2.14.0+cpu` aarch64）。https://github.com/pytorch/torchtitan/issues/4688
- **行为效应**：无 crash、无 NaN、lint 全过——前向数值正确、**梯度静默错值**；凡走 GraphTrainer + SAR 且重算激活涉及「clone 后就地写」型 functional 算子（fused/自定义 RoPE 等 in-place 包装）即中招。与 ACT.05（partitioned recompute 参数表错位）同族：**重算执行了但重算状态 ≠ 原前向**；差异在这边是 dataflow 分析不建模副作用写入、那边是序列化折叠 bug；与 ACT.03（replay 图与前向分叉）同症状面，这边给出了 pass 级最小因果链。
- **发现来源**：2026-09-16 每日扫描

### ACT.07 `offload_reload_missing_stream_ordering_edge`

stage: `shared` · Cov: `NEW` · 置信度 `verified`

- **机制**：Megatron 细粒度激活 offload 的 `ChunkOffloadHandler.reload()` 把 H2D 拷贝排进 caller 当前流，对 `d2h_stream` 上记录的 `_offload_event` **没有任何排序边**。`bulk_reload_group`（prefetch 路径）自己补这条边；`tensor_pop` 的 **inline reload 没有**——backward 先于 prefetch 到达某 offloaded group 时，H2D 可能在 D2H 仍在写 pinned staging buffer 时就读它，重载进 GPU 的 saved activation 是**陈旧 buffer 内容**（CPU tensor pool 回收复用使污染内容确定化）。与特性自身文档承诺直接矛盾（user-guide L180：「compute stream waits for the transfer to complete」——inline 路径不等）。B300 上 20 trial × 2 臂：无边树 19/20 损坏（wrong 3.2M–67M/67M 元素，uniq 值 [-7.0, 1.0] 混合 = 新旧内容掺杂）；加排序边 0/20。warmup 迭代**全部** reload 走无序 inline 路径（24/24），steady 态才由 prefetch 覆盖；现有测试 `test_gpt_fine_grained_activation_offloading_*` 恰好把 warmup 迭代结构性丢弃后才断言梯度——`tests/` 无一读 `_offload_event`。修复 PR #7399。
- **来源**：Megatron-LM #7398（2026-09-16；mcore `79097b457`，torch 2.11.0+cu130，B300/A40 单卡单进程；确定性机制演示：stall d2h_stream + SENTINEL 预污染 → `tensor_pop` 返回污染值；缺陷自 #1913 引入即有）。https://github.com/NVIDIA/Megatron-LM/issues/7398 ；修复 PR #7399（2026-09-16 已提；Connor-XY 指派 lhb8125 评审，并要求复查 warmup vs steady、当轮未 offload 的 group、reset 行为——修复待 maintainer review）。https://github.com/NVIDIA/Megatron-LM/pull/7399
- **行为效应**：无 raise、loss 曲线照降——warmup 迭代（及任何 backward 跑赢 prefetch 的时刻）的梯度**静默算在陈旧激活上**；「每块激活 = 新旧值混合」意味着错误幅度连续可变、无全有全无信号。与 ACT.05/ACT.06 同族「重算/重载状态 ≠ 原前向」：那边错在参数表/dataflow 建模，这边错在**流同步契约**（文档承诺 vs inline 路径实现）；与 KER.09（掩码越界静默写）同为 B300 新硬件面上被旧假设掩盖的流/坐标契约破裂。
- **发现来源**：2026-09-17 每日扫描

---

# KER · 训练 kernel

## KER

### KER.01 `fa_backward_mantissa_truncation`

stage: `pretrain` · Cov: `NEW` · 置信度 `verified`

- **机制**：在 FlashAttention backward 里对保存的 O 和/或 dO 做尾数截断（`bitshift`）或就近舍入（`round2nearest`）。`softmax_d = rowsum(dO⊙O)` 被污染，dq/dk/dv 系统性偏。前向保持干净。本实验室把它当作低精度注意力的可控阳性。
- **来源**：`opensource_training_megatron` `fault_injected_attention.py` / `fault_injection.py`；`tests/test_fault_injection.py`。
- **行为效应**：不 NaN。QK 谱 climb→collapse，`realized_var` 抬升，MoE `per_token_entropy` 先塌，然后才是 grad norm / loss spike。
- **发现来源**：2026-08-17 基础调研（本地实验）

### KER.02 `fa_backward_dual_site_O_and_dO`

stage: `pretrain` · Cov: `var:fa_backward_mantissa_truncation` · 置信度 `verified`

- **机制**：同一 kernel 上 O 与 dO 两个注入点不是同一个故障。O 通过 softmax_d 间接污染；dO 直接进 backward GEMM。环境变量互斥：`bitshift_*` 与 `round2nearest_*`。
- **来源**：同 KER.01；`debugging_fault_injection.py`。
- **行为效应**：同样剂量下两条曲线的谱/鞅时钟不同；对照实验必须声明注入点。
- **发现来源**：2026-08-17 基础调研（本地实验）

### KER.03 `training_kernel_sync_semantic_wrong_value`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Triton/编译器在合法 API 下同步范围错了，MMA 未完成就读，或 `inline_asm_elementwise` 对复制元素执行多次副作用。训练 GEMM/attention 会吃到有界错值。**训练框架同型实例（2026-08-20 补）**：Megatron grouped fused MLP 的 TP=1 路径，两个未注册执行视图（normalization 视图 + fused MLP 视图）共享 FC1 参数存储，descendant pre-forward hook 只在 fused 边界转发 → 开 distributed-optimizer 参数 gather 时，normalization 可以在 FC1 参数同步 hook 运行**之前**读到旧存储——纯控制流/事件序错值，数值本身全对。
- **来源**：交叉：推理编目 `1.9 kernel_sync_semantic_silent_error`（triton #11047 / #11065）。训练同样走这些核。Megatron-LM PR #6654（配对回归测试：控制组确定性失败于「normalization 先于 FC1 pre-hook」，修复组通过；50-step 8×B200 TP=1 DP=8 FP8-MX + full-iteration CUDA graphs 验证 finite loss/grad、无 NaN）。https://github.com/NVIDIA/Megatron-LM/pull/6654
- **行为效应**：无 crash、数字合理但错；像轻度 SDC。#6654 实例：validation run 本身 finite loss、无 NaN、无 assert——只有确定性事件序回归测试抓得住。
- **发现来源**：2026-08-17 基础调研（从推理编目回链）

### KER.04 `rocm_wheel_silent_loss_parity_regression`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：同一份 torchtitan 训练代码、同一组 Qwen3 MoE FSDP+TP+EP 配置，仅因把 ROCm 7.2 PyTorch wheel 换成 7.14 wheel，训练即产出**极大/NaN 梯度范数**并 loss parity 失败——错值藏在闭源发行版 kernel/builder 差异里，训练框架侧无从 assert。对齐 LOSS/NUM 类数值故障的**发行版供应链**来源：不是超参也不是模型结构，而是「环境升级 + 无 golden 对照」这一组合。
- **来源**：torchtitan #4205（ROCm 8-GPU feature test：Qwen3 MoE FSDP+TP+EP loss parity against reference 失败；7.2 对照稳定；维护者标 duplicated by pytorch/pytorch#194004 转上游）。https://github.com/pytorch/torchtitan/issues/4205
- **行为效应**：部分 run NaN，另一些 run 仅 loss-parity 偏差——不跑 parity golden 就只有「梯度范数异常大」这一间接信号；本地诊断极难归因到 wheel。
- **发现来源**：2026-08-23 每日扫描

### KER.05 `framework_version_drift_silent_loss_divergence`

stage: `rl` · Cov: `var:rocm_wheel_silent_loss_parity_regression` · 置信度 `documented`

- **机制**：同一份训练脚本、同数据同种子，verl `v0.9.0` → `HEAD`（1dda039b）的 distillation loss 从 step 1 起静默分叉：多教师 OPD（GRPO + FSDP2 + vLLM rollout/teacher）在 NVIDIA B300 与寒武纪 MLU 两套硬件栈复现。本编目用 compare API 核对两 ref 间 40 个 ahead commit / 297 个变更文件：`examples/on_policy_distillation_trainer/` 下只有启动脚本（uv 包装，报告者经 `ray job submit` 绕开）变更，蒸馏数值路径的漂移藏在共有 trainer/数据/kernel 路径里，框架侧无 assert、无警告。把 KER.04 的「发行版升级 + 无 golden 对照」族推广到「框架 ref 漂移」向量：升级后不跑 step-1 parity 对照就只有「曲线看起来在训」。
- **来源**：verl #7540（双硬件 A/B 复现脚本 + 曲线截图，0 评论未分诊；compare 核验见 `daily/raw/2026-08-26/issues/verl_compare_*.json`）。https://github.com/verl-project/verl/issues/7540 ；配置面线索（2026-08-30 补）：verl #7597——#7540 报告命令里嵌套 `actor.fsdp_config.strategy=fsdp2` 在 unmodified main 上被**静默丢弃**（请求 FSDP2、实际跑 FSDP1，effective 全部落 fsdp），A/B 两版本的有效引擎策略可能根本不同；#7597 使 `actor.strategy` 成为可执行真源并拒绝冲突嵌套值（该 PR 明确不主张解释 #7540 的 loss 分叉本身，蒸馏数值代码两版本间未变）。https://github.com/verl-project/verl/pull/7597
- **行为效应**：`distillation/loss`、`teacher_mass`、`student_mass` 与基线版本自第一步起数值不一致，但全程有限、无 NaN、无 crash；root cause 尚未定位（issue 未分诊）。
- **发现来源**：2026-08-26 每日扫描

### KER.06 `torch_compile_float_kwarg_stale_constant`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：`megatron.core.fusions.fused_bias_swiglu.clamped_swiglu(_back)` 经 `torch.compile` 包装后，Python float 标量 kwargs `clamp_value` 在 PyTorch 默认 `specialize_float=False` 配置下被当作编译期常量捕获。首个被观察到的值（如 0.5）固化进 Inductor 生成代码，之后传入 1.0 仍执行按 0.5 编译的 kernel——前向、AOTAutograd 生成的反向、显式编译的 `clamped_swiglu_back` 三处全部静默错值，无异常无警告。训练侧触发条件不是假设：StepFun Step-3.5-Flash / Step-3.7-Flash 的 `swiglu_limits`（routed=7）与 `swiglu_limits_shared`（shared=16）按层/按专家类型取不同 clamp，K-EXAONE-2.0-750B 也用 layer-indexed `swiglu_limits` 且只在深层开 clamp；Megatron Bridge `step35_provider.py` 已按层深拷贝 config 传入不同 `activation_func_clamp_value`——第二组层起全部用旧 clamp 值训练。
- **来源**：Megatron-LM #6918（reproducer 直接 import 原函数、fresh process CPU/CUDA 双复现、不 reset dynamo 不改 specialize_float；维护者初步回复「clamp_value 被视为常量不算 bug」，社区补充 per-layer 真实用例）；上游 pytorch/pytorch#194976；DeepSeek-V4 technical report §4.2.3（SwiGLU Clamping 与 Anticipatory Routing 并列的两大训练稳定手段：clamping「有效消除 outlier、显著稳定训练且不损性能」，见 LOSS.06——生产级训练已把 per-layer clamp 当基础设施依赖，该依赖正落在被冻结的编译期常量上）。https://github.com/NVIDIA/Megatron-LM/issues/6918 · https://github.com/pytorch/pytorch/issues/194976 · https://arxiv.org/abs/2606.19348
- **行为效应**：无 crash、无 NaN、无警告；SwiGLU 前向/反向输出系统性偏移（reproducer 中 forward/autograd diff 非零），训练曲线照常。与 KER.03 同族（编译栈在合法配置下静默错值），但根因是标量捕获语义而非事件序。
- **发现来源**：2026-08-28 每日扫描

### KER.07 `activation_offload_bypasses_absorbed_mla`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron 细粒度 activation offload 的模块选择逻辑不覆盖 absorbed MLA 投影——配置里开了 offload 的模块在 absorbed MLA 路径下被**静默绕过**：不报「不支持」，也不下装，激活照常驻留显存。正确性面不是 offload 本身算错，而是「配置声明 ≠ 执行覆盖」：同配置在非 MLA 模型上生效、在 absorbed MLA 上无效，A/B 显存对比成了唯一暴露手段；若用户为满足显存而叠加其它妥协（减小 batch / 开启未验证的 recompute 组合），错的输入进优化决策。修复 = offload 逻辑延伸到 absorbed MLA 投影。
- **来源**：Megatron-LM PR #7090（两节点 B300 GLM 5.3 生命周期 XID 1050342 完整 fwd/bwd/step/ckpt 验证）。https://github.com/NVIDIA/Megatron-LM/pull/7090
- **行为效应**：无 crash、无日志差；activation offload 的显存收益静默缺失。边界条目：单独发生是容量/性能面，但它是 CKPT.06 家族（配置被静默吞掉）在执行层的镜像——开关通过校验、行为不变。记录在案以防后续与 recompute/精度问题组合成静默错值面。
- **发现来源**：2026-09-06 每日扫描

### KER.08 `overflow_grad_poisons_muon_momentum`

stage: `pretrain` · Cov: `var:loss_scaler_swallows_overflow` · 置信度 `documented`

- **机制**：DeepSpeed ZeRO-1/2 的 Muon 路径在 `independent_gradient_partition_epilogue` 里、于**溢出检查之前**对累积梯度跑 `muon_update`：`momentum.lerp_(grad, 1-β)` 把 inf/NaN 折进 momentum（此后无人重置，永久毒化）；nesterov 分支再把 NaN 原地写回梯度。被丢弃的 step 本应不动优化器状态——这里既毒了 momentum 又让**下一个** step 无论 loss scale 多小都必然再溢出：失败自持，loss scale 一路减到最小值后 `Exception` 退出，全程**零参数更新**。fp16 + Muon + ZeRO-1/2（官方 `tests/unit/ops/muon/test_muon.py` 自用的配置只多跑几步）即中招。测试套件绿的另一个原因：参数变更断言在 `deepspeed.initialize`（fp16 cast）**之前**克隆初始参数，fp32 vs fp16 的 `torch.equal` 恒 False——断言零步训练也能通过。
- **来源**：DeepSpeed #8432（隔离表：同配置仅 `initial_scale_power` 不同——默认 16（scale 65536）0/10 参数变更、momentum 非有限、死于最小 scale；scale 1（无溢出）10/10 正常训练）。https://github.com/deepspeedai/DeepSpeed/issues/8432
- **行为效应**：fp16 面最终以 loss-scale 耗尽 crash 收场（首溢出前的训练与延迟可见面）；测试侧的静默面更宽：断言被 dtype cast 洗白，任何「run 太短够不到 crash」的验证都报告成功。与 NUM.04/NUM.10（scaler 吞溢出/静默 skip）同族：溢出处理契约与优化器状态生命周期错配——那边丢的是 step、这边毒的是永久缓冲；测试断言漂移与 OBS 面（观测骗过）交叉。
- **发现来源**：2026-09-07 每日扫描

### KER.09 `triton_block_local_mask_oob_rope_write`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `fused_mla_yarn_rope_apply.py` 四个 Triton kernel 的边界掩码用**块内** head 坐标（`tl.arange(0, BLOCK_H)`）去比**全局** head 数：`mask = local_off < head_num * stride`，正确条件应为 `pid_head * BLOCK_H + local_h < head_num`。`pid_head==0` 恰好正确、之后的块掩码恒过宽——最后一个不整除块的全部 lane 通过。`BLOCK_H` 在 {1..128} autotune，任何非 2 的幂 head_num 都存在 `head_num % BLOCK_H != 0` 的 tiling；现有单测全用 `num_heads=32` 所以漏网。bwd kernel 经同一掩码 `tl.store`：幻影 lane 是**写**不是读——内部 token 静默污染下一 token 的 head 0–3 rope 切片（与合法 writer 竞争），行尾 token 越界写到分配之外。`restore_value=["Q"]` 只恢复张量自身存储、救不回写到外面的部分；autotuner 首次调用把每个候选 tiling 都执行一遍，不安全配置无论胜出与否都已跑过。
- **来源**：Megatron-LM #7103（5 处掩码位点逐行定位；standalone 复现：head_num=12/BLOCK_H=8 时 4 条幻影 lane 落在 +128/+320/+512/+704；H100 `CUDA_LAUNCH_BLOCKING=1` 下无分配 slack 即 illegal memory access，有 slack 则静默污染后续内存——后者才是到达真实训练的形态）。https://github.com/NVIDIA/Megatron-LM/issues/7103
- **行为效应**：训练侧无 precursor：有分配 slack 时进程存活、内存被静默写坏，NaN 或 illegal access 在数百 step 后才出现且与根因完全解耦；autotune 基准 pass 首次调用即执行越界。与 KER.01/02（FA bwd 数值错）同为训练 kernel 静默错值，差异在这边是**内存安全边界错**而非浮点精度；测试盲区来自「只测 2 的幂 head_num」，autotune 依赖面与 NUM.07 共享。
- **发现来源**：2026-09-08 每日扫描

### KER.10 `dsa_cudnn_packed_cp_wrong_values_flaky_masked`

stage: `pretrain/sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron DSA（DeepSeek Sparse Attention）cuDNN real-kernel 路径在 **packed-THD + context-parallel（CP>1）** 组合下产出错误数值：top-k 索引测试 8/8 元素全错（应 bottom-right 对齐却整体错位）、full-fusion 输出与参考路径 cosine 相似度仅 **0.365**（两条路径对同一输入严重分歧）；张量全部有限——kernel 正常运行完、值是错的。两测试自 #5099 合入起就带 `@pytest.mark.flaky` / `flaky_in_dev`（注释引用 import error 与「CI 里的确定性 CUDA error」），GB200/SM100 实机上标记下藏的是**真数值失败**。cuDNN 版本对照已做（torch 编译于 9.23 vs 装的 9.21，换 9.23 后同样 2 failed）——非版本错配。该 packed+CP 组合正是 GB200 GLM-5.2 SFT recipe 的形状（`dsa_kernel_backend=cudnn`、CP4、packed、`cp_comm_type=allgather`）；32 节点训练形状对照显示 10 步内 lm loss 差 ≤0.7%、100 步 plateau 同型（0.646 vs 0.615）——**哪些形状中招未能定位**，影响面悬置。
- **来源**：Megatron-LM #7105（`tests/unit_tests/transformer/experimental_attention_variant/test_dsa_native_parity.py` 78 passed / 2 failed 实测 + 复现环境全列；#5099 引入 CP-via-THD，#4878 仍 open）。https://github.com/NVIDIA/Megatron-LM/issues/7105 ；同 kernel 家族 crash 面（2026-09-22 补）：Megatron-LM #7463——同 packed+CP 路径在 256K 序列（同配置 32K 成功）于 indexer top-k `cudaErrorIllegalAddress`（nemo2608 镜像 / Megatron-LM 0.19）；09-21 oncall Connor-XY 分配 hxbai triage、要求补 cuDNN Frontend/CUTLASS 版本与 cu_seqlens 元数据——illegal-address 本身尚不能区分 shape/packing 问题与上游 kernel 缺陷。https://github.com/NVIDIA/Megatron-LM/issues/7463
- **行为效应**：训练不崩、loss 轨迹看不出——生产形状实测「碰巧」贴住参考路径，但单测证明部分 packed+CP 配置下 kernel 输出错值：使用形状与中招集合的交集未刻画前，任何 cuDNN-DSA packed+CP run 都不可信。`flaky` 标记让 CI 永远报绿，故障对测试基建**隐形**——KER 错值与 OBS 掩体的复合：标记本身成了掩体。与 KER.01/02/09 同为训练 kernel 静默错值，差异在根因分层未知（cuDNN kernel vs Megatron 调用序 vs THD+CP 布局契约）且**实机硬件相关**（GB200/SM100）；测试标记掩盖面与 KER.08 的「断言被 cast 洗白」同族——验证基建自己失效。
- **发现来源**：2026-09-09 每日扫描

### KER.11 `precision_override_never_reaches_fused_grouped_mlp`

stage: `pretrain` · Cov: `NEW` · 置信度 `verified`

- **机制**：Megatron op-fuser 的 grouped-MLP（fused MoE）kernel 是 **FP8/NVFP4-only** 且从**全局 autocast state** 取量化 recipe——用户经 `--te-precision-config-file` 对特定层做的 precision override **永远到不了**这些 kernel：配置声称 bf16（或其它非默认 recipe）的 grouped expert 层照样按全局 autocast 量化。纯 TE 路径下**静默**（层被量化、无任何提示）；GTP 路径下 backward 交给 kernel 一个未量化权重 → `AttributeError: '_columnwise_data'` 才炸。修复 = `_with_fused_impl` 同时检查 fc1/fc2 的 `will_execute_quantized()`，任一 opt-out 即回退 unfused bf16 gmm。
- **来源**：Megatron-LM PR #7212（2026-09-10，closed merged；修复前 casting+mxfp8-gmm-fusion trace vs 修复后 bf16 unfused gmm trace 截图对照）。https://github.com/NVIDIA/Megatron-LM/pull/7212
- **行为效应**：静默面（纯 TE）：per-layer precision 配置在该层**完全失效**——以为在 bf16 训练的 expert 层实际跑 FP8/NVFP4，数值行为与配置意图背离，无日志无报错。与 NUM.09（混精 cast 扫过量化参数）同族：**精度配置的作用域与实际量化决策点脱节**；差异在这边根因是 fused kernel 的 recipe 读取通道（全局 autocast）不感知 per-layer override；与 CKPT.14（量化 extra_state 加载即弃）互补：那边加载侧丢精度状态、这边执行侧读不到精度配置。
- **发现来源**：2026-09-12 每日扫描

### KER.12 `cudnn_sdpa_backward_padded_tile_scores_zero`

stage: `shared` · Cov: `NEW` · 置信度 `verified`

- **机制**：cuDNN SDPA **backward** kernel（torch 2.13 默认 SDPA 后端）在三个条件同时满足时算错：① 显式 `attn_mask`（全 True 的 bias 路径也触发）、② `seq_len % 128 == 64`（tile 对齐断裂）、③ 存在 logsumexp < −88.7 = ln(FLT_MAX) 的 query 行（如 Qwen2.5「massive activation」头：logits ±1600 量级）。kernel 把最后一个 128 宽 key tile 的 **padding 半区打 0 分而不是 −inf**，`exp(0 − lse)` 在 fp32 溢出，`inf × 0 = NaN` 进 dQ。forward 完全正确（cuDNN 前向误差 = mem-efficient 级），只有 `grad_q` 坏；同 q/k/v/mask 走 EFFICIENT 或 MATH 后端干净。**有限垃圾窗口**：lse 高于阈值时 pad 半区的错误值是有限的、乘 0 后不可见——大量形状下 kernel 错而不 NaN、梯度静默带错。12 行 torch-only 复现：`L=704`（L%128==64）非有限梯度元素 45056，`L=640/768` 干净，`enable_cudnn_sdp(False)` 或 `is_causal=True` 干净。
- **来源**：trl #7181（DPO/KTO on Qwen2.5-0.5B bf16 SDPA：~60 步内 grad_norm 永久 NaN；维护者 qgallouedec 2026-09-11 定位为 cuDNN SDPA backward kernel bug、非 trl/transformers 问题，bisect 到 DPO micro-batch 的 layer 8 head 3 + 12 行 torch repro）。https://github.com/huggingface/trl/issues/7181
- **行为效应**：可见面 = grad_norm NaN（DPO/KTO 任何带 padding mask 的 batch，all-True mask 同样触发）；**静默面 = lse 未过阈的错值梯度**——上限取决于模型激活分布（massive activation 头越多越易过阈）。与 KER.10（DSA cuDNN packed-THD+CP 错值被 flaky 标记掩盖）同族：cuDNN attention 路径在非规范形状下错值 + 现有测试形状恰好免疫（这边是 `L%128==64`，那边是 packed+CP 组合）；与 KER.09（Triton 掩码块内/全局坐标混淆）同病：**tile padding 半区的边界处理**决定对错。
- **发现来源**：2026-09-13 每日扫描

### KER.13 `chunked_ce_backward_inplace_overwrites_logits`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：ms-swift 序列并行 `ChunkedCrossEntropyLoss`（`swift.sequence_parallel.utils` 导出，`CELOSS_PARALLEL_SIZE > 0` 时被 `per_token_loss_func_sp` 使用）为省显存把每个梯度 chunk **就地写进前向保存的 `logits`**，再把该输入张量原样作为梯度返回——违反 autograd「backward 返回独立梯度、保留 saved input」的契约。后果按调用形状分三层：① 叶子张量（raw `requires_grad=True` logits）触发 in-place `RuntimeError`（fail-fast 面）；② 非叶子输入（如 `raw.tanh()`）报 version-counter 错；带辅助项（`logits.square().mean()`）同样炸；③ **简单 linear head、单一 logits 消费者时静默跑通**——backward 把调用方的 logits 原位替换成梯度值，前向图若被复用（aux loss / retain_graph / 第二次 backward）即消费被污染的值。`torch.nn.functional.cross_entropy` 接受同样的输入。
- **来源**：ms-swift #10129（6 行最小复现 `ChunkedCrossEntropyLoss.apply(logits, labels, 3).mean().backward()`；提交者自带含梯度/参照测试的修复，代价是额外 logits-sized 梯度 buffer；macOS arm64 / PyTorch 2.14 / CPU+MPS，无需权重或分布式；closed 2026-09-15，无评论、无关联 commit）。https://github.com/modelscope/ms-swift/issues/10129
- **行为效应**：可见面 = 复用前向图或带 aux loss 时 crash；**静默面 = 单一消费者的 SP 训练路径 backward 后调用方 logits 已被梯度覆写**——任何后续依赖前向 logits 的计算（distill 温度缩放、软标签缓存）拿到的是梯度值；标准单 backward 下返回的梯度数值本身正确，loss 曲线无异常信号。与 KER.12（cuDNN SDPA backward 写坏 dQ）同为 backward 路径错误，差异在这边是 Python 侧自定义 autograd Function 的 saved-tensor 契约破坏（写穿调用方张量）、那边是闭源 kernel 内部错值；与 DATA.09（SWA fallback 无条件覆写 caller attention mask）同病：**库代码就地复用调用方张量**。
- **发现来源**：2026-09-15 每日扫描

### KER.14 `vocab_parallel_ce_label_smoothing_local_shard`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `vocab_parallel_cross_entropy` 的 label-smoothing 尾部按 **TP 本地词表分片**计算：`vocab_size = exp_logits.size(-1)` 取 per-rank 分片宽度（cross_entropy.py:158）、`log_probs.mean(dim=-1)` 只在本地 shard 上平均（L175）、backward 复用 `ctx.vocab_size`（L191）。base CE 的 max/target-logit/exp-sum 都正确 all-reduce，**只有 smoothing 尾是局部的**——局部均值是 Partial 值，却与已完整的 CE 基项直接相加、无归约。TP=2、smoothing 0.2 repro：两 rank loss `[1.4401896, 1.0951819]`（互不相等）、全局公式 `[0.8401898, 0.8618487]`，max 梯度误差 0.1333。修复 = 对 TP 组 sum 本地 log-prob 和再除全局词表（PR #5522 提出过精确该修复、被关未合）；#737（2026-09-17 有第三方独立同分析评论）。
- **来源**：Megatron-LM #7452（2026-09-17；finding 1，126 行 CPU/Gloo repro 附 spmd_types `Found types: [I, P]` 拒绝见证）；上游线索 #737 / PR #5522。https://github.com/NVIDIA/Megatron-LM/issues/7452 ；修复 PR（2026-09-22 补）：#7466 by gss10282025（draft，uses the global vocabulary size and global mean log-prob）——Connor-XY 09-21 在 #7452 建立 oncall tracking map 列 finding 1 已挂该 PR。https://github.com/NVIDIA/Megatron-LM/pull/7466 ；独立社区复现（2026-09-24 补）：Megatron-LM #7464——与 finding 1 无依赖的第三方报告，RTX 5090 + FP32、TP1 vs TP2 全梯度相对 L2 `0.0515`、三台主机复现，修复后 discrepancy 降至原值 ~1.7e-6 并对齐 float64 参照；再引 #737 / #5522 相同上游线索。https://github.com/NVIDIA/Megatron-LM/issues/7464
- **行为效应**：无 crash——**每个 TP rank 打出不同的 loss**、梯度带 0.1 量级误差；单卡 vs TP>1 的 run 从第一步起不可比，且 per-rank loss 差异常被当作「数值噪声/归约顺序」忽略。与 KER.13（fused CE 路径就地覆写）同文件域不同断点：那边破坏 autograd 契约、这边是**分布式归约域错误**（Partial 进非线性组合）；与 PAR.04（Ulysses 头数锁死）同型「TP 切分下 per-rank 常数被当全局常数」。
- **发现来源**：2026-09-18 每日扫描

### KER.15 `gathered_logits_fed_to_vocab_parallel_ce`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `GPTModel` 在 `parallel_output=False` 或 `runtime_gather_output=True` 时以 `gather_output` 建输出层（gpt_model.py:292/579/671）——logits 是**全词表、TP replicated**；但带 labels 的路径 `compute_language_model_loss` 无条件调 vocab-parallel CE（language_module.py:127），该 CE 把最后一维当作**本地词表分片**：TE fused 路径有 assert 挡住、**非 fused 路径没有**。TP=2、logits `[0,1,2,3]`、label 3 repro：observed loss 1.1333369 = 正确值 0.4401897 **+ log(2)**（replicated logits 被当 shard 求和/归一，系统性偏 log(TP)）；物理梯度 `[0.0160,0.0436,0.1184,0.3220]` vs 参照 `[0.0321,0.0871,0.2369,-0.3561]`——target 项丢失。两个触发项都是公开模型选项。修复 = assert sharded、gathered 时回退普通 CE 或拒绝组合。
- **来源**：Megatron-LM #7452（2026-09-17；finding 2，113 行 CPU/Gloo repro；未发现既有上游报告）。https://github.com/NVIDIA/Megatron-LM/issues/7452
- **行为效应**：无 crash——loss 恒偏 **+log(TP)**、梯度结构性错误（target 梯度项消失）；「能训、loss 略高」的形态使它极易被归因为超参/数据。与 KER.14 同函数族但断点相反：那边输入确是 shard、归约缺失；这边输入是 **replicated 却被当 shard**——分布式类型混淆的两个方向；与 DATA.06（tokenizer 本地 vocab 分片错位）症状面相邻（词表域错值）但机制在 CE kernel 入口契约。
- **发现来源**：2026-09-18 每日扫描

### KER.16 `triton_int32_offset_wraparound_oob_rw`

stage: `shared` · Cov: `var:triton_block_local_mask_oob_rope_write` · 置信度 `documented`

- **机制**：DeepSpeed Triton kernel 用 int32 算元素偏移：`swiglu_triton.py`（#8244 引入，fwd L38-40 / bwd L53-55）`offsets = pid * BLOCK_SIZE + arange`，`n_elements > 2^31` 时 `pid*2048` 在 `pid=1,048,576` 处溢出为**负数**，而掩码 `offsets < n_elements` **不拒绝负偏移**——mask 放行，kernel 对**张量起始之前**的内存 load+store（bf16 输入 1.25×2^31 元素时越界窗 4.0→3.0 GiB 在张量前方）。`autoep_fused_token_ops.py`（#8326，opt-in `combine_impl="fused_weighted_sum"` 路径）`token * out_stride`（stride=hidden）同型：tokens×hidden > 2^31 即溢出（expert-row 地址已 int64，只有 token 乘积中招）。触发阈值=元素数>2^31（bf16 即 >4 GiB 张量；MoE 大 hidden / 长序列可行）。
- **来源**：DeepSpeed #8590（2026-09-18；swiglu 是 GroupedExperts 默认路径；`program_id` 转 int64 修复后 2^31 以下 bitwise 相同、kernel 耗时不变）。https://github.com/deepspeedai/DeepSpeed/issues/8590 ；2026-09-19 补：修复 PR DeepSpeed #8591（`Use int64 program ids in the SwiGLU and AutoEP fused restore Triton kernels`）已于 2026-09-19 04:02 UTC merge（merge commit `bb5c23e0255c`），#8590 同刻 closed completed。
- **行为效应**：两侧可见面：CUDA illegal memory access（本报告 repro）或**负偏移落进合法映射区时静默读写错位内存**（KER.09 的 H100 观察同型：有分配 slack 则静默污染、无 slack 才 crash）——写入侧打坏**别的张量**、读取侧把垃圾当梯度。与 KER.09（Triton 掩码用块内坐标漏全局边界）同族「Triton 边界/寻址算术错」第 2 个实例，断点不同：那边 mask 域错（局部 vs 全局坐标）、这边**偏移量本身溢出回绕 + mask 不查负数**；与 NUM.07（autotune 非确定）无共享根因但同处「现有单测规模太小漏网」的测试盲区。
- **发现来源**：2026-09-19 每日扫描

---

# MOE

## MOE

### MOE.01 `router_entropy_collapse`

stage: `pretrain` · Cov: `NEW` · 置信度 `verified`

- **机制**：router logits 锐化，token 涌向少数专家，其余专家饿死。负载均衡损失关了、过小、或被错误的 aux 系数关掉时会发生。
- **来源**：本实验室 `stability_monitor`：`per_token_entropy` 从 ~4.0（64 专家近均匀）掉到 <1.0，早于 grad 爆炸。
- **行为效应**：总 loss 仍可降（活着的专家在过拟合）；长跑后能力塌。无 NaN。
- **发现来源**：2026-08-17 基础调研（本地实验）

### MOE.02 `expert_weight_id_misbind`

stage: `pretrain` · Cov: `var:expert_reshard_id_permutation` · 置信度 `documented`

- **机制**：运行时 EP 布局与 ckpt / HF 转换的专家序号约定不一致。router 选 expert k，算的是另一套权重。
- **来源**：同 CKPT.01。
- **行为效应**：训练在「错绑专家」上自洽地继续，评测像换了模型。
- **发现来源**：2026-08-17 基础调研

### MOE.03 `train_infer_router_desync`

stage: `rl/pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：训练 mini-step 重算与 rollout 的 router 决策（top-k、jitter）不一致。R3 的修法是回放推理期路由。
- **来源**：Ma et al., "Stabilizing MoE reinforcement learning by aligning training and inference routers", arXiv:2510.11370。https://arxiv.org/abs/2510.11370
- **行为效应**：同权重 MoE 在 RL 里等效于另一张稀疏图；reward 先升后塌。
- **发现来源**：2026-08-17 基础调研

### MOE.04 `dropless_capacity_silent_drop`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：宣称 dropless 的实现在容量/对齐约束下仍丢 token，或把溢出 token 静默改路由。统计上的「dropped=0」不可信。
- **来源**：Megatron / DeepSpeed MoE 容量因子文献；需对具体实现核 dropless 路径。种子期标 documented。
- **行为效应**：部分 token 的 MoE 输出近零或走错专家，loss 略噪，无报错。
- **发现来源**：2026-08-17 基础调研

### MOE.05 `expert_bias_buffer_dropped_on_replacement`

stage: `sft/rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：DeepSpeed AutoEP 替换 MoE 模块时迁移 `e_score_correction_bias`（DeepSeek-V3 aux-loss-free 路由的 top-k **选择项**偏置）要求 `isinstance(..., nn.Parameter)`——transformers 5.15.1 里 22 个声明该张量的模型族有 **18 个存成 `nn.Buffer`**（含全部 DeepSeek/GLM/MiniMax/Nemotron 变体），另有 4 族挂在 `gate` 以外位置（block 直挂 / `moe_statics` / `router`），两个独立缺陷叠加使 **20/22 族替换后静默丢偏置**，专家选择退回未训练的均衡态（gating 权重本身不受影响，变的只是选哪些专家）。
- **来源**：DeepSpeed #8358（transformers 全模型族普查表 + meta-device 无需真实权重的最小检查脚本：DeepSeek-V3.2 打印 `Tensor -> AutoEP would DROP it`；8×H200 复现；另注 `supports_expert_bias=False` 的 preset 意味着训练中也不更新该偏置）；修复 #8369（parser 记录 bias owner 实际路径，替换时按原样保留 Parameter/buffer 及持久性，`gate`/block/`router`/`gate.moe_statics` 位置全覆盖；真实 Transformers 5.15.1 `DeepseekV3MoE` 验证路由选择一致、单卡全 MoE forward 最大绝对差 0.0）。https://github.com/deepspeedai/DeepSpeed/issues/8358
- **行为效应**：无报错、无警告、loss 照降——路由偏置归零后专家选择系统性错位，行为像「换了一批专家」（MOE.02 症状），但根因是**模块替换时的张量类型/位置判定**而非 id 排列错位。任何 DeepSeek-V3 系 checkpoint + ZeRO-3 + AutoEP 即中招。
- **发现来源**：2026-08-30 每日扫描

### MOE.06 `expert_bias_hook_implicit_semantics_shift`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：torchtitan #3666 起 Qwen3 registry 无条件注册 aux-loss-free 负载均衡 hook（`post_optimizer_build_fn=register_moe_load_balancing_hook`），optimizer pre-hook 每步用累计 `tokens_per_expert_E` 更新持久 buffer `expert_bias_E` 并清零计数。行为变更完全隐式：`load_balance_coeff` 默认 1e-3 照旧、无独立开关、ckpt/配置元数据不记录更新模式——同 ckpt + 同数据 + 同 seed + 确定性执行，升级前后从 step 2 起 loss 静默分叉（Qwen3-235B-A22B 1000 步 MAE 0.042、最大差 0.129@step 955；grad_norm 首差 step 19、最大差 5.31；Qwen3-30B-A3B 复现 step 2 首差），pre-#3666 run 因此不可复现且无任何警告。消融闭合因果边界：两侧关 #3666 后残差缩到 MAE 0.012；#3666+#3386 双关时 1000/1000 步 loss/grad_norm/batch fingerprint 全部 bitwise 全等。
- **来源**：torchtitan #4394（235B/30B 双模型受控对比 + 四组消融 + freeze/迁移契约诉求，明确不要求 revert）。https://github.com/pytorch/torchtitan/issues/4394
- **行为效应**：无 crash、无 NaN、无日志差；路由选择所依赖的训练态被 registry 默认值隐式换掉，升级即静默换一条训练轨迹，且 ckpt 里非零 `expert_bias_E` 的 freeze 语义未定义。与 KER.05（版本漂移致发散、机制未定位）的区别：这边因果已闭合到单一 hook 注册，缺的是显式契约与迁移可见性；与 MOE.05 的区别：那边是模块替换时丢偏置张量，这边是偏置**更新语义**被隐式改变。
- **发现来源**：2026-09-02 每日扫描

### MOE.07 `quantized_expert_layout_republish_breaks_sync`

stage: `sft/rl` · Cov: `var:expert_reshard_id_permutation` · 置信度 `documented`

- **机制**：verl DeepSeek-V4 RL 栈里 Megatron 桥接的 routed experts 量化（MXFP4 ckpt → 训练侧 bf16）要求 rollout 端保持 ckpt 自带 packed 布局，但 SGLang `process_weights_after_loading` 在模型加载时跑一次 CUTLASS interleave、把四个专家参数 republish 为**全新 Parameter 对象**——`set_weight_attrs` 挂的 `weight_loader` 等属性全丢，后续每次训练→rollout 权重同步在 `deepseek_v4.py` 撞 AttributeError（fail-fast 面）。真正的静默面在**量化布局契约**：训练侧从 ckpt 直接训 packed MXFP4 专家、rollout 侧在 interleave 前后两种布局间同步，双方若任一侧静默容忍布局差异（如 sentinel 协议回退），权重被按错误 interleave 解释。修复方向 = 自定义 weight loader 在每次 sync 前 stage 恢复 ckpt 布局 buffer（interleave 是置换、字节守恒，可原地 reinterpret）。
- **来源**：verl PR #7747（H200/SM90 FlashInfer CUTLASS W4A16 验证；SGLang 侧 vLLM 对应实现 `vllm_fp4_utils.py` 已存在）。https://github.com/verl-project/verl/pull/7747
- **行为效应**：fail-fast 面（AttributeError）之外，布局错配若被容忍则 rollout 采样自**错误解释的专家权重**——与 RL-RO.09（fused expert view 冻结在捕获时刻）同族：训练/rollout 两侧对同一份 MoE 权重的视图或布局语义不一致；与 CKPT.09/NUM.09 同链（量化表示在存取/初始化路径被破坏）。此处为 draft PR、机制以布局契约记录在案。
- **发现来源**：2026-09-05 每日扫描

### MOE.08 `hash_router_aux_loss_optimizes_undispatched_route`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：hash-routed MoE 层的 token dispatch 按 `tid2eid` 哈希表执行；但 aux load-balancing loss 开启时，router 侧从 learned logits 另算一份 top-k routing map 进 aux loss——**优化的是一份从未被 dispatch 执行的路由**。两个路由图（执行 vs 优化）可以任意分叉，aux loss 的梯度把 router 往与实际 token 分配无关的方向拉。修复 = hash MoE 层跳过 aux loss。
- **来源**：Megatron-LM PR #7079（dev 分支，draft；机制为「learned routing map can differ from the hash routing map that was actually executed」——修复理由直接给出机制）。https://github.com/NVIDIA/Megatron-LM/pull/7079
- **行为效应**：无 crash、无 NaN；aux loss 数值正常下降但负载均衡目标与真实 dispatch 脱钩——router logits 被训成「纸上均衡」，实际 token 分布不变。与 MOE.03（train/rollout 路由不一致）同族不同位：那边是**时间维**（训练重算 vs rollout 执行），这边是**同一步内**（loss 优化的路由 vs forward 执行的路由）。
- **发现来源**：2026-09-06 每日扫描

### MOE.09 `load_balance_hook_scans_wrong_container`

stage: `pretrain` · Cov: `var:expert_bias_hook_implicit_semantics_shift` · 置信度 `documented`

- **机制**：torchtitan MoE 负载均衡 optimizer hook 只扫 `model.layers`；DeepSeek MTP decoder 块在 FSDP setup 后保留在 `model.mtp_layers`——MTP MoE 层的 `expert_bias_E`/`tokens_per_expert_E` **从不上报、从不更新**：bias 停在初值、本地计数器不归约不消费。修复 = hook 同时扫 `layers` 与 `mtp_layers`（ModuleDict/ModuleList 通吃），同一层迭代器保证发现/聚合/更新一致。
- **来源**：torchtitan PR #4494（已合并；两 rank Gloo reproducer：修复前 MTP bias 不动、counter 停 [3,1]，修复后 bias [-0.3,0.3]、counter 归零；4×4090D `deepseek_v3_mtp_fsdp+ep+compile` 10/10 步）。https://github.com/pytorch/torchtitan/pull/4494
- **行为效应**：MTP 专家的 aux-loss-free 偏置冻结在陈旧值、跨 rank token 计数静默失真——主干预练正常、MTP 头的路由均衡静默失守。与 MOE.06（hook 隐式语义漂移）同根：负载均衡 hook 的作用域/语义契约脆弱；本条是**容器枚举漏掉一整组层**，机制上更接近 MOE.05 的「部分专家被更新机制遗漏」。
- **发现来源**：2026-09-06 每日扫描

### MOE.10 `colocate_reload_swallows_post_load_processing`

stage: `rl` · Cov: `var:quantized_expert_layout_republish_breaks_sync` · 置信度 `documented`

- **机制**：ms-swift `finish_vllm_weight_reload` 用裸 `except Exception: return` 包住 vLLM `process_weights_after_loading`——**吞掉一切失败**。Megatron colocate 路径这不是理论防御：Megatron `get_current_device()` 返回 **CUDA ordinal (int)**，而 vLLM 加载上下文要 `torch.device` 并读 `.type`——int 直接令处理在 try 块内抛异常，宽 handler 把它藏掉，于是**权重 packing、MoE fusion、量化 prep 在 colocate 路径从未真正跑过**，rollout 引擎带着未处理权重采样。修复 = 归一化 `target_device`（int/str → `torch.device`）+ 加 `strict` 旗标，Megatron colocate caller 传 `strict=True` 让真失败炸出来，其他 caller 保持非严格。
- **来源**：ms-swift #10098、#10099；修复 PR #10113（与 CKPT.23 同 PR；colocate caller strict 化）。https://github.com/modelscope/ms-swift/issues/10098
- **行为效应**：无 crash、无日志——训练侧照常推进，rollout 引擎每次 weight reload 后都在**未 pack/未 fusion/未量化 prep 的权重**上采样，生成分布静默偏离训练策略。与 MOE.07（`process_weights_after_loading` republish 丢 `weight_loader` 属性、破坏 MXFP4 布局契约）**同函数不同断点**：那边是处理本身执行了但破坏下游契约、这边是处理**根本没执行**且被 except 掩盖；「宽 except 吞加载失败」的观测面与 OBS.04/OBS.09（校验/读取异常被吞后返回 success）同型——修复的共同方向是 strict 化。
- **发现来源**：2026-09-13 每日扫描

### MOE.11 `global_aux_loss_denominator_uses_current_microbatch`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron-Core `--global-aux-loss` 开启时，router aux loss 的 expert 计数在微批间**累积**（`global_tokens_per_expert` 跨 microbatch 累加），但归一化分母是 `ga_steps × 当前 microbatch 的有效 token 数`——把「历史每个 microbatch 都有当前 token 数」当作假设。各 microbatch 有效 token 数不同（含物理等长 padding 下）时分母即错：6+2 token 两微批后分子覆盖 8 token、分母却按 `2×2=4` 计，专家频率估计被系统性放大。修复 = 伴随累积计数维护累积 token 数（同 scope all-reduce），或等价换算 `累积计数 × 当前 token 数 / 累积 token 数` 传入 loss helper。
- **来源**：Megatron-LM #7213（open；core_v0.18.2 router.py 定位 + TopKRouter 最小复现：balanced `[4,4,4]` vs uneven `[6,2,4]` 同 12 token、router 梯度相对 L2 差 **0.2500001**，修复后 2.25e-7、FP64 参考 1.33e-15；修复 PR #7214 open 未合并）；#7197（同文本 issue，closed 标 community-request、重命名并入 #7213 跟踪）。https://github.com/NVIDIA/Megatron-LM/issues/7213
- **行为效应**：无 crash、无 NaN——aux loss 数值正常、可下降，但**负载均衡压力随 microbatch 有效 token 分布的方差系统性漂移**：router 梯度带错值，专家均衡目标被过/欠施加；物理等长 padding 完全掩盖触发条件。与 MOE.07（计数经 BF16 buffer 舍入）同为「累积 expert 计数的数值语义错误」但断点不同：那边是**存储精度**（float buffer 转 BF16）、这边是**归一化分母的时序语义**（累积分子配瞬时分母）；「梯度对 microbatch 划分方式敏感」的表现面与 LOSS.09（G 缩放漂移）同型的隐式超参依赖。
- **发现来源**：2026-09-14 每日扫描

### MOE.12 `router_aux_loss_inplace_reduce_overwrites_token_count`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `get_tokens_per_expert_and_token_count`（moe_utils.py:269）把 `local_tokens_per_expert` 传给 `reduce_from_tensor_model_parallel_region`——`tensor_parallel/mappings.py` 的 `_reduce` 对 contiguous 输入**就地归约**（代码注释自认）。带 `padding_mask` 时下游（L283）又从这张**已被覆写为全局和**的张量推导 `local_num_tokens`：两个返回计数都成了全局值。`router.py:452/509/555` 把这个「local」计数当 `valid_token_count` 传进 aux loss——**aux loss 梯度被额外乘上 TP/CP 组大小**。TP=2、8 token 中 5 有效 repro：两 rank 都得 `local_num_tokens=5`，router 梯度范数 2.2258655 vs 正确 1.1129328（**恰好 TP× 过大**）。触发只需 MoE aux loss + padding_mask（数据调度器的 sequence packing 即产生）+ per-token loss + TP/CP>1——任何非空 mask 都选中该分支。2026-09 的 `/`→`//` 改动（#7081）不影响此缺陷。修复 = reduce 前 clone、或先算 local 计数再调集合通信。
- **来源**：Megatron-LM #7452（2026-09-17；finding 10；#6111 是同路径不同 crash 已关）。https://github.com/NVIDIA/Megatron-LM/issues/7452 ；#7452 finding 10 独立确认 + 修复 PR（2026-09-18 补）：0z5a 在当前 main `d564dd01d` 上真实 NCCL 2-rank/EP=2/topk∈{1,2,4} 复现——rank0 3 有效行 / rank1 1 行，修复前两 rank「local」计数都得 **4**（= 组全局值）、修复后 3/1；`all_reduce` 作用于返回 buffer 会原地改变它，即别名污染的直接证据；隔离实验（routing map/scores/aux 公式/进程组全同、只改报告计数）下 aux 梯度随报告值**精确线性**缩放：local/total=7/10 → 1.428571×、3/10 → 3.333333×（恰为 `global/local`）；`//topk` floor（#7183）是独立改动、别名污染早于它存在；修复 PR #7481 在集合通信前把 rank-local 标量算进独立存储，base 5 failed → PR 后 11 passed。https://github.com/NVIDIA/Megatron-LM/pull/7481 ；2026-09-22 补：Connor-XY 09-21 在 #7481 请 zhongbozhu 评审「reduce 前的 local token-count 捕获及其在 router aux loss 中的使用」（邻近其 #7183 dtype 修复）并检查新增分布式回归 fixtures——PR 仍 open 未 merge。
- **行为效应**：无 crash、无 NaN——router/aux loss 梯度**恒放大于正确值 TP× 或 CP×**（取决于归约组），负载均衡压力随并行度漂移：同模型 TP=1 与 TP=2 的 run 行为不可比，专家均衡目标被过量施加。与 MOE.11（分母用当前 microbatch）同打击 aux loss 归一化语义、不同断点：那边分母时序错、这边**上游张量被集合通信就地污染后复用**；「in-place collective 污染调用方张量」与 KER.13（backward 就地覆写 logits）/DATA.09（覆写 caller mask）同一病根——**库代码就地复用调用方缓冲**；与 PAR.11（GTP 梯度 ×GTP 放大）症状面同为「梯度恒乘组大小」。
- **发现来源**：2026-09-18 每日扫描

### MOE.13 `overdispersed_routing_importance_signal_collapse`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：**激进负载均衡 → over-dispersed routing** 的静默退化证据：训练期 aux loss 把 token 分布推向近均匀时，router 概率作为专家重要性信号**塌缩**（近均匀分布下专家间区分度消失）；对 gpt-oss-20B 做专家剪枝：**最低 perplexity 的剪枝配置给出最差数学推理**、最高 perplexity 配置反而保住能力——**perplexity 与下游能力的关系在 over-dispersion 下反转**，标准路由（Mixtral-8x7B-Instruct）无此现象（ppl 与精度同降）。
- **来源**："When Load-Balancing Goes Too Far: Expert Pruning in Over-Dispersed Mixture-of-Experts Models", arXiv:2609.04453（2026-09-03，本轮窗口检出）。https://arxiv.org/abs/2609.04453
- **行为效应**：训练侧完全正常（这正是设计目标达成）——伤害在**训练目标与模型能力的错位**：负载均衡压力过大时学到的「均匀使用专家」使下游任何依赖 router 概率做重要性判断的路径（剪枝/合并/蒸馏专家选择）拿到无区分度信号，且 ppl 掩盖退化方向。与 MOE.01（均衡失守 → 熵塌缩 → 专家死）是同一「负载均衡压力 × 路由分布」轴的**反端**：那边均衡不足路由锐化、这边均衡过强路由过散，两端都毁能力而 loss 曲线都「正常」；与 LOSS 家族的差异：无任何步级数值症状，只有训后能力/可剪枝性可见——**长跑静默退化的 MoE 特异形态**；与 RL-KL.01「探针贴零不等于健康」同型观测陷阱（ppl 贴正常≠能力在）。
- **发现来源**：2026-09-19 每日扫描

### MOE.14 `aux_loss_coef_noop_without_router_logit_flag`

stage: `sft` · Cov: `var:dead_or_shadowed_arg_silent_noop` · 置信度 `documented`

- **机制**：trl `SFTTrainer` 用 `getattr(text_config, "output_router_logits", None) is not None` 推断是否 MoE 并启用 aux loss——但该字段是**可选声明**：transformers main 上 64 个 MoE config 有 30 个不声明（DeepSeek-V2/V3/V3.2、GLM4-MoE、GLM4V-MoE、Qwen3-VL-MoE、Mistral4、dots1、Kimi-Linear、Step3p7 全部在内），对这些架构 `router_aux_loss_coef` 是**静默 no-op**：无 aux loss 项、无警告。同库 `compute_flops_per_token` 已用 `num_experts_per_tok` 作权威 MoE 标记，检测路径间自身不一致。
- **来源**：trl #7222（2026-09-16；双 tiny 模型 repro：Qwen3Moe 记 aux_loss=True、DeepseekV3 False；config 声明面清点 34 declare / 30 not）。https://github.com/huggingface/trl/issues/7222
- **行为效应**：无 crash、无 warning——近半 MoE 架构的 SFT run 负载均衡项从未生效，专家利用分布不受约束地漂移，训练 loss 曲线完全正常；与 SFT.07（LoRA target_modules 静默欠匹配）同库同型「配置声明了、检测谓词漏一类目标」，差异在这边打击 aux loss（路由层）而非 PEFT 挂载面；「is_moe 推断依赖可选 config 字段」与 CKPT.07 死配置族共享病根但方向相反：CKPT.07 是 key 无人消费、这边是**消费了但布尔谓词对一半架构恒 False**；维护者 qgallouedec 已表态内部处理（不收外部 PR），拟用 `num_experts` 类权威标记 + 未应用时告警。
- **发现来源**：2026-09-23 每日扫描

---

# LOSS · 发散与长跑退化

## LOSS

### LOSS.01 `unscaled_embedding_grad_spike`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：子层谱范数上界大、残差捷径相对小，梯度范数会突然跳。Takase et al. 证明需要「小子层 + 大捷径」：用标准 LLM 初始化压子层，并用 embedding scale / embed-LN 让 embedding 标准差近 1。
- **来源**：Takase et al., "Spike No More", arXiv:2312.16903。https://arxiv.org/abs/2312.16903
- **行为效应**：loss 尖峰，严重时毁掉整段预训练。尖峰前常有 grad norm 先跳。
- **发现来源**：2026-08-17 基础调研

### LOSS.02 `attention_logit_growth_without_qk_norm`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：没有 QK-norm / logit softcap 时，attention logits 随深度/长度增长，softmax 变单峰，反传尖峰。NoPE 等消融上尤其明显。
- **来源**：QK-Norm / PaLM-scale 训练稳定性实践；NoPE+QK-norm 不稳定对照。
- **行为效应**：周期性 loss spike，有时能恢复，有时进入重复/乱码 attractor。
- **发现来源**：2026-08-17 基础调研

### LOSS.03 `post_ln_vanishing_then_explode`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Post-LN 残差在深网上梯度消失，一加大 LR 就爆炸。Pre-LN + 上述稳定化是现行默认；用 Post-LN 权重按 Pre-LN 图训（或反过来）是配置层故障。
- **来源**：Spike No More 附录对 Post-LN 的对照；交叉：推理编目 pre-norm/post-norm 配错。
- **行为效应**：要么训不动，要么一个 spike 之后不可恢复。
- **发现来源**：2026-08-17 基础调研

### LOSS.04 `long_run_quality_drop_without_nan`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：loss 仍在缓降或震荡在可接受带，下游能力已经掉（数据污染、学习率过期、MoE 塌了、SDC 累积）。单看标量 loss 会判「健康」。
- **来源**：本实验室 monitor 的存在理由：谱/熵/鞅时钟在 loss 之前动。ByteDance 2509.16293 把「轨迹偏离、无显式失败」列为 implicit failure。
- **行为效应**：ckpt 能 resume，评测却一代不如一代。
- **发现来源**：2026-08-17 基础调研

### LOSS.05 `weight_norm_criticality_spike`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：归一化带来的 scale-invariant 权重 × weight decay 持续收缩参数范数 → 接近临界边界时 loss landscape sharpness 快速抬升 → 突发 loss spike。与学习率临界性（EoS）正交的第二条临界轴：weight decay 系数再大一点就跨界。
- **来源**："Weight-norm Criticality: A Mechanism for Loss Spikes Induced by the Normalization and Weight Decay", arXiv:2607.21005。https://arxiv.org/abs/2607.21005
- **行为效应**：临界前曲线完全正常；spike 可恢复但反复出现。论文给出可检验预测（scale-invariant 权重范数轨迹），是比「超参没调好」低一层的机制级来源。
- **发现来源**：2026-08-19 每日扫描

### LOSS.06 `router_backbone_update_coupling_spike`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：MoE 训练中 router（gating 网络）与 backbone 同步更新构成正反馈：论文原表述「spike 的出现与 MoE 层 outlier 一致相关，且 routing 机制本身加剧这些 outlier 的涌现」——outlier 放大路由扰动、路由扰动再催生 outlier，恶性循环以 loss spike 爆发。DeepSeek-V4 的缓解是 **Anticipatory Routing**：step t 的特征计算用当前参数 θ_t，路由 index 用历史参数 θ_{t−Δt} 计算并缓存（数据在 t−Δt 提前取、路由 index 预计算后回放），从而解开两条网络的同步更新；生产部署为 spike 自动检测 → 短 rollback → 仅 spike 期间启用，额外 wall-clock 开销约 20%，据称不损模型性能。
- **来源**：DeepSeek-V4 technical report §4.2.3 "Mitigating Training Instability — Anticipatory Routing"，arXiv:2606.19348（1.6T / 284B 双模型 32T token 预训练的一手生产经验）；torchtitan RFC #4349 提议把该技术引入 torchtitan，动机段复述同一机制（spike 常需 rollback，否则 derail run 或被迫保守超参）。https://arxiv.org/abs/2606.19348 · https://github.com/pytorch/torchtitan/issues/4349 ；RFC #4495（2026-09-06 补：具体实现草案——`anticipatory_cache`/`anticipatory_indices` 瞬态运行时属性 + prefetch→replay→forward-only 编排，显式禁用与 PP/CUDA graph 的组合；另见 #4494 揭示的 hook 容器盲区 MOE.09——同一 RFC 落地时的作用域陷阱已在真实代码出现）。https://github.com/pytorch/torchtitan/pull/4495
- **行为效应**：反复 loss spike、常需 rollback，否则 derail run 或收敛到保守超参；无 crash。与 MOE.01（router 熵塌）区分：这里路由**功能正常**，病根是路由与主干**更新耦合**放大 MoE outlier。
- **发现来源**：2026-09-01 每日扫描

### LOSS.07 `zero_init_final_norm_blanks_first_backward`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：torchtitan Muse Glimmer 的 final output norm 用 `gain_center=0.0` 且继承 `_gain_norm` 的**零初始化 weight**——有效 scale = 0+0 = 零：final hidden states 与 logits 全零，首个 backward 的 decoder/lm_head 梯度也全零。内部 norm 的零初始化是有意的 gain-center 语义，final norm 复用同一初始化路径则是错的：输出侧需要单位 scale。修复 = 仅 final norm weight 初始化为 1、参数化不变。
- **来源**：torchtitan PR #4489（已合并；standalone RMSNorm forward/backward 检查：修复前输出与梯度全零、修复后非零）。https://github.com/pytorch/torchtitan/pull/4489
- **行为效应**：预训练从「第一步对所有下游参数零梯度」起步——loss 平但不学（首步优化器更新为零），warmup 期损失曲线看似正常爬升、实际起步期已被浪费；更糟的路径是与先验初始化假设耦合的隐式质量损失。无 crash、无 NaN。与 LOSS.04（长跑静默退化）不同：本条是**初始化即静默错**，收敛掩盖起步缺陷；与 CKPT.03（随机初始化）对照：零初始化比随机更隐蔽——数值范围看起来完全正常。
- **发现来源**：2026-09-06 每日扫描

### LOSS.08 `gkd_mean_of_means_token_reweighting`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：ms-swift Megatron 后端 GKD（full-vocab Reverse-KL 蒸馏，`rlhf_type='gkd'`）的损失归约是「均值的均值」而非全局 token 均值：`gkd_loss` 返回未归一 `(total_loss, num_valid)`；`loss_func` 先 `cp_reduce(total_loss, num_valid)` 在**每个 micro-batch 内部**除以自己的有效 token 数（per-mb token mean）；Megatron scheduler 再对 micro-batch 等权平均（`loss /= num_microbatches`），且 RLHF 路径 `calculate_per_token_loss` 默认 False、`finalize_model_grads` 收到 `num_tokens=None`。最终 `L = (1/M)·Σ_mb[Σ_t KL_t / N_mb]`——每个 token 的梯度权 `1/(M·N_mb)` 取决于**所在 micro-batch 的长度**。`micro_batch_size=1` 时退化为序列均值：10-token 与 1000-token 回复分属不同 mb 时 per-token 权差 100×，训练被短回复支配。同 seed 切全局 token 归一后 loss 0.499→0.396、grad norm 42.77→25.21（比值 ~0.79 vs ~0.59 不一致，证实是 token 重加权而非常数缩放）。
- **来源**：ms-swift #10076（ms-swift main `426f897`，teacher Qwen3.5-27B → student Qwen3-4B，TP=4，解析推导 + smoke A/B）。https://github.com/modelscope/ms-swift/issues/10076
- **行为效应**：无 crash、无 NaN；蒸馏 loss 正常下降，但隐式目标从「每 token 等权」变成「短回复 token 加权」——蒸馏出的分布偏短答行为，长度分布漂移不可见。与 RL-ADV.03（token 长度偏置）同症状面；与 OBS.11（trl loss-path 指标按 micro-batch 平摊）同机制不同后果：那边错指标、这边错梯度；修复 = GKD 强制 `calculate_per_token_loss=True` 走 Megatron token-count-aware 终化路径。
- **发现来源**：2026-09-10 每日扫描

### LOSS.09 `dapo_family_gas_double_scaling`

stage: `rl` · Cov: `var:gkd_mean_of_means_token_reweighting` · 置信度 `documented`

- **机制**：ms-swift GRPO 的 `loss_type='dapo'`（`cispo`/`fipo` 同族）loss 已按**整个梯度累积窗口**的 completion token 数归一：`normalizer = num_items_in_batch / num_processes`，其中 `num_items_in_batch` 由 `ga_batch_encoded_inputs` 汇总、覆盖所有 micro-batch × 所有进程——单个 micro-batch 只是全局 token 均值的一部分。但 `transformers.Trainer.training_step` 在 backward 前还会**再除一次** `current_gradient_accumulation_steps`（`Accelerator` 的 `num_steps` 被强制为 1，无二次缩放对冲），于是 loss 与梯度都被额外缩小 **G 倍**：期望梯度 g、`gradient_accumulation_steps=4` 时实际只有 g/4。`grpo`/`sapo`/`bnpo`/`dr_grpo` 按 micro-batch 取均值，不受影响。修复 PR #10116：补偿 `current_gradient_accumulation_steps`，附单测 21 passed。
- **来源**：ms-swift #10117（解析推导 + 最小验证 `(loss/4).backward()` 复现 Trainer 行为；`--loss_type dapo --gradient_accumulation_steps 4` 实测更新量为 G=1 的 1/4）+ 修复 PR #10116。https://github.com/modelscope/ms-swift/issues/10117
- **行为效应**：无 crash、无 NaN——**有效学习率随 `gradient_accumulation_steps` 漂移**（G 越大更新越弱），日志 loss 也被除以 G；同配置改 G 的两次 run 收敛速度不可比，且无任何信号指示更新量错。与 OPT.10（ZeRO-3 Muon 每 micro-batch 跑一次 NS——momentum 变 βⁿ）同病：**梯度累积旋钮静默改变有效优化量**，那边放大、这边缩小；归一化域错配（全局 token 已归一再被 per-mb 逻辑二次缩放）与 LOSS.08（mean-of-means）互为镜像——那边漏归、这边重除。
- **发现来源**：2026-09-13 每日扫描

### LOSS.10 `cp_local_loss_means_averaged`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron `forward_step_calc_loss`（schedules.py:295）在 `calculate_per_token_loss=False`（默认）时把每个 CP rank 的 masked loss 和**除以自己的有效 token 数**——CP-local mean；`pretrain_gpt.py` 返回 CP-local sums/counts，DDP 随后跨 DP×CP 平均梯度。有效 token 数在 CP 分片间不等（`--eod-mask-loss`、任意 loss masking、Wan 真实数据路径 pad 到 `2*CP` 再 mask padding、CP=4/8 recipe）时，实际算的是 `(L0/n0 + L1/n1)/2` 而非 `(L0+L1)/(n0+n1)`：**mean-of-means**。8 token、CP counts 4/1、4 参数线性预测器 repro：observed 梯度 `[0.8369,-0.0900,0.2948,-0.1063]` vs 参照 `[0.5110,-0.1440,0.0577,-0.1701]`——**梯度方向改变**（不只是尺度）。修复 = 像 per-token 路径那样先跨 CP sum 有效计数再除。per-token loss 开启时不受影响。
- **来源**：Megatron-LM #7452（2026-09-17；finding 9）。https://github.com/NVIDIA/Megatron-LM/issues/7452 ；修复 PR（2026-09-22 补）：#7513 by zupengwang（three-value loss callback 返回 CP-local sums/counts 时先跨 CP 汇总再归一）——挂 09-21 oncall tracking map。https://github.com/NVIDIA/Megatron-LM/pull/7513
- **行为效应**：无 crash、无 NaN——**梯度方向随 CP 分片的有效 token 分布畸变**：序列边界/eod 位置决定了哪个 CP 分片贡献多少权重，长跑下优化目标系统性偏离全局均值目标；普通 GPT + `--context-parallel-size 2 --eod-mask-loss` 即触发。与 LOSS.08（GKD mean-of-means 跨 micro-batch）/RL-ADV 家族的组内统计错值同病根「局部均值的均值 ≠ 全局均值」，这边打击**CP 轴**且是默认 loss 路径（per-token 关闭时）；与 KER.14（smoothing 尾局部化）同为「归约域在并行轴上缺一跳」，那边 TP 轴、这边 CP 轴。
- **发现来源**：2026-09-18 每日扫描

### LOSS.11 `mtp_per_token_loss_local_token_ratio`

stage: `pretrain` · Cov: `NEW` · 置信度 `documented`

- **机制**：Megatron MTP 路径（multi_token_prediction.py:1113 `original_num_tokens = loss_mask.sum()`、L1216 把每个 MTP loss 乘 `original_num_tokens / num_tokens_safe`）——**两个计数都是 DP/CP rank 本地的**。`finalize_model_grads` 后续的全局归一除以全局 main-token 计数，**解不开一个已形成的非线性本地比率**（比率先作用于 per-token loss、再进梯度）。MTP + `calculate_per_token_loss=True` + DP/CP 间有效 token 数不均（SFT 式 masking 即产生）时触发。repro：两 DP worker main/MTP 计数 4/3 与 2/1、未缩放 MTP 梯度和 6 与 10 → 实现梯度 4.6666665，单 worker 参照 4.0。修复 = 先在与梯度归一相同的 DP/CP 组上 all-reduce 两个计数、再构成比率。#4896 报过同型比率平均但只在 MTP loss **日志**（已关）；本条证明**梯度路径同病**；#3943/#1532 是更早的 MTP 缩放 bug（已修）。
- **来源**：Megatron-LM #7452（2026-09-17；finding 3）。https://github.com/NVIDIA/Megatron-LM/issues/7452
- **行为效应**：无 crash、无 NaN——MTP 梯度被本地 token 比率加权，**有效 MTP 损失权重随各 rank 的 main/深 token 分布漂移**；多卡 vs 单卡 run 不可比、MTP head 的训练强度系统性错位。与 LOSS.10（CP-local mean-of-means）同打击「局部比率/均值先于全局归一进入非线性组合」，这边是 **MTP 双计数比率**、那边是 CP 均值；与 MOE.11（累积分子配瞬时分母）同为「比率的两项取自不同归约域」；「日志路径已报过、梯度路径漏修」与 OBS 家族「可见指标正常、隐藏路径坏」互为镜像。
- **发现来源**：2026-09-18 每日扫描

---

# SFT

## SFT

### SFT.01 `chat_template_train_serve_mismatch`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：训练用模板 A（含/不含 generation prompt、系统角色、tool 头），上线用模板 B。模型在错误的控制 token 分布上被优化。
- **来源**：交叉：推理编目模板类。训练侧是根因。工具参数序列化实例（2026-08-30 补）：ms-swift #9968——Qwen 原生模板要求非字符串工具参数用 JSON 字面量（`true`/`null`），Swift formatter 只对 dict/list 走 JSON 序列化，bool/None 落回 Python repr（`True`/`None`）——训练语料里的工具调用格式与模板/Qwen 原生输出静默分叉，模型学到的参数格式部署侧解析不了。https://github.com/modelscope/ms-swift/pull/9968
- **行为效应**：SFT 验证 loss 好，对话/工具调用一上线就退化。
- **发现来源**：2026-08-17 基础调研

### SFT.02 `packed_next_sample_as_completion`

stage: `sft` · Cov: `var:packing_cross_document_attention` · 置信度 `documented`

- **机制**：packing 时 label 在文档边界不断开，模型把下一条样本的 token 当本条 completion 来学。
- **来源**：与 DATA.02 成对。HF TRL packing / 多种 SFT 框架的已知坑。
- **行为效应**：训练 CE 异常低；线上出现「回答下一题」。
- **发现来源**：2026-08-17 基础调研

### SFT.03 `assistant_mask_drops_last_turn`

stage: `sft` · Cov: `var:loss_mask_includes_prompt` · 置信度 `documented`

- **机制**：多轮 mask 逻辑漏了最后一轮 assistant，或把 tool 结果当 assistant 学。有效监督信号被砍掉或标错角色。
- **来源**：chat template + 多轮 SFT 的边界 bug 族。
- **行为效应**：多轮能力明显弱于单轮，loss 看不出来。
- **发现来源**：2026-08-17 基础调研

### SFT.04 `preference_pair_swapped_or_identical`

stage: `sft/rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：DPO/ORPO 对 chosen/rejected 列对调，或两条相同。梯度符号反了或为零，仍会「在训」。
- **来源**：偏好数据管道的经典静默错误。
- **行为效应**：reward margin 变负或钉在 0；模型风格往差的一边走。
- **发现来源**：2026-08-17 基础调研

### SFT.05 `left_pad_scoring_uses_arange_positions`

stage: `sft/rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：左 padding batch 里做 logprob 打分时只传 `input_ids`+`attention_mask`、不传 `position_ids`，模型侧 fallback 到 `torch.arange` → 每个 padded 行的位置被整体平移 pad 数。关键非对称：`generate()` 从 mask 推 position（`cumsum-1`），打分侧用 arange——**采样与打分在两套位置上分歧**，而不是一致地错。
- **来源**：trl #6800（gpt2 左 pad 13 行实测：max|Δlogp|=1.17e1、序列 logprob 差 -34.0；RoPE 类（Qwen2-0.5B）与绝对位置类 delta 恰为 0；PPO 路径正确构造 position_ids，OnlineDPO/XPO/Nash-MD 均漏）。https://github.com/huggingface/trl/issues/6800
- **行为效应**：对 RoPE 是 no-op（这也是它长期潜伏的原因）；对位置敏感实现（ALiBi、学习式/绝对 position embedding）整个 batch 的左 pad 行 logprob 全错、偏移量随 pad 宽度增长。无 crash、无 NaN。与 RL-RO.01/RL-RO.07 不同：权重与分布定义都一致，错的是**喂进模型的位置索引**。
- **发现来源**：2026-08-20 每日扫描

### SFT.06 `truncation_forges_identical_preference_pair`

stage: `sft/rl` · Cov: `var:preference_pair_swapped_or_identical` · 置信度 `documented`

- **机制**：OpenRLHF `RewardDataset` 对 chosen/rejected **各自独立**截断到 `max_length` 再把末位保留 token 强制覆写为 EOS。偏好差异位于截断边界之后、或恰在末位保留 token 上时，两条模型输入变得**完全相同**——数据集过滤只查 `prompt is None` 与 DPO 的 prompt 长度，变换后无任何校验，坏 pair 被静默保留并训练。这是 SFT.04「identical pair」的一个**生成机制**：不是管道列错，是合法 max_len 变换自己造出来的。
- **来源**：OpenRLHF #1311（`reward_dataset.py` 行号级根因；RM/DPO 公共 CLI 默认 `--data.max_len=512` 即可达；实测 Qwen3.5 复现）。https://github.com/OpenRLHF/OpenRLHF/issues/1311
- **行为效应**：RM/DPO 的 pair loss 恒 `ln(2)`、主偏好梯度精确为 0；混合 batch 里这些常数损失样本稀释梯度、扭曲 accuracy/step 口径。无 crash 无 NaN，loss 曲线只是「降得慢」。与 SFT.04 互补：SFT.04 收现象族，本条收**变换内生**的产生路径（默认 512 max_len 下长答案数据集必然渗入）。
- **发现来源**：2026-08-21 每日扫描

### SFT.07 `lora_target_modules_silent_partial_match`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：LoRA `target_modules` 本质是**模块名选择器**：列表里匹配不到任何模块的条目被静默跳过，无零匹配/近零匹配警告。TRL 官方 Nemotron 3 SFT 示例（`examples/scripts/sft_nemotron_3.py`）对 NemotronH 混合架构给出 Llama 式 MLP 名（`gate_proj`/`up_proj`/`down_proj`），而 Mamba-2 层实为 `in_proj`/`out_proj`、MoE 层另名——adapter 只挂上 56 层中的 4 层 attention（7.1%）。与 CKPT.06/CKPT.07 族（死参数/被覆盖）的边界：key 被 PEFT **正常消费**，配置无错位；错在**选择器值对架构静默欠匹配**且框架层缺「列出项未命中任何模块」护栏，用户复制官方示例即中招。
- **来源**：trl #6773（DPO LoRA 实测：attention-only target_modules，adapter 覆盖 4/56 层；提议 TRL 加 target_modules 未匹配运行时告警；关联 PEFT PR #3289 NemotronH 默认 target_modules）。https://github.com/huggingface/trl/issues/6773
- **行为效应**：DPO LoRA 25 步 loss 钉死 0.6931=ln(2)、模型行为零变化（与 SFT.04/SFT.06 同症状：有效偏好/监督信号趋零）；SFT 场景则表现为「在训但只训了 7%」——loss 会动但远弱于预期，曲线无法区分。全程无 crash、无 warning。
- **发现来源**：2026-08-24 每日扫描

### SFT.08 `dpo_len_norm_denominator_unscored_token`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl `DPOTrainer._compute_loss` 的 `ipo` 与 `sigmoid_norm` 分支里长度归一化**分子分母掩码错位**：token logprob 用 `completion_mask[..., 1:]` 移位掩码（首 token 无前驱状态、不可打分），序列得分却除以**未移位**的 `completion_mask.sum()`。当截断使序列首位恰好是 completion token（`truncation_mode="keep_end"` 截掉整个 prompt、或空 prompt）时，分母多计 1 个从未进入分子的 token。普通非空 prompt 首位是 prompt token（completion_mask=0），数值不变——故障被门控在 keep_end/空 prompt 路径上，常规短测全绿。
- **来源**：trl #6879（最小复现：completion_mask `[1,1,1]` vs scored `[1,1]`，ipo loss 实际 0.0278 / 正确值 0，梯度 -0.222 非零；作者备两行修复 + 参数化回归测试，同时点名 `sigmoid_norm` 同错）。https://github.com/huggingface/trl/issues/6879
- **行为效应**：受影响样本的 ipo/sigmoid_norm **loss 值与参数梯度**系统性偏移（不止指标），无 crash 无 NaN；含短/空 prompt 的数据集里逐样本稀释梯度、扭曲 DPO 指标口径。与 OBS.06（纯诊断面、loss 路径正确）互补：这条直接改 loss 与梯度本身。
- **发现来源**：2026-08-24 每日扫描

### SFT.09 `nondiverging_pair_prompt_split_off_by_one`

stage: `sft/rl` · Cov: `var:preference_pair_swapped_or_identical` · 置信度 `documented`

- **机制**：trl `extract_prompt`（DPO/KTO 偏好预处理公共路径）扫描 chosen/rejected 首个分歧点；两者**从不分歧**（其一为另一个的完整前缀、或完全相同）时循环无 break，Python 把 `idx` 留在最后一次迭代值 `min(len)-1` 而非 `min(len)`——最后一条公共消息被挤出 prompt、**逐字复制进 chosen 和 rejected 两侧**。
- **来源**：trl #6960（`for...else` 修复使穷尽循环落到真前缀长度；两条新回归测试在 main 上红、修复后绿；Cursor Bugbot 评级 Medium Risk："wrong splits could corrupt training data"）。https://github.com/huggingface/trl/pull/6960
- **行为效应**：偏好对 prompt 边界系统性错位：本该属于条件的末轮消息进了两个 completion，DPO 在错位的对比目标上训练，无 crash 无 NaN。与 SFT.06（截断内生造 identical pair）互补——同属 SFT.04 现象族的**变换内生**生成路径：这条收「前缀包含/相同」输入下 split 算术的 off-by-one。
- **发现来源**：2026-08-30 每日扫描

### SFT.10 `zero_diff_probe_slice_drops_system_prompt`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：verl `initialize_system_prompt` / `extract_system_prompt_and_generation` 用两次 probe render 的长度差切 system prompt 前缀：`token1[: -(len(token2) - len(token1))]`。模板对「1 轮 vs 2 轮相同 probe」**渲染不增长**（diff==0）时表达式变成 `token1[:-0]`，Python 语义是 `token1[:0]` = **空表**而非「全保留」——system prompt 被静默丢成 `[]`。
- **来源**：verl #7592（修 #6477；生产路径 `MultiTurnSFTDataset` 据此计算每个非首轮 turn 该剥的 leading tokens 数=0，**每一轮都被塞进重复的 system+首轮前缀**；Jinja2 假 tokenizer CPU 复现 red→green 2 failed→4 passed；此前修复尝试 #6487 被关闭未合，bug 在 main 两处调用点均在）。https://github.com/verl-project/verl/pull/7592 ；修复 PR #7756（2026-09-06 补，已合并）。https://github.com/verl-project/verl/pull/7756
- **行为效应**：多轮 SFT 样本从第二轮起 token 流结构损坏（前缀重复注入），训练无报错、loss 正常。与 SFT.01（训练/服务模板错位）不同：模板本身没错，错在**从模板渲染反推前缀长度**的探针算术在合法模板（只渲染首条消息类）上翻车——`x[:-0]` 陷阱是 Python 切片语义与「diff 可为 0」假设的组合。
- **发现来源**：2026-08-30 每日扫描

### SFT.11 `auto_processor_ignores_model_revision`

stage: `sft/rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl 各 trainer 以字符串 model id + `model_init_kwargs.revision` 加载时，模型按指定 revision 取权重，自动创建的 `processing_class` 却走 `AutoProcessor.from_pretrained(get_config_model_id(model.config))` **不带 revision**——从仓库默认 revision 拉 tokenizer/chat template。同一 trainer 内权重与模板来自不同 revision；同一模式存在于 stable DPO/GRPO/KTO/RLOO/SFT/Distillation trainer。
- **来源**：trl #6974（GRPOTrainer 等价代码 + 复现形状；修复 PR #6978 转发 revision 至全部 stable + experimental trainer（含 Reward），逐 trainer 加 `test_init_auto_processing_class_uses_model_revision` 回归）。https://github.com/huggingface/trl/issues/6974 ；verl 实例（2026-09-04 补）：PR #7658——student/teacher Hub id 不 pin revision 时随 `main` 漂移，家族（Gemma-4）在 `main` 上移动 tokenizer/`generation_config` 后两边加载不同 snapshot 训练即坏；修复 = `HFModelConfig`/teacher config 加 `revision` 并贯穿 tokenizer/processor/`AutoConfig`/`generation_config`/FSDP/vLLM/reward tokenizer。https://github.com/verl-project/verl/pull/7658
- **行为效应**：checkpoint revision 的 chat template 与默认 revision 不同时（模板修复后的 fork/发布 revision 常见），训练全程用错模板格式化对话——这是 SFT.01（训服模板错位）的**加载侧生成机制**：不是人配错，是「按 revision 取权重」与「按默认取模板」两条加载路径静默分叉。无任何警告；显式传 `processing_class` 才能绕开。
- **发现来源**：2026-08-30 每日扫描

### SFT.12 `trajectory_split_default_scale_reweights_turns`

stage: `sft/rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：ms-swift 多轮调度器把同一条对话轨迹**拆成多条训练样本**（每轮一条）时，默认 `loss_scale` 权重使每一轮的 assistant 内容都按全权计入 loss：同一轨迹的前几轮在多条样本中被**重复训练**、非末轮拿到与末轮相同的权重——loss 面向的不是「每条轨迹一次」而是「每轮一次、前段轮次被放大」。维护者确认拆分场景必须显式 `loss_scale=last_round`，「不然前面轮次可能会多次训练」；该约束无警告、默认值不跟随拆分模式，且该调度路径已被标记将移除。叠加面：拆分 + 序列并行不兼容（非整除维度直接 assert，fail-fast；`padding_free` 同不支持）。
- **来源**：ms-swift #9996（GRPO 多轮重载 run 拆轨迹，SP 下 `AssertionError: The dimension to split (1533) is not a multiple of world size (2)`；维护者 hjh0119 2026-08-28 确认「多轮的和sp还没兼容」「megatron 的 sp/cp 是支持的，实现不同。但是 megatron 多轮不支持拆分为多条的场景」「拆分多条的场景，使用 last_round」；日志含 deprecation 警告）。https://github.com/modelscope/ms-swift/issues/9996
- **行为效应**：静默面（不叠 SP 时）无报错：早期/中间轮次 token 的有效梯度权重按拆分轮数放大，训练静默偏向对话前段行为，无日志指出权重语义已从「每轨迹一次」变成「每轮一次」。与 SFT.03（mask 丢末轮）对偶：那边丢轮次、这边重复计权；与 DATA.03（prompt 计入 loss）同族——样本内容没错，错在**哪些 token 以多大权重进 loss** 的语义随调度器行为静默分叉。
- **发现来源**：2026-09-03 每日扫描

### SFT.13 `per_message_tokenization_drops_think_block`

stage: `sft` · Cov: `var:chat_template_train_serve_mismatch` · 置信度 `documented`

- **机制**：verl `MultiTurnSFTDataset._process_single_message()` 对**每条消息单独**调 `apply_chat_template([message])` 再拼接 input_ids——但 Qwen3 系模板对 `reasoning_content` 的渲染依赖完整前缀上下文：对含 reasoning + tool_call 的 assistant 轮，逐消息渲染丢掉 `<think>reasoning</think>` 块，全对话渲染则保留，即 `apply_chat_template([msg]) != apply_chat_template(conv[:i+1])`。训练序列里 reasoning/final-answer 边界因此错位，loss mask 按错位序列监督。自带的 `sanity_check()` 本会 `AssertionError` 拦下该不一致，但 `ignore_input_ids_mismatch=True` 一键压掉检查后训练照常进行。
- **来源**：verl #7807（Qwen3 多轮 SFT，`enable_thinking=True`，user→assistant(reasoning+tool_call)→tool_response×2→final；逐消息 vs 全对话模板渲染 diff 实证）。https://github.com/verl-project/verl/issues/7807 ；verl PR #7844（2026-09-12 补：SGLang 侧同族 token 恒等性破坏——多模态 hash pad（radix-cache 身份用的 OOV id）在 0.5.14 的 prompt-logprob 元数据里被 clip 成 0，蒸馏 teacher 在视觉位看到错 token 而 logprob 值仍按位对齐；修复 = tokenize 后在 request 上记录视觉 offset、提取时恢复 canonical id，一致位不 alias 保持 fail-closed）。https://github.com/verl-project/verl/pull/7844
- **行为效应**：静默面无 crash：训练正常跑完，训后模型把最终答案写进 reasoning 通道（vLLM 返回 `content=None, reasoning="…导航已开始…"`），推理侧表现为「答案藏进思考、正式回复为空」。与 SFT.01（train/serve 模板不一致）同族——都在 tokenization 层把训练分布推离模板语义；与 SFT.12（拆分重复计权）互补：那边权重错、这边内容错；「压掉 sanity check 继续训」的逃逸阀与 RL-KL.04 修复方向（fail-fast 化）正相反，构成检查可被一键绕过的实例。
- **发现来源**：2026-09-10 每日扫描

### SFT.14 `padding_free_ld_prefix_uses_physical_segment_length`

stage: `sft` · Cov: `var:dpo_len_norm_denominator_unscored_token` · 置信度 `verified`

- **机制**：ms-swift padding-free LD-DPO 用**物理 packed 段长**（`logits_to_keep` 产出的段切分）确定共享 completion 前缀长度——prompt / masked 位可以挤进该前缀、改变有效 response token 的权重。实测 4 chosen + 2 rejected response token、`logits_to_keep` 产出段长 `[4, 3]`：`ld_alpha=0` 时 chosen 分数计入 **3 个 response token 而非 2 个**（多出的一个是 prompt 位）；不带 `logits_to_keep` 同样错位。修复 = 共享长度改用既有 **effective completion token 计数**、reduce 前用 `loss_mask` 过滤 logprobs，使 padding-free 与 padded 路径权重对齐。
- **来源**：ms-swift PR #10072（2026-09-09 closed；新增 padded/padding-free × logits_to_keep 开关的参数化回归 + 修正后的既有集成测试**修复前全挂**；修后 `test_rlhf_loss.py` 12 passed / 32 subtests；离线 CPU FP32 参照 loss/全参数梯度/单步 SGD 更新对照 + H800 FP32/BF16 trainer 路径核对）。https://github.com/modelscope/ms-swift/pull/10072
- **行为效应**：无 crash——LD-DPO 的 label-discount 权重系统性落在错误 token 集上：prompt/masked 位被当 completion 折扣、真实 response 位权重错移，训出的偏好边界偏移。与 SFT.08（`dpo_len_norm_denominator_unscored_token`——长度归一化分子分母掩码错位）同族同库：**DPO 前缀/掩码的 token 集合与 loss 权重集合不一致**；「物理段 vs 有效 token」的语义错换与 LOSS.08/CKPT.23 一脉：padding-free/packing 几何下的计数契约失守。
- **发现来源**：2026-09-13 每日扫描

### SFT.15 `explicit_loss_type_path_drops_token_weights`

stage: `sft` · Cov: `var:loss_scale_reweights_rounds` · 置信度 `verified`

- **机制**：ms-swift `--loss_type cross_entropy`（显式选择）时 `Seq2SeqTrainer` 把对齐后的 token weights 传给 `CustomCrossEntropyLoss`，但回调在**重算** cross entropy 时丢弃 weights：非均匀权重在**默认 trainer 路径**会改变 objective、在显式选择该 loss 时却**毫无效果**——甚至零权重 token（本该被 mask 掉的位）照样计入 loss。修复 = 在 reduce 前把 `loss_scale` 应用到 per-token loss，保留 valid-token / `num_items_in_batch` 分母。
- **来源**：ms-swift PR #10100（2026-09-10 closed；8 组合回归：absent/unit/nonuniform/zero weights × local/explicit 归一，4 个加权用例修复前挂；离线 tiny Qwen3 经 `Seq2SeqTrainer._prepare_inputs`+`compute_loss` 与独立平移 PyTorch CE 的 loss/全参数梯度/单步 SGD 对照，CPU FP32 + H800 FP32/BF16）。https://github.com/modelscope/ms-swift/pull/10100
- **行为效应**：静默——显式 `loss_type` 用户配置的 token 权重（如多轮 `last_round`、工具位降权）不生效，模型按均匀权重训练，预期强调的 token 被稀释、该 mask 的位泄漏进梯度。与 SFT.12（多轮拆分默认 `loss_scale` 重复计权）同族同库：**`loss_scale` 权重链路在 trainer 分支间断裂**——那边默认路径重复计权、这边显式路径全丢；「配置面存在、执行面缺席」的死配置面与 CKPT.07/CKPT.20 同型，但打击的是 objective 本身。
- **发现来源**：2026-09-13 每日扫描

---

# RL-RWD · reward

## RL-RWD

### RL-RWD.01 `format_reward_hacking`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：格式分（XML 标签、`think` 块）权重大于正确性，策略专卷格式。
- **来源**：RLVR / R1 复现中的常规失败；reward hacking 文献。
- **行为效应**：格式分→1，任务分→0。看起来 reward 在涨。
- **发现来源**：2026-08-17 基础调研

### RL-RWD.02 `length_reward_explosion`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：GRPO 的序列级平均让更长的错误回答惩罚不足，策略把回复拉到 `max_tokens` 并重复。DrGRPO / DAPO 为此改归一化。
- **来源**：Liu et al., "Understanding R1-Zero-like Training", arXiv:2503.20783；DAPO。https://arxiv.org/abs/2503.20783
- **行为效应**：平均长度冲顶，下游分数掉到 0。无 NaN。
- **发现来源**：2026-08-17 基础调研

### RL-RWD.03 `reward_model_overoptimization`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：策略过拟合 RM 的虚假特征，RM 分涨、人类/黄金分掉。
- **来源**：经典 RM overoptimization；与格式黑客不同，这里 RM 自己是错的坐标。新实测：Rubric Dropout arXiv:2608.11669（GRPO + rubric judge，训练 judge 分持续升、gold judge 分先峰后降：HealthBench-Hard −3 / ResearchQA −22）。https://arxiv.org/abs/2608.11669
- **行为效应**：代理奖励单调好，真实评测变差。
- **发现来源**：2026-08-17 基础调研

### RL-RWD.04 `tool_schema_constraints_silently_weakened`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：verl `OpenAIFunctionToolSchema.model_validate()` 接受含标准 JSON Schema 约束的工具定义，但底层窄 Pydantic 载体只声明 `type/description/enum`（属性层）与 `type/properties/required`（参数层），`model_dump()` 把未声明键（`minimum`/`maximum`/`additionalProperties` 等）**整键静默丢弃**。agent-RL 循环里模型可见的工具契约在 prompt 渲染前被单方面削弱：没有了 `limit ≤ 128`、`offset ≥ 0` 这类约束，合法的退化参数空间变大。`@function_tool(..., schema=...)` 文档承诺 "as-is"，实际同路丢失。CPU 可复现，无 GPU/rollout 依赖。
- **来源**：verl #7505（`model_validate → model_dump` round-trip 断言复现；主分支 + 独立 checkout 双确认，`schemas.py` blob 一致）。https://github.com/verl-project/verl/issues/7505 ；修复 PR #7766（2026-09-06 补：两个 carrier 加 `ConfigDict(extra="allow")`，未声明键保留且计入 `exclude_unset`——`minimum`/`maximum`/`pattern`/`items`/嵌套 `properties`/根级 `additionalProperties` 全部存活；已声明字段仍走校验）。https://github.com/verl-project/verl/pull/7766
- **行为效应**：工具调用仍正常解析执行、训练无异常——变的只是模型看到的约束定义（issue 自评：非 tool-parser / 执行失败，是「模型可见契约被静默改写」）。行为族与 RL-RWD.01（格式黑客）同向：奖励可涨、约束遵循能力不受测；但机制在**环境侧 schema 序列化有损**而非奖励函数设计。reward 只认「调用成功」时，越界/退化参数即成合法得分路径（机制为代码级确认；RL 端到端效应为推断，issue 未跑完整 RL 实验）。
- **发现来源**：2026-08-22 每日扫描

### RL-RWD.05 `prefilled_tag_reward_dead_zone`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：模型 chat template 在 generation prompt 末尾**预填** `<think>`（如 DeepSeek-R1-Distill 官方模板结尾 `<｜Assistant｜><think>\n`）。TRL GRPO 只解码**生成的** token 当 completion 送 reward（`grpo_trainer.py`），模板预填的开标签留在 prompt 侧；而 `think_format_reward` 用锚定正则 `^<think>(?!.*<think>)(.*?)</think>.*$` 要求 completion 以 `<think>` **开头**。开标签被模板吃掉后任何 completion 都无法匹配 → 格式 reward 恒 0.0，无报错无警告。奖励函数的「completion 含完整标记」假设与生成侧 prompt 预填事实脱节。
- **来源**：trl #6966（issue 含 CPU 复现脚本 + TRL 自带 `deepseek_r1_distill.jinja` 证据）。https://github.com/huggingface/trl/issues/6966
- **行为效应**：GRPO 该 reward 通道全程 0；若组内其他 reward 也无区分 → advantage 全 0（叠加 RL-ADV.01 `grpo_group_std_zero`），格式奖励静默失效。训练不 crash；指标上 reward 恒 0 易被误读为「模型没学会格式」而非「奖励死区」。
- **发现来源**：2026-08-29 每日扫描

### RL-RWD.06 `reward_tokenizer_vocab_mismatch_silent_scores`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl `XPOTrainer`/`NashMDTrainer`（Online DPO 变体）的 `get_reward()` 把**策略 tokenizer 的 input_ids 直接喂 RM forward**，不做 decode/re-tokenize。RM 词表较小时要么 embedding 越界 IndexError（fail-fast 面），要么**同尺寸不同映射时静默给错位 token 打分**（silent 面）。且 `XPOTrainer` 接受 `reward_processing_classes` 却从不使用、`NashMDTrainer` 硬编码成 policy tokenizer——「换 reward tokenizer」两条路都是静默 no-op。
- **来源**：trl #6952 + #6955（同缺陷的补功能/加校验一对 PR：`get_reward` 行级代码对照；正路参照 `OnlineDPOTrainer._calculate_rewards_from_functions` 的 decode→re-tokenize）。https://github.com/huggingface/trl/pull/6952 · https://github.com/huggingface/trl/pull/6955 ；母 issue trl #6951（2026-09-03 补：XPO 接受 `reward_processing_classes` 却从不使用、NashMD 硬接 policy tokenizer 的签名级证据；异词表 RM 的 late-fail 是 embedding `IndexError`，同词表时即落入本条静默面；报告者实测 Llama RM + Qwen policy 命中，trainer 自带测试因 RM 与 policy 同 model id 而 CI 不可见）。https://github.com/huggingface/trl/issues/6951
- **行为效应**：silent 面无任何信号——RM 在错位 token 序列上打分，reward 是**量级正常的真实数字**但语义无效，偏好梯度被随机化优势驱动，reward 曲线照样能涨。与 RL-RWD.01（策略钻奖励空子）正交：这是**奖励输入管线静默错接**；与 DATA.04（tokenizer id mismatch）同根因，落在 RL reward 路径上。
- **发现来源**：2026-08-30 每日扫描

### RL-RWD.07 `verifier_error_group_correlation_sign_flip`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：RLVR 用自动 verifier 给组内多条 completion 打分，GRPO advantage 把这些分数当**独立测量**做组内中心化——但同一 verifier 对同一 prompt 的组内错误是**相关的**（答案形式聚类：fraction/radical/symbolic/interval 错成一片，unit/percent 较轻）。24,998 组 × 8 completions（Qwen2.5-1.5B，MATH/GSM8K/DeepMath-103K）实测组内 verifier 错误相关 ρ=0.530（95% CI 0.500-0.560）：可交换误差模型下 8 样本组的设计效应把有效样本量压到 **1.70**——组内基线（GRPO 的方差归一分母、leave-one-out 均值）建立在远比名义 n=8 弱的统计量上。跨四种 rule-based verifier 重放组相对 advantage：**至少一条 advantage 符号翻转**的组达 0.83%。
- **来源**：arXiv:2609.06386（2026-09-06，"Are Verifier Errors Independent Within a GRPO Group? Evidence from Qwen2.5 Rollouts"）。https://arxiv.org/abs/2609.06386
- **行为效应**：无 crash、无 NaN；reward 与训练指标一切正常。低烈度但系统性的方向污染：符号翻转的 advantage 把策略往 verifier 错判的方向推，且组内相关使「多采样平均」的去噪效率比名义低约 4.7×。与 RL-RWD.03（RM 过优化）不同：错不在 reward 模型被钻、在**测量误差的组内结构**被组相对估计器假设掉；与 RL-ADV.08（guess 与 reasoning 同 advantage）互补：那边是 outcome 无法区分路径，这边是 outcome 本身带相关噪声。
- **发现来源**：2026-09-10 每日扫描

### RL-RWD.08 `unscored_rollout_defaults_to_zero_reward`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl OpenReward 环境 `environment.py:192` 把 `self.reward` 初始化为 `0.0`、只在 `_call_ors_tool`（L266-296）拿到非 null reward 时覆写；默认 reward 函数 `_outcome_only_reward_func`（`_spec.py:69-75`）无条件返回 `env.reward`。从未调过打分工具的 rollout（放弃 / 撞 `max_tool_calling_iterations` / 全部工具调用抛异常）与「真被打 0 分」的 rollout **bit 级同值**——sentinel 与合法测量坍缩成同一个数。修复形状已有先例：PR #6430 的 `has_reward` 标志（仅工具返回非 null 时置位、`reset()` 复位、未打分返回 `None` 使 `unscorable_mask` 正确剔除）曾被维护者正面评审为 "a clean, well-reasoned fix for a real reward-hacking vector"，后因无关测试覆盖问题被关未合；`environment.py`/`_spec.py` 此后无 commit，缺陷在当前 HEAD 仍活。
- **来源**：trl #7364（2026-09-24；逐函数源替换 repro 三例对照：A 未打分 vs B 真错 vs C 真对——A/B 下游全同；附 PR #6430 兴衰史）。https://github.com/huggingface/trl/issues/7364 ；trl #6430（先行修复、正评后被关）。https://github.com/huggingface/trl/pull/6430
- **行为效应**：无 crash——组内中心化后「未测」与「测了且零分」同为 0.0：genuinely-failed 样本无法与 never-scored 区分，`unscorable_mask` 剔除逻辑失效；放弃/超限 rollout 被当作真实负信号参与 advantage，按 0 分中心化的组统计把「评分覆盖缺口」转成方向性梯度偏置（放弃行为可能被错误奖励或惩罚，取决于组内其余得分）。与 RL-RWD.05（prefilled tag 使 reward 恒 0 的死区）行为效应同型「合法 0 分与未测坍缩」，差异在这边是**环境侧哨兵值**而非 tokenizer/模板错位；与 RL-RWD.07（verifier 误差组内相关）互补：那边是测量的噪声结构、这边是测量的**缺席**被编码为测量值。
- **发现来源**：2026-09-24 每日扫描

---

# RL-ADV · advantage

## RL-ADV

### RL-ADV.01 `grpo_group_std_zero`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：一组 rollout 奖励全相同（全对或全错），组标准差为 0。实现若直接相除 → Inf/NaN；若悄悄跳过 → 该 prompt 零学习信号却仍占 batch。
- **来源**：TRL GRPOTrainer 文档（zero std）；OpenRLHF #1272 / #1270。https://huggingface.co/docs/trl/en/grpo_trainer
- **行为效应**：NaN 一步毒化，或「在训但学不到难题」。
- **发现来源**：2026-08-17 基础调研

### RL-ADV.02 `grpo_std_difficulty_bias`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：用组内 std 去除 advantage，低方差题（几乎全对/全错）的 advantage 被放大。DrGRPO 去掉 std 项。
- **来源**：arXiv:2503.20783。https://arxiv.org/abs/2503.20783
- **行为效应**：模型在「已经会的题」上步子过大，难题信号被压。
- **发现来源**：2026-08-17 基础调研

### RL-ADV.03 `seq_mean_token_length_bias`

stage: `rl` · Cov: `var:length_reward_explosion` · 置信度 `documented`

- **机制**：对每条序列先对 token 取平均再进组，长序列的 per-token 梯度被压小。DAPO 改为对全局 token 归一。
- **来源**：DAPO；TRL `loss_type`：`grpo` / `dr_grpo` / `dapo`。
- **行为效应**：与 RL-RWD.02 同一家族，落在 loss 归约而不是奖励设计。
- **发现来源**：2026-08-17 基础调研

### RL-ADV.04 `drop_group_ignores_nonbinary_std`

stage: `rl` · Cov: `var:grpo_group_std_zero` · 置信度 `documented`

- **机制**：动态过滤「无对比」组时只看是否全 0/1，非二值奖励下仍可能 std≈0。OpenRLHF #1270 要求 drop 时看标准差。
- **来源**：https://github.com/OpenRLHF/OpenRLHF/issues/1270
- **行为效应**：软奖励任务里「几乎同分」的组仍进更新，噪声优势。
- **发现来源**：2026-08-17 基础调研

### RL-ADV.05 `flat_group_zero_learning_signal`

stage: `rl` · Cov: `var:grpo_group_std_zero` · 置信度 `documented`

- **机制**：即使实现防了除零，全对/全错组对 GRPO 仍是零对比。DAPO dynamic sampling 的动机。过滤侧也有静默面（2026-08-21 补）：OpenRLHF #1310——dynamic filtering 在 prompt loader 耗尽后拒绝一个组时，`SamplesGenerator._generate_vllm()` 直接返回空列表，**把已累积的 accepted_experiences 整批丢弃**并 cancel 全部 pending refs（含已生成完、本会被接受的组）；shipped DAPO/REINFORCE recipe 开着该过滤，小数据集上唯一一个生成 batch 可被清空 → 0 次 optimizer update，run 仍「正常」走完。
- **来源**：DAPO dynamic sampling；TRL / OpenRLHF 过滤开关；OpenRLHF #1310（shipped DAPO recipe 数值推演：20000%128==32 的末批含过滤组的概率可观）。https://github.com/OpenRLHF/OpenRLHF/issues/1310
- **行为效应**：有效 batch 比名义 batch 小很多，进度虚标。
- **发现来源**：2026-08-17 基础调研

### RL-ADV.06 `replay_buffer_shape_contract_degenerates`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl 实验 trainer `GRPOWithReplayBufferTrainer` 的真实调用路径与 replay 模块的**形状契约**脱节：`_generate_and_score_completions` 传 1-D `(B,)` per-sample 张量，`update_with_replay_buffer` 及其 helper 全部假设 `(num_groups, num_generations)` 2-D。`(B,)` 输入下 `group_std_rewards.max(dim=0).values > 0` 退化成整 batch 一个 0-dim 标量 bool；`nonzero()[0]` 只返回 `[0]`，**每批只有「组 0」进 buffer**，存的还是单样本 advantage 冒充整组；替换触发条件变成「整批零方差」，触发时换掉全部 `num_generations` 行 token 却只覆写一个样本的 advantage，其余行带着别人的零方差 advantage 续命；buffer 优先级 `(abs·std).sum(-1)` 也成整批标量。单测直接用手搓 2-D 张量调 `update_with_replay_buffer()`，CI 全绿。
- **来源**：trl #6804（代码审计 + 张量行为逐条验证：`torch.tensor([0.5,0.5,0.0,0.0]).max(dim=0).values>0` → 标量 True）。https://github.com/huggingface/trl/issues/6804
- **行为效应**：replay 功能「开着」但内容与优先级无意义；run 无报错继续训，replay 样本的 advantage 标签错位。与 RL-KL.04 同为实验 trainer 复制粘贴漂移家族，但漂移点在**数据结构契约**而非 loss 项：形状对不上时张量语义静默换义。
- **发现来源**：2026-08-23 每日扫描

### RL-ADV.07 `singleton_group_vectorized_zero_path`

stage: `rl` · Cov: `var:grpo_group_std_zero` · 置信度 `documented`

- **机制**：verl `rloo` 与 `rloo_vectorized` 文档声明可互换、等价测试也绿，但对**单样本组**行为相反：标量版 `if response_num > 1` 让 singleton 保留原始分数（GRPO 同约定：mean=0/std=1 归一后原分数存活）；向量化版 `loo * (c > 1)` 把这些行**乘成 0**。`rollout.n == 1` 时每组都是 singleton → advantage 全零张量、策略无梯度；过滤后组内只剩一条幸存样本的 ragged 组同样中招。
- **来源**：verl #7581（等价测试 `_make_group_index` 断言每组 ≥2 样本，**恰好绕开唯一不存在 leave-one-out baseline 的组规模**；修复 `torch.where(c > 1, loo, scores)`，红→绿 3 failed→30 passed）。https://github.com/verl-project/verl/pull/7581 ；verl #7793（2026-09-10 补：同机制的独立 issue——overlong/dynamic filtering 把组内滤到只剩一条时，向量化版末尾 `* (c > 1)` 把 advantage 乘成 0，标量版 `id2mean = 0` 分支让幸存样本保留原始分数；报告者验证多样本组的 leave-one-out 代数两边等价，仅 singleton 分歧行为，与 GRPO singleton 约定 mean=0/std=1 一致的是标量版）。https://github.com/verl-project/verl/issues/7793
- **行为效应**：无报错、无 NaN；`n=1` 的 RLOO run「正常训练但永远不学习」——advantage 恒 0 下 loss 仍有限、指标齐全。#7793 面向**过滤后**的 ragged singleton：模型对被过滤路径之外的幸存 rollout 也拿不到梯度。与 RL-ADV.01（组内无对比零信号）同族，但机制在**向量化掩码错杀非退化组**：分数本可用，被实现乘没了；`n` 从不与估计器选择做校验。
- **发现来源**：2026-08-30 每日扫描（#7793 来源 2026-09-10 补）

### RL-ADV.08 `guess_path_spurious_advantage`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：GRPO 的组内 advantage 只看 outcome reward 统计：一条 rollout **靠猜**命中正确答案（bounded-answer 小候选集、open-answer 里的 bounded 子题、搜索 agent 预算内多路径撞对）与**靠推理**命中，拿到完全相同的高 advantage——「正确答案」信号里混进了与能力无关的猜测分量，策略被推向 guess-like 行为。机制覆盖三类常见任务面；提案 SIGNBALANCE 保留 verifier 符号、用全局 scale、per-class stop-gradient 重标定恢复零均值。
- **来源**：Wang et al., "Spurious Advantage Hidden in GRPO", arXiv:2609.04063（2026-09-03；数学 + 搜索 agent 双 benchmark，SIGNBALANCE 在 bounded-answer 与 search agent 上超 GRPO、open-answer 持平）。https://arxiv.org/abs/2609.04063
- **行为效应**：reward 曲线上涨、benchmark 分数可能也涨，但策略的**能力构成**在漂移（guess 分量被强化）；与 RL-RWD.03（reward 过优化）同症状（指标涨、能力掉），机制在估计器而非 reward 本身；与 RL-ADV.02（std 放大已会题）互补：那边错在方差归一，这边错在 outcome 无法区分到达路径。
- **发现来源**：2026-09-05 每日扫描

### RL-ADV.09 `filter_metric_predicate_mismatch_phantom_advantage`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：GRPO 系动态采样靠 filter metric 丢弃「无对比」组，但实现把 metric 做成可配置项——**按 shaped 训练分过滤时谓词语义失配**：复合 shaped reward 下 all-fail 组（任务全失败）内部仍有非零组内离差（shaping 分量差异），谓词判「有对比」放行；std 归一化把失败组内部的 shaping 差异**提升为满幅 phantom advantage**，策略在「全是错解、只有格式/过程分不同」的组上拿到完整学习信号。verl native recipe/dapo trainer 实测（Qwen2.5-1.5B LoRA + GSM8K 自定 shaped hook，模型/数据/reward/trainer 全固定、只换 metric）：score 指标臂 40 步内 0 次 batch refill、终值 EM 0.160；accuracy 指标臂 29/40 步 refill、终值 EM 0.763；无过滤臂 EM 0.080±0.112。作者定位为 metric-谓词语义错配（区别于已知 shaping 放大与 all-fail 过滤），并直接instrument了 native deletion/refill telemetry。
- **来源**：arXiv:2609.13866（2026-09-12；受控 GSM8K 对照 + verl native trainer 复现）。https://arxiv.org/abs/2609.13866
- **行为效应**：无 crash、指标看似正常推进——**训练信号在系统性学习「失败组内的 shaping 差异」**，收敛终点被拉向格式/过程分黑客方向（EM 0.160 vs 0.763 同预算同配置）；filter 命中率 telemetry（refill 次数）是唯一可观测异常。与 RL-ADV.04（drop 谓词只认二值 std）同族：**过滤谓词与奖励语义错配**，那边漏掉软奖励近同分组、这边反向放行全失败组；与 RL-ADV.08（guess 路径伪优势）互补：那边是 outcome 不区分路径、这边是 filter 不区分 outcome 与 shaping；对 RL-RWD 家族（shaped reward 黑客）是 estimator 侧的放大器。
- **发现来源**：2026-09-16 每日扫描

### RL-ADV.10 `pfppo_v1_resampled_advantage_trajectory_mispair`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：verl v1 trainer 在 `algorithm.use_pf_ppo=true` 且 GAE advantage 路径下，PF-PPO 对整批数据有放回重采样；v1 的 `_compute_advantage` 却先从 TransferQueue 取**原始行序**的 `response_mask`，再把重采样后的 `advantages` / `returns` 依原键写回。原轨迹的 response、old/ref log-prob 留在队列原位置，因而更新时 advantage 与其对应轨迹错配；v0 trainer 将重采样后的整个 `DataProto` 一起传下去，不属于本问题。
- **来源**：verl #7956（2026-09-18；在 main `23a341e3` 上的代码路径核对与无 GPU 的 CPU 复现；截至 2026-09-24 issue 为 open）。https://github.com/verl-project/verl/issues/7956
- **行为效应**：来源的 CPU 路径复现了优势值与轨迹键错配而没有异常；在上述配置下，策略梯度会把 credit assignment 赋给错误轨迹。**未提供长程训练回报或任务指标实测，未评估检测器效果**。与 RL-ADV.09 的 shaped reward 过滤谓词错配不同，这里是有放回重采样后跨键写回破坏样本身份。
- **发现来源**：2026-09-18 远程扫描回填；2026-09-24 合并复核

---

# RL-KL

## RL-KL

### RL-KL.01 `k1_cancels_tim_early_collapse`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：K1 = −log r 有符号，batch 内正负抵消。TIM 已在扭曲目标时，K1/K3 仍可贴着零基线。VeXact 文：recompute 模式前 700 step 奖励已掉，KL 探针几乎不动。
- **来源**：arXiv:2605.14220 §4.1。https://arxiv.org/abs/2605.14220
- **行为效应**：看板说 KL 健康，策略已在塌。
- **发现来源**：2026-08-17 基础调研

### RL-KL.02 `reverse_kl_as_loss_wrong_is_weight`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：把 reverse-KL 当损失时，梯度形式和 off-policy 重要性权重写错，更新指向错误的散度。
- **来源**：OpenRLHF #1309。https://github.com/OpenRLHF/OpenRLHF/pull/1309
- **行为效应**：KL 项在降，策略并不靠近参照；或反过来。
- **发现来源**：2026-08-17 基础调研

### RL-KL.03 `entropy_collapse_to_repetition`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：策略熵塌缩，解码陷入重复循环。与长度爆炸经常一起出现。
- **来源**：GRPO 失败复盘（Interconnects / R1-zero-like）。
- **行为效应**：reward 或格式分仍高，文本不可用。
- **发现来源**：2026-08-17 基础调研

### RL-KL.04 `stale_loss_copy_silently_drops_config`

stage: `rl` · Cov: `var:config_default_swallows_cli_override` · 置信度 `documented`

- **机制**：变体 trainer（GSPO-token / GMPO 等复制粘贴 `_compute_loss` 的实验实现）与主实现**漂移**：新增的正则项/开关在副本里没有对应代码，配置项通过校验、resolved config 里正确，但在 loss 里**静默 no-op**——KL 偏置校正系数不乘、off-policy mask（OPSM）不应用、entropy bonus 整块缺失、MoE router aux loss 不加。
- **来源**：trl #6807（gspo_token `_compute_loss` 是旧版 GRPO 的 stale copy：`use_bias_correction_kl`/`off_policy_mask_threshold`/`entropy_coef`/`router_aux_loss_coef` 全部静默无效，代码对照复现）。https://github.com/huggingface/trl/issues/6807 ；trl #6808（GMPO：`use_liger_kernel=True` 时基类 `compute_loss` 绕过 GMPO 覆写、静默训成普通 GRPO clipped-surrogate；MoE aux loss / entropy bonus 同样缺失）。https://github.com/huggingface/trl/issues/6808 ；trl #7009（2026-09-03 补，GRPOTrainer **本体**同一分叉：`compute_loss` 按 Liger flag 提前 return，唯一加 aux loss 的 `_compute_loss` 不再执行、Liger 路径从不引用 `router_aux_loss_coef`，而 `aux_loss_enabled` 仍为 True、`router_aux_loss_coef` 默认 0.001 使 MoE 模型**零配置即中招**；2×2 实测（tiny-Qwen3MoE，同 H100 同 seed）：coef 0.0→0.1 标准 path loss 移动 0.2239 且与 `coef×aux_loss` 对到 float32 分辨率，Liger path 两个 loss **bit 级相同**（2^-26）——系数对目标完全无效；DPO/KTO 对该组合已 raise，GRPO 副本没有守卫）。https://github.com/huggingface/trl/issues/7009 ；修复 #7037/#7038（2026-09-04 补：GRPO 对 MoE aux + Liger 组合补 fail-fast 守卫 + `GRPOConfig` 文档写明须 `router_aux_loss_coef=0.0`，静默 no-op → 显式拒绝）。https://github.com/huggingface/trl/pull/7038 ；trl #7062（2026-09-06 补：vendored Liger 路径同族缺陷——DPO label smoothing/DiscoPOP tau 不转发进 fused loss、KTO 非默认 class weight 被**静默忽略**、KTO 多 example chunk 指标错、PEFT 参考路径被误关：#7059 vendor 进 `trl.losses` 时把配置面凿穿，「fused 快路径丢弃配置」是本条机制的第 5 个实例）。https://github.com/huggingface/trl/pull/7062 ；trl #7270（2026-09-19 补：GMPO + Liger 的**完整更新级**独立复现——固定 ckpt + 固定 rollout 下单步 SGD 对照，Liger 开启的 GMPO 与纯 GRPO 的 loss/全梯度/终态**完全一致**、与 GMPO 目标的梯度相对 L2 差 0.1996558210；覆盖 `compute_loss` 调 GMPO 自身 `_compute_loss` 后差降到 1.93e-7；维护者 qgallouedec 确认与 #6808 同病、关闭为 duplicate，修复走向 = #7163 init 期拒绝该 flag + #7063/#7077 整体移除 fused Liger 路径后 `compute_loss` 不再绕过 `_compute_loss`——「结构性移除分叉面」而非补丁）。https://github.com/huggingface/trl/issues/7270
- **行为效应**：无报错；用户以为在跑 GSPO-token/GMPO + KL 校正 + 负载均衡，实际 loss 退化成缺项的旧目标。MoE 场景 router aux 缺失会沿 MOE.01（负载均衡失守 → 专家死）方向放大。与 CKPT.06/07 同族但发生在**损失计算路径**：错误每一步都进梯度。机制面与 OPT.01（矩/权重错配）互补：那边是恢复期错位，这边是副本代码漂移。
- **发现来源**：2026-08-20 每日扫描

### RL-KL.05 `entropy_bonus_unscaled_logits_temperature_split`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：OpenRLHF `Actor.forward` 在 `log_probs_from_logits` 把 logits **原地**除以 `rollout.temperature` **之前**调用 `compute_entropy(output["logits"])`——entropy bonus（`entropy_coef`）与日志 `entropy_loss` 描述的是 `softmax(logits)`，而训练的策略与 vLLM 采样器都在 `softmax(logits/T)` 上。两套温度各算各的：T≠1.0 时 entropy 项系统性错值（tiny table model 实测 per-token 偏差 T=0.5 达 1.03 nats、T=1.5 达 0.44 nats），entropy 正则施加在**与实际策略不同的分布**上。修复 = fp32 cast 后先统一除 T，entropy 与 log-prob 用同一份 scaled logits（与 trl GRPOTrainer 行为对齐）。
- **来源**：OpenRLHF PR #1319（CPU 回归测试：新测试在 main 上 fail、修复后 pass；#857 的 in-place 优化保留）。https://github.com/OpenRLHF/OpenRLHF/pull/1319
- **行为效应**：无 crash；T<1 时 entropy 奖励被高估（真实策略熵本已低）、T>1 时被低估——探索/坍缩平衡调不对却查不出原因，`entropy_loss` 曲线本身也在报错分布的数。与 RL-KL.04（stale copy 丢配置）不同：代码路径都执行了，错在**同一前向里两个消费者用了不同温度的 logits**；与 RL-RO.01（TIM）互补：那边训练/rollout 两侧引擎不一致，这边单引擎内部时序把温度分裂成两份。
- **发现来源**：2026-09-06 每日扫描

### RL-KL.06 `kl_estimator_choice_varies_by_trainer`

stage: `rl` · Cov: `var:rl_diagnostics_unmasked_padding_positions` · 置信度 `documented`

- **机制**：trl `GRPOTrainer` 用低方差 k3 近似（Schulman 2020，PPOTrainer 已在 PR #3240 切换），`RLOOTrainer` 仍用高方差 k1 式估计——同一仓库、同一接口、同一 KL 语义，不同 trainer 的 KL 项噪声特性不同：k1 的方差使 KL 惩罚实际强度抖动，跨 trainer 对比「KL 系数调多少」会把估计器差异当成算法/任务差异。非 crash、数值有限，属目标实现的选择性漂移而非单点 bug。
- **来源**：trl #7086（0 评论、刚开，作者自带一行修复方案；官方文档 rloo_trainer 页自述当前用的是较差估计器）。https://github.com/huggingface/trl/issues/7086
- **行为效应**：RLOO 的 KL 信号噪声更大、惩罚时紧时松；与 RL-KL.01（K1 符号抵消）同根——K 系估计器的方差/偏差特性直接决定「KL 探针贴零」类观测的可靠性；与 RL-KL.04（变体 trainer stale copy）互补：那边副本丢配置，这边主实现之间估计器选择未对齐。
- **发现来源**：2026-09-07 每日扫描

### RL-KL.07 `k3_clip_saturates_but_gradient_nans`

stage: `rl` · Cov: `NEW` · 置信度 `verified`

- **机制**：OpenRLHF 的 k3 KL 估计器在算完指数后对**输出**做 clamp——极端但有限的 log-prob 差先过 `exp` 变成 `inf`，clamp 把返回值拉回有限，但 autograd 图里留着 `exp` 的 NaN 梯度（`0 × inf` 型）：**KL 值有限、backward 产出 NaN**，一次 KL-regularized optimizer 更新即可污染原本有限的参数。修复 = 只在「输出 cap 已饱和」的区间内对**局部 log-ratio 拷贝**先 bound 再 exp（保持被 clip 的目标函数与其正常区梯度不变；原始 log ratio 仍留给可选的 corrected-gradient surrogate）。FP16/BF16/FP32 值/梯度对 float64 参照 + corrected-gradient 覆盖 + CPU AdamW 更新测试全过。
- **来源**：OpenRLHF PR #1335（2026-09-11；「返回的 KL clip 后仍有限，但 backward 经 `0*inf` 产生 NaN」）。https://github.com/OpenRLHF/OpenRLHF/pull/1335
- **行为效应**：无 crash——KL 指标一切正常（clamp 住了），grad_norm 突然 NaN、参数被污染；触发面是极端 log-prob 差（off-policy 漂移 / rollout-train mismatch 大时更可能），恰好是最需要 KL 惩罚兜底的场景。与 NUM.04（scaler 吞溢出）互补：那边前向溢出被吞、这边**反向** inf 被 forward-side clip 掩盖；与 RL-RO.04（GSPO 比值指数溢出）同根（无界指数），差异在那边直接 inf、这边被 clamp 转成静默 NaN 梯度。
- **发现来源**：2026-09-12 每日扫描

### RL-KL.08 `sequence_mean_is_weight_broadcast_to_token_kl`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl GRPOTrainer `use_bias_correction_kl=True`（PR #6503 起默认开）时，KL bias correction 用 `coef_1 = exp(log_importance_weights)` 乘 per-token KL。`importance_sampling_level="sequence"` 下 `log_importance_weights` 是**序列均值** log-ratio，shape `(B, 1)`——广播到 `(B, T)` per-token KL 后，总 KL 的 token 级梯度多出一项 `(Σ_t k3_t) · ∇(mean_t log π_t)`：既非 DeepSeek-V3.2 论文的 per-token 修正、也非无偏 reverse-KL 梯度，纯 spurious。`level="token"` 时权重本就 per-token、无此 bug。修复（PR #6594，closed 未合并）= 无论 level 一律用 per-token `exp(log_ratio)` 乘 KL。
- **来源**：trl PR #6594（closed unmerged；Closes #6586；arXiv:2608.02786 将其引为「loss 正常下降、梯度被腐蚀」的训练期实例）。https://github.com/huggingface/trl/pull/6594 ；#6586。https://github.com/huggingface/trl/issues/6586
- **行为效应**：无 crash、loss 曲线正常下降——每个 token 的 KL 惩罚梯度携带全序列聚合的伪梯度分量，KL 约束的施加方向系统性偏移（sequence-level IS 配置下默认即中招）；与 RL-KL.07（k3 clamp 前向有限反向 NaN）同打击 KL 修正路径，差异在这边是**离散的形状契约破坏**（(B,1) 权重 × (B,T) KL）、那边是无界指数的数值面；与 RL-ADV 家族 IS 权重类（trl #6945 幸存者偏差 ratio）同根：**IS 修正项的粒度/形状与损失定义不匹配**。
- **发现来源**：2026-09-16 每日扫描

### RL-KL.09 `fp8_noise_clips_negative_tokens_out_of_trust_region`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：全管线 FP8 RL 中，训练侧与 rollout 侧各自的 FP8 量化噪声都进入 importance ratio，**逐级复合**后 ratio 系统性偏移——负 advantage token 的 ratio 被不成比例地推出 trust region（clip 上界），梯度被错误归零：病态输出**得不到惩罚**且随训练累积。表现为训练中期 entropy 涌现式激增 + 输出乱码。BF16 rollout + FP8 trainer（FP8-RL 栈）无此不稳定性——复合噪声是全管线 FP8 特有。缓解 = Calibrated Clipping：按高精度 BF16 分布对齐 FP8 clip 下界分位数并再平衡上界（8B–32B、GRPO/DAPO、多 scaling 粒度验证恢复至 BF16 基线水平）。
- **来源**："Towards Full Pipeline FP8 Reinforcement Learning for LLMs", arXiv:2609.22870（2026-09-19）。https://arxiv.org/abs/2609.22870
- **行为效应**：无 crash、无 NaN——entropy surge + 乱码输出这类「看似探索性崩溃」的症状，根因在**量化噪声 × clip 机制的符号不对称交互**：惩罚通道被静默关闭、奖励通道完好，策略单向漂移。与 RL-KL.03（entropy 塌缩到重复）表象同族但机制不同：那边是策略收敛动力学、这边是 clip 掩蔽负梯度；与 RL-RO.01（TIM：量化 rollout 的 logprob 偏差，TIS/MIS 校正对象）互补：TIS 校正 logprob **数值**、本条指出 ratio 进入**非线性 clip** 后校正残差被符号不对称放大——低精度栈设计需同时管校正与 clip 边界；与 RL-KL.07（k3 前向有限反向 NaN）同打击「梯度经有界算子后语义翻转」，差异在这边是确定性的方向偏置而非数值爆炸。
- **发现来源**：2026-09-23 每日扫描

---

# RL-RO · 训推不一致（TIM）

## RL-RO

### RL-RO.01 `tim_rollout_trainer_logprob_delta`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：rollout（vLLM/SGLang）与 trainer（FSDP/Megatron）对**同一权重同一序列**给出不同 token 概率。来源：核实现不同 + 非 batch-invariant 归约。δₜ = log π_train − log π_rollout，均值小，极值可到 ~1.0，甚至 argmax 翻转。
- **来源**：Zhong et al., "Diagnosing Training Inference Mismatch", arXiv:2605.14220。https://arxiv.org/abs/2605.14220 ；量化 rollout 放大版：QaRL arXiv:2604.07853（低精度 rollout + 全精度 trainer，长回复退化为重复/乱码 error token）。https://arxiv.org/abs/2604.07853 ；目标错位视角：MIPI/MIPU arXiv:2606.29526（TIM 造成常驻 off-policyness，训练引擎的改进不保证部署侧推理策略改进，需推理侧 gap proxy 选择性接受候选更新）。https://arxiv.org/abs/2606.29526 ；动态演化视角（2026-08-20 补）：arXiv:2602.01826（TIM 不是静态数值差：梯度噪声与 mismatch 随训练**同步增长**，IS 在长跑中失效；缩小更新尺寸可压制——mismatch 是与优化动力学耦合的动态故障，LR 调度即缓解）。https://arxiv.org/abs/2602.01826 ；结构化极限（2026-08-21 补）：arXiv:2606.09821（长尾词表下 importance ratio 是 distributional shift 的劣质代理，PPO/GRPO 的 ratio-clipping 只近似 trust region；DPPO 用 divergence-based mask 替换 ratio 剪裁更稳）。https://arxiv.org/abs/2606.09821 ；谱系（2026-08-22 补）：FP8-RL arXiv:2601.18150（veRL 生态 FP8 rollout 生产栈：blockwise W8A8 + FP8 KV-cache + per-step QKV scale 重校准，mismatch 用 token 级 TIS/MIS 校正——TIM 已被当作低精度 rollout 的**设计约束**而非偶发缺陷）。https://arxiv.org/abs/2601.18150 ；DVP arXiv:2512.23087（证明 TIM 的 logprob 散度界 ∝ (1−p)：高频 token 界趋零、长尾 token 界显著，采样尾部 token 引入系统性偏差误差并沿序列累积——为 RL-RO.06 重尾 / RL-RO.07 截断偏置给出统一理论：动态剪掉词表极尾部即稳定）。https://arxiv.org/abs/2512.23087 ；TP 尺寸维度（2026-08-30 补）：arXiv:2511.17826（serving 框架跨 TP 尺寸非确定：浮点非结合性 + 跨 GPU 归约序不一致，RL 场景 trainer TP=1 / rollout 多卡 TP 是天然错配源，与 batch-invariant kernel 已解决的 batch 维非确定性正交）。https://arxiv.org/abs/2511.17826 ；量化误差源分解（2026-09-17 补）：QUADS arXiv:2607.15810（NVFP4 rollout + BF16 trainer ~150 步崩塌且 rollout-trainer logprob gap 迅速增长；受控消融定位**激活误差而非权重误差**为主导——权重可经共享 quant-dequant 路径对齐、激活在线重算且误差被粗 E2M1 网格放大；缓解 = trainer 侧非对称 QAT 只伪量化权重 + rollout 侧残差补偿：低精度 mismatch 的**误差源分解 + 双侧对齐**设计范式）。https://arxiv.org/abs/2607.15810 ；clip 交互面（2026-09-23 补）：arXiv:2609.22870（全管线 FP8 的复合量化噪声使 ratio 系统性偏移，负 advantage token 被推出 trust region——TIM 残差经非线性 clip 放大成符号不对称的梯度掩蔽，见 RL-KL.09）。https://arxiv.org/abs/2609.22870
- **行为效应**：REINFORCE 下仅 TIM 就能让 MoE 验证奖励从 0.29 掉到 0.07；VeXact（零错配）继续升到 0.53。
- **发现来源**：2026-08-17 基础调研

### RL-RO.02 `recompute_vs_bypass_objective_skew`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：π_old 有两条取法。recompute：分母是 trainer 重算 → 优势加权的零中心贡献对正负样本不对称扭曲。bypass：分母是 rollout，但分子和 ∇logπ 仍在 trainer 的数值景观上，更新到 rollout 侧不兑现。
- **来源**：同 RL-RO.01 §4.1。https://arxiv.org/abs/2605.14220
- **行为效应**：recompute 可把奖励塌到近 0；bypass 掉到 ~0.4 且不一定伴随 loss 尖峰。PPO clip 盖不住。
- **发现来源**：2026-08-17 基础调研

### RL-RO.03 `seq_mask_tis_skips_token_clamp`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：OpenRLHF `seq-mask-tis` 注释写「序列几何平均只做过滤，系数仍走 token TIS」，实现却 `exp(rollout_log_ratio)` **不 clamp**。中等比值已偏离文档语义；极端 off-policy 上 `exp` 溢出：接受序列 → inf 梯度，拒绝序列 → `0 * inf = NaN`。
- **来源**：OpenRLHF #1295（2026-08-08），修复向 #1298。https://github.com/OpenRLHF/OpenRLHF/issues/1295
- **行为效应**：单条陈旧序列毒化整步 PPO。opt-in 但一开就中招。
- **发现来源**：2026-08-17 基础调研

### RL-RO.04 `gspo_ratio_exponent_overflow`

stage: `rl` · Cov: `var:seq_mask_tis_skips_token_clamp` · 置信度 `documented`

- **机制**：GSPO 序列级重要性比把对数比乘长度再 exp，未绑定指数 → 非有限梯度。与 #1295 相邻但在另一条分支。
- **来源**：OpenRLHF #1293。https://github.com/OpenRLHF/OpenRLHF/pull/1293
- **行为效应**：长回复一步 inf/NaN。
- **发现来源**：2026-08-17 基础调研

### RL-RO.05 `moe_router_not_replayed`

stage: `rl` · Cov: `var:train_infer_router_desync` · 置信度 `documented`

- **机制**：TIM 的 MoE 特例：连 top-k 专家集合都可以不一致。不回放路由，重要性采样校正的是一张错图。
- **来源**：R3 arXiv:2510.11370。https://arxiv.org/abs/2510.11370 ；框架实例（2026-08-20 补）：verl #7463——`actor.router_replay.mode`（官方 Ascend 文档教的键）是死 key，engine 只读 `actor.megatron.router_replay.mode`，按文档配置的 routing replay 静默 `disabled`，大 MoE 的 train/infer 路由不一致无人拦截（机制详见 CKPT.07 实例三）。https://github.com/verl-project/verl/issues/7463 ；verl PR #7805（2026-09-12 补：R2 router replay 的**半修复回归史**——首版实现查 process-global router registry，replay 关闭时普通 Megatron SFT/RL job 因空 registry 直接 RuntimeError 而被 #7786 revert；重引入版改为 model-scoped + 显式 opt-in，空 registry 在 disabled 下视为 no-op、R2 活跃时缺 routing 数据保持硬失败——「正确性修复被回滚」本身即此类契约脆弱性的实例）。https://github.com/verl-project/verl/pull/7805
- **行为效应**：MoE RL 比 dense 更早塌。
- **发现来源**：2026-08-17 基础调研

### RL-RO.06 `untruncated_is_heavy_tail`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：重要性权重未截断时，TIM 的重尾 δₜ 变成重尾比。TIS + 基于 r_corr 的序列拒绝能贴近零错配基线，但仍是事后扔掉样本。
- **来源**：arXiv:2605.14220 §4.2；Yao et al. rollout-training mismatch。
- **行为效应**：偶发巨大梯度；或大量样本被拒，有效数据远小于名义 batch。
- **发现来源**：2026-08-17 基础调研

### RL-RO.07 `nucleus_truncation_is_bias`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：top_p / top_k / min_p 截断采样时，vLLM 返回的是**截断后重归一化**分布上的 logprob，trainer 侧重算的是全词表 log-softmax。对幸存 token，rollout 侧 logprob 系统性高出 `-log(幸存概率质量)` → `log π_θ − log π_behaviour` 带一个**逐 token 恒负**的偏置，直接乘进 importance_sampling_ratio 与 per-token loss。温度是对称处理的（#4159 的修复），截断不是。
- **来源**：trl #6789（含两侧代码行号；指出另两个 RL 框架已修同问题、vLLM 原生修复已合未发版）。https://github.com/huggingface/trl/issues/6789 ；AsyncGRPO 实例（2026-08-28 补）：trl #6945——TRL 无条件给所有 vLLM server 强加 `--logprobs-mode processed_logprobs`（理由只适用于有 IS 修正的 trainer），而 `AsyncGRPOTrainer` **没有** IS 修正，直接拿 rollout logprob 当 PPO 分母 `old_log_probs`（`async_grpo_trainer.py:1113`）。同策略未动时 `coef_1` 即读出幸存质量 S：top_p=0.9→0.9、0.8→0.8、0.7→0.7（落入 clip 外），截断越强偏差越大，负优势 token 的 clip 每步都误触发；且 `raw_logprobs` 是**任何处理前**（温度/惩罚之前）的分布，换模式只是换成温度错配（评论区核 vLLM `v1/sample/sampler.py` 证伪「换 raw 即修」）。#6944 加了警告但未改 flag。https://github.com/huggingface/trl/issues/6945
- **行为效应**：无 NaN、无 clip 迹象；IS ratio 整体偏移，off-policy 校正系统性失真。与 RL-RO.01（数值核差）正交：这是**分布定义不一致**，同权重同 kernel 也存在。top_p<1 / top_k>0 / min_p 设了就中。
- **发现来源**：2026-08-19 每日扫描

### RL-RO.08 `adapter_omitted_from_weight_sync`

stage: `rl` · Cov: `var:moe_router_not_replayed` · 置信度 `documented`

- **机制**：verl disaggregated trainer/rollout（`rollout.checkpoint_engine.backend != "naive"`，即所有 async RL recipe）+ LoRA `merge=False`（默认）时，**每次权重同步推的都是冻结的 base model，LoRA adapter 从不传输**。rollout 引擎整个 run 从初始策略采样；训练侧为当前策略算 policy gradient，数据却永远来自初始策略——RL 循环被静默切断，mismatch 随训练单调增长。colocated 路径（`backend: naive`）正确：同配置、不同 mode。
- **来源**：verl #7495（生产 verl 0.8.0：3 trainer + 1 rollout GPU、fully_async、Qwen3-4B + LoRA r32；0.8.0 与 main 相关代码 byte-identical）。https://github.com/verl-project/verl/issues/7495
- **行为效应**：训练「正常完成」——loss 降、reward 动、grad_norm/entropy/ESS 都看着合理、无任何警告。行为上是 RL-RO.01 的**极端形式**（rollout↔train 错配不随 off-policy 校正收敛，而是结构性钉死在 π₀），但机制上不是数值核差，是**同步载荷不完整**：训练的 adapter 根本没进同步路径。RL-RO.05 的对偶：那边是「该回放的不回放」，这边是「该同步的不同步」。
- **发现来源**：2026-08-21 每日扫描

### RL-RO.09 `fused_expert_view_frozen_at_capture`

stage: `rl` · Cov: `var:adapter_omitted_from_weight_sync` · 置信度 `documented`

- **机制**：Megatron colocated RL（trainer 与生成引擎同进程）里，serving 侧此前**在捕获时另行 fuse 了一份 expert 张量**；distributed optimizer 之后更新的是自己的参数 buffer，那份独立 fuse 的张量值**冻结在捕获时刻**——生成引擎一直用陈旧专家权重采样，训练侧却在更新当前策略。修复：DDP 构造前把 per-expert 权重分配为相邻 view 并在参数 buffer 中保持无间隙布局，serving 可恢复 zero-copy fused view 而不改 ckpt 参数名。
- **来源**：Megatron-LM #6963（PR 自述「causing colocated RL generation to use stale expert weights」；替代被关的 #6931）。https://github.com/NVIDIA/Megatron-LM/pull/6963
- **行为效应**：rollout 与 train 权重静默脱钩——与 RL-RO.08（同步载荷缺 adapter）同行为面：RL 循环被切断、mismatch 随训练增长、无警告；机制不同：不是同步载荷缺项，是**存储别名**——合法 zero-copy 优化下被捕获的 fused 视图不随优化器写回刷新。
- **发现来源**：2026-08-30 每日扫描

### RL-RO.10 `zero3_layout_mutation_not_persisted`

stage: `rl` · Cov: `var:adapter_omitted_from_weight_sync` · 置信度 `documented`

- **机制**：DeepSpeed Hybrid Engine（ZeRO-3）在 rollout 前后做 QKV 布局互转：`HybridMegatronContainer.transform_for_inference()` / `transform_for_training()` 在 `GatheredParameters` 上下文内就地改写 gather 后的 QKV 参数，但**不指定 `modifier_rank`**——ZeRO-3 repartition 时无从广播修改后的全参数，布局转换不被持久化：GPT-NeoX head-interleaved QKV 存活进推理转换，再被 DeepSpeed 推理层按 all-Q/all-K/all-V 连续布局解读。修复 = 两个方向都传 `modifier_rank=0`，让就地转换在全参数态固化后再 repartition。
- **来源**：DeepSpeed #8391（Pythia-410M HE + ZeRO-3 on MI250：原生 HF 首 token `187`、注入 HE 转换前 `39318`；QKV `max_abs=20.5625`/`mean_abs=1.115`，首个不匹配在 layer-0 attention；修复后 `max_abs=0`、首 token 一致、20 步 `eval()↔train()` 布局往返稳定；附回归测试）。https://github.com/deepspeedai/DeepSpeed/issues/8391 ；修复 PR #8392（2026-09-04 补）。https://github.com/deepspeedai/DeepSpeed/pull/8392
- **行为效应**：rollout 用**布局损坏的权重**采样，训练侧重算的 logprob 与真实采样分布脱节且无 crash、无警告；表象易被归因为「RL 训坏了」。与 RL-RO.08/09 同族（rollout↔train 权重静默脱钩），但根因是**参数 repartition 协议缺 `modifier_rank`**：不是同步载荷缺项、也不是别名冻结，而是合法就地转换在 ZeRO-3 分片协议下不被写回。与 CKPT.02（ZeRO 分片图与加载布局错位）同域，但发生在训练循环内的引擎切换而非 ckpt 加载。
- **发现来源**：2026-09-03 每日扫描

### RL-RO.11 `decode_reencode_drifts_trajectory`

stage: `rl` · Cov: `var:tim_rollout_trainer_logprob_delta` · 置信度 `documented`

- **机制**：agentic 多步 rollout 把引擎采样的 assistant token ids **decode 成文本**，构造后续推理历史或训练输入时再 **re-encode**。tokenization 非双射（特殊 token 拼接、空格归并、工具观察插入、确定性前缀、多模态 chunk 重建处均可漂移），round-trip 后的 token 序列 ≠ 采样序列。RL 里 token 轨迹是训练样本本体：policy 被按**它没有产生过的序列**计算 logprob/优势并更新。泄漏点横跨 colocate/server 多轮 rollout、scheduler hook（要求文本形态）、agentic 续写、多模态 chunk 重建。修复 = 引擎采样的 token ids 全程作为 source of truth，文本仅在 scheduler hook 需要时临时物化；对确定性前缀/scheduler 追加内容用对齐的 loss mask 表达。
- **来源**：ms-swift PR #10012（引 verl agent-loop 文档的 chat-completion vs token-in/token-out 区分；覆盖 scheduler 变更、工具观察、确定性 response prefix、agentic 续写、多模态 chunk 重建五个泄漏点的回归测试）。https://github.com/modelscope/ms-swift/pull/10012
- **行为效应**：无 crash、无 NaN；logprob/IS ratio 带一个结构性 bias（训练序列不是采样序列），轨迹越长、工具轮次越多偏得越大，off-policy 校正永远补不齐。与 RL-RO.01（数值核差）正交：权重和分布都对，错的是**序列恒等性**——同一 token 串在采样侧与训练侧是两个不同的 int 序列；与 RL-RWD.06（RM 吃错位 token）同根因（decode/re-encode 非双射），那边污染 reward 输入，这边污染训练轨迹。
- **发现来源**：2026-09-04 每日扫描

### RL-RO.12 `stale_completion_rolls_into_next_weight_version`

stage: `rl` · Cov: `var:adapter_omitted_from_weight_sync` · 置信度 `documented`

- **机制**：verl Mooncake 传输引擎的权重同步在 buffer 复用跨 weight version 时，旧 completion marker 可在**下一个接收者读完新内容之前释放 buffer**：`send_weights()` 只消费最后一个 bucket 的 slot，中间转发 rank 的末尾下游 slot 从不被消费，update 间 barrier 也不清 marker——READ/WRITE 全部返回 success，接收 rank 读到的却是**上一版本的张量**（复现：rank 2 tensor 0 应为 2.0、实际读到 tensor 8 的 2.125）。修复 = 每次 update 返回前 drain 全部在途 completion slot（sender 与转发 rank 两侧）。
- **来源**：verl PR #7764（双主机 8×16MiB bucket 双版本原始代码 fail、H20 BF16 组件级验证；与 #6813「marker 存储位置」修复互补，本条修的是**消费时序**）。https://github.com/verl-project/verl/pull/7764
- **行为效应**：传输 API 全绿（success 返回值），训练侧用部分陈旧权重 rollout——采样分布与正在训练的 policy 静默错位，TIM/IS 校正全部失准。与 RL-RO.08（adapter 漏同步）同族：训练→rollout 权重同步契约不完整；本条错在**时序**（旧版本残留）而非**内容缺失**，且 failure 被 transfer 层的 success 语义掩盖。
- **发现来源**：2026-09-06 每日扫描

### RL-RO.13 `teacher_engine_never_receives_real_weights`

stage: `rl` · Cov: `var:from_pretrained_silent_random_init` · 置信度 `documented`

- **机制**：verl 多教师 OPD（蒸馏）经 sparse Hydra override 构造 `DistillationTeacherModelConfig` 时，缺省的 inference 字段从 `RolloutConfig` 继承默认 `load_format="dummy"`（单教师 YAML 路径的值是 `"auto"`）；actor rollout 引擎后来有 weight sync 纠正 dummy 加载，**teacher 引擎没有任何后续同步**——受影响教师在整段训练中持续用随机初始化权重出蒸馏信号。修复 = teacher 缺省 `load_format="auto"` + 显式 `dummy*` 一律拒绝启动。
- **来源**：verl PR #7742（CPU 回归测试；文档补写 teacher/actor 两种加载生命周期差异）。https://github.com/verl-project/verl/pull/7742
- **行为效应**：蒸馏 loss 正常计算、正常下降——学生被拉向**随机教师**的输出分布；无 crash、无警告，唯一线索是蒸馏收益远低于预期。与 CKPT.03（missing key 随机初始化）同下游（随机权重当真值用），机制在**配置继承把占位加载格式带进无自愈路径**；与 RL-RO.08（LoRA adapter 漏同步）互补：那边同步了部分权重，这边整个引擎从未拿到真权重。
- **发现来源**：2026-09-06 每日扫描

### RL-RO.14 `rollout_context_compression_conditions_diverged_history`

stage: `rl` · Cov: `var:decode_reencode_drifts_trajectory` · 置信度 `documented`

- **机制**：生产 agent harness（Claude Code / Qwen-Agent 类）在 rollout 中**压缩上下文**（evict 旧轮次），训练对象因此不是序列而是**树**——每次 eviction 都把有效历史分叉。现有线性化都不保真：保留最右路径 → time-travel leakage（被训的条件里有 rollout 时已不存在的历史）；depth-first 重放 → 与生成时条件序不一致的 train-inference mismatch。论文给出两个梯度等价的精确修正（LogitTree：分段 K-forward 遍历，需 K+1 次 backward；packed 4D attention mask：需自定义 kernel 与白盒 eviction 记录）+ 单 backward 的变分松弛 SDCC（eviction 处对压缩 student 与 stop-gradient teacher 在重建历史上最小化前向 KL）。
- **来源**：MemoryWalker, "Stop Training Agents on Contexts They Never Saw", arXiv:2609.00865（2026-09-01，窗口内）。https://arxiv.org/abs/2609.00865
- **行为效应**：无 crash；agent RL 在「训练条件 ≠ 生成条件」上更新——策略被优化去利用训练侧虚构的完整历史，部署侧（真压缩）行为漂移，且与 RL-RO.01 的数值 TIM 叠加后更难归因。与 RL-RO.11（decode/re-encode 非双射破坏序列恒等性）同族的结构性 TIM：那边 token 序列被改写，这边**条件历史本身**在 rollout 与训练两侧是不同对象；eviction 密度越高（长程 agent 任务）偏差越大。
- **发现来源**：2026-09-07 每日扫描

### RL-RO.15 `fsdp2_to_empty_unties_tied_embeddings`

stage: `rl` · Cov: `NEW` · 置信度 `verified`

- **机制**：verl `fsdp2_load_full_state_dict` 用 `to_empty()` 物化非零 rank 参数——`to_empty()` 给每个参数**全新存储**，tied `lm_head` 对 embedding 的别名就此断开：**无任何失败**（rank 0 的 state dict 里 `lm_head.weight` 是别名、数值仍正确），但 embedding 从此**只收 input 侧那一半梯度**，broadcast ranks 安静地按「untied」训练。该隐藏缺陷正是一个旧 guard 存在的原因：verl 对 `tie_word_embeddings` 模型跳过 FSDP2 meta init（连带引发 host RAM 随 rank 数爆炸，#7834），此前 #5746 提议只删 guard、35 小时后自撤。修复 = meta init 无条件保留 + `to_empty()` 后**重新 tie**。
- **来源**：verl PR #7835（2026-09-10；「(1) without (2) would have silently untied every model this path is meant to help」；61GB tied MoE 8-rank 1437GB/1511GB host RAM OOM 的关联 issue #7834）。https://github.com/verl-project/verl/pull/7835 ；verl #7834（2026-09-13 补：关联 issue 正文补全——Part 2 实测 untied 路径 `embed_tokens.grad norm` 0.3548→0.2271，`set_model_state_dict` 数值仍对、模型照跑、loss 照降；触发面为大且 tied 的 ckpt，North-Mini-Code-1.0 一例）。https://github.com/verl-project/verl/issues/7834
- **行为效应**：无 crash、无指标异常——tied 模型在 FSDP2 权重加载后**静默 untied 训练**：input embedding 与 output head 梯度各走各的，训出的模型与「tied」语义下的预期不可比；ckpt 里 `lm_head.weight` 是否还被当别名处理决定下游加载是否继续错。与 CKPT.10（ZeRO-3 分片后惰性初始化读空 shard）同族：**参数物化时机的别名/绑定契约断裂**；与 OPT.05（优化器遗忘参数）互补：那边参数没进组、这边参数进了组但绑定关系没了；「修复一半比不修更糟」的陷阱面与 verl #7805（R2 replay 半修复回归）同型。
- **发现来源**：2026-09-12 每日扫描

### RL-RO.16 `response_prefix_role_flips_prompt_to_completion`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：ms-swift GRPO 的 `response_prefix`（如 forcing `<think>`）在**生成侧**拼进 prompt——vLLM 从 prefix 之后开始采样，返回的 `generate_ids` **不含** prefix；`Template.decode()` 又把 prefix **拼回**解码文本，整串编码为 `{{RESPONSE}}` 供训练前向——prefix token 的 labels 被置位、落在 `completion_mask` 内。**策略被按它从未采样过的 token 计分**：vLLM 对 N 个 token 算 log-prob，trainer 对 `len(prefix)+N` 个算；模型若认为 prefix 不太可能，这几个 token 就主导序列 log-prob，本应 ≈1 的 `rollout_correction/ppl_ratio`（colocated 严格 on-policy 的健康指标）爆炸。缺陷在 `swift rlhf` 与 `megatron rlhf` 共享的模板层。修复 PR #10107 对齐训练侧 rollout response 前缀。
- **来源**：ms-swift #10106（2026-09-10；`swift/template/base.py` 三处代码定位：生成拼 prompt L1449 / decode 拼回 L803 / 训练整串当 response L1423 + mask 由 labels 派生）。https://github.com/modelscope/ms-swift/issues/10106 ；修复 PR #10107。https://github.com/modelscope/ms-swift/pull/10107
- **行为效应**：无 crash、无 NaN；被污染的每条序列的 IS ratio/logprob 带系统性 prefix 偏置——rollout-train 一致性仪表（ppl_ratio）失真，优势与 KL 都在「部分非采样 token」上计算。与 RL-RO.11（decode/re-encode 破坏序列恒等）同族：**训练序列 ≠ 采样序列**；差异在这边不是 tokenize 非双射，而是 prefix 的 **角色归属**在两侧翻转（prompt vs completion）；与 SFT.12（拆分重复计权）同在「轮次/区段归属」层；与 RL-RO.05（router replay 死键）都属「rollout 侧语义在训练侧未被忠实重放」。
- **发现来源**：2026-09-12 每日扫描

### RL-RO.17 `vllm_encoder_cache_survives_weight_sync`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl `VLLMGeneration.sync_weights` 权重同步后只调 `reset_prefix_cache()` 清 KV prefix cache；但 vLLM 对**多模态模型的 vision encoder 输出**（vision tower + projector 的 image embedding）另有一层 per-image cache——训练含 vision encoder 时，同图复现会命中**旧权重算出的视觉 embedding**，rollout 从陈旧视觉表征采样。issue 提供最小验证：扰动 vision encoder 权重后 logprob 不变（cache 命中），调 `reset_encoder_cache()` 才恢复。训练侧 policy 对新权重前向、rollout 侧前缀却冻结在同步前——同权重两个 logprob 的 TIM 模式在多模态路径的实例。
- **来源**：trl #7198（logprob 不变式 repro：扰动 encoder 权重 → logprobs 逐 token 不动、`reset_encoder_cache()` 后恢复；引用 vllm_generation.py L513-L517）。https://github.com/huggingface/trl/issues/7198
- **行为效应**：无 crash、无 NaN；多模态 RL（如 GRPO 多模态变体）里图像复现样本的 rollout 分布落后训练策略——IS/比率校正、KL 估计在视觉条件上系统性失准，且误差随 weight sync 周期与图像复现率增长。与 RL-RO.12（Mooncake buffer 旧版本滚入下一同步）同族：**同步契约漏了一类缓存/在途状态**；差异在这边是**多模态派生激活缓存**（vLLM 内部、engine 层不可见），不是权重版本本身——`sync_weights` 语义完整性缺陷与 RL-RO.08（adapter 漏传）同型。
- **发现来源**：2026-09-14 每日扫描

### RL-RO.18 `vllm_sleep_resume_corrupts_rollout_weights`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：verl colocated 模式下 `free_cache_engine=true`（**默认值**）让每次 weight sync 前后对 vLLM engine 执行 `sleep(level=1)` / `resume`——sleep 释放 weights/KV 显存、resume 重建。verl 0.8.0 + vllm 0.28.0：从**第二次 `update_weights` 起**，rollout 输出退化为多语言 token 乱码（真实词表 token 的 CJK/西里尔/阿拉伯文混合）；逐 step 归因：gs=0（初始 val）0/60 损坏 → gs=1 0/32 → gs=2 **32/52 全损坏**、最终 greedy val 全 0 分。五重排除链：发送端 merged 权重 vs 手工 HF merge 396 张量 0 超 1e-3；接收端 `_update_weights` 收到的权重与发送端 dump **逐 key max diff 0.0**；vLLM 磁盘/内存加载路径单独重放 + 8 并发两段生成 96 utterances 零损坏；LoRA delta 本身 ~1e-6 无辜；唯一剩余变量 `free_cache_engine`——关掉后同训练 0 损坏、reward 健康（E/A 非零 78%）。损坏发生在 sleep/resume 周期内部（resume 重建的权重 ≠ 刚同步的权重），与 ms-swift PR #7017（GRPO sleep/wake 乱码修复）现象一致。verl 0.9.0 同路径字节级相同、默认仍 true。
- **来源**：verl #7904（2026-09-17；Qwen3-8B bf16 + LoRA r=64 merge=true、GRPO n=4 多轮 agent loop、A100-80GB 单卡；逐步归因表 + 双 config 对照）。https://github.com/verl-project/verl/issues/7904
- **行为效应**：无 error、无 warning——rollout 静默变成乱码、reward 全零，**外部表象 = 「GRPO reward collapse」**（报告者自述被该表象骗了数轮调试）；训练侧 advantage 全 0、策略无学习信号但 job 绿。与 RL-RO.17（encoder cache 漏清）同族「weight sync 后 rollout engine 状态陈旧/损坏」，差异在这边打击 **sleep/resume 生命周期本身**（权重级、非缓存级）、且是默认配置；与 RL-RO.16（verl Mooncake completion slot 不 drain）同为 verl 引擎生命周期管理的静默正确性缺口；「表象是 reward collapse」的伪装面与 RL-RWD 家族诊断陷阱同型。
- **发现来源**：2026-09-18 每日扫描

### RL-RO.19 `weight_sync_ack_precedes_tp_barrier_buffer_reuse`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：ms-swift Ray Megatron 原生 IPC 权重同步：接收侧每个 TP worker 从共享 bucket 拷出张量后 `cuda.synchronize()`（只等**本进程**拷贝）→ driver rank 即刻 socket ACK → **之后**才进 `tp_size>1` 的 TP barrier；发送端 `_flush()` 收到 ACK 就 reset offset 复用**同一块**共享 buffer 写下一桶。合法交错：driver rank 拷完桶 A 发 ACK → 发送端开始写桶 B → 慢的非 driver TP rank 还在读桶 A → barrier 事后完成、撤销不了覆写。发送端不是 TP barrier 参与方、本地 CUDA sync 跨不了进程——不变量「所有 TP rank 读完前不得复用 bucket」被 ACK 时序结构性破坏；张量名/形状合法、值损坏、无同步异常。环境症状：Ray + Megatron GRPO **首次 rollout 后输出损坏**（AMD MI308X/ROCm、Qwen3.6-27B、训练 TP=4/PP=2、rollout 4×vLLM TP=4）；报告者明确标注尚未证明该 race 即根因、修复未端到端验证。
- **来源**：ms-swift #10204（2026-09-19；收发两侧源码交错分析，upstream main `654e24f17` 同序）。https://github.com/modelscope/ms-swift/issues/10204
- **行为效应**：若坐实：无 error——rollout 引擎收到名形合法、值被部分覆写的权重，采样静默跑在混合新旧版本参数上；表象与 RL-RO.18（sleep/resume 损坏 rollout 权重）同型「首 rollout 后输出损坏」，打击面在**传输缓冲生命周期**（复用 vs 重建）。与 RL-RO.12（completion slot 不 drain、旧版本张量滚入下一版本）共享「生产者复用缓冲先于消费者读完」病根；「本地 sync 代替全局同步」的越界与 ACT.07（offload reload 缺排序边）同型。当前定位：机制成立 + 症状在案、因果与数值验证待修复落地。2026-09-20 补：修复 PR ms-swift #10207（`fix(rollout): acknowledge IPC buckets after all TP copies`——ACK 移到 TP barrier 之后）已于 2026-09-20 10:28 UTC merge（merge commit `d7e24a13df6b`），#10204 同刻 closed completed；条目维持 `documented`，等端到端验证报告后评估升 `verified`。
- **发现来源**：2026-09-20 每日扫描

---

# OBS · 观测性

## OBS

### OBS.01 `stability_monitor_env_lost_in_torchrun`

stage: `pretrain` · Cov: `NEW` · 置信度 `verified`

- **机制**：`export STABILITY_MONITOR_ENABLED=1` 经 torchrun 子进程丢失。`_is_enabled()` 读到空就跳过初始化，**不报错**。JSONL 是 0 字节文件。
- **来源**：`opensource_training_megatron` / megatron-lm-training skill。修复：必须传 `--enable-stability-monitor`。
- **行为效应**：目录里「有监控文件」，内容为空。排障的人以为没故障。
- **发现来源**：2026-08-17 基础调研（本地）

### OBS.02 `empty_metrics_read_as_healthy`

stage: `shared` · Cov: `var:stability_monitor_env_lost_in_torchrun` · 置信度 `verified`

- **机制**：监控路径写了文件但 0 字节 / 只有 header，dashboard 当「无报警」。
- **来源**：同 OBS.01。
- **行为效应**：KER.01 一类注入可以整段跑完无人知道。
- **发现来源**：2026-08-17 基础调研（本地）

### OBS.03 `job_green_zero_step_progress`

stage: `shared` · Cov: `NEW` · 置信度 `documented`

- **机制**：调度器/健康检查看进程活着、GPU util 非零（可能在 hang 在集体通信或数据加载），step 计数不涨。训练正确性监控没有新样本。
- **来源**：ByteDance "Robust LLM Training Infrastructure" arXiv:2509.16293 对 implicit failure（hang、轨迹偏离、无明确报错）的分类。https://arxiv.org/abs/2509.16293
- **行为效应**：任务显示 Running，ckpt 停留在旧 iteration。
- **发现来源**：2026-08-17 基础调研

### OBS.04 `zero_signal_run_reports_success`

stage: `rl` · Cov: `var:grpo_group_std_zero` · 置信度 `documented`

- **机制**：全组同分 → `id2std==0` → advantage 全 0 → `actor/grad_norm≈0`，GRPO 整个 run 是 no-op。fit loop 对 grad_norm 只记日志不设护栏：run 正常 exit 0，ckpt 出的模型与初始化相同。advantage 侧机制见 RL-ADV.01 / RL-ADV.05，本条收「观测性骗过」面：零信号 run 报成功。
- **来源**：verl #7405（RFC：opt-in fail-fast detector；引用 #5211 `reward=0, loss=0, grad_norm=0` 真实实例）。https://github.com/verl-project/verl/issues/7405
- **行为效应**：任务显示成功、指标图全平，烧完整轮 GPU 才从图表发现什么都没学。与 OBS.02 / blindspots B.3 同向：有指标、没护栏读它。
- **发现来源**：2026-08-18 每日扫描

### OBS.05 `train_loader_geometry_reused_for_val`

stage: `sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：verl 两个 SFT trainer（`sft_trainer.py` SPMD + `sft_trainer_ray.py`）都用**训练**全局 batch（`data.train_batch_size`，默认 256）构造**验证** loader，且 sampler 与 dataloader 双 `drop_last=True`；配置注释写明 `micro_batch_size_per_gpu: 4  # this is also val batch size`，但该字段在 val 路径从未被读。当 val 集小于每 rank train batch 尺寸时 loader 产出 **0 个 batch**，收尾 `torch.mean(torch.tensor([]))` = NaN。同仓 PPO 的 val loader（`drop_last=False` + 独立 `val_batch_size`）是现成的正确对照。
- **来源**：verl #7464（读 main 复现 + CPU 最小复现；两入口同错；含具体数值表：Ray 路径 200 样本 val × batch 256 → 0 batch → `val/loss=NaN`）。https://github.com/verl-project/verl/issues/7464
- **行为效应**：`val/loss = NaN` 无警告、run exit 0；val 集略大于 batch 时静默丢尾部样本（300 样本 val 丢 5）。与 CKPT.07（死参数/影子参数）的边界：那边配置**声明了但没被消费**，这边字段在 train 侧正常消费、只是 train 的几何参数漏进了 val 构造——错的是 loader 装配而非配置面。观测面与 OBS.02 同型（图表被读成健康），但根因是「验证从未发生」而非空文件。
- **发现来源**：2026-08-22 每日扫描

### OBS.06 `rl_diagnostics_unmasked_padding_positions`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl RL trainer 的**指标路径**与 loss 路径掩码不一致：Online DPO `training_step` 里 loss 用 `cr_logprobs * ~cr_padding_mask`，stats 块却用原始未掩码张量算 `objective/kl`、`non_score_reward`、`rlhf_reward`、`entropy`；PPO 侧 #6121 修了 `objective/entropy` 但优化循环里 `policy/approxkl_avg`、`policy/entropy_avg`、`policy/ratio_avg` 仍在含 padding 的全位置上平均（隔壁 `pg_loss`/`pg_clipfrac` 已用 masked_mean）；XPO 的 `logps/chosen`、`logps/rejected` 干脆把 policy 与 reference logprob **加在一起**上报。所有监控数字系统性偏离真实量。
- **来源**：trl #6809（代码审计；明确「不影响训练 loss、全部 loss 路径掩码正确」，纯诊断面；Online DPO 已有 #6803 跟踪）。https://github.com/huggingface/trl/issues/6809 ；trl #6803（2026-08-24 补，Online DPO 实例独立 issue：`objective/kl`/`entropy`/`non_score_reward`/`rlhf_reward` 对全 completion 位置含 padding 求和；与 PPO 变体的差异——无 `INVALID_LOGPROB` 哨兵，pad 位置贡献普通有限负值而非 +1.0，entropy 单纯膨胀；PPO #6121 与 XPO/Nash-MD 已修，Online DPO 是残留未修副本）。https://github.com/huggingface/trl/issues/6803
- **行为效应**：训练本身健康，但看板上的 KL/entropy/reward 曲线被 padding 拉偏——正是用来发现 RL-KL.01/RL-KL.03 塌缩的那几个探针。观测层骗过而非训练层出错：与 OBS.02（空文件读作健康）互补，这条是**错值读作真相**。
- **发现来源**：2026-08-23 每日扫描

### OBS.07 `eval_replays_training_captured_graph`

stage: `pretrain/sft` · Cov: `NEW` · 置信度 `documented`

- **机制**：训练进程为加速捕获的 CUDA graph 里**冻结了 capture 时刻的执行语义**：Python 分支走向、buffer 突变都在内。普通验证（`model.eval()` + `torch.no_grad()`）本应切到 eval 语义，却仍**重放训练态（grad 模式开启）捕获的 graph**——capture 期执行的 MoE router token-count 累积这类 buffer 写被原样重放进验证前向，eval 指标混入训练态副作用。修复 = eval/no_grad 下回退 eager 执行；显式推理捕获与 ckpt-forward replay 保持原行为。
- **来源**：Megatron-LM PR #6992。https://github.com/NVIDIA/Megatron-LM/pull/6992
- **行为效应**：无 crash、无 NaN；val loss/metrics 被 replay 的训练态 buffer 污染——用被污染的验证曲线做早停/lr 决策/回滚判断，信号源本身就是错的。与 ACT.03（replay 图与训练图分叉）互为镜像：那边训练路径 replay 错范围、这边验证路径 replay 错模式；与 OBS.06（诊断路径错值）同层：训练健康，**观测层拿的是另一个计算的输出**。与 RL-RO.09（fused view 冻结在捕获时刻）共享「capture 冻结旧语义」机制，但冻结的是控制流/buffer 而非权重。
- **发现来源**：2026-09-04 每日扫描

### OBS.08 `loss_path_metrics_weighted_by_window_size`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：trl `GRPOTrainer` 把 loss-path 指标（`policy_loss`、`entropy` 等）**每个 micro-batch append 一次**，`log()` 对 logging interval 内全部条目取平摊平均——而 transformers 5.11 会把 epoch 最后一个 accumulation window 截短到 remainder（`current_gradient_accumulation_steps` = remainder）：长窗贡献 3 条、短窗贡献 1 条，flat mean 按 3:1 加权，与 per-optimizer-window 平均**差 2 倍**（实测 `policy_loss` logged −0.000557 vs window-mean −0.001114）。基础 `loss` 不受影响，偏差只打击 RL 诊断指标。同窗口内另有一层：`policy_loss` 的 capture 时点随 `loss_type` 分支而异——`grpo/bnpo/dr_grpo/sapo/luspo` 在 rescale 前 capture（window 量级），`cispo/dapo/vespo` 在 normalizer 折入后 capture（micro-batch 量级），两种量级差 `grad_accum / steps_per_generation` 倍，代码与文档均未声明。
- **来源**：trl #7047（4-micro-batch / gas=3 受控实测，两种聚合方式对照）+ 同族 #7011（八种 loss_type 的 capture-scale 对照）。https://github.com/huggingface/trl/issues/7047 ；https://github.com/huggingface/trl/issues/7011 ；修复 PR #7053/#7042（optimizer-window 粒度统一记录）+ #7066（per-token entropy 指标修复顺带）。
- **行为效应**：无 crash；RL 训练的 policy_loss/entropy 曲线系统性错值——跨 run 对比、跨 loss_type 对比、与 loss 交叉验证全部给出错误答案，且错值方向与 epoch 尾窗相关（不均态、随数据量波动）。与 OBS.06（RL 诊断不掩 padding）同族：训练本体健康，诊断层系统性错值；差异在机制是**聚合粒度错位**而非掩码遗漏。
- **发现来源**：2026-09-05 每日扫描

### OBS.09 `gathered_mean_metrics_double_count_padding`

stage: `sft/rl` · Cov: `var:loss_path_metrics_weighted_by_window_size` · 置信度 `documented`

- **机制**：`Accelerator.gather_for_metrics` 靠沿 batch 维切片去掉 accelerate 给最后一批补的 padding 去重——**rank-local 标量已经没有 batch 维**，重复样本折进数字里后截断无从剥离。数据集不整除 rank 数时，每个 epoch 的最后一批上，所有「先本地 reduce 再 gather」的指标都带这份偏差。trl 修复一次性扫描出六个 trainer 的 11 处（RewardTrainer accuracy/margin、XPO log-probs/KL/entropy/margins、OnlineDPO scores/KL/RLHF reward、CPO/ORPO logits means、TPO per-token entropy……）；同日姊妹 PR 把 PPO micro-batch 指标的「槽位均值」改成按 token/样本数加权池化，顺带修掉 unwritten 槽位零值污染（`val/ratio_var` 在全 1 ratio 下 log 出 0.2667 方差）。
- **来源**：trl PR #7045（fixes #7010）+ 姊妹 PR #7041（fixes #7012，closed 后并入后续修复）。https://github.com/huggingface/trl/pull/7045
- **行为效应**：无 crash；每个 epoch 尾批的评测指标（accuracy、KL、margins、entropy……）系统性偏移，尾批占比越大偏得越狠——小数据集/大 world_size 的实验读数最受污染，恰好是调参决策最依赖看板的场景。与 OBS.08 同族（诊断层聚合错值），机制互补：那边错在**时间维**（micro-batch 条目数加权），这边错在**空间维**（跨 rank gather 的 padding 去重对已聚合标量失效）。
- **发现来源**：2026-09-06 每日扫描

### OBS.10 `tp_replicated_tokens_summed_in_metrics`

stage: `sft/rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：TP 下每个 rank 收到**同一份** batch，但 trl 的 `num_tokens` 指标对全部进程求和——每个 token 被计 `tp_size` 次。`Trainer` 基类对 `num_input_tokens_seen` / `num_items_in_batch` 已做 TP 除法，trl 自建路径漏了同一处理。EP 同理（每 EP rank 见全量 token）。最小复现：`tp_size=2` 下 `num_tokens` 恰为真实值 2 倍，`tp_size=32` 下 32 倍。
- **来源**：trl #7100（2-rank repro 脚本；期望行为引用 `Trainer.num_input_tokens_seen` 的先例）。https://github.com/huggingface/trl/issues/7100
- **行为效应**：无 crash、训练本体正确；tokens/s、token 预算、数据配方核算（「N token 训完」）全部按 TP 度系统性虚高，wall-clock 效率对比与成本核算错值。与 OBS.09 同族（指标聚合几何与并行几何错配），差异在这边是**复制维**（TP 复制被当独立样本累加）而非 padding 维；弱于 OBS.06/08 一档（不动 loss/KL，只动计数面），但直接影响「训了多少 token」这一最常被引用的 run 属性。
- **发现来源**：2026-09-08 每日扫描

### OBS.11 `synthetic_padding_dilutes_rollout_correction_metrics`

stage: `rl` · Cov: `NEW` · 置信度 `documented`

- **机制**：verl V1 `_step_once` 无条件调用 `_balance_batch`，其 padding 模板设 `response_mask=0`、`is_padding=True`；`_compute_advantage` 把含 padding 行的整批传给 `compute_offpolicy_metrics`，后者算 masked per-sequence 均值后**对所有行取平均**——零有效 token 的 padding 行贡献 logprob 均值 0 / PPL 1，而不是被剔除。下游 `_compute_metrics` 的 `is_padding` 过滤救不回这些已算进均值的指标。padding 行越多，报告的训练-推理 gap（`rollout_corr/log_ppl_abs_diff`、`training_ppl`）显得越小——而每个真实 token 的 logprob 一字未动。
- **来源**：verl #7771（CPU-only 合成张量复现：同一 batch 加 0/1/3 条 padding 行，指标单调漂移；调用链逐行定位 padding_utils / trainer_base / rollout_corr_helper）。https://github.com/verl-project/verl/issues/7771
- **行为效应**：无 crash；RL-RO 诊断面板系统性低估训练-推理偏移，padding 量随 batch 几何波动→gap 指标带非模型因素的伪趋势。与 OBS.09（gathered 均值重复计 padding）同族但方向相反：那边**多算**、这边把真实差异**稀释**；与 RL-RO 条目（TIM 本体）的关系是诊断层：本条不改训练目标，只让最常用来发现 TIM 的仪表失准——观测骗过直接削弱 RL-RO 家族的检测面。
- **发现来源**：2026-09-09 每日扫描

### OBS.12 `missing_dataloader_silently_disables_resampling`

stage: `shared` · Cov: `NEW` · 置信度 `verified`

- **机制**：DeepSpeed ZenFlow 的列重选间隔按 `'epoch'` 策略表达（`'auto'` 默认解析到它），需要 epoch 步数——从 `engine.training_dataloader` 读，而该属性**只有调用方把 dataloader 交给 `deepspeed.initialize()` 时才非空**；自管 dataloader（常见路径）走 else 分支 `select_interval = 0`，`is__zenflow_select_boundary()` 把 0 读作「永不重选」：**第 0 步选一次、整个 run 不再换列**。无 error、无 warning，默认配置即中招。40 步对照：默认 `auto`（无 dataloader）重选 1 次 vs 正确路径 4 次。掩体一：**全部既有 ZenFlow 测试都跑在这条错路径上**（测试在 init 之后自建 dataloader），断言「不 crash」恒绿；掩体二：`raise Warning(...)`——`Warning` 是 `Exception` 子类，本想 warn 的分支实际**直接抛异常**，紧随其后的 fallback 行不可达（engine.py / engine_stage3.py / 第三处共三份拷贝）。
- **来源**：DeepSpeed #8456（2026-09-08；间隔解析表、`raise Warning` 语义坑、测试盲区三段式分析）。https://github.com/deepspeedai/DeepSpeed/issues/8456
- **行为效应**：无 crash、无日志——自管 dataloader 的 ZenFlow run 用**第一步选出的列**训完整个 run：重要性列从不轮换，训练动态与配置语义（按 epoch 重选）背离；`auto`+整数 interval 组合还会在初始化期以一条「描述 fallback 的报错」崩掉，掩盖真正语义。与 OBS.02（空指标当健康）同族但更隐蔽：**指标全正常、行为整体退化**；「测试断言不 crash 而非断言语义」的掩体与 OBS.07/KER.10 的假绿同型；与 CKPT.07（配置被静默忽略）的差异：这边不是死配置，是**配置语义依赖一个未传入的运行时对象**，签名上完全合法。
- **发现来源**：2026-09-12 每日扫描

