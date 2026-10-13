# 人工形式检查清单

只检查格式和写法，**不需要判断技术内容是否正确**（正确性已由独立裁定和忠实性审计负责）。
每条在不符合的项后标 ✗：

- A 标题是一句话、能看懂在说什么
- B 原因行为只写缺陷本身，没有混入后果
- C 影响详情只写外部表现和触发条件，没有重复原因
- D 原因对象/影响对象的取值看起来与文字描述相称（凭字面判断即可）
- E 没有乱码、截断、中英混排失控等形式问题
- F 有来源链接或文字引用

### TRN-0128（旧号 ACT.07）激活 offload inline reload 缺流排序读到陈旧数据

- 原因对象：重计算/卸载 / megatron-lm / tensor_pop inline reload
- 原因行为：ChunkOffloadHandler.reload() 的 H2D 拷贝与 D2H 的 _offload_event 无排序边；tensor_pop 的 inline reload 未补此边，可在 D2H 写 buffer 时读取。
- 影响对象：梯度、隐状态（静默错误）
- 影响详情：无报错、loss 照降，warmup 迭代及 backward 跑赢 prefetch 时重载的激活是新旧内容混合，梯度静默算在陈旧激活上；B300 上 20 次试验 19 次损坏，现有测试丢弃 warmup 迭代而漏检。
- 来源：https://github.com/NVIDIA/Megatron-LM/issues/7398；https://github.com/NVIDIA/Megatron-LM/pull/7399
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0050（旧号 CKPT.01）专家重分片或排序约定不一致，router expert-id 指向错专家权重

- 原因对象：权重/检查点 / deepspeed / 专家权重重分片与排序映射
- 原因行为：以不同于保存时的 EP degree 加载、num_experts % ep_size != 0，或 Megatron↔HF 专家排序约定不同时，router 的 expert-id 指向另一组专家权重。
- 影响对象：模型权重、输出内容（质量退化）
- 影响详情：加载成功无报错；MoE 模型输出流畅但能力像换了一批专家，router 熵可能仍看起来健康。
- 来源：https://github.com/deepspeedai/DeepSpeed/issues/5794
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0084（旧号 DATA.02）packing 未按文档边界切断 attention，后一条样本可见前一条

- 原因对象：数据管线 / 通用 / packing 文档边界 mask 与位置 id
- 原因行为：把多条样本 pack 进同一序列以提高利用率时，attention mask 与 position id 没有按文档边界切断。
- 影响对象：跨请求/跨样本数据、训练轨迹（质量退化）
- 影响详情：后一条样本的 token 能看见前一条的内容，模型学会“抄下一条的答案”；验证集单条样本分数虚高，真实对话掉点。
- 来源：（引用）SFT packing 的标准正确性陷阱；
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0001（旧号 HW.01）训练 SDC：不健康节点把模型推向另一局部最优

- 原因对象：硬件 / 硬件 / 生产集群不健康节点的计算单元
- 原因行为：被自动化清扫出的不健康节点仍在跑训练，其静默数据损坏扰动子模块计算与单次 optimizer step，且故障往往不在矩阵乘单元。
- 影响对象：梯度、训练轨迹（静默错误）
- 影响详情：无 Xid、无 NaN 也能训下去，权重轨迹分叉进入另一局部最优，最终 ckpt 与健康节点同配置不同权重；部分节点打出 loss spike，微调阶段可见不同幅度 spike；ABFT 对 GEMM 的覆盖大多失败。
- 来源：https://arxiv.org/abs/2502.12340
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0133（旧号 KER.02）FA backward 的 O 与 dO 是两个不同注入点

- 原因对象：算子/内核 / 通用 / FA backward O 与 dO 注入点
- 原因行为：同一 FA backward kernel 上 O 与 dO 两处低精度注入机制不同：O 经 softmax_d 间接污染梯度，dO 直接进入 backward GEMM；bitshift 与 round2nearest 环境变量互斥。
- 影响对象：梯度、训练轨迹（质量退化）
- 影响详情：同样剂量下 O 与 dO 两个注入点的谱与鞅时钟曲线不同，对照实验不声明注入点则结果不可比。
- 来源：（引用）同 KER.01；
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1229（旧号 K1.1）vLLM cascade 注意力漏传 sink，softmax 分母错误

- 原因对象：算子/内核 / vllm / flash_attn cascade 路径
- 原因行为：cascade attention 的共享前缀 pass 与后缀 pass 都没有传入学到的 sink 张量，经 merge_attn_states/LSE 合并后的 softmax 分母缺少 sink 项
- 影响对象：数值结果、输出内容（质量退化）
- 影响详情：开启 prefix caching 走 cascade 路径且模型带 attention sink（GPT-OSS）时出现：GPT-OSS-20B 的 MMLU-Pro 约 0.68→0.58，并出现退化的重复循环；非 cascade 路径正常
- 来源：https://github.com/vllm-project/vllm/issues/26203
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1342（旧号 K10.31）Gemini 显式上下文缓存不追踪 GCS 对象变更致用旧 KV

- 原因对象：KV缓存 / gemini-api / 显式上下文缓存的缓存 id 失效
- 原因行为：显式上下文缓存按 TTL（默认 60 分钟）存储内容的 KV，缓存 id 不追踪可变内容；TTL 内源 GCS 对象被改动后，缓存资源仍指向由过期字节算出的 KV。
- 影响对象：KV状态、输出内容（静默错误）
- 影响详情：用户更新源文件后继续提问，回答仍基于编辑前版本，最长持续一个 TTL 且无任何提示；隐状态完全健康，只有对比一次全新重算才能检测。
- 来源：（引用）来源：ai.google.dev/gemini-api/docs/caching（原文警告）。
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1232（旧号 K2.2）SlidingWindowCache 守卫差一，提前 roll 留下零值槽

- 原因对象：KV缓存 / transformers / HybridCache 滑窗更新守卫条件
- 原因行为：_sliding_update 守卫 cache_position >= max_cache_len - 1 在 seq_len 首次达到窗口时提前一步触发，缓存未满就被 torch.roll 平移，旧 token 被移出并留下未初始化零槽
- 影响对象：KV状态、输出内容（质量退化）
- 影响详情：长上下文解码中 seq_len 越过 SWA 窗口后，窗口边界附近的注意力被破坏，零值 K/V 槽与错位一格的旧 token 使生成持续劣化
- 来源：https://github.com/huggingface/transformers/issues/37574
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1244（旧号 K3.7）TRT-LLM 强开 INT8 KV 复用，约 3000 请求后 SIGABRT

- 原因对象：KV缓存 / tensorrt-llm / paged-context INT8 KV 路径
- 原因行为：KV 复用依赖不支持 INT8 KV 的 paged-context FMHA（有断言守卫）；强行启用后，该 INT8 路径在复用期间错误处理量化块布局与 scale
- 影响对象：KV状态、可用性/进度（报错）
- 影响详情：强行启用后能运行，约 3000 个请求后在 GPTAttentionPlugin::enqueue 中 SIGABRT 中止；源文档称崩溃前复用块的 INT8 反量化给出粗糙的 KV
- 来源：https://github.com/NVIDIA/TensorRT-LLM/issues/2166；https://github.com/NVIDIA/TensorRT-LLM/issues/2112
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1248（旧号 K4.1）vLLM 前缀缓存哈希用 lora_name，同名适配器复用他人 KV

- 原因对象：KV缓存 / vllm / LoRA 前缀缓存额外哈希键
- 原因行为：_gen_lora_extra_hash_keys 用 lora_name 而非唯一的 lora_int_id 生成块哈希，LoRARequest 的 __eq__/__hash__ 也只看 lora_name，同名不同权重适配器哈希相同
- 影响对象：KV状态、跨请求/跨样本数据（静默错误）
- 影响详情：前缀缓存开启且同名适配器被热切换/重载后，适配器 B 的请求命中 A 权重下算出的 KV 块，输出语法正常但内容偏向另一适配器，无一致性校验报警
- 来源：https://github.com/vllm-project/vllm/issues/30931；https://github.com/vllm-project/vllm/issues/42125
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1259（旧号 K5.3）PD 分离 decode 端页复用早于 KV 落地致读到过期 KV

- 原因对象：KV缓存 / sglang / decode 端 KV 页释放与分配
- 原因行为：（源 issue 未确认的假设机制）decode 端经 Mooncake RDMA 收 KV，已完成请求释放的页可能在新请求入站传输落地前分配给它，调度器读到回收页上的过期字节；bootstrap_room 元数据匹配，故无告警。
- 影响对象：KV状态、输出内容、跨请求/跨样本数据（质量退化）
- 影响详情：输出随请求计数渐进坍缩为乱码：准确率从第 1–50 个请求约 94–96% 降到第 151 个以后 25–41%，损害跨请求累积；重启 PD 可恢复。
- 来源：https://github.com/sgl-project/sglang/issues/23020
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1269（旧号 K6.4）Ada-KV：逐头均匀 KV 预算致弥散头被饿死

- 原因对象：KV缓存 / 通用 / KV 驱逐的逐头预算分配
- 原因行为：驱逐方法把一层的预算在各注意力头间均匀切分，忽略各头注意力集中度的差异，使稀疏头被过度服务、弥散头被饿死。
- 影响对象：KV状态、输出内容（质量退化）
- 影响详情：原文按机制推断：同等总预算下注意力输出偏离全注意力，损失主要落在弥散头上，表现为静默的长上下文质量下降而非崩溃。
- 来源：https://arxiv.org/abs/2407.11550
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1276（旧号 K7.1）vLLM ROCm 后端前缀缓存命中与重算结果不一致

- 原因对象：算子/内核 / vllm / ROCm 注意力后端缓存 KV 归约
- 原因行为：cache-HIT 路径加载缓存前缀 KV、只算新 token，部分后端对缓存 KV 的归约以与 cache-MISS 全序列重算不同的顺序/精度累加，两条路径数值不等价。
- 影响对象：数值结果、输出内容（不可复现）
- 影响详情：同一 prompt 在缓存命中与未命中两次运行下输出不同，偏离最早在位置 3 翻转 argmax 后级联放大，无报错；仅 AMD MI355X/MI325X 上的 TRITON_ATTN、ROCM_AITER_FA 受影响，CUDA 不受影响。
- 来源：https://github.com/vllm-project/vllm/issues/33123
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1279（旧号 K8.2）推测解码 bonus token 未写入 draft KV 致上下文滞后一位

- 原因对象：解码/采样 / vllm / 推测解码 bonus token 处理
- 原因行为：提议全部被接受时会从 verifier 采样 bonus token，但对自持 KV 的 proposer（draft model/MTP），该 token 的 K/V 从未在 draft 中生成，draft KV 比被接受序列少一项。
- 影响对象：KV状态、输出内容（静默错误）
- 影响详情：draft 与 target 上下文永久错位，后续每次提议都基于落后一位的上下文，接受率下降、输出静默偏移；仅在提议全部被接受且 proposer 自持 KV 时触发。
- 来源：https://github.com/vllm-project/vllm/issues/4212
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1311（旧号 K9.20）vLLM CUDA graph 在完整缓存命中时跳过 KV 传输同步

- 原因对象：运行时/执行 / vllm / 逐层异步 KV 传输同步
- 原因行为：完整缓存命中场景下 CUDA graph 重放跳过了 KV 传输同步，一个逐层异步 KV 操作未被等待，decode 读取尚未传输完成的 KV。
- 影响对象：KV状态、输出内容（不可复现）
- 影响详情：跨节点 KV 传输且完整缓存命中时偶发输出错误，损坏集中在命中前缀段、后缀正常；因是竞态，复现依赖时序，服务无报错。
- 来源：https://github.com/vllm-project/vllm/issues/29608；https://github.com/vllm-project/vllm/pull/31057
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0173（旧号 LOSS.09）ms-swift DAPO 族 loss 在梯度累积下被二次除以 G

- 原因对象：损失计算 / ms-swift / GRPO dapo 族 loss 归一化
- 原因行为：dapo/cispo/fipo loss 已按整个梯度累积窗口的 completion token 数归一，Trainer.training_step 在 backward 前又除以梯度累积步数且无补偿，形成二次缩放。
- 影响对象：梯度、训练信号（静默错误）
- 影响详情：loss_type=dapo 且梯度累积步数 G>1 时 loss 与梯度被缩小 G 倍（G=4 时更新量为 G=1 的 1/4），有效学习率随 G 漂移、日志 loss 也被除以 G；无 crash、无 NaN。grpo/bnpo/dr_grpo 不受影响。
- 来源：https://github.com/modelscope/ms-swift/issues/10117
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0164（旧号 MOE.15）trl SFT 的 MoE aux loss 被卡数与累积步双重放大

- 原因对象：损失计算 / trl / SFTTrainer 的 aux loss 缩放
- 原因行为：compute_loss 把微批级 aux loss 并入 token 归一的 CE 后再乘 num_processes：aux 项未除以卡数；用 num_items_in_batch 时又不除以累积步，跨微批被求和。
- 影响对象：训练信号、梯度（静默错误）
- 影响详情：无 crash；aux 项放大 num_processes × gas 倍，router 更新实测 1 GPU 为 4.94、2 GPU 为 9.88、2 GPU+gas2 为 19.76，修复后均为 4.94；aux 系数效果随并行配置漂移。DPO/KTO 不受影响。
- 来源：https://github.com/huggingface/trl/pull/7570
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0020（旧号 NUM.11）CPU offload 下 RNG tracker 设备错配致随机值重复

- 原因对象：运行时/执行 / pytorch / DTensor RNG tracker 设备谓词
- 原因行为：DTensor RNG tracker 以 CUDA 设备 fork_rng 包裹随机操作；CPU offload 下 mesh 仍为 CUDA 而存储在 CPU，CPU 随机操作被错路由，退出时无条件恢复 CPU 生成器状态。
- 影响对象：数值结果、模型权重、可用性/进度（静默错误）
- 影响详情：FSDP2 开 CPU offload 时触发：offload 后的 CPU 随机操作产生完全相同的值，初始化/dropout 等去相关失效，训练期无 crash、无 NaN；torchtitan 中 lm_head 的 trunc_normal_ 反复得到相同越界值，初始化挂死。
- 来源：https://github.com/pytorch/pytorch/pull/196072；https://github.com/pytorch/torchtitan/issues/4511
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0263（旧号 OBS.14）DeepSpeed autocast fp16 ZeRO-0 溢出跳步记为已应用

- 原因对象：优化器 / deepspeed / _take_model_step 溢出探测
- 原因行为：_take_model_step 只查 optimizer.overflow 属性探测溢出，该路径为裸 torch.optim、无此属性，overflow 恒 False；GradScaler 已跳过 step，engine 仍记已应用。
- 影响对象：优化器状态、训练轨迹、指标/日志（静默错误）
- 影响详情：溢出步 was_step_applied() 为 True、skipped_steps 恒 0、lr_scheduler 照常推进而参数未变；GradScaler 默认 init_scale 65536 使前几步常溢出，warmup 被静默烧掉、schedule 相位前移。fp16 配置块路径正常。
- 来源：https://github.com/deepspeedai/DeepSpeed/issues/8723
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0104（旧号 OPT.04）全局 clip_grad_norm 把损坏梯度重缩放回阈值，抹掉幅值证据

- 原因对象：优化器 / megatron-lm / 全局 clip_grad_norm
- 原因行为：clip_grad_norm 只看整体 L2 范数，把被抬高范数的损坏梯度（如 PAR.05 的 2^N 倍 expert 梯度）按 clip_coef 重缩放回阈值，不校验方向与逐参数幅值，不触发 skipped iteration。
- 影响对象：梯度、模型权重（静默错误）
- 影响详情：clip 开启时 grad_norm 图正常、skipped iterations=0（实测 ×2^16 expert 梯度被重缩放）；N≤~64 时幅值假的更新持续写入权重，N=128 时范数溢出为 inf、clip_coef≈0，run 零梯度空转。
- 来源：https://github.com/NVIDIA/Megatron-LM/issues/6660
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0046（旧号 PAR.18）DeepSpeed ZeRO ckpt 重建缺别名子模块参数

- 原因对象：权重/检查点 / deepspeed / _get_shared_params 子模块遍历
- 原因行为：_get_shared_params() 用按对象身份去重的 named_children() 遍历，子模块被第二个属性名别名时其子树不被访问、参数不记为 shared，与 state_dict() 的遍历结果不一致。
- 影响对象：模型权重（静默错误）
- 影响详情：ckpt 无错写出；zero_to_fp32.py 等重建的 state_dict 缺别名子模块（如 self.layer2 = self.layer1）参数；之后 load_state_dict 报错，或容忍 missing keys 时静默加载成不完整模型。
- 来源：https://github.com/deepspeedai/DeepSpeed/pull/8686
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0205（旧号 RL-ADV.03）先序列内 token 平均的 loss 归约压小长序列梯度

- 原因对象：损失计算 / 通用 / GRPO 策略 loss 的 token 归约
- 原因行为：每条序列先对 token 取平均再进组聚合，而非按全局 token 数归一
- 影响对象：梯度、训练轨迹（静默错误）
- 影响详情：长序列中每个 token 的梯度贡献被压小，形成与长度相关的偏置，与 RL-RWD.02 同族但落在 loss 归约而非奖励设计
- 来源：（引用）DAPO；
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0215（旧号 RL-KL.02）OpenRLHF reverse-KL 损失的梯度与 IS 权重写错

- 原因对象：RL训练回路 / openrlhf / reverse-KL 损失及 IS 权重
- 原因行为：把 reverse-KL 当损失时，梯度形式和 off-policy 重要性权重写错，更新指向错误的散度
- 影响对象：梯度、训练轨迹（静默错误）
- 影响详情：KL 项在下降但策略并不靠近参照策略，或反过来
- 来源：https://github.com/OpenRLHF/OpenRLHF/pull/1309
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0233（旧号 RL-RO.11）rollout 的 decode/re-encode 使训练序列偏离采样

- 原因对象：RL训练回路 / ms-swift / rollout decode/re-encode
- 原因行为：多步 rollout 把引擎采样的 assistant token ids decode 成文本，构造后续历史或训练输入时再 re-encode；tokenization 非双射，round-trip 后的 token 序列不等于采样序列。
- 影响对象：输入编码、训练信号（静默错误）
- 影响详情：无 crash、无 NaN；logprob/IS ratio 带结构性偏置，策略按自己未产生过的序列更新，轨迹越长、工具轮次越多偏差越大，off-policy 校正补不齐；漂移点含特殊 token 拼接、空格归并、工具观察插入、多模态 chunk 重建。
- 来源：https://github.com/modelscope/ms-swift/pull/10012
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0198（旧号 RL-RWD.05）模板预填 <think> 使 think_format_reward 恒为 0

- 原因对象：RL训练回路 / trl / think_format_reward 锚定正则
- 原因行为：模板在 generation prompt 末尾预填 <think>，GRPO 送 reward 的 completion 只含生成 token，而 think_format_reward 的锚定正则要求以 <think> 开头。
- 影响对象：训练信号、训练轨迹（静默错误）
- 影响详情：使用预填模板（如 DeepSeek-R1-Distill）时该 reward 通道全程为 0.0，无报错无警告；若组内其他 reward 也无区分则 advantage 全 0。reward 恒 0 易被误读为模型没学会格式。
- 来源：https://github.com/huggingface/trl/issues/6966
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### TRN-0192（旧号 SFT.15）ms-swift 显式 CE loss_type 丢弃 token 权重

- 原因对象：损失计算 / ms-swift / CustomCrossEntropyLoss
- 原因行为：Seq2SeqTrainer 把 token weights 传给 CustomCrossEntropyLoss，但回调重算 CE 时丢弃 weights，loss_scale 未在 reduce 前作用于 per-token loss。
- 影响对象：训练信号、梯度（静默错误）
- 影响详情：显式 --loss_type cross_entropy 时用户配置的 token 权重（如 last_round、工具位降权）不生效，按均匀权重训练，零权重 token 也计入 loss 与梯度；默认 trainer 路径正常。静默、无报错。
- 来源：https://github.com/modelscope/ms-swift/pull/10100
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-0033（旧号 1.33）ttnn.i1/erfc 输入钳位无痕迹致域外返回单一常量

- 原因对象：算子/内核 / tt-metal / ttnn.i1/erfc SFPU 输入钳位
- 原因行为：i1 与 erfc 把输入钉在近似有效范围边缘再求值，钳位不留 flag 或 NaN；ttnn.i1 钉在 ±88.5，取的是 exp 的 FP32 溢出点而非 i1 自身的溢出点（91.90626）。
- 影响对象：数值结果（静默错误）
- 影响详情：域外输入一律返回同一个有限常量，下游无法检测；穷举 bf16 时 65024 个正常 pattern 中 31132 个（47.9%）返回 1.15477e+37，其中 31120 个正确答案为 ±Inf。无 crash、无 NaN。
- 来源：https://github.com/tenstorrent/tt-metal/pull/52202
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-0954（旧号 10.38）vLLM DSV4-Flash 深推理档+auto 工具静默空响应

- 原因对象：输出解析 / vllm / deepseek_v4 推理/工具解析边界
- 原因行为：深推理档拉长推理段后，推理→动作的解码边界处理与 tool-calling 语法未对齐：模型在推理中选定工具后流即终止，未产出可提取的 tool call 或 content；具体代码位置原文未定位。
- 影响对象：输出内容（静默错误）
- 影响详情：deepseek_v4 推理/工具 parser + tool_choice=auto + reasoning_effort xhigh/max 时高概率返回 finish_reason=stop、空 content、无 tool_calls；两条件缺一则不高频，live 流量可复现。
- 来源：https://github.com/vllm-project/vllm/issues/53831
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1114（旧号 11.91）vLLM CPU triton top-p 丢弃应保留的 token

- 原因对象：解码/采样 / vllm / apply_top_k_top_p_triton
- 原因行为：triton 实现的 top-p 对累计概率质量边界的包含性判断错误，把 top-p 必须保留的 token 截掉；既有测试只比 kept 计数不比 kept 集合，且只覆盖 V≥1024。
- 影响对象：输出分布（静默错误）
- 影响详情：CPU 构建（triton-cpu）下用 top_p 采样：8-token 行中携 7.5% 质量的 token 被丢；引擎级 4000 请求该 token 采样 0 次，精确分布约 300 次；V≤4096 时 0–4% 行受影响。
- 来源：https://github.com/vllm-project/vllm/issues/59785
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐

### INF-1163（旧号 12.47）vLLM 忽略 LoRA rank/alpha_pattern 与未实现特性

- 原因对象：权重/检查点 / vllm / PEFTHelper adapter 配置解析
- 原因行为：PEFTHelper 读取 rank_pattern/alpha_pattern 后丢弃，统一按 lora_alpha/r 缩放；from_dict 只留已知键，只拒 DoRA/modules_to_save，其余未实现特性被忽略。
- 影响对象：模型权重、输出分布（质量退化）
- 影响详情：此类 adapter 无错加载但以错误缩放服务：OLMoE-1B-7B（64 experts）held-out PPL 为 PEFT 10.06、vLLM 11.28（base 13.09），丢约 40% 增益；把缩放折叠进 lora_B 后恢复 10.06。
- 来源：https://github.com/vllm-project/vllm/issues/59799
- 检查：A ☐ B ☐ C ☐ D ☐ E ☐ F ☐
