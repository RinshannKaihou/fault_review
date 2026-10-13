# 全量迁移运行报告

## 覆盖

- 条目总数 1607，语义字段已填 1607
- 来源：{'pilot-majority': 45, 'extract': 1223, 'adjudicated': 339}
- 已填条目中仍为 `待定` 的字段（原文缺信息）：{'原因子系统': 9, '影响对象': 1, '可见性': 5}

## 第一遍抽取与裁定

- 第一遍标为拿不准：277/1562（18%）
- 裁定后改动：50/277（18%），改动字段 {'cause.objects': 36, 'effect.detail': 4, 'effect.objects': 9, 'effect.visibility': 6, 'cause.behavior': 2, 'vocab_gap': 1, 'kind': 1, 'title': 1}

## 忠实性审计（随机 5%）

- 审计 78 条，发现问题并已改正 12/78（15%）；按 95% Wilson 区间，未审计条目的问题率估计在 9%–25%
- 问题类型：{'忠实性': 8, '形式': 2, '分类': 5}
  - INF-0004 `effect.detail` [忠实性] 原文为“有时波及很多通道”，第一遍写成“常波及多通道”，夸大了频度
  - INF-0132 `cause.behavior` [形式] behavior 写入了后果“宽 fp32_dest_acc 行下 L1 超限”，与 effect.detail 的“静态 L1 分配超限，构建期报错”重复
  - INF-0578 `effect.detail` [忠实性] 29.44→32.06 GiB、4.50M→4.90M token 是修复前→修复后的对照，第一遍紧接“KV 池少约 8%”书写、未注明，读起来方向相反
  - INF-0578 `cause.objects` [分类] 修复点在 GLM 模型专属调用点（对照 deepseek_v4/attention.py），属 模型定义 而非 算子/内核
  - INF-0858 `cause.behavior` [忠实性] 原文为“跨 warp MMA 可在共享 Q/K tile 的写同步前读”（可能性），第一遍写成确定的“先读后写”且漏了 Q/K tile
  - INF-0914 `cause.behavior` [忠实性] 原文说被守卫的动态更新对本 checkpoint 为 no-op、检查本身仍同步，第一遍写成“该检查实为 no-op”
  - INF-0943 `effect.visibility` [分类] 主失效是 in/startswith 条件不对称导致的确定性静默错分，one-shot 下即发生；不可复现只是流式路径的次要表现，应取 静默错误
  - TRN-0027 `cause.behavior` [忠实性] 原文是默认 dtype 由 param_dtype 改为 reduce_dtype，仅在 torchtitan 混精策略下等于 fp32，原写“改为 fp32”丢了条件
  - TRN-0027 `effect.objects` [分类] 原文称端到端任务影响未实测，只证明 trainer 间 bitwise 分叉，训练轨迹（收敛/质量）无依据，删去
  - TRN-0033 `cause.behavior` [忠实性] 原文 expert-DP 组退化为单 rank 的条件是 EP=DP，原写漏掉该条件
  - TRN-0060 `effect.detail` [形式] detail 末句写的是同族另一故障（MFSDP v2 缺 no_sync，PR #7186）的原因与后果，不属本条影响；改为原文的触发条件（MTP 跨 depth 复用 + grad-accum fusion，关闭即消失）
  - TRN-0133 `kind` [分类] 条目讲注入点选择与对照实验须声明注入点，是设计结论，应为 结论 而非 故障
  - TRN-0133 `cause.objects[0].stack` [分类] 本地注入实验、机制与栈无关，应写 通用 而非 待定
  - TRN-0206 `effect.detail` [忠实性] 原文只说几乎同分组仍进更新、噪声优势，没有「无报错」这一事实
  - TRN-0261 `effect.detail` [忠实性] 原文「全部既有 ZenFlow 测试都跑在这条错路径上」，第一遍写成「未覆盖此路径」，意思相反

## 分布

| 原因子系统 | 条数 |
|---|---|
| 算子/内核 | 313 |
| KV缓存 | 277 |
| 量化 | 150 |
| 解码/采样 | 115 |
| 运行时/执行 | 73 |
| 调度/服务 | 64 |
| 权重/检查点 | 58 |
| RL训练回路 | 58 |
| 输出解析 | 52 |
| 模型定义 | 48 |
| 编译器 | 47 |
| 位置编码 | 38 |
| 分词/模板 | 34 |
| MoE路由 | 33 |
| 硬件 | 31 |
| 优化器 | 30 |
| 并行切分 | 29 |
| 观测/评测 | 27 |
| 数据管线 | 25 |
| 损失计算 | 18 |
| 外部攻击 | 18 |
| 重计算/卸载 | 16 |
| 模型自身 | 16 |
| 集合通信 | 14 |
| 待定 | 9 |
| 配置/参数 | 8 |
| 构建/发布 | 6 |

| 影响主对象 | 条数 |
|---|---|
| 数值结果 | 377 |
| 输出内容 | 224 |
| 可用性/进度 | 204 |
| KV状态 | 203 |
| 模型权重 | 109 |
| 性能/资源 | 93 |
| 梯度 | 79 |
| 隐状态 | 57 |
| 输出分布 | 57 |
| 训练信号 | 46 |
| 输入编码 | 45 |
| 跨请求/跨样本数据 | 33 |
| 训练样本流 | 25 |
| 指标/日志 | 20 |
| 训练轨迹 | 15 |
| 优化器状态 | 10 |
| 测试判定 | 9 |
| 待定 | 1 |

- kind：{'故障': 1585, '结论': 10, '方法': 8, '指针': 1, '统计': 3}
- 可见性：{'静默错误': 977, '质量退化': 181, '数值异常': 76, '报错': 190, '不可复现': 53, '性能退化': 89, '挂起': 36, '待定': 5}

## 旧分类 → 新原因子系统

| 旧分类 | 条数 | 新子系统分布（前 4） |
|---|---|---|
| cat1 | 293 | 算子/内核 201、编译器 32、运行时/执行 18、硬件 14 |
| cat6 | 280 | KV缓存 137、调度/服务 34、运行时/执行 29、算子/内核 18 |
| cat7 | 111 | 量化 96、算子/内核 4、解码/采样 2、运行时/执行 2 |
| cat10 | 108 | 输出解析 42、分词/模板 26、解码/采样 18、调度/服务 17 |
| cat8 | 99 | KV缓存 42、算子/内核 34、量化 9、模型定义 6 |
| cat11 | 92 | 解码/采样 60、KV缓存 9、输出解析 8、运行时/执行 4 |
| cat12 | 52 | 权重/检查点 27、量化 10、调度/服务 4、解码/采样 3 |
| cat5 | 46 | 并行切分 8、KV缓存 8、算子/内核 7、集合通信 6 |
| cat13 | 42 | 模型定义 23、算子/内核 10、MoE路由 4、位置编码 2 |
| cat9 | 33 | 位置编码 27、数据管线 1、编译器 1、运行时/执行 1 |
| CKPT | 33 | 权重/检查点 18、数据管线 7、位置编码 2、RL训练回路 2 |
| cat4 | 31 | 算子/内核 14、编译器 5、运行时/执行 3、模型定义 2 |
| KV10 | 31 | KV缓存 21、量化 4、算子/内核 3、位置编码 2 |
| RL-RO | 27 | RL训练回路 21、权重/检查点 2、分词/模板 1、量化 1 |
| PAR | 21 | 并行切分 15、集合通信 1、编译器 1、算子/内核 1 |
| OPT | 21 | 优化器 17、并行切分 2、权重/检查点 1、重计算/卸载 1 |
| cat3 | 20 | 外部攻击 15、硬件 5 |
| KV9 | 20 | KV缓存 11、量化 3、算子/内核 3、位置编码 1 |
| NUM | 19 | 优化器 5、量化 3、RL训练回路 2、并行切分 2 |
| DATA | 18 | 数据管线 9、分词/模板 2、RL训练回路 2、算子/内核 1 |
| KER | 18 | 算子/内核 5、损失计算 3、编译器 2、位置编码 2 |
| SFT | 16 | 数据管线 7、分词/模板 3、损失计算 3、权重/检查点 2 |
| OBS | 16 | 观测/评测 10、优化器 2、数据管线 1、运行时/执行 1 |
| MOE | 15 | MoE路由 10、RL训练回路 2、权重/检查点 1、量化 1 |
| KV8 | 14 | 解码/采样 9、KV缓存 4、运行时/执行 1 |
| LOSS | 13 | 损失计算 5、模型定义 4、观测/评测 1、优化器 1 |
| RL-ADV | 11 | RL训练回路 10、损失计算 1 |
| cat15 | 10 | 模型自身 9、KV缓存 1 |
| KV3 | 10 | KV缓存 10 |
| KV6 | 10 | KV缓存 7、量化 2、算子/内核 1 |
| ACT | 10 | 重计算/卸载 8、运行时/执行 1、权重/检查点 1 |
| KV4 | 9 | KV缓存 6、位置编码 1、调度/服务 1、硬件 1 |
| KV5 | 9 | KV缓存 7、集合通信 2 |
| HW | 9 | 硬件 9 |
| RL-RWD | 9 | RL训练回路 9 |
| RL-KL | 9 | RL训练回路 7、观测/评测 1、损失计算 1 |
| cat14 | 8 | 模型自身 7、算子/内核 1 |
| KV2 | 7 | KV缓存 7 |
| cat2 | 3 | 硬件 2、外部攻击 1 |
| KV1 | 2 | 算子/内核 2 |
| KV7 | 2 | 算子/内核 1、KV缓存 1 |

## 词表缺口（第一遍提出 91 条）

- INF-0019: subsystem 缺“设备固件”：ERISC 固件启动代码既非调度执行也非通信逻辑，暂归最接近的 运行时/执行；effect_object 缺“设备固件状态”，原文也未给出对外影响，暂写 待定。
- INF-0126: effect.objects 缺“内存安全/越界写”：本条除错值外还有源程序不会执行的越界 store，现只能并入 数值结果。
- INF-0173: subsystem 缺“解释器/参考执行器”类值：Triton interpreter 是 CPU 模拟执行路径，既非代码生成 pass 也非测试 harness，暂取最接近的 编译器。
- INF-0174: subsystem 缺“解释器/参考执行器”类值：Triton interpreter 是 CPU 模拟执行路径，既非代码生成 pass 也非测试 harness，暂取最接近的 编译器。
- INF-0197: effect_object 的 输出内容 只列 token/文本/结构化输出，本条坏的是生成的视频/音频，希望有“多模态生成输出”或扩展 输出内容 的说明；暂选 输出内容。
- INF-0210: subsystem 缺“发布/版本分发”类：缺陷在 release 分支未回吸已有修复，而非某个编译器 pass 本身，暂归最接近的 编译器。
- INF-0334: effect_object 缺「生成媒体内容」（diffusion 生成的视频/音频），输出内容 只覆盖 token/文本/结构化输出，暂用 输出内容
- INF-0357: effect_object 缺「生成媒体内容」（diffusion 生成的视频/音频），输出内容 只覆盖 token/文本/结构化输出，暂用 输出内容
- INF-0362: effect_object 缺“设备资源记账读数”：hipMemGetInfo 显存读数被驱动算错并被用于容量决策，暂取最接近的 指标/日志
- INF-0393: effect object 缺少“生成图像/多模态输出”，本条为 diffusion 图像损坏，暂用 输出内容（其描述只覆盖 token/文本/结构化输出）。
- INF-0412: subsystem 缺“配置/参数解析”：环境变量、CLI/YAML 配置静默不生效类缺陷暂归 调度/服务（API/前端）；effect 缺“配置生效状态”，暂归 性能/资源（配置静默不生效）。
- INF-0431: subsystem 缺“配置/参数解析”：环境变量、CLI/YAML 配置静默不生效类缺陷暂归 调度/服务（API/前端）；effect 缺“配置生效状态”，暂归 性能/资源（配置静默不生效）。
- INF-0454: subsystem：缺少“MoE 专家计算/融合 MoE kernel”类值；MoE路由 只描述 router/分派/负载均衡，暂取最接近的 MoE路由。
- INF-0461: subsystem：缺少“扩散推理流水线/步级加速缓存（如 Cache-DiT）”类值，暂取最接近的 运行时/执行。
- INF-0495: subsystem 暂选最接近的 运行时/执行：LazyDict 是 Python 通用容器/配置层，希望有「配置/通用工具」类 subsystem；effect.objects 缺「配置状态」类对象，原文未给出下游具体影响，故写待定。
- INF-0496: subsystem 暂选最接近的 运行时/执行：缺陷在 wheel 变体解析/安装层，希望有「构建/安装」类 subsystem。
- INF-0498: subsystem 暂选最接近的 调度/服务：cluster storage 是服务集群的元数据存储，希望有「集群协调/元数据存储」类 subsystem；effect.objects 缺「服务元数据」类对象，原文未说明下游影响，故写待定。
- INF-0523: effect_object 希望有“池化/embedding 输出”：pooling 请求的最终向量既非生成文本也非中间张量，暂归 输出内容
- INF-0549: subsystem 缺“构建/依赖/打包”：根因是依赖解析器静默回退版本，词表无对应值，暂按复活的缺陷（TP>1 丢 down_proj all-reduce）归 并行切分
- INF-0600: subsystem 缺“配置/参数解析”（环境变量、CLI 参数规范化与 deprecation 转发），暂归 调度/服务；effect 缺“配置生效”（用户设置被静默丢弃），暂用 性能/资源。
- INF-0616: effect_object 缺“生成媒体（视频/图像帧）”：输出内容 的描述只覆盖 token/文本/结构化输出，暂选 输出内容。
- INF-0637: effect_object 缺“测量基线/控制面操作语义”：admin API 报成功但缓存未真正回到冷态，输出与计数器本身都正确，暂选 指标/日志、测试判定。
- INF-0645: effect object 缺“对外事件流/接口契约”类值：变错的是发给外部消费者的 KV 事件流，暂用最接近的 KV状态
- INF-0646: effect object 缺“对外事件流/接口契约”类值：变错的是发给外部消费者的 KV 事件流，暂用最接近的 KV状态
- INF-0663: effect_object 缺“多模态输入数据”（图像/音频像素或特征）：输入编码 的说明只覆盖分词、模板和 token 序列，此处暂选最接近的 输入编码。
- INF-0740: subsystem 缺“发布/版本载体”类值：本条现实缺口是修复已在 main 但发布 tag 未携带，暂按出错代码所在功能域归 量化。
- INF-0830: subsystem 希望有“配置/参数解析”：缺陷在配置旋钮的优先级解析，不在量化计算本身，暂归量化（所控对象是 indexer KV 的低位格式）
- INF-0847: subsystem 希望有“配置/能力校验”：启动期能力矩阵检查缺失不专属于量化代码，现有值中只能就近归 量化
- INF-0897: effect_object 缺「嵌入/池化输出」：embedding 模型返回的向量不是生成 token/文本，暂归 输出内容
- INF-0901: effect_object 缺「模型配置/校验结果」：本条变错的是 config 内容与参数校验结论，暂取 模型权重（检查点内容）
- INF-0909: effect_object 缺「嵌入/池化输出」：embedding 模型返回的向量不是生成 token/文本，暂归 输出内容
- INF-0927: subsystem 缺“输出解析（reasoning/tool parser）”：解析器把生成文本切分为 content/reasoning/tool_calls，既非 detokenize 也非请求调度，暂归 调度/服务（API/前端）
- INF-0930: subsystem 缺“输出解析（reasoning/tool parser）”：解析器把生成文本切分为 content/reasoning/tool_calls，既非 detokenize 也非请求调度，暂归 调度/服务（API/前端）
- INF-0932: subsystem 缺“输出解析（reasoning/tool parser）”：解析器把生成文本切分为 content/reasoning/tool_calls，既非 detokenize 也非请求调度，暂归 调度/服务（API/前端）
- INF-0933: subsystem 缺“模型文件获取/缓存”：Hub 下载与缓存回退属文件获取层，不是权重内容或格式转换，暂归 权重/检查点（加载）
- INF-0936: subsystem 缺“输出解析（reasoning/tool parser）”：解析器把生成文本切分为 content/reasoning/tool_calls，既非 detokenize 也非请求调度，暂归 调度/服务（API/前端）
- INF-0938: subsystem 缺“输出解析（reasoning/tool parser）”：解析器把生成文本切分为 content/reasoning/tool_calls，既非 detokenize 也非请求调度，暂归 调度/服务（API/前端）
- INF-0939: subsystem 缺“输出解析（reasoning/tool parser）”：解析器把生成文本切分为 content/reasoning/tool_calls，既非 detokenize 也非请求调度，暂归 调度/服务（API/前端）
- INF-0940: subsystem 缺“输出解析（reasoning/tool parser）”：解析器把生成文本切分为 content/reasoning/tool_calls，既非 detokenize 也非请求调度，暂归 调度/服务（API/前端）
- INF-0942: subsystem 缺“输出解析（reasoning/tool parser）”：解析器把生成文本切分为 content/reasoning/tool_calls，既非 detokenize 也非请求调度，暂归 调度/服务（API/前端）
- INF-0943: subsystem 缺“输出解析（reasoning/tool parser）”：解析器把生成文本切分为 content/reasoning/tool_calls，既非 detokenize 也非请求调度，暂归 调度/服务（API/前端）
- INF-0945: subsystem 缺「输出解析」（tool/reasoning parser）：暂归 调度/服务 的 API/前端，但 parser 缺陷与请求调度、批处理无关，单列便于聚类
- INF-0946: subsystem 缺「输出解析」（tool/reasoning parser）：暂归 调度/服务 的 API/前端，但 parser 缺陷与请求调度、批处理无关，单列便于聚类
- INF-0947: subsystem 缺「输出解析」（tool/reasoning parser）：暂归 调度/服务 的 API/前端，但 parser 缺陷与请求调度、批处理无关，单列便于聚类
- INF-0949: effect object 缺「多模态输入（像素/音频特征）」：输入编码 只覆盖 token 序列与模板渲染，此处变错的是预处理后的像素张量，暂归 输入编码
- INF-0951: subsystem 缺「输出解析」（tool/reasoning parser）：暂归 调度/服务 的 API/前端，但 parser 缺陷与请求调度、批处理无关，单列便于聚类
- INF-0954: subsystem 缺「输出解析」（tool/reasoning parser）：暂归 调度/服务 的 API/前端，但 parser 缺陷与请求调度、批处理无关，单列便于聚类
- INF-0958: subsystem 缺「输出解析」（tool/reasoning parser）：暂归 调度/服务 的 API/前端，但 parser 缺陷与请求调度、批处理无关，单列便于聚类
- INF-0960: subsystem 缺「输出解析」（tool/reasoning parser）：暂归 调度/服务 的 API/前端，但 parser 缺陷与请求调度、批处理无关，单列便于聚类
- INF-0962: subsystem 缺「输出解析」（tool/reasoning parser）：暂归 调度/服务 的 API/前端，但 parser 缺陷与请求调度、批处理无关，单列便于聚类
- INF-0965: subsystem 缺「输出解析」（tool/reasoning parser）：暂归 调度/服务 的 API/前端，但 parser 缺陷与请求调度、批处理无关，单列便于聚类
- INF-0971: subsystem 缺「输出解析（tool/reasoning parser）」：parser 既非分词/模板也非请求调度，暂归 调度/服务（API/前端）
- INF-0973: subsystem 缺「输出解析（tool/reasoning parser）」：parser 既非分词/模板也非请求调度，暂归 调度/服务（API/前端）
- INF-0976: subsystem 缺「输出解析（tool/reasoning parser）」：parser 既非分词/模板也非请求调度，暂归 调度/服务（API/前端）
- INF-0978: subsystem 缺「输出解析（tool/reasoning parser）」：parser 既非分词/模板也非请求调度，暂归 调度/服务（API/前端）
- INF-0985: subsystem 缺「输出解析（tool/reasoning parser）」：parser 既非分词/模板也非请求调度，暂归 调度/服务（API/前端）
- INF-0988: subsystem 缺「输出解析（tool/reasoning parser）」：parser 既非分词/模板也非请求调度，暂归 调度/服务（API/前端）
- INF-0991: subsystem 缺「输出解析（tool/reasoning parser）」：parser 既非分词/模板也非请求调度，暂归 调度/服务（API/前端）
- INF-0995: subsystem 缺「输出解析（tool/reasoning parser）」：parser 既非分词/模板也非请求调度，暂归 调度/服务（API/前端）
- INF-0998: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1000: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1003: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1006: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1011: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1012: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1013: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1014: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1015: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1016: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1020: subsystem 缺「输出解析」（tool/reasoning parser 对生成文本的后处理）一类；暂取最接近的 调度/服务（API/前端）。
- INF-1021: 希望增加 subsystem「输出解析」：tool-call/reasoning parser 等把模型输出解析为结构化响应的服务层后处理，既非请求调度也非分词/模板，暂归 调度/服务
- INF-1022: 希望增加 subsystem「输出解析」：tool-call/reasoning parser 等把模型输出解析为结构化响应的服务层后处理，既非请求调度也非分词/模板，暂归 调度/服务
- INF-1023: effect_object 缺少「生成图像/多模态输出」：扩散生成的图像结果无对应值，暂用 输出内容
- INF-1024: effect_object 缺少「生成图像/多模态输出」：扩散生成的图像结果无对应值，暂用 输出内容
- INF-1035: 希望增加 subsystem「输出解析」：tool-call/reasoning parser 等把模型输出解析为结构化响应的服务层后处理，既非请求调度也非分词/模板，暂归 调度/服务
- INF-1037: 希望增加 subsystem「输出解析」：tool-call/reasoning parser 等把模型输出解析为结构化响应的服务层后处理，既非请求调度也非分词/模板，暂归 调度/服务
- INF-1038: 希望增加 subsystem「输出解析」：tool-call/reasoning parser 等把模型输出解析为结构化响应的服务层后处理，既非请求调度也非分词/模板，暂归 调度/服务
- INF-1067: subsystem 希望有「扩散加速缓存」（TeaCache 等 timestep cache），原文称其为 KV/prefix cache 之外的第三类推理缓存，暂归 KV缓存；effect 的 输出内容 未覆盖扩散生成的图像/视频
- INF-1068: subsystem 希望有「扩散加速缓存」（Spectrum/TeaCache 等 forecaster/timestep cache），暂归 KV缓存；effect 的 输出内容 未覆盖扩散生成的图像/视频
- INF-1096: subsystem 缺“模型执行器（model runner 输入准备、hidden state 收集）”类值；V1/V2 runner 的 PLE 输入与 MTP getter 缺陷暂归 运行时/执行
- INF-1097: subsystem 缺“工具调用/推理输出解析器”：tool parser 属生成后的输出解析，暂归 调度/服务（API/前端）
- INF-1113: subsystem 缺“LoRA/适配器服务”：多 LoRA 的请求→adapter 映射既非加载（权重/检查点）也非计算 kernel（算子/内核），暂归 运行时/执行
- INF-1153: effect_object 缺“输入嵌入/软提示”：prompt embedding 表不是 token 序列，输入编码 只能近似覆盖
- INF-1168: effect_object 缺“API 契约/控制语义”：请求参数被忽略、响应字段恒 0，并非监控指标，只能近似归 指标/日志
- INF-1192: effect_object 希望有「多模态输入（音频/图像特征）」：输入编码 只涵盖分词/模板/token 序列，本条变错的是音频波形，暂取最接近的 输入编码。
- TRN-0024: subsystem 希望有「训练配置/精度策略」：缺陷是训练参数把混精决定经进程环境泄漏给后续配置，不属于设备调度执行，暂选最接近的 运行时/执行。
- TRN-0056: subsystem 缺「配置/参数管线」：死参数与被硬编码覆盖的参数横跨 MTP 结构、critic 引擎、router replay、SP、FSDP 策略、chat template 等多个功能域，按首个实例归 模型定义 不能代表整条。
- TRN-0121: subsystem 重计算/卸载 描述只列激活/参数 offload，未明确覆盖 ZeRO 的梯度/优化器 CPU offload；建议在说明中补上梯度 offload
- TRN-0184: subsystem 缺「PEFT/adapter 配置与注入」：LoRA target_modules 匹配属于 adapter 挂载，不是权重加载/保存/转换，暂归最接近的 权重/检查点。
- TRN-0197: subsystem 缺「agent 环境/工具」：agent-RL 的工具定义与环境交互不在 RL训练回路 现有描述（reward/advantage/KL/IS/同步）内，也不是模板本身，暂归最接近的 RL训练回路。
- TRN-0246: subsystem 缺「训练精度/混合精度配置」：训练引擎未消费 dtype 配置字段，不属于切分、算子或量化，暂归最接近的 运行时/执行。
