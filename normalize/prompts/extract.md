# 故障条目结构化抽取说明

你要把故障编目里的自由文本条目，抽取成 FMEA 式结构化字段。受控词表在
`/workspace/RAG/fault_review/normalize/vocab.yaml`，先完整读一遍：每个枚举值的说明都是判定依据，末尾的 `rules` 是必须遵守的判定规则，与下面的字段规则冲突时以 `rules` 为准。

## 输入

一个 JSONL 文件，每行一个条目：`id`、`legacy_id`、`slug`、`repo`、`old_section`（旧分类，**仅供参考，旧分类经常是错的**）、`meta`（原元数据行）、`stack_hint`（从来源 URL 推出的栈，可能不对）、`original`（原文各字段）。

## 输出

每个输入条目输出一行 JSON，顺序与输入相同，写入指定的输出文件：

```json
{"id": "INF-0044",
 "title": "...",
 "kind": "故障",
 "cause": {"objects": [{"subsystem": "算子/内核", "stack": "tt-metal", "component": "ttnn.topk 单核 factory"}],
           "behavior": "..."},
 "effect": {"objects": ["数值结果"], "detail": "...", "visibility": "静默错误"},
 "vocab_gap": "",
 "uncertain": ""}
```

## 字段规则

- **title**：一句话中文标题，不超过 40 字；关键英文标识符（算子名、框架名）保留原文。
- **kind**：`故障` / `结论` / `统计` / `方法` / `指针`。条目在讲检测器、验证器、防护方案本身 → `方法`；讲基率或频率 → `统计`；讲设计结论 → `结论`。
- **cause.objects**：缺陷所在的位置。
  - `subsystem` 必须取自词表。判定口诀是问**“修复要改哪里”**，不要问“看起来像什么”。例如 tt-metal 的 SFPU 数学函数写错，归 `算子/内核`，不归 `硬件`；测试 golden 或容差写错，归 `观测/评测`。
  - `stack`：先核对 `stack_hint`。若缺陷实际在另一个项目（比如 issue 提在 trl，但根因在 vLLM），改成真实所在的栈。优先取词表值，词表里没有就写小写项目名。`模型` 只配 subsystem=模型自身；不依赖具体栈的论文类机制写 `通用`。
  - `component`：自由短语，不超过 25 字，写到具体模块、算子或函数。
  - 只有修复必须同时改两个**不同 subsystem** 时才写多项，最多 3 项；同一 subsystem 的多个组件合写进一个 component。
- **cause.behavior**：缺陷本身做错了什么，不超过 120 字。**只写原因，不写后果**。
- **effect.objects**：哪个东西变错了。取词表值，1–3 项，按主次排序。
- **effect.detail**：外部能观察到什么、在什么条件下触发，不超过 150 字。不要重复 behavior 的内容。
- **effect.visibility**：取词表值。
- **vocab_gap**：如果词表里没有合适的 subsystem 或 effect object，先选最接近的值，再在这里写你希望有的值和理由；没有缺口就留空字符串。
- **uncertain**：只在按 `rules` 判定后，仍有两个词表值同样成立时填写：写字段名、两个候选值和理由。正常的取舍不要写，没有就留空字符串。这个字段会进入人工复核队列。

## 纪律

- 只用原文里的信息，不补充原文没有的事实，不臆测修复方式。
- 原文确实没有给出的信息写 `待定`；对本条不适用的写 `N/A`（`结论`/`统计`/`方法` 类条目先按 `rules` 填所针对故障的对象和影响，确实对应不上才写 `N/A`）。
- 所有自然语言字段用中文。
- 输出必须是合法 JSONL：每行一个对象，不要包 markdown 代码块。
- **长度上限是硬约束**：title ≤40 字、component ≤25 字、behavior ≤120 字、detail ≤150 字，超出会被校验器打回。写完用 python3 逐条检查长度，超了就改短。
