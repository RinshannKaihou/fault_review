# 新条目写法（结构化，JSON 输出）

每收录一条新故障，输出一个 JSON 对象，由辅助脚本分配编号、补来源链接和原文字段，并转存为 YAML。你不要写编号、来源链接和原文。
受控词表在下方。分类字段必须取词表里的值；原文确实没有的信息写 `待定`，对本条不适用的写 `N/A`。

## 模板

```json
{
  "slug": "<snake_case 名>",
  "title": "<一句话中文标题，≤40 字，关键英文标识符保留原文>",
  "kind": "<故障|结论|统计|方法|指针>",
  "cause": {
    "objects": [{"subsystem": "<词表值>", "stack": "<软件栈>", "component": "<具体模块/算子/函数，≤25 字>"}],
    "behavior": "<缺陷本身做错了什么，≤120 字；只写原因，不写后果>"
  },
  "effect": {
    "objects": ["<词表值>"],
    "detail": "<外部能观察到什么、在什么条件下触发，≤150 字；不重复原因>",
    "visibility": "<词表值>"
  },
  "mitigation": {"detect": "待定", "fix": "待定"},
  "labels": {
    "family": "<infra_off|model_on|ambiguous|N/A；训练编目写 N/A>",
    "trigger": "<config|yes|partial|other_stack|hw_specific|no|not_target；训练编目写 N/A>",
    "trigger_review": "<true|false；训练编目写 N/A>",
    "trigger_note": "<一行触发方式；训练编目写 N/A>",
    "stage": ["<pretrain|sft|rl|shared>，推理编目整个字段写 \"N/A\">"],
    "confidence": "<verified|documented|speculative>",
    "coverage": "<NEW|var|rationale|existing>",
    "discovered": "<YYYY-MM-DD>"
  },
  "relations": [],
  "remarks": ""
}
```

## 规则

- 只写来源里有的事实，不补充、不推断；不改变程度和确定性；保留触发条件。
- 只有修复必须同时改两个不同 subsystem 时，`cause.objects` 才写多项。
- 长度上限是硬约束：title ≤40、component ≤25、behavior ≤120、detail ≤150 字。
- 只输出这一个 JSON 对象，不要解释。`relations` 只在 coverage=var 时写 `[{"kind": "var", "target": "<已有条目新编号>"}]`。
- `mitigation` 两项一律写 "待定"：改进措施另有专门阶段补全，扫描时不填。

{{VOCAB}}
