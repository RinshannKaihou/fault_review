# 抽取结果裁定与审计说明

你是第一遍抽取之后的第二道关。人工复核只能检查格式，无法判断内容对错，**所以你的裁定就是最终结果**。

## 你会拿到

- 原文：若干 `in` 文件，每行一个条目（字段说明见抽取说明）。
- 第一遍抽取结果：与每个 `in` 一一对应的 `out` 文件。
- 词表与判定规则：`/workspace/RAG/fault_review/normalize/vocab.yaml`，末尾有 `rules`。
- 抽取说明：`/workspace/RAG/fault_review/normalize/prompts/extract.md`，里面有字段定义和长度上限。
- 一个审计 id 列表。

## 处理对象

- **裁定**：`out` 文件里 `uncertain` 非空的条目。
- **审计**：审计 id 列表里的条目。

同一条目两类都占时，两件事都做。`out` 文件缺失或无法解析时跳过该批，并在返回的 `notes` 里写明。

## 步骤

顺序很重要：先独立判断，再看第一遍结果，避免被它带偏。

1. **独立判断。** 读词表、rules 和抽取说明。只读 `in` 文件里处理对象的原文，**先不要打开 `out` 文件**。按词表和规则写出你对以下字段的判断：
   - `kind`；
   - `cause.objects` 的 `subsystem` 和 `stack`；
   - `effect.objects`；
   - `effect.visibility`。

   写到草稿文件。
2. **比较。** 打开 `out` 文件，逐字段对比你的独立判断和第一遍结果。
3. **裁定（裁定类）。** 对 `uncertain` 指出的字段，以及你和第一遍不一致的分类字段，回到原文和 rules 做最终决定：
   - 理由要指明依据，是哪条 rule，或原文里的哪句话。
   - 两个值都说得通时，按 rules 选定一个，**不要留 `待定`**。只有原文确实缺信息时才写 `待定`。
4. **审计（审计类）。** 逐字段检查第一遍结果：
   - **忠实性**：`title`、`component`、`behavior`、`detail` 里的每个事实（数字、版本、触发条件、组件名）都要能在原文找到依据，不能有原文没有的推断。
   - **分类**：`kind`、`subsystem`、`stack`、`effect.objects`、`visibility` 是否符合词表说明和 rules。
   - **形式**：`behavior` 是否只写原因，`detail` 是否只写影响，有没有超长。

   发现问题就直接改正，并记下问题类别。
5. **写出最终记录。** 把每个处理过的条目的**完整最终记录**写入指定的 judged 文件，每行一个 JSON。字段与抽取输出相同，另加一个 `judge` 对象：

   ```json
   "judge": {"reason": ["uncertain", "audit"],
             "changed": ["cause.objects", "effect.visibility"],
             "why": "依据和理由",
             "audit_issues": [{"field": "effect.detail", "type": "忠实性", "detail": "原文没有 W≥65535 这个条件"}]}
   ```

   - `reason` 只列实际适用的项。没改的字段照抄第一遍结果；`changed` 为空表示维持原判。
   - 裁定后把 `uncertain` 置为空字符串；仍有保留意见的写进 `judge.why`。
   - `audit_issues` 的 `type` 取值：`忠实性` / `分类` / `形式`；非审计条目写空列表。
6. **校验。** 用 python3 检查 judged 文件：每行是合法 JSON，枚举值都在词表内，长度上限都满足。不通过就修正后重写。

## 返回

按 schema 返回汇总。
