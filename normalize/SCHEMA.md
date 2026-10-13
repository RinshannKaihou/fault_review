# 故障模式条目结构 v0（草案）

每个故障模式一个 YAML 文件：`entries/<ID>.yaml`。Markdown 总参考与 CSV/XLSX 由这些文件生成，不再手写。
受控词表见 [`vocab.yaml`](vocab.yaml)，校验见 `validate.py`。

## 编号

- 格式 `INF-0001` / `TRN-0001`，序号只表示分配顺序，**不含分类语义**。分类改动不会改编号。
- `INF` / `TRN` 是命名空间，只表示由哪个编目仓库分配；两个每日扫描独立追加，靠命名空间避免撞号。
- 新条目编号 = 本命名空间最大序号 + 1；编号一经分配不复用，条目合并或退役后保留文件并标 `status`。
- 旧编号（`1.44`、`HW.01`，以及已并入的 `1.70`）全部记在 `legacy_ids`，检索时可按旧号查到新条目。

## 占位符

任何字段都可以写：

| 值 | 含义 |
|---|---|
| `待定` | 应当有值，但尚未填写或尚未核实 |
| `N/A` | 对本条不适用（如 `结论` 类条目没有修复措施） |

## 字段

| 键 | 对应参考格式 | 说明 |
|---|---|---|
| `id` | 编号 | 见上 |
| `legacy_ids` | — | 旧编号列表 |
| `slug` | — | 原 snake_case 名，作别名保留 |
| `title` | — | 一句话中文标题（≤40 字） |
| `kind` | — | `故障` / `结论` / `统计` / `方法` / `指针` |
| `status` | — | `active` / `merged`（配 `merged_into`）/ `retired` |
| `cause.objects[]` | 故障原因（对象） | 每项含 `subsystem`（严格枚举）、`stack`（半受控）、`component`（自由短语）。接口或契约类故障可以写多项 |
| `cause.behavior` | 故障原因（行为） | 缺陷本身做错了什么，≤120 字，只写原因，不写后果 |
| `effect.objects[]` | 故障影响（对象） | 严格枚举，1–3 项，按主次排序 |
| `effect.detail` | 故障影响（详情） | 用户/系统看到什么、触发条件，≤150 字 |
| `effect.visibility` | — | 可选枚举：静默错误 / 质量退化 / 数值异常 / 挂起 / 报错 |
| `mitigation.detect` | 改进措施（检测） | 怎么发现。**本期统一 `待定`**；推理库原“隐藏态签名”保留在 `original.signature`，补全时作检测线索 |
| `mitigation.fix` | 改进措施（修复/规避） | **本期统一 `待定`**，后续单独回源补全 |
| `labels` | remarks | 项目既有维度：`family`、`trigger`、`trigger_review`、`trigger_note`、`stage`、`confidence`、`coverage`、`discovered` |
| `remarks` | remarks | 自由备注 |
| `evidence[]` | — | 来源链接：`url`、`kind`（github_issue / github_pr / arxiv / paper / other） |
| `relations[]` | — | `{kind: var / cross, target: <新 ID>}`；目标无法解析时保留 `target_raw` |
| `original` | — | 迁移前各字段原文，原样保留，供检索和回溯 |
| `migration` | — | 可选。语义字段的来源：`extract`（单遍抽取）/ `adjudicated`（独立裁定或审计后定稿）/ `pilot-majority`（试点三路多数票），附改动字段与审计问题 |

## 设计约束

1. **模式与实例分层**：现阶段一条旧条目对应一个模式（1:1 迁移），`var` 变体先作为 `relations` 保留。之后经人工确认，再把同一机制的条目合并成“一个模式挂多条 evidence”，被并入的条目标 `status: merged`。
2. **结构化字段是新增，不是替换**：`original` 保留全部原文，RAG 同时索引两者，压缩不会损失检索细节。
3. **原因与影响分开写**：`cause.behavior` 只写缺陷，`effect.detail` 只写后果；同一句话不要同时出现在两处。
