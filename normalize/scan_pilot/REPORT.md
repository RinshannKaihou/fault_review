# 每日扫描结构化写入：离线试点报告（outputs.jsonl）

- 模型：zhipu/glm-5.3；条目 61（新鲜 21、参考 40）
- 输出可解析：{'ok': 61}；输出 token 中位数 905，最大 3091

## 格式与词表

- 通过校验（0 错误）：51/61
- 超长字段数：{'behavior': 37, 'component': 37, 'title': 53, 'detail': 34}；缺 visibility：10
- 错误类型：{'effect.visibility': 10}
- 告警类型（含超长）：{'title 超过 40 字': 53, 'cause.behavior 超过 120 字': 37, 'effect.detail 超过 150 字': 34}

## 标签（与扫描自己写的元数据行对照）

- confidence / coverage 一致：122/122；不一致 无

## 分类与审计结论的一致率（参考组）

- kind：40/40（100%）
- subsystem：34/40（85%）
- effect：36/40（90%）
- visibility：32/40（80%）

## 分类与网关三票一致结论的一致率（新鲜组，仅三票一致的字段）

- kind：21/21
- subsystem：19/20
- effect：16/17
- visibility：15/21

## 文本字段忠实性（网关三票）

| 字段 | 一致没问题 | 一致有问题 | 分歧 |
|---|---|---|---|
| title | 58 | 1 | 2 |
| component | 61 | 0 | 0 |
| behavior | 56 | 0 | 5 |
| detail | 46 | 5 | 10 |
