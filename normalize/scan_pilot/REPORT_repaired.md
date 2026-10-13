# 每日扫描结构化写入：离线试点报告（outputs_repaired.jsonl）

- 模型：zhipu/glm-5.3；条目 61（新鲜 21、参考 40）
- 输出可解析：{'ok': 60, 'json_error: Extra data line 1 col 1089': 1}；输出 token 中位数 2079，最大 9019

## 格式与词表

- 通过校验（0 错误）：60/60
- 超长字段数：{'detail': 4, 'behavior': 10, 'title': 3, 'component': 3}；缺 visibility：0
- 错误类型：无
- 告警类型（含超长）：{'cause.behavior 超过 120 字': 10, 'title 超过 40 字': 3, 'effect.detail 超过 150 字': 4}

## 标签（与扫描自己写的元数据行对照）

- confidence / coverage 一致：120/120；不一致 无

## 分类与审计结论的一致率（参考组）

- kind：39/39（100%）
- subsystem：33/39（85%）
- effect：35/39（90%）
- visibility：37/39（95%）

## 分类与网关三票一致结论的一致率（新鲜组，仅三票一致的字段）

- kind：21/21
- subsystem：19/20
- effect：16/17
- visibility：19/21

## 文本字段忠实性（网关三票）

| 字段 | 一致没问题 | 一致有问题 | 分歧 |
|---|---|---|---|
| title | 58 | 0 | 2 |
| component | 59 | 0 | 1 |
| behavior | 49 | 1 | 10 |
| detail | 44 | 3 | 13 |
