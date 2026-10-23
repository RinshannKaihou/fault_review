# fault-rag 网页版（GitHub Pages）

把 `fault-rag` 的检索搬进浏览器的静态站点：输入症状、机制或编号，在推理 / 训练两个故障编目里找先例，读完整条目，复制合规的引用行。没有服务器，所有检索在用户浏览器里完成。

不输入检索词时是**浏览模式**：选任一筛选条件（编目、原因子系统、置信度，或详情页的"只看这一族"），直接列出符合条件的全部条目和总数，按编号或最新首录日期排序，每页 50 条；也可以"浏览全部"。浏览不经过任何排名，过滤规则与检索共用 `src/search.js` 的 `filter()`（与 CLI 的 `_passes` 一致，指针条目默认不列）。筛选条件、排序和选中的条目都写在 URL 里，可以直接分享。

检索结果只供参考，和命令行版一样：引用前打开条目原文核对，遵守 [AGENTS.md](../AGENTS.md) 的引用纪律。页面把这些纪律放在上下文里：负结果、指针条目、speculative、待审计、同名条目都有标记与说明，详情页给出带编目、编号、名称、源文件的引用行。

## 两档检索

| 档位 | 浏览器下载 | 做什么 |
|---|---|---|
| 关键词（默认） | 首屏约 2.3 MB（gzip） | BM25，中文按相邻两字（bigram）切分 |
| 关键词 + 语义（用户勾选才加载） | 再加约 135 MB 模型，存进浏览器缓存 | 加一路 `multilingual-e5-small`（int8）向量，两路名次 RRF 融合 |

命令行版的交叉编码器重排（`bge-reranker-v2-m3`，每次查询要读 30 条候选全文，约 1.6 万 token）在浏览器 wasm 里要几十秒到几分钟，网页版不做。需要最好的召回，用命令行版。

### 质量（2026-10-23，快照 inference `e9b7871` 2117 条 + training `2671871` 364 条）

R@5 / R@10 / MRR@10：

| 管线 | bench 36 条 | indy 63 条 | 留出 50 条（只观察） |
|---|---|---|---|
| 网页 · 关键词 | 1.000 / 1.000 / 0.954 | 0.905 / 0.921 / 0.830 | 0.80 / 0.82 / 0.673 |
| 网页 · 关键词 + 语义 | 1.000 / 1.000 / 0.944 | 0.921 / 0.937 / 0.844 | 0.76 / 0.86 / 0.681 |
| 命令行 `--no-rerank`（参照） | 1.000 / 1.000 / 1.000 | 0.937 / 0.952 / 0.863 | 0.78 / 0.86 / 0.665 |
| 命令行（含重排，参照） | 1.000 / 1.000 / 1.000 | 0.952 / 0.968 / 0.881 | 0.88 / 0.90 / 0.769 |

- 语义档的增益很小：两套题的 R@10 各多 1–2 条，留出集 R@5 反而少 2 条。语料向量只要有余弦 0.998 级的数值差，留出集 R@5 就会在 0.76 与 0.82 之间摆动，所以别把两档之间的差别读成定论。它保留为可选项，供口语化描述时扩大召回窗口用。
- 与命令行版的逐条差距（indy，R@5）：indy-001 在两档都掉出前 10（命令行第 1），是 bigram 跨词边界的噪声（"始错"一类低频组合，见 `eval/ACCEPTANCE.md` 2026-10-22）；indy-014、indy-065 靠命令行的重排或 bge-m3 才进前 10；indy-059、indy-063 在语义档回到第 2。
- 任何无重排的档位都会让 indy-014 掉出（它靠重排回到第 4），所以网页版用自己的基线（`web/eval/baselines/`），不对命令行基线比。

## 设计取舍（都有实测依据）

- **bigram 而不是 jieba**：浏览器不用带分词器（jieba-wasm 包 16 MB），两端分词也不会漂移；同一快照上留出集 R@5 0.80 对 jieba 版 0.68，indy R@5 持平（0.905）。代价是 indy-001 这类跨词边界组合的噪声。`check_web.py` 每次核对 JS 与 Python 两份分词器在全部评测 query 与 2,495 段语料上逐 token 相同。
- **语义模型选 me5-small**：在同一套题上对比过 bge-m3 int8（570 MB，查询向量与 fp32 只有余弦 0.985，indy 还丢 1 条）、bge-small-zh（24 MB，中英混排很差，留出 R@5 0.62）、bge-m3 静态 token 向量（崩）。me5-small 118 MB 的 int8 是体积与质量的平衡点。模型版本钉死在 `src/model.json` 的 `revision`；transformers.js 4.3.1 取文件清单时不带 revision，所以 `semantic.js` 把版本也写进了 URL 模板。
- **所有推理都跑浏览器那份 wasm**：原生 onnxruntime-node 的 int8 内核与浏览器的 wasm 不同，查询向量只有余弦 ~0.999，足以让近邻条目换位，门禁测的就不是用户拿到的了（实测 149 条里 115 条前 10 列表不同）。现在语料嵌入、门禁和浏览器都用 onnxruntime-web 同一个 wasm 二进制、单线程：Node 与 Chromium 的查询向量逐位相同，`browser_parity.py` 实测 149/149 条前 10 完全一致。wasm 也不随 CPU 指令集变化，本机与 CI 构建出的向量相同。
- **语料逐条嵌入**：int8 模型按整批张量量化激活，批内每条向量都受邻居和 padding 影响（余弦 ~0.998）。逐条嵌入后每条向量只取决于自己的文本，新增条目不再扰动其他向量，也能多 worker 分片而结果逐位不变。
- **运行时随站点发布**：transformers.js 与 onnxruntime-web 的 wasm 放在站点 `vendor/` 下，不走 jsdelivr；字体自托管（Barlow Semi Condensed，OFL）。它们的许可证随站点发布在 `src/licenses/`（`README.txt` 是索引）。onnxruntime 的 npm 包不带许可文本，`onnxruntime-MIT.txt` 与 `onnxruntime-ThirdPartyNotices.txt` 取自 microsoft/onnxruntime 在 `node_modules/onnxruntime-web/__commit.txt` 所记的提交；升级 transformers.js 或 onnxruntime-web 时，要重新确认 `transformers.min.js` 打包了哪些依赖，并按新提交重取这两个文件。站点只在加载语义档时访问模型源（huggingface.co 或 hf-mirror.com，页面上可切换）。GitHub Pages 发不了多线程 wasm 需要的 COOP/COEP 头，所以浏览器里单线程运行，一次查询是一次短前向，够用。

## 本地构建与预览

```bash
cd web && ONNXRUNTIME_NODE_INSTALL=skip npm ci && cd ..   # onnxruntime-node 是用不到的传递依赖
python3 web/build_site.py                  # 关键词索引、条目页、语义向量 → web/dist（约 2–5 分钟）
python3 web/build_site.py --no-vectors     # 只要关键词档：不需要 node / 模型，约 30 秒
python3 -m http.server -d web/dist 8000    # 打开 http://127.0.0.1:8000/
```

`build_site.py` 只读 `catalogs/` 快照（经 `fault_rag.parse`，与 `reindex` 读到的条目相同），不依赖 `data/`、`models/`。语义向量由 `embed_corpus.mjs` 计算，模型来源：

- 默认从 huggingface.co 下载，缓存在 `web/node_modules/@huggingface/transformers/.cache/`；
- `HF_ENDPOINT=https://hf-mirror.com`：走镜像。本机进程若设了代理，Node 默认不读代理环境变量，需加 `NODE_USE_ENV_PROXY=1`，并把 `hf-mirror.com` 加进 `NO_PROXY`（镜像会 302 到 xet CDN，后者要走代理）；
- `FAULT_RAG_WEB_MODEL_DIR=<dir>`：离线读 `<dir>/Xenova/multilingual-e5-small/{config.json,tokenizer.json,tokenizer_config.json,onnx/model_quantized.onnx}`。

## 网页质量门禁

```bash
python3 web/eval/check_web.py                     # 两档都测：bench 聚合口径、indy 逐条口径、留出集只报告
python3 web/eval/check_web.py --write-baselines   # 经审查的快照同步或检索改动之后重写基线
python3 web/eval/browser_parity.py                # 在 headless Chromium 里跑同样的 query，核对与门禁逐条相同
```

- `check_web.py` 用站点自己的 `src/search.js`（经 `eval/rank.mjs` 在 Node 里跑）给评测题排序，再交给 `eval/run_eval.py` 的同一套指标与门禁规则：bench 聚合口径（R@5/R@10 不降，MRR/nDCG 容差 0.02，绝对下限与负结果分片），indy 逐条口径（基线通过的题必须继续通过）。基线在 `web/eval/baselines/`，豁免名单 `web/eval/fix_list.json` 与 `eval/fix_list.json` 同一格式、同一纪律（`weak_gold` 要单路证据，`accepted_regression` 要用户确认）。
- `browser_parity.py` 需要 Playwright 与 Chromium，不进 CI；升级 transformers.js、onnxruntime-web 或模型版本后必须跑一次。
- 和命令行门禁一样：旧题通过不代表新增、修订、退役的条目语义正确，发布前要做本次变更的定向检查。

## 发布

`.github/workflows/pages.yml`：`main` 上 `catalogs/`、`web/`、`fault_rag/`、`eval/` 有改动时，CI 从已提交的快照构建站点（约 5–10 分钟，语料嵌入占大头），跑 `check_web.py`，通过才部署到 GitHub Pages。门禁失败时线上保持上一版，报告作为 `web-gate-report` 产物上传。仓库需在 Settings → Pages 里把 Source 设为 GitHub Actions。

快照同步流程不变（`scripts/sync_catalogs.sh`，见主 README）。推送新快照后若 CI 的网页门禁失败，在本地 `build_site.py` + `check_web.py` 复现，按命令行门禁同样的规则处理：真退化不得登记为 `weak_gold`。

## 文件

```text
build_site.py        catalogs/ → web/dist：docs.json、bm25.json、entry/<repo>/<id>.json、vectors.f32、meta.json
embed_corpus.mjs     语料向量（onnxruntime-web wasm，逐条，多 worker）
node_env.mjs         Node 端 transformers.js：注入 onnxruntime-web，模型来源配置
src/search.js        检索内核（过滤、BM25、RRF、lookup、related），浏览器与门禁共用
src/semantic.js      语义档：模型加载与查询编码，浏览器与 Node 共用
src/app.js           页面：检索、编号匹配、条目详情、引用行
src/model.json       语义模型与固定 revision
src/licenses/        随站点发布的许可证：本项目与 vendor/ 里第三方运行时（README.txt 是索引）
eval/check_web.py    网页门禁；eval/rank.mjs 是它调用的 Node 排序器
eval/browser_parity.py   浏览器与门禁逐条一致性检查
```
