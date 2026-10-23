// fault-rag static site: search, lookup, entry sheet. Retrieval itself lives in search.js (shared
// with the Node eval); the semantic tier (semantic.js + vendored transformers.js) loads on demand.
import { Index } from "./search.js";
import { configureWeb, createEmbedder } from "./semantic.js";

const REPO_LABEL = { inference: "推理", training: "训练" };
const CONF_NOTE = {
  verified: "已复现",
  documented: "有文档记载，不等于已复现",
  speculative: "推测，不能当作检测结论",
};
const LINK_KIND = {
  var: "变体（来源作者断言，机制未必已核实相同）",
  cross: "交叉引用",
  mentions: "正文提及",
};
const ORIGINAL_LABEL = {
  mechanism: "机制", behavior: "行为", signature: "签名", trigger_surface: "触发面",
  repro: "复现", source: "来源", discovery: "发现记录", notes: "回访记录", locus: "位置",
  priority: "优先级", meta: "元信息", category: "旧分类", section: "旧章节",
};
const LABEL_LABEL = {
  confidence: "置信度", trigger: "触发标签", trigger_note: "触发说明", trigger_review: "触发待复核",
  family: "家族", stage: "阶段", coverage: "覆盖", discovered: "首录日期",
};
const EXAMPLES = ["loss spike 但无 NaN", "GRPO advantage std zero", "前缀缓存命中后输出串到别的请求",
                  "INF-0044", "HW.05"];
const PAGE = 10;

// ---------- tiny DOM helpers (text only: catalog content never becomes HTML) ----------
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : String(kid));
  }
  return el;
}
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : null);
const isTbd = (v) => v == null || v === "" || /^待定/.test(String(v));
// Catalog prose uses `code` spans; render them as <code>, everything else as text.
function prose(text) {
  const parts = String(text).split(/`([^`]+)`/);
  return parts.map((p, i) => (i % 2 ? h("code", {}, p) : p));
}
function value(v) {
  if (isTbd(v)) return h("span", { class: "tbd" }, v == null || v === "" ? "未记录" : String(v));
  if (Array.isArray(v)) return v.join("、");
  if (typeof v === "object") return JSON.stringify(v);
  return prose(v);
}

// ---------- state <-> URL hash ----------
const FIELDS = ["q", "repo", "category", "confidence", "trigger", "pattern", "mode", "no_neg", "pointers", "e"];
function readState() {
  const p = new URLSearchParams(location.hash.slice(1));
  const s = Object.fromEntries(FIELDS.map((f) => [f, p.get(f) || ""]));
  s.k = Number(p.get("k")) || PAGE;
  return s;
}
function writeState(s, { push = false } = {}) {
  const p = new URLSearchParams();
  for (const f of FIELDS) if (s[f]) p.set(f, s[f]);
  if (s.k && s.k !== PAGE) p.set("k", s.k);
  const hash = "#" + p.toString();
  if (hash === location.hash) return render();
  if (push) location.hash = hash;
  else { history.replaceState(null, "", hash); render(); }
}

// ---------- data ----------
let meta, index, docs;
let semantic = { state: "off", embedder: null, error: null };
const getJSON = (p) => fetch(p).then((r) => {
  if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
  return r.json();
});

async function boot() {
  [meta, docs] = await Promise.all([getJSON("data/meta.json"), getJSON("data/docs.json")]);
  const bm25 = await getJSON("data/bm25.json");
  index = new Index(docs, bm25);
  renderSnapshot();
  initForm();
  window.addEventListener("hashchange", render);
  render();
}

function renderSnapshot() {
  const c = meta.counts.by_repo;
  const src = meta.sources.map((s) => `${s.repo.replace("_error_review", "")} ${s.commit.slice(0, 7)}（${s.date.slice(0, 10)}）`);
  $("#snapshot").replaceChildren(
    "推理 ", h("b", {}, (c.inference || 0).toLocaleString()), " 条、训练 ", h("b", {}, (c.training || 0).toLocaleString()),
    " 条（含负结果与指针）。快照对应上游提交 ", src.join("，"), "。");
  const built = meta.site_commit ? `站点构建于 ${meta.built_at}，提交 ${meta.site_commit.slice(0, 7)}。` : `站点构建于 ${meta.built_at}。`;
  $("#build-info").replaceChildren(built, " ",
    meta.repo_url ? h("a", { href: meta.repo_url }, "源码与编目快照") : null,
    "。编目真源在两个上游仓库，本站读取经人工审查后同步的只读快照，可能滞后于上游。");
}

const $ = (s) => document.querySelector(s);
const form = () => $("#search-form");

function initForm() {
  $("#examples").replaceChildren("试试：", ...EXAMPLES.flatMap((q, i) => [
    i ? "、" : "",
    h("button", { type: "button", onclick: () => writeState({ ...readState(), q, e: "", k: PAGE }, { push: true }) }, q),
  ]));
  const triggers = meta.filters.trigger;
  $("#f-trigger").append(...triggers.map((t) => h("option", { value: t }, t)));
  fillCategories("");
  $("#f-repo").addEventListener("change", (ev) => fillCategories(ev.target.value));

  if (meta.semantic) {
    $("#semantic-size").textContent = `首次需下载约 ${meta.semantic.download_mb} MB`;
  } else {
    $("#mode-hybrid").disabled = true;
    $("#semantic-size").textContent = "本次构建未包含语义检索";
  }
  const saved = localStorage.getItem("fault-rag-model-source");
  if (saved) $("#model-source").value = saved;
  $("#model-source").addEventListener("change", (ev) => {
    localStorage.setItem("fault-rag-model-source", ev.target.value);
    if (semantic.state === "failed") { lastSearchKey = null; render(); }   // retry from the new source
  });
  for (const r of form().querySelectorAll('input[name="mode"]')) {
    r.addEventListener("change", () => {
      if (r.checked) writeState({ ...collect(), e: readState().e });
    });
  }
  for (const sel of form().querySelectorAll("select:not(#model-source), input[type=checkbox]")) {
    sel.addEventListener("change", () => readState().q && writeState({ ...collect(), e: readState().e }));
  }
  form().addEventListener("submit", (ev) => {
    ev.preventDefault();
    writeState({ ...collect(), e: "" }, { push: true });
  });
  $("#q").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); form().requestSubmit(); }
  });
}

function fillCategories(repo) {
  const sel = $("#f-category");
  const keep = sel.value;
  const lists = repo ? [meta.filters.category[repo]] : Object.values(meta.filters.category);
  const merged = new Map();
  for (const list of lists) for (const [c, n] of list) merged.set(c, (merged.get(c) || 0) + n);
  sel.replaceChildren(h("option", { value: "" }, "全部"),
    ...[...merged].sort((a, b) => b[1] - a[1]).map(([c, n]) => h("option", { value: c }, `${c}（${n}）`)));
  sel.value = merged.has(keep) ? keep : "";
}

function collect() {
  const f = new FormData(form());
  const s = readState();
  return {
    q: (f.get("q") || "").trim(), repo: f.get("repo") || "", category: f.get("category") || "",
    confidence: f.get("confidence") || "", trigger: f.get("trigger") || "", pattern: s.pattern,
    mode: f.get("mode") === "hybrid" ? "hybrid" : "", no_neg: f.get("no_neg") ? "1" : "",
    pointers: f.get("pointers") ? "1" : "", k: PAGE,
  };
}

function syncForm(s) {
  if (document.activeElement !== $("#q")) $("#q").value = s.q;
  $("#f-repo").value = s.repo;
  fillCategories(s.repo);
  $("#f-category").value = s.category;
  $("#f-confidence").value = s.confidence;
  $("#f-trigger").value = s.trigger;
  $("#f-noneg").checked = !!s.no_neg;
  $("#f-pointers").checked = !!s.pointers;
  form().querySelector(`input[name="mode"][value="${s.mode === "hybrid" ? "hybrid" : "bm25"}"]`).checked = true;
  $("#source-wrap").hidden = s.mode !== "hybrid";
  const pf = $("#pattern-filter");
  pf.hidden = !s.pattern;
  if (s.pattern) {
    pf.replaceChildren("只看故障模式 ", h("code", {}, s.pattern), " ",
      h("button", { type: "button", onclick: () => writeState({ ...readState(), pattern: "", k: PAGE }) }, "取消"));
  }
}

// ---------- semantic tier ----------
async function ensureSemantic() {
  if (semantic.state === "ready" || !meta.semantic) return semantic.state === "ready";
  if (semantic.loading) return semantic.loading;
  semantic.loading = loadSemantic();
  return semantic.loading;
}

async function loadSemantic() {
  const box = $("#semantic-progress");
  box.hidden = false;
  box.classList.remove("error");
  const bar = $("#semantic-bar"), status = $("#semantic-status");
  const files = new Map();
  const show = (text, frac) => { status.textContent = text; if (frac != null) bar.style.width = `${Math.round(frac * 100)}%`; };
  show("正在载入语义检索组件…", 0.02);
  try {
    const [lib, vec] = await Promise.all([
      import("./vendor/transformers.min.js"),
      fetch("data/vectors.f32").then((r) => { if (!r.ok) throw new Error(`vectors: HTTP ${r.status}`); return r.arrayBuffer(); }),
    ]);
    configureWeb(lib, $("#model-source").value);
    const embedder = await createEmbedder(lib, meta.semantic, {
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          files.set(p.file, [p.loaded, p.total]);
          let got = 0, all = 0;
          for (const [l, t] of files.values()) { got += l; all += t; }
          show(`正在下载模型 ${(got / 2 ** 20).toFixed(0)} / ${(all / 2 ** 20).toFixed(0)} MB（下载后存进浏览器缓存）`, 0.05 + 0.9 * got / all);
        } else if (p.status === "ready") show("正在初始化模型…", 0.97);
      },
    });
    await embedder.query("warm up");
    index.vectors = new Float32Array(vec);
    index.dim = meta.semantic.dim;
    semantic = { state: "ready", embedder };
    show("语义检索已就绪。", 1);
    setTimeout(() => { if (semantic.state === "ready") box.hidden = true; }, 1500);
    return true;
  } catch (err) {
    semantic = { state: "failed", error: err };
    lastSearchKey = null;          // the next search (or source switch) retries instead of reusing
    box.classList.add("error");
    show(`语义检索没能载入（${err.message || err}），下面是关键词结果。切换“模型来源”会自动重试。`, 0);
    return false;
  }
}

// ---------- render ----------
let lastSearchKey = null;
let renderSeq = 0;

async function render() {
  const s = readState();
  const seq = ++renderSeq;
  syncForm(s);
  const key = JSON.stringify({ ...s, e: "" });
  if (key !== lastSearchKey) {
    lastSearchKey = key;
    await renderResults(s, seq);
  } else {
    markCurrent(s.e);
  }
  if (seq === renderSeq) renderSheet(s.e);
}

async function renderResults(s, seq) {
  const out = $("#results");
  if (!s.q) {
    out.replaceChildren();
    return;
  }
  let mode = "bm25", qv = null;
  if (s.mode === "hybrid") {
    $("#search-form .primary").disabled = true;
    const ok = await ensureSemantic();
    $("#search-form .primary").disabled = false;
    if (seq !== renderSeq) return;
    if (ok) { mode = "hybrid"; qv = await semantic.embedder.query(s.q); }
  }
  const opts = {
    topk: s.k, mode, repo: s.repo || undefined, category: s.category || undefined,
    confidence: s.confidence || undefined, trigger: s.trigger || undefined, pattern: s.pattern || undefined,
    include_neg: !s.no_neg, include_pointers: !!s.pointers,
  };
  const hits = index.search(s.q, opts, qv);
  const exact = index.exactLookup(s.q);
  const nodes = [];

  if (exact.length) {
    nodes.push(h("p", { class: "group-title" }, "编号或名称完全匹配"), ...exact.map((row) => hitNode({ ...docs[row], row }, null)));
    if (exact.length > 1 || docs[exact[0]].name_ambiguous) {
      nodes.push(h("p", { class: "notice" }, "同一个名称对应多条记录：引用时务必写全编目、编号、名称和源文件。"));
    }
  } else if (index.isIdShaped(s.q)) {
    nodes.push(h("p", { class: "notice" }, `编目里没有编号为 ${s.q} 的条目。编号查询不做模糊匹配；若是上游刚写入的新条目，要等下一次快照同步。`));
  }

  nodes.push(h("div", { class: "results-head" },
    h("span", {}, mode === "hybrid" ? "关键词 + 语义（RRF 融合）" : "关键词（BM25）"),
    h("span", {}, hits.length ? `前 ${hits.length} 条` : "")));
  if (!hits.length) {
    nodes.push(h("p", { class: "notice" }, mode === "bm25"
      ? "没有任何关键词命中。这不代表编目里没有相关故障：换个说法，或改用“关键词 + 语义”再查。"
      : "当前过滤条件下没有合格条目，放宽过滤再试。"));
  }
  nodes.push(...hits.map((r) => hitNode(r, mode)));
  if (hits.length === s.k && s.k < 50) {
    nodes.push(h("button", { type: "button", class: "more", onclick: () => writeState({ ...readState(), k: s.k + PAGE }) }, "显示更多"));
  }
  if (seq !== renderSeq) return;
  out.replaceChildren(...nodes);
  markCurrent(s.e);
}

function stampFor(e) {
  if (e.neg) return h("span", { class: "stamp negative", title: "这条路被否过，不是可复用的故障模式" }, "负结果");
  if (e.entry_type === "pointer") return h("span", { class: "stamp pointer", title: "指向其他记录的说明，不是独立故障" }, "指针");
  const c = e.confidence;
  if (!c || isTbd(c)) return h("span", { class: "stamp none" }, "置信度待定");
  return h("span", { class: `stamp ${c}`, title: CONF_NOTE[c] || "" }, c);
}

function summaryLines(summary) {
  if (!summary) return [];
  if (summary.startsWith("负结果：")) return [h("span", {}, summary)];
  return summary.split(/｜(?=原因：|修复：)/).map((part) => {
    const m = part.match(/^(现象|原因|修复)：([\s\S]*)$/);
    if (!m) return h("span", {}, part);
    if (m[1] === "修复" && m[2] === "未记录") {
      return h("span", { title: "编目没写修复，不等于上游没修；详情页的证据链接里可能有已合入的 PR" }, h("b", {}, "修复"), "编目未记录");
    }
    return h("span", {}, h("b", {}, m[1]), m[2]);
  });
}

function hitNode(r, mode) {
  const key = `${r.repo}/${r.id}`;
  const facts = [
    h("span", {}, "编目", h("b", {}, REPO_LABEL[r.repo] || r.repo)),
    r.category ? h("span", {}, "子系统", h("b", {}, r.category)) : null,
    r.legacy_ids && r.legacy_ids.length ? h("span", {}, "旧号", h("b", {}, r.legacy_ids.join("、"))) : null,
    r.pattern ? h("span", { title: "同族只表示同一底层问题的一族，不等于已核实同机制" }, "故障模式", h("b", {}, r.pattern)) : null,
    r.status === "pending_audit" ? h("span", { class: "warn" }, "待审计（每日扫描新写入）") : null,
    r.name_ambiguous ? h("span", { class: "warn" }, `同名：${r.same_name_ids.join("、")}`) : null,
  ];
  if (mode === "hybrid") {
    facts.push(r.bm25_rank == null
      ? h("span", { class: "warn" }, "仅语义命中，无关键词命中：读原文再判断")
      : h("span", {}, "名次", h("b", {}, `关键词 ${r.bm25_rank + 1}、语义 ${r.dense_rank + 1}`)));
  }
  return h("button", {
    type: "button", class: "hit" + (r.neg ? " neg" : ""), "data-key": key,
    onclick: () => writeState({ ...readState(), e: key }, { push: true }),
  },
  h("span", { class: "hit-head" }, h("span", { class: "acc" }, r.id), h("span", { class: "snake" }, r.name), stampFor(r)),
  r.title ? h("p", { class: "hit-title" }, r.title) : null,
  h("p", { class: "hit-summary" }, summaryLines(r.summary)),
  h("p", { class: "facts" }, facts));
}

function markCurrent(key) {
  for (const b of document.querySelectorAll(".hit")) b.setAttribute("aria-current", b.dataset.key === key ? "true" : "false");
}

// ---------- entry sheet ----------
async function renderSheet(key) {
  const sheet = $("#sheet");
  if (!key) {
    sheet.replaceChildren(h("div", { class: "sheet-empty" },
      h("p", {}, "选中一条结果，在这里阅读完整条目：现象、原因、证据链接、原始说明和关联条目。"),
      h("p", {}, "需要引用时，请写全编目、编号、名称和源文件，详情页底部给出可直接复制的引用行。")));
    return;
  }
  const [repo, id] = key.split("/");
  const row = index.byKey.get(`${repo}\u0000${id}`);
  if (row === undefined) {
    sheet.replaceChildren(h("p", { class: "notice" }, `本快照里没有 ${key}。`));
    return;
  }
  let d;
  try {
    d = await getJSON(`data/entry/${encodeURIComponent(repo)}/${encodeURIComponent(id)}.json`);
  } catch (err) {
    sheet.replaceChildren(h("p", { class: "notice" }, `条目没能载入：${err.message}`));
    return;
  }
  if (readState().e !== key) return;
  sheet.replaceChildren(...sheetNodes(d, row));
  if (matchMedia("(max-width: 960px)").matches) sheet.scrollIntoView({ block: "start" });
  sheet.scrollTop = 0;
}

function section(title, ...body) {
  const kids = body.flat(Infinity).filter((x) => x != null && x !== false);
  return kids.length ? h("section", {}, h("h3", {}, title), kids) : null;
}
function dl(pairs) {
  const rows = pairs.filter(([, v]) => v !== undefined);
  return rows.length ? h("dl", {}, rows.map(([k, v]) => [h("dt", {}, k), h("dd", {}, value(v))])) : null;
}

function sourceUrl(d) {
  if (!meta.repo_url || !meta.site_commit) return null;
  const lines = d.file.endsWith(".md") && d.line ? `#L${d.line}-L${d.line_end || d.line}` : "";
  return `${meta.repo_url}/blob/${meta.site_commit}/${d.file}${lines}`;
}

function citation(d) {
  const legacy = d.legacy_ids && d.legacy_ids.length ? `旧号 ${d.legacy_ids.join("、")}；` : "";
  const where = d.file.endsWith(".md") && d.line ? `${d.file}:${d.line}-${d.line_end || d.line}` : d.file;
  const conf = d.neg ? "负结果" : d.entry_type === "pointer" ? "指针条目" : `置信度 ${d.confidence || "待定"}`;
  return `${REPO_LABEL[d.repo]}编目 ${d.id} ${d.name}（${legacy}${where}；${conf}）`;
}

function sheetNodes(d, row) {
  const y = d.yaml || {};
  const nodes = [
    h("p", { class: "sheet-acc" }, d.id, h("small", {}, `${REPO_LABEL[d.repo]}编目`)),
    h("p", { class: "snake-line" }, h("code", {}, d.name)),
    d.title ? h("h2", {}, d.title) : null,
    h("div", { class: "stamps" }, stampFor(d),
      d.status === "pending_audit" ? h("span", { class: "stamp pending" }, "待审计") : null,
      y.kind ? h("span", { class: "stamp none" }, y.kind) : null),
  ];
  if (d.neg) nodes.push(h("p", { class: "banner" }, "负结果：它告诉你“这条路被否过”，不要当成可复用的故障模式引用。"));
  if (d.entry_type === "pointer") nodes.push(h("p", { class: "banner" }, "指针条目：指向其他记录的说明，不是独立故障，不能作为故障证据引用。顺着下面的关联条目找到真实条目。"));
  if (d.status === "pending_audit") nodes.push(h("p", { class: "banner caution" }, "每日扫描刚写入、尚未经过次日审计的条目，引用时请注明。"));
  if (d.confidence === "speculative") nodes.push(h("p", { class: "banner caution" }, "置信度 speculative：推测，不能当作检测结论。"));
  if (d.name_ambiguous) nodes.push(h("p", { class: "banner caution" }, `同一编目内有同名条目（${d.same_name_ids.join("、")}），它们是不同的来源记录。引用时必须写全编目、编号、名称和源文件。`));

  if (d.neg) {
    nodes.push(section("原文", h("pre", {}, d.fields.content || "")));
  } else {
    const eff = y.effect || {};
    nodes.push(section("现象", dl([
      ["影响对象", eff.objects], ["表现", eff.detail], ["可见性", eff.visibility],
    ])));
    const cause = y.cause || {};
    const objs = cause.objects || [];
    const n = (i) => (objs.length > 1 ? `（${i + 1}）` : "");
    nodes.push(section("原因", dl([
      ...objs.flatMap((o, i) => [["子系统" + n(i), o.subsystem], ["技术栈" + n(i), o.stack], ["组件" + n(i), o.component]]),
      ["行为", cause.behavior],
    ])));
    const mit = y.mitigation || {};
    nodes.push(section("检测与修复", dl([["检测", mit.detect], ["修复", mit.fix]]),
      isTbd(mit.fix) ? h("p", { class: "note" }, "编目没写修复不等于上游没修：抽样中约三分之一的来源 issue 其实已有合入的 PR。需要修复信息时请看下面的证据链接。") : null));
    nodes.push(section("证据", h("ul", { class: "links" }, (y.evidence || []).map((ev) => {
      const url = safeUrl(ev.url);
      return h("li", {}, url ? h("a", { href: url, rel: "noopener", target: "_blank" }, url) : prose(ev.text || ev.url || ""),
        ev.kind ? h("span", { class: "kind" }, ev.kind) : null);
    }))));
    const orig = y.original || {};
    const order = ["mechanism", "behavior", "signature", "trigger_surface", "repro", "source", "discovery", "notes",
                   "locus", "priority", "meta", "category", "section"];
    const keys = [...order.filter((k) => k in orig), ...Object.keys(orig).filter((k) => !order.includes(k))];
    nodes.push(section("原始说明",
      h("p", { class: "note" }, "扫描写下的完整原始叙述；上面的结构化字段是从这里抽取的摘要，下结论前请读这一节。"),
      dl(keys.map((k) => [ORIGINAL_LABEL[k] || k, orig[k]]))));
    const labels = y.labels || {};
    nodes.push(section("标签", dl(Object.keys(labels).map((k) => [LABEL_LABEL[k] || k, labels[k]]))));
    if (!isTbd(y.remarks)) nodes.push(section("备注", h("p", {}, prose(y.remarks))));
  }

  if (d.pattern) {
    nodes.push(section("故障模式",
      h("p", {}, "属于 ", h("code", {}, d.pattern), "。 ",
        h("button", { type: "button", class: "more", onclick: () => writeState({ ...readState(), pattern: d.pattern, k: PAGE }) }, "只看这一族")),
      h("p", { class: "note" }, "模式成员由变体关系派生，只表示同族，不等于已核实同机制。")));
  }

  const rel = index.relatedFrom([row], 1);
  nodes.push(section("关联条目",
    rel.length ? h("ul", { class: "links" }, rel.map(({ row: r, link_kind }) => {
      const e = docs[r];
      return h("li", {},
        h("button", { type: "button", onclick: () => writeState({ ...readState(), e: `${e.repo}/${e.id}` }, { push: true }) },
          `${e.id} ${e.title || e.name}`),
        h("span", { class: "kind" }, `${REPO_LABEL[e.repo]}，${LINK_KIND[link_kind] || link_kind}`));
    })) : h("p", { class: "note" }, "没有记录在案的关联。"),
    rel.length ? h("p", { class: "note" }, "关联只用于导航，不能据此断定两个故障机制相同。") : null));

  const cite = citation(d);
  const copyBtn = h("button", { type: "button", onclick: async (ev) => {
    try { await navigator.clipboard.writeText(cite); ev.target.textContent = "已复制"; }
    catch { ev.target.textContent = "请手动复制"; }
  } }, "复制");
  const src = sourceUrl(d);
  nodes.push(section("引用",
    h("div", { class: "cite" }, h("code", {}, cite), copyBtn),
    src ? h("p", { class: "note" }, h("a", { href: src, rel: "noopener", target: "_blank" }, "在 GitHub 查看源文件"),
      "（本站快照中的版本）") : null));
  return nodes;
}

boot().catch((err) => {
  $("#snapshot").textContent = `编目数据没能载入：${err.message}`;
});
