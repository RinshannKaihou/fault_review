// Module-centric LLM lifecycle causal workbench for fault-rag.
// Supports both static-site fetch("data/causal_graph.json") and single-file offline mode (window.__INLINE_CAUSAL_GRAPH__).

const MODULES = [
  // Row 1: Training, Checkpoint, Quantization, RL
  {
    id: "m_data", name: "数据管线", sub: "采样 / Packing", x: 8, y: 34, w: 134, h: 56, row: 1,
    subs: ["数据管线"],
    carriers: ["训练样本流"],
    probe: "核对单步 DataLoader 输出的 sample 顺序、position_ids 与 packed 文档边界 loss_mask 是否越界串扰。",
    walkthrough: "数据与样本流异常主要由【数据管线】自身（采样器丢种子、packing 未切因果掩码）或 RL 在线样本回灌引起。",
  },
  {
    id: "m_loss", name: "损失/优化器", sub: "Loss / 梯度 / 重算", x: 176, y: 34, w: 148, h: 56, row: 1,
    subs: ["优化器", "损失计算", "重计算/卸载"],
    carriers: ["训练信号", "梯度", "优化器状态", "训练轨迹"],
    probe: "开启非零 LR 单步对比 TP=1/无重算基线的逐层 grad norm，隔离「损失/优化器配置缺陷」与「上游前向激活或底层算子算错」。",
    walkthrough: "顺箭头倒推：① 先查【损失/优化/重算】归约与混合精度缩放 → ② 若跑 RL，查右上【RL训练回路 ← 解码采样】→ ③ 查下方【模型前向层 ← 分词 ← 数据管线】→ ④ 重点查脚下【算子/内核】与【并行通信】（下游推理模块已自动排除）。",
  },
  {
    id: "m_ckpt", name: "权重/检查点", sub: "保存 / 转换 / 分片", x: 366, y: 34, w: 146, h: 56, row: 1,
    subs: ["权重/检查点"],
    carriers: ["模型权重"],
    probe: "比对加载后各层参数 shape/dtype/分片哈希与源 ckpt 键名表（排查静默随机初始化、EP 专家错位或训练期坏更新继承）。",
    walkthrough: "检查点是训推交界枢纽：既可能因【权重/检查点】自身转换/分片映射写错，也可能继承了左侧【训练优化器/硬件SDC】写坏的参数。",
  },
  {
    id: "m_quant", name: "量化模块", sub: "FP8 / INT4 / Scale", x: 552, y: 34, w: 146, h: 56, row: 1,
    subs: ["量化"],
    carriers: ["模型权重"],
    probe: "切换回 BF16/FP16 原生权重对拍首步隐状态与 logits，一键二分是否由 PTQ scale/zero-point 或低位量化核引起。",
    walkthrough: "量化位于检查点与推理执行之间：重点核查 PTQ 导出校准、融合权重共用 scale 丢失，以及在线 FP8/INT4 低位算子精度。",
  },
  {
    id: "m_rl", name: "RL训练回路", sub: "Reward / KL / 同步", x: 730, y: 34, w: 142, h: 56, row: 1,
    subs: ["RL训练回路"],
    carriers: ["训练信号", "训练轨迹"],
    probe: "冻结同一批 Rollout 采样文本与 logprobs 离线重算 advantage/KL，并在权重同步后强制清空推理侧 Prefix Cache 复测。",
    walkthrough: "RL 是训推紧耦合闭环：① 查【RL训练回路】advantage/KL 归约与权重同步 → ② 顺 Rollout 箭头查推理侧【解码/采样 ← KV缓存】logprobs 一致性 → ③ 查底层训推并行度不同导致的【算子/通信】浮点漂移。",
  },

  // Row 2: Unbroken Left-to-Right Forward & Serving Pipeline
  {
    id: "m_tok", name: "分词/模板", sub: "Tokenizer / 模板", x: 8, y: 136, w: 134, h: 56, row: 2,
    subs: ["分词/模板"],
    carriers: ["输入编码"],
    probe: "在进入 Embedding 层前直接打印 exact input_ids 与多模态张量哈希，核对训推 chat template、特殊 token 与图像归一化是否一致。",
    walkthrough: "源头输入编码异常（82% 为静默错误）：一旦模板渲染、特殊 token 或多模态预处理错位，后续所有健康模块都会跟着算出错结果。",
  },
  {
    id: "m_model", name: "模型前向层", sub: "架构 / RoPE / MoE", x: 176, y: 136, w: 148, h: 56, row: 2,
    subs: ["MoE路由", "模型定义", "位置编码", "模型自身"],
    carriers: ["数值结果", "隐状态"],
    probe: "按层二分导出 hidden_states 与 FP32 eager 参考实现对比余弦相似度，定位首个偏离层（区分架构/RoPE/MoE 逻辑与底层算子/编译问题）。",
    walkthrough: "前向激活与数值异常是全库最大汇流面：除【模型定义 / RoPE / MoE路由】自身外，超过一半根因来自正下方的【算子/内核】与【编译器】地基，以及上方加载的【量化 / 检查点权重】。",
  },
  {
    id: "m_kv", name: "调度/KV缓存", sub: "Prefill / 前缀缓存", x: 366, y: 136, w: 146, h: 56, row: 2,
    subs: ["KV缓存", "调度/服务"],
    carriers: ["KV状态", "跨请求/跨样本数据", "可用性/进度", "性能/资源"],
    probe: "关闭 Prefix Caching 与 Chunked Prefill，并设并发数 concurrency=1 重跑；若异常消失，即可锁定在调度批处理与 KV 块管理层。",
    walkthrough: "顺箭头倒推：① 首查【KV缓存 & 调度服务 (485条)】块表分配、前缀哈希命中与并发槽位隔离 → ② 查上游【量化 KV scale】与【模型前向 RoPE/注意力】→ ③ 查底层【Attention 算子/运行时】。",
  },
  {
    id: "m_sample", name: "解码/采样", sub: "Logits / 投机解码", x: 552, y: 136, w: 146, h: 56, row: 2,
    subs: ["解码/采样"],
    carriers: ["输出分布"],
    probe: "设 temperature=0 并关闭投机解码（Speculative Decoding）与语法掩码，直接导出首步 top-k logprobs 对拍。",
    walkthrough: "顺箭头倒推：① 查【解码/采样】惩罚项、投机草稿验收分布与语法约束掩码 → ② 往左查【调度/KV缓存】与【模型前向/量化】传来的 logits 是否已经偏离 → ③ 查底层采样算子。",
  },
  {
    id: "m_parse", name: "输出解析/评测", sub: "解析器 / 评测门禁", x: 730, y: 136, w: 142, h: 56, row: 2,
    subs: ["观测/评测", "输出解析"],
    carriers: ["输出内容", "测试判定", "指标/日志"],
    probe: "第一步先看引擎原始输出 token IDs：若 token IDs 正确而 API 字段错，直接锁定【输出解析/评测脚本】；若 token IDs 已错，则顺箭头往左二分查采样与 KV。",
    walkthrough: "这是用户最常看到异常的终点站（超千条故障在此暴露）：① 先核原始 token 排除【输出解析/评测】→ ② 设 temp=0 查【解码/采样】→ ③ 关前缀缓存单并发查【调度/KV缓存】→ ④ 对拍激活查【模型前向/量化/检查点】与底部【算子/内核】地基。",
  },

  // Row 3: Infrastructure Pedestal
  {
    id: "b_hw", name: "硬件/安全/配置", sub: "SDC / 攻击 / 配置", x: 8, y: 248, w: 182, h: 56, row: 3,
    subs: ["硬件", "外部攻击", "配置/参数", "构建/发布"],
    carriers: [],
    probe: "核对多副本共识离群度（检测硬件 SDC 位翻转）、权重文件哈希以及运行时实际生效配置是否被默认值静默覆盖。",
    walkthrough: "物理与外生底座：包含 GPU/NPU 静默数据损坏（SDC）、Rowhammer 权重位翻转攻击、CLI/YAML 参数静默未生效及 wheel 打包漏修。",
  },
  {
    id: "b_dist", name: "运行时 & 并行通信", sub: "Graph / TP·EP / 通信", x: 204, y: 248, w: 216, h: 56, row: 3,
    subs: ["运行时/执行", "并行切分", "集合通信"],
    carriers: [],
    probe: "对比 TP=1/EP=1 单卡 Eager 基线与多卡 CUDA Graph 路径，检查 all-reduce/all-to-all 归约次数、stream 同步与缓存复用。",
    walkthrough: "分布式与设备执行层：涵盖 CUDA/ACL Graph 捕获重放错位、显存分配器复用竞态、TP/PP/DP/EP 双重归约或漏归约及集合通信死锁/错值。",
  },
  {
    id: "b_comp", name: "编译器", sub: "Triton / Inductor", x: 434, y: 248, w: 182, h: 56, row: 3,
    subs: ["编译器"],
    carriers: [],
    probe: "关闭 torch.compile / torchair / XLA 退回 Eager 逐算子执行对比输出；检查融合 pass 是否误消边界掩码或固化静态 shape。",
    walkthrough: "编译优化层（222 条）：Triton / Inductor / XLA / torchair 在算子融合、循环切分或自动降精度 pass 中引入静默数值偏差。",
  },
  {
    id: "b_kern", name: "算子/内核 (最大根因)", sub: "占全库故障 31%", x: 630, y: 248, w: 242, h: 56, row: 3,
    subs: ["算子/内核"],
    carriers: [],
    probe: "固定完全相同的输入 shape/stride/dtype，单独摘出可疑算子与 PyTorch FP32 参考实现做边界 tiling 与非连续张量对拍。",
    walkthrough: "全库第一大故障根源（1,011 条，80% 为静默错误）：自身不直接面向用户，但一旦 MatMul/Attention/MoE/Norm 内核算错，上方所有训练和推理模块都会跟着出问题。",
  },
];

const EDGES = [
  { from: "m_data",   to: "m_tok",    lbl: "样本",     d: "M 75 90 L 75 134",    lx: 92,  ly: 116 },
  { from: "m_tok",    to: "m_model",  lbl: "Token",    d: "M 142 164 L 174 164", lx: 158, ly: 156 },
  { from: "m_model",  to: "m_kv",     lbl: "隐状态",   d: "M 324 164 L 364 164", lx: 344, ly: 156 },
  { from: "m_kv",     to: "m_sample", lbl: "KV读取",   d: "M 512 164 L 550 164", lx: 531, ly: 156 },
  { from: "m_sample", to: "m_parse",  lbl: "文本",     d: "M 698 164 L 728 164", lx: 713, ly: 156 },

  { from: "m_model",  to: "m_loss",   lbl: "激活",     d: "M 250 136 L 250 92",  lx: 270, ly: 116 },
  { from: "m_loss",   to: "m_ckpt",   lbl: "梯度更新", d: "M 324 62 L 364 62",   lx: 344, ly: 54 },
  { from: "m_ckpt",   to: "m_quant",  lbl: "导出",     d: "M 512 62 L 550 62",   lx: 531, ly: 54 },
  { from: "m_ckpt",   to: "m_model",  lbl: "加载权重", d: "M 392 90 L 306 134",  lx: 368, ly: 116 },
  { from: "m_quant",  to: "m_kv",     lbl: "量化权重", d: "M 578 90 L 492 134",  lx: 556, ly: 116 },

  { from: "m_sample", to: "m_rl",     lbl: "Rollout",  d: "M 672 136 L 762 92",  lx: 736, ly: 118 },
  { from: "m_rl",     to: "m_loss",   lbl: "RL信号回流", d: "M 800 34 Q 545 12 278 34", lx: 565, ly: 21 },

  { from: "b_hw",     to: "b_dist",   lbl: "",         d: "M 190 276 L 202 276" },
  { from: "b_dist",   to: "b_comp",   lbl: "",         d: "M 420 276 L 432 276" },
  { from: "b_comp",   to: "b_kern",   lbl: "",         d: "M 616 276 L 628 276" },
];

const PRUNER_RULES = [
  { id: "single_gpu",    label: "单卡运行 (排除并行切分/通信)", subs: ["并行切分", "集合通信"] },
  { id: "single_req",    label: "单并发关前缀缓存 (排除缓存池)", subs: ["KV缓存"] },
  { id: "no_quant",      label: "原生 BF16/FP16 (排除量化)",    subs: ["量化"] },
  { id: "eager_mode",    label: "Eager 模式 (排除编译器)",      subs: ["编译器"] },
  { id: "greedy_nospec", label: "Temp=0 关投机 (排除采样器)",   subs: ["解码/采样"] },
  { id: "raw_tok_bad",   label: "Raw Token 已错 (排除输出解析)", subs: ["输出解析"] },
];

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

const $ = (s) => document.querySelector(s);
const MOD_MAP = Object.fromEntries(MODULES.map((m) => [m.id, m]));
const SUB_TO_MOD = {};
MODULES.forEach((m) => m.subs.forEach((s) => { SUB_TO_MOD[s] = m.id; }));

let graph, dicts, stToGroupMask;
const detailCache = new Map();

const state = {
  mode: "backward",      // 'backward' (symptom -> root cause) | 'forward' (module -> its own faults & blast radius)
  selectedMod: "m_parse",
  pinnedSub: null,       // specific vocab.yaml subsystem filter
  stage: "all",
  stackGroup: "all",
  vis: "all",
  pruners: new Set(),
  tab: "entries",
  expandedId: null,
};

async function boot() {
  if (window.__INLINE_CAUSAL_GRAPH__) {
    graph = window.__INLINE_CAUSAL_GRAPH__;
    const navLink = document.querySelector(".masthead .nav-pill");
    if (navLink) navLink.hidden = true;
  } else {
    const res = await fetch("data/causal_graph.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    graph = await res.json();
  }
  dicts = graph.dictionaries;

  stToGroupMask = {};
  dicts.stacks.forEach((sname, idx) => {
    let m = 0;
    dicts.stack_groups.forEach((grp, gi) => {
      if (grp.members.includes(sname)) m |= (1 << gi);
    });
    stToGroupMask[idx] = m;
  });

  // Precompute symptom_mods for each fault row once so runtime interaction is <1ms
  const srvIdx = dicts.stages.indexOf("serving");
  const rlIdx = dicts.stages.indexOf("rl");
  const preIdx = dicts.stages.indexOf("pretrain");

  graph.faults.forEach((r) => {
    const sNames = r[4].map((i) => dicts.subsystems[i]);
    const cNames = r[5].map((i) => dicts.carriers[i]);
    const isSrv = r[6].includes(srvIdx);
    const isRlFocused = (r[6].includes(rlIdx) && !r[6].includes(preIdx)) || sNames.includes("RL训练回路");

    const smods = new Set();
    cNames.forEach((c) => {
      if (c === "训练样本流") smods.add("m_data");
      else if (c === "输入编码") smods.add("m_tok");
      else if (c === "隐状态" || c === "数值结果") smods.add("m_model");
      else if (c === "KV状态") smods.add("m_kv");
      else if (c === "跨请求/跨样本数据") smods.add(isSrv ? "m_kv" : "m_data");
      else if (c === "输出分布") smods.add("m_sample");
      else if (c === "输出内容" || c === "测试判定" || c === "指标/日志") smods.add("m_parse");
      else if (c === "训练信号") smods.add(isRlFocused ? "m_rl" : "m_loss");
      else if (c === "梯度" || c === "优化器状态" || c === "训练轨迹") {
        smods.add(sNames.includes("RL训练回路") ? "m_rl" : "m_loss");
      } else if (c === "模型权重") {
        smods.add(sNames.includes("量化") ? "m_quant" : "m_ckpt");
      } else if (c === "可用性/进度" || c === "性能/资源") {
        smods.add(isSrv ? "m_kv" : "m_loss");
      }
    });
    r._symMods = [...smods];
    r._causeMods = [...new Set(sNames.map((s) => SUB_TO_MOD[s]).filter(Boolean))];
  });

  initControls();
  renderAll();
}

function initControls() {
  const stkSel = $("#sel-stack");
  dicts.stack_groups.forEach((g, idx) => {
    stkSel.append(h("option", { value: String(idx) }, g.label));
  });
  const visSel = $("#sel-vis");
  dicts.visibilities.forEach((v, idx) => {
    if (v !== "待定") visSel.append(h("option", { value: String(idx) }, v));
  });

  $("#btn-mode-back").addEventListener("click", () => {
    state.mode = "backward";
    if (MOD_MAP[state.selectedMod].row === 3) state.selectedMod = "m_parse";
    state.pinnedSub = null;
    renderAll();
  });
  $("#btn-mode-fwd").addEventListener("click", () => {
    state.mode = "forward";
    state.pinnedSub = null;
    renderAll();
  });

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mode = "backward";
      state.selectedMod = btn.dataset.preset;
      state.pinnedSub = null;
      state.expandedId = null;
      renderAll();
    });
  });

  $("#sel-stage").addEventListener("change", (e) => { state.stage = e.target.value; renderAll(); });
  $("#sel-stack").addEventListener("change", (e) => { state.stackGroup = e.target.value; renderAll(); });
  $("#sel-vis").addEventListener("change", (e) => { state.vis = e.target.value; renderAll(); });

  $("#btn-reset").addEventListener("click", () => {
    state.stage = "all"; state.stackGroup = "all"; state.vis = "all";
    state.pinnedSub = null; state.pruners.clear(); state.expandedId = null;
    $("#sel-stage").value = "all"; $("#sel-stack").value = "all"; $("#sel-vis").value = "all";
    $("#entry-q").value = "";
    renderAll();
  });
  $("#btn-unpin").addEventListener("click", () => { state.pinnedSub = null; renderAll(); });
  $("#btn-clear-pruners").addEventListener("click", () => { state.pruners.clear(); renderAll(); });

  const prnBox = $("#pruner-pills");
  PRUNER_RULES.forEach((r) => {
    prnBox.append(h("button", {
      type: "button",
      id: `prn-${r.id}`,
      class: "prn-pill",
      onclick: () => {
        if (state.pruners.has(r.id)) state.pruners.delete(r.id);
        else state.pruners.add(r.id);
        renderAll();
      },
    }, r.label));
  });

  ["entries", "patterns", "mysteries"].forEach((t) => {
    $(`#tab-${t}`).addEventListener("click", () => { state.tab = t; renderTabs(); });
  });
  $("#entry-q").addEventListener("input", () => renderTabs());
}

function getPrunedSubs() {
  const s = new Set();
  if (state.mode === "backward") {
    PRUNER_RULES.forEach((r) => {
      if (state.pruners.has(r.id)) r.subs.forEach((x) => s.add(x));
    });
  }
  return s;
}

function filterBaseRows() {
  const stgIdx = state.stage === "all" ? -1 : dicts.stages.indexOf(state.stage);
  const stkBit = state.stackGroup === "all" ? -1 : Number(state.stackGroup);
  const visIdx = state.vis === "all" ? -1 : Number(state.vis);
  const prunedSubs = getPrunedSubs();

  return graph.faults.filter((row) => {
    if (stgIdx >= 0 && !row[6].includes(stgIdx)) return false;
    if (stkBit >= 0) {
      let m = 0;
      for (const si of row[7]) m |= (stToGroupMask[si] || 0);
      if (!(m & (1 << stkBit))) return false;
    }
    if (visIdx >= 0 && row[8] !== visIdx) return false;
    if (prunedSubs.size > 0 && row[4].length > 0 && row[4].every((si) => prunedSubs.has(dicts.subsystems[si]))) {
      return false;
    }
    return true;
  });
}

function traceGraph(modId, mode, prunedSubs) {
  const activeNodes = new Set([modId]);
  const activeEdges = new Set();
  const dist = { [modId]: 0 };

  if (mode === "backward") {
    const queue = [modId];
    while (queue.length) {
      const cur = queue.shift();
      EDGES.forEach((e, idx) => {
        if (e.to === cur) {
          // Pure online serving symptom does not loop back into training optimizer/RL unless inspecting ckpt/loss/rl directly
          if (modId === "m_parse" && (e.from === "m_loss" || e.from === "m_rl")) return;
          if (modId === "m_kv" && (e.from === "m_loss" || e.from === "m_rl")) return;
          const srcMod = MOD_MAP[e.from];
          if (srcMod && srcMod.subs.every((s) => prunedSubs.has(s))) return;
          activeEdges.add(idx);
          if (!activeNodes.has(e.from)) {
            activeNodes.add(e.from);
            dist[e.from] = dist[cur] + 1;
            queue.push(e.from);
          }
        }
      });
    }
    // Include unpruned foundation modules
    ["b_hw", "b_dist", "b_comp", "b_kern"].forEach((bid) => {
      if (!MOD_MAP[bid].subs.every((s) => prunedSubs.has(s))) activeNodes.add(bid);
    });
    [12, 13, 14].forEach((i) => activeEdges.add(i));
  } else {
    // Forward blast radius from modId
    if (MOD_MAP[modId].row === 3) {
      MODULES.forEach((m) => { if (m.row !== 3 || m.id === modId) activeNodes.add(m.id); });
      EDGES.forEach((_, idx) => { if (idx < 12) activeEdges.add(idx); });
    } else {
      const queue = [modId];
      while (queue.length) {
        const cur = queue.shift();
        EDGES.forEach((e, idx) => {
          if (e.from === cur) {
            activeEdges.add(idx);
            if (!activeNodes.has(e.to)) {
              activeNodes.add(e.to);
              dist[e.to] = dist[cur] + 1;
              queue.push(e.to);
            }
          }
        });
      }
    }
  }
  return { activeNodes, activeEdges, dist };
}

let currentScopedRows = [];
let currentSliceRows = [];

function renderAll() {
  const rows = filterBaseRows();
  currentSliceRows = rows;
  const prunedSubs = getPrunedSubs();
  const silentIdx = dicts.visibilities.indexOf("静默错误");

  // Compute per-module root_cause & symptom_seen counts under active filters
  const modCauseCnt = Object.fromEntries(MODULES.map((m) => [m.id, 0]));
  const modSymCnt = Object.fromEntries(MODULES.map((m) => [m.id, 0]));
  const modSilentCnt = Object.fromEntries(MODULES.map((m) => [m.id, 0]));
  const subCntGlobal = Object.fromEntries(dicts.subsystems.map((s) => [s, 0]));

  rows.forEach((r) => {
    const isSil = r[8] === silentIdx;
    r._causeMods.forEach((mid) => {
      modCauseCnt[mid]++;
      if (isSil) modSilentCnt[mid]++;
    });
    r._symMods.forEach((mid) => {
      modSymCnt[mid]++;
      if (MOD_MAP[mid].row !== 3 && isSil) modSilentCnt[mid] = modSilentCnt[mid]; // keep cause silent on foundation
    });
    r[4].forEach((si) => { subCntGlobal[dicts.subsystems[si]]++; });
  });

  $("#btn-mode-back").classList.toggle("active", state.mode === "backward");
  $("#btn-mode-fwd").classList.toggle("active", state.mode === "forward");
  document.querySelectorAll(".preset-btn").forEach((b) => {
    b.classList.toggle("active", state.mode === "backward" && state.selectedMod === b.dataset.preset);
  });

  const pedTotal = ["b_hw", "b_dist", "b_comp", "b_kern"].reduce((a, k) => a + modCauseCnt[k], 0);
  $("#pedestal-caption").textContent =
    `↑ 底层软硬件承重地基（当前切片共 ${pedTotal.toLocaleString()} 条故障源于此地基；点任一地基模块可看其自身故障库）`;

  $("#causal-scope").replaceChildren(
    "当前切片 ", h("b", {}, rows.length.toLocaleString()), " / ",
    graph.normalization_stats.total_faults.toLocaleString(), " 条故障"
  );

  const trace = traceGraph(state.selectedMod, state.mode, prunedSubs);
  renderSVG(modCauseCnt, modSymCnt, trace, prunedSubs);
  renderAllSubsStrip(subCntGlobal, prunedSubs);
  renderInspector(rows, modCauseCnt, modSymCnt, prunedSubs);
}

function renderSVG(modCauseCnt, modSymCnt, trace, prunedSubs) {
  const edgesG = $("#svg-edges");
  const nodesG = $("#svg-nodes");
  edgesG.replaceChildren();
  nodesG.replaceChildren();

  EDGES.forEach((e, idx) => {
    const hi = trace.activeEdges.has(idx);
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", e.d);
    p.setAttribute("fill", "none");
    p.setAttribute("stroke", hi ? "var(--action)" : "var(--muted)");
    p.setAttribute("stroke-width", hi ? "2.3" : "1.2");
    p.setAttribute("opacity", hi ? "0.95" : "0.2");
    p.setAttribute("marker-end", hi ? "url(#arr-act)" : "url(#arr-dim)");
    edgesG.append(p);

    if (e.lbl) {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", e.lx);
      t.setAttribute("y", e.ly);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("font-size", "12");
      t.setAttribute("fill", hi ? "var(--ink)" : "var(--muted)");
      t.setAttribute("opacity", hi ? "0.92" : "0.26");
      t.textContent = e.lbl;
      edgesG.append(t);
    }
  });

  MODULES.forEach((m) => {
    const isSel = m.id === state.selectedMod;
    const allPruned = m.subs.every((s) => prunedSubs.has(s));
    const hi = !allPruned && trace.activeNodes.has(m.id);

    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "m-node");
    g.setAttribute("opacity", allPruned ? "0.18" : (hi ? "1" : "0.24"));
    g.addEventListener("click", () => {
      if (m.row === 3) {
        state.mode = "forward";
      }
      state.selectedMod = m.id;
      state.pinnedSub = null;
      state.expandedId = null;
      renderAll();
    });

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", m.x);
    rect.setAttribute("y", m.y);
    rect.setAttribute("width", m.w);
    rect.setAttribute("height", m.h);
    rect.setAttribute("rx", "6");
    rect.setAttribute("fill", "var(--paper)");
    rect.setAttribute("stroke", isSel ? "var(--speculative)" : (hi ? "var(--action)" : "var(--rule)"));
    rect.setAttribute("stroke-width", isSel ? "2.6" : (hi ? "1.6" : "1"));
    g.append(rect);

    const mkText = (txt, x, y, anchor, weight, fill) => {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", x);
      t.setAttribute("y", y);
      if (anchor) t.setAttribute("text-anchor", anchor);
      t.setAttribute("font-size", "12");
      if (weight) t.setAttribute("font-weight", weight);
      t.setAttribute("fill", fill);
      t.textContent = txt;
      return t;
    };

    g.append(
      mkText(m.name, m.x + 8, m.y + 18, null, "700", "var(--ink)"),
      mkText(m.sub, m.x + 8, m.y + 34, null, null, "var(--muted)"),
      mkText(
        m.row === 3 ? `底层根因 ${modCauseCnt[m.id]} 条` : `表象 ${modSymCnt[m.id]} · 根因 ${modCauseCnt[m.id]}`,
        m.x + 8, m.y + 50, null, "600", isSel ? "var(--action)" : "var(--ink)"
      )
    );

    if (allPruned) {
      g.append(mkText("已排除", m.x + m.w - 6, m.y + 18, "end", "600", "var(--muted)"));
    } else if (isSel) {
      g.append(mkText(state.mode === "backward" ? "★观测点" : "★当前模块", m.x + m.w - 6, m.y + 18, "end", "700", "var(--speculative)"));
    } else if (trace.dist[m.id] > 0) {
      const tag = state.mode === "backward" ? `上游-${trace.dist[m.id]}` : `波及+${trace.dist[m.id]}`;
      g.append(mkText(tag, m.x + m.w - 6, m.y + 18, "end", "600", "var(--action)"));
    }

    nodesG.append(g);
  });

  // Walkthrough box below SVG
  const curMod = MOD_MAP[state.selectedMod];
  $("#wt-badge").textContent = state.mode === "backward"
    ? `当前观测异常位置：${curMod.name}`
    : `当前查看模块：${curMod.name}`;
  const inactiveNames = MODULES
    .filter((m) => m.row !== 3 && !trace.activeNodes.has(m.id))
    .map((m) => m.name);
  $("#wt-pruned").textContent = state.mode === "backward" && inactiveNames.length
    ? `已按因果方向自动免查下游模块：${inactiveNames.join("、")}`
    : "高亮箭头表示故障传播方向";
  $("#wt-steps").textContent = curMod.walkthrough;
}

function renderAllSubsStrip(subCntGlobal, prunedSubs) {
  const strip = $("#all-subs-strip");
  strip.replaceChildren();
  Object.entries(subCntGlobal)
    .sort((a, b) => b[1] - a[1])
    .forEach(([sname, cnt]) => {
      const isPruned = prunedSubs.has(sname);
      const isSel = state.pinnedSub === sname || (state.mode === "forward" && MOD_MAP[state.selectedMod].subs.length === 1 && MOD_MAP[state.selectedMod].subs[0] === sname);
      strip.append(h("button", {
        type: "button",
        class: `sub-btn ${isPruned ? "pruned" : ""} ${isSel ? "sel" : ""}`,
        onclick: () => {
          state.mode = "forward";
          state.selectedMod = SUB_TO_MOD[sname];
          state.pinnedSub = MOD_MAP[state.selectedMod].subs.length > 1 ? sname : null;
          state.expandedId = null;
          renderAll();
        },
      }, sname, h("b", {}, String(cnt))));
    });
}

function renderInspector(rows, modCauseCnt, modSymCnt, prunedSubs) {
  const m = MOD_MAP[state.selectedMod];
  const silentIdx = dicts.visibilities.indexOf("静默错误");
  const isStandalone = Boolean(window.__INLINE_CAUSAL_GRAPH__);

  PRUNER_RULES.forEach((r) => {
    const el = $(`#prn-${r.id}`);
    const on = state.pruners.has(r.id);
    el.classList.toggle("on", on);
    el.textContent = `${on ? "✓ " : ""}${r.label}`;
  });

  if (state.mode === "backward") {
    $("#pruner-box").hidden = false;
    $("#sub-filter-wrap").hidden = true;

    const baseSymRows = rows.filter((r) => r._symMods.includes(m.id));
    let scoped = baseSymRows;
    if (state.pinnedSub) {
      const pIdx = dicts.subsystems.indexOf(state.pinnedSub);
      scoped = baseSymRows.filter((r) => r[4].includes(pIdx));
      $("#btn-unpin").hidden = false;
      $("#insp-badge").textContent = "锁定真凶路径";
      $("#insp-title").textContent = `根因【${state.pinnedSub}】➔ 在【${m.name}】暴露`;
    } else {
      $("#btn-unpin").hidden = true;
      $("#insp-badge").textContent = "按表象逆向溯因";
      $("#insp-title").textContent = `在【${m.name}】看到异常时`;
    }
    $("#insp-desc").textContent = `包含编目内表象落在 ${m.carriers.join(" / ")} 的全部案例（含本模块自身缺陷、上游传入及底层地基算错）。`;

    const silCnt = scoped.filter((r) => r[8] === silentIdx).length;
    const prbCnt = scoped.filter((r) => (r[11] & 12) !== 0).length;
    const mysCnt = baseSymRows.filter((r) => (r[11] & 64) !== 0).length;
    $("#insp-kpis").replaceChildren(
      h("div", { class: "kpi-box" }, h("span", {}, "暴露案例 (未定悬案)"), h("strong", {}, `${scoped.length} (${mysCnt})`)),
      h("div", { class: "kpi-box" }, h("span", {}, "静默无报错占比"), h("strong", {}, `${scoped.length ? Math.round((silCnt / scoped.length) * 100) : 0}%`)),
      h("div", { class: "kpi-box" }, h("span", {}, "已有检测/签名"), h("strong", {}, `${scoped.length ? Math.round((prbCnt / scoped.length) * 100) : 0}%`))
    );

    $("#probe-title").textContent = "二分插桩检查建议";
    $("#probe-text").textContent = m.probe;
    const methods = [];
    const seenM = new Set();
    m.carriers.forEach((c) => {
      (graph.methods_by_carrier[c] || []).forEach((mt) => {
        if (!seenM.has(mt.id)) { seenM.add(mt.id); methods.push(mt); }
      });
    });
    $("#probe-methods-cnt").textContent = methods.length ? `可用检测工具 ${methods.length} 个` : "";
    $("#probe-methods").replaceChildren(...methods.map((mt) =>
      isStandalone
        ? h("span", { title: mt.title }, `${mt.id} ${mt.name}`)
        : h("a", { href: `index.html#e=${encodeURIComponent(mt.id)}`, title: mt.title }, `${mt.id} ${mt.name}`)
    ));

    // Rank true root-cause subsystems causing symptoms at module m
    const subHits = {};
    let totHits = 0;
    baseSymRows.forEach((r) => {
      r[4].forEach((si) => {
        const sname = dicts.subsystems[si];
        if (prunedSubs.has(sname)) return;
        subHits[sname] = (subHits[sname] || 0) + 1;
        totHits++;
      });
    });

    $("#bars-heading").textContent = `真凶模块排行榜（在 ${baseSymRows.length} 条暴露案例中谁是真正根因）`;
    const bars = Object.entries(subHits).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([sname, cnt]) => {
      const pct = totHits ? ((cnt / totHits) * 100).toFixed(1) : "0.0";
      const srcMod = MOD_MAP[SUB_TO_MOD[sname]];
      const role = srcMod.id === m.id ? "自身" : (srcMod.row === 3 ? "地基" : "上游");
      return h("button", {
        type: "button",
        class: `bar-row ${state.pinnedSub === sname ? "pinned" : ""}`,
        onclick: () => { state.pinnedSub = state.pinnedSub === sname ? null : sname; renderAll(); },
      },
        h("span", {}, sname, h("span", { class: "role-dot" }, `·${role}`)),
        h("span", { class: "bar-track" }, h("span", { class: "bar-fill", style: `width:${Math.max(4, Number(pct))}%` })),
        h("span", { class: "bar-num" }, `${cnt} (${pct}%)`)
      );
    });
    $("#bars-list").replaceChildren(...(bars.length ? bars : [h("p", { class: "panel-sub" }, "当前筛选与排除条件下无匹配案例。")]));
    currentScopedRows = scoped;

  } else {
    // Forward / Module Fault Library View
    $("#pruner-box").hidden = true;
    const baseCauseRows = rows.filter((r) => r._causeMods.includes(m.id));

    // Show sub-module pills if this node groups multiple vocab subsystems
    const subWrap = $("#sub-filter-wrap");
    if (m.subs.length > 1) {
      subWrap.hidden = false;
      subWrap.replaceChildren(
        h("button", {
          type: "button",
          class: `sub-btn ${!state.pinnedSub ? "sel" : ""}`,
          onclick: () => { state.pinnedSub = null; renderAll(); },
        }, "本框全部", h("b", {}, String(baseCauseRows.length))),
        ...m.subs.map((s) => {
          const sIdx = dicts.subsystems.indexOf(s);
          const n = baseCauseRows.filter((r) => r[4].includes(sIdx)).length;
          return h("button", {
            type: "button",
            class: `sub-btn ${state.pinnedSub === s ? "sel" : ""}`,
            onclick: () => { state.pinnedSub = state.pinnedSub === s ? null : s; renderAll(); },
          }, s, h("b", {}, String(n)));
        })
      );
    } else {
      subWrap.hidden = true;
    }

    let scoped = baseCauseRows;
    if (state.pinnedSub) {
      const pIdx = dicts.subsystems.indexOf(state.pinnedSub);
      scoped = baseCauseRows.filter((r) => r[4].includes(pIdx));
      $("#btn-unpin").hidden = false;
      $("#insp-badge").textContent = "细分模块故障库";
      $("#insp-title").textContent = `${m.name} · ${state.pinnedSub}`;
    } else {
      $("#btn-unpin").hidden = true;
      $("#insp-badge").textContent = m.row === 3 ? "底层地基根因库" : "模块自身缺陷库";
      $("#insp-title").textContent = m.name;
    }
    $("#insp-desc").textContent = `包含编目中真实修复点（cause.subsystem）归属在 ${m.subs.join("、")} 的全部故障。`;

    const silCnt = scoped.filter((r) => r[8] === silentIdx).length;
    const fixCnt = scoped.filter((r) => (r[11] & 16) !== 0).length;
    $("#insp-kpis").replaceChildren(
      h("div", { class: "kpi-box" }, h("span", {}, "自身根因故障数"), h("strong", {}, String(scoped.length))),
      h("div", { class: "kpi-box" }, h("span", {}, "静默无报错占比"), h("strong", {}, `${scoped.length ? Math.round((silCnt / scoped.length) * 100) : 0}%`)),
      h("div", { class: "kpi-box" }, h("span", {}, "已记录修复方案"), h("strong", {}, `${scoped.length ? Math.round((fixCnt / scoped.length) * 100) : 0}%`))
    );

    $("#probe-title").textContent = "该模块验证与隔离方法";
    $("#probe-methods-cnt").textContent = "";
    $("#probe-text").textContent = m.probe;
    $("#probe-methods").replaceChildren();

    // Downstream symptom manifestation breakdown
    const symHits = {};
    scoped.forEach((r) => {
      r._symMods.forEach((sm) => { symHits[sm] = (symHits[sm] || 0) + 1; });
    });
    const maxS = Math.max(...Object.values(symHits), 1);
    $("#bars-heading").textContent = "该模块出错后，异常表象最终暴露在哪些模块";
    $("#bars-list").replaceChildren(...Object.entries(symHits).sort((a, b) => b[1] - a[1]).map(([smId, cnt]) =>
      h("button", {
        type: "button",
        class: "bar-row",
        onclick: () => {
          state.mode = "backward";
          state.selectedMod = smId;
          state.pinnedSub = m.subs.length === 1 ? m.subs[0] : state.pinnedSub;
          renderAll();
        },
      },
        h("span", {}, MOD_MAP[smId].name),
        h("span", { class: "bar-track" }, h("span", { class: "bar-fill", style: `width:${Math.max(5, Math.round((cnt / maxS) * 100))}%` })),
        h("span", { class: "bar-num" }, `${cnt} 条暴露`)
      )
    ));
    currentScopedRows = scoped;
  }

  renderTabs();
}

async function loadEntryDetail(repo, eid) {
  if (detailCache.has(eid)) return detailCache.get(eid);
  try {
    const r = await fetch(`data/entry/${repo}/${eid}.json`);
    if (!r.ok) return null;
    const d = await r.json();
    detailCache.set(eid, d);
    return d;
  } catch {
    return null;
  }
}

function renderEntryCard(r) {
  const eid = r[0];
  const repo = r[1] === 0 ? "inference" : "training";
  const conf = dicts.confidences[r[9]];
  const vis = dicts.visibilities[r[8]];
  const subs = r[4].length ? r[4].map((i) => dicts.subsystems[i]).join("+") : "待定";
  const cars = r[5].length ? r[5].map((i) => dicts.carriers[i]).join(" → ") : "待定";
  const fileRel = `catalogs/${repo}/entries/${eid}.yaml`;
  const citeLine = `${eid} ${r[2]} (${fileRel}) [${conf}]`;
  const isExp = state.expandedId === eid;
  const isStandalone = Boolean(window.__INLINE_CAUSAL_GRAPH__);

  const meta = h("div", { class: "entry-meta" },
    h("span", { class: "entry-id" }, eid),
    h("span", { class: "badge-mini" }, `根因: ${subs}`),
    h("span", { class: "badge-mini" }, conf),
    h("span", { class: "panel-sub" }, vis),
    (r[11] & 2) ? h("span", { class: "badge-mini badge-warn" }, "INF→训练挂载") : null,
    (r[11] & 32) ? h("span", { class: "badge-mini" }, "跨库关联") : null
  );

  const card = h("div", { class: "entry-card" },
    h("div", {
      class: "entry-card-top",
      onclick: async () => {
        state.expandedId = isExp ? null : eid;
        if (!isExp && !isStandalone) await loadEntryDetail(repo, eid);
        renderTabs();
      },
    },
      h("div", {}, meta, h("div", { style: "font-weight:600" }, r[3]), h("div", { class: "entry-chain" }, `传播链：${cars}`)),
      h("span", { class: "snake" }, isExp ? "▲" : "▼")
    )
  );

  if (isExp) {
    const d = detailCache.get(eid);
    const y = d?.yaml;
    const drawer = h("div", { class: "entry-detail-drawer" },
      y?.cause?.behavior ? h("div", {}, h("strong", {}, "原因行为："), y.cause.behavior) : (r[12] ? h("div", {}, h("strong", {}, "原因行为："), r[12]) : null),
      y?.effect?.detail ? h("div", {}, h("strong", {}, "影响详情："), y.effect.detail) : (r[13] ? h("div", {}, h("strong", {}, "影响详情："), r[13]) : null),
      h("div", { class: "cite-line" }, citeLine),
      !isStandalone ? h("div", { style: "display:flex;gap:0.75rem;align-items:center;flex-wrap:wrap" },
        h("a", { href: `index.html#e=${encodeURIComponent(eid)}` }, "在检索页打开完整条目 →"),
        r[10] >= 0 ? h("a", { href: `index.html#pattern=${encodeURIComponent(dicts.patterns[r[10]].id)}` }, `同族模式 ${dicts.patterns[r[10]].id} →`) : null
      ) : (r[10] >= 0 ? h("div", { class: "panel-sub" }, `所属模式族：${dicts.patterns[r[10]].id}（${dicts.patterns[r[10]].title}）`) : null)
    );
    card.append(drawer);
  }
  return card;
}

function renderTabs() {
  const mysteries = currentSliceRows.filter((r) => (r[11] & 64) !== 0);
  const patSet = new Set(currentScopedRows.map((r) => r[10]).filter((pi) => pi >= 0));
  const isStandalone = Boolean(window.__INLINE_CAUSAL_GRAPH__);

  $("#tab-entries").textContent = `匹配案例 (${currentScopedRows.length})`;
  $("#tab-patterns").textContent = `常见故障模式族 (${patSet.size})`;
  $("#tab-mysteries").textContent = `根因未定悬案 (${mysteries.length})`;
  ["entries", "patterns", "mysteries"].forEach((t) => $(`#tab-${t}`).classList.toggle("active", state.tab === t));

  const q = ($("#entry-q").value || "").trim().toLowerCase();
  const body = $("#tab-body");
  body.replaceChildren();

  if (state.tab === "entries") {
    let list = currentScopedRows;
    if (q) list = list.filter((r) => r[0].toLowerCase().includes(q) || r[2].toLowerCase().includes(q) || r[3].toLowerCase().includes(q) || (r[12] || "").toLowerCase().includes(q));
    list.slice(0, 60).forEach((r) => body.append(renderEntryCard(r)));
    if (!list.length) body.append(h("p", { class: "panel-sub" }, "无匹配故障案例。"));
    else if (list.length > 60) body.append(h("p", { class: "panel-sub" }, `仅展示前 60 / ${list.length} 条，可用右上搜索框过滤。`));
    return;
  }

  if (state.tab === "patterns") {
    const counts = {};
    currentScopedRows.forEach((r) => { if (r[10] >= 0) counts[r[10]] = (counts[r[10]] || 0) + 1; });
    let items = Object.entries(counts).map(([pi, cnt]) => ({ pi: Number(pi), pat: dicts.patterns[Number(pi)], cnt })).sort((a, b) => b.cnt - a.cnt);
    if (q) items = items.filter((x) => x.pat.id.toLowerCase().includes(q) || x.pat.title.toLowerCase().includes(q));
    items.slice(0, 50).forEach((x) => {
      body.append(h("div", {
        class: "entry-card",
        style: "display:flex;justify-content:space-between;align-items:center;gap:0.5rem;cursor:pointer",
        onclick: () => {
          currentScopedRows = currentScopedRows.filter((r) => r[10] === x.pi);
          state.tab = "entries";
          renderTabs();
        },
      },
        h("div", {}, h("span", { class: "entry-id" }, x.pat.id), " ", x.pat.title),
        isStandalone
          ? h("span", { class: "snake" }, `${x.cnt} 条 ➔`)
          : h("a", { href: `index.html#pattern=${encodeURIComponent(x.pat.id)}`, class: "snake", onclick: (ev) => ev.stopPropagation() }, `${x.cnt} 条 →`)
      ));
    });
    if (!items.length) body.append(h("p", { class: "panel-sub" }, "当前节点下暂无聚合模式族。"));
    return;
  }

  let mys = mysteries;
  if (q) mys = mys.filter((r) => r[0].toLowerCase().includes(q) || r[2].toLowerCase().includes(q) || r[3].toLowerCase().includes(q));
  mys.forEach((r) => body.append(renderEntryCard(r)));
  if (!mys.length) body.append(h("p", { class: "panel-sub" }, "当前切片下无待定根因悬案。"));
}

boot().catch((err) => {
  $("#causal-scope").textContent = `加载数据失败：${err.message}`;
});
