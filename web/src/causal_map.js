// Unified lifecycle causal map lens for the single-page fault-rag site.
// Consumes the same `docs` (data/docs.json, enriched with `.causal`) and `meta` as app.js,
// shares URL hash state, and bridges directly into keyword/semantic search and entry sheets.

export const MODULES = [
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
    walkthrough: "顺箭头倒推：① 首查【KV缓存 & 调度服务】块表分配、前缀哈希命中与并发槽位隔离 → ② 查上游【量化 KV scale】与【模型前向 RoPE/注意力】→ ③ 查底层【Attention 算子/运行时】。",
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
    walkthrough: "编译优化层：Triton / Inductor / XLA / torchair 在算子融合、循环切分或自动降精度 pass 中引入静默数值偏差。",
  },
  {
    id: "b_kern", name: "算子/内核 (最大根因)", sub: "占全库故障 31%", x: 630, y: 248, w: 242, h: 56, row: 3,
    subs: ["算子/内核"],
    carriers: [],
    probe: "固定完全相同的输入 shape/stride/dtype，单独摘出可疑算子与 PyTorch FP32 参考实现做边界 tiling 与非连续张量对拍。",
    walkthrough: "全库第一大故障根源（超千条，80% 为静默错误）：自身不直接面向用户，但一旦 MatMul/Attention/MoE/Norm 内核算错，上方所有训练和推理模块都会跟着出问题。",
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

const VIS_OPTIONS = ["静默错误", "质量退化", "数值异常", "不可复现", "性能退化", "挂起", "报错"];

const MOD_MAP = Object.fromEntries(MODULES.map((m) => [m.id, m]));
const SUB_TO_MOD = {};
MODULES.forEach((m) => m.subs.forEach((s) => { SUB_TO_MOD[s] = m.id; }));

export function createCausalMap({ h, docs, meta, index, readState, writeState, stampFor, summaryLines }) {
  const stackGroups = meta.causal?.stack_groups || [];
  const stackGroupById = Object.fromEntries(stackGroups.map((g) => [g.id, new Set(g.members)]));

  // Precompute module mappings once over `docs`
  const methodsByCarrier = {};
  const faultRowIndices = [];

  docs.forEach((d, idx) => {
    const c = d.causal;
    if (!c) return;
    if (c.kind === "method") {
      (c.effs || []).forEach((ef) => {
        (methodsByCarrier[ef] ||= []).push(d);
      });
      return;
    }
    const sNames = c.subs || [];
    const cNames = c.effs || [];
    const stgs = c.stages || [];
    const isSrv = stgs.includes("serving");
    const isRlFocused = (stgs.includes("rl") && !stgs.includes("pretrain")) || sNames.includes("RL训练回路");

    const smods = new Set();
    cNames.forEach((cn) => {
      if (cn === "训练样本流") smods.add("m_data");
      else if (cn === "输入编码") smods.add("m_tok");
      else if (cn === "隐状态" || cn === "数值结果") smods.add("m_model");
      else if (cn === "KV状态") smods.add("m_kv");
      else if (cn === "跨请求/跨样本数据") smods.add(isSrv ? "m_kv" : "m_data");
      else if (cn === "输出分布") smods.add("m_sample");
      else if (cn === "输出内容" || cn === "测试判定" || cn === "指标/日志") smods.add("m_parse");
      else if (cn === "训练信号") smods.add(isRlFocused ? "m_rl" : "m_loss");
      else if (cn === "梯度" || cn === "优化器状态" || cn === "训练轨迹") {
        smods.add(sNames.includes("RL训练回路") ? "m_rl" : "m_loss");
      } else if (cn === "模型权重") {
        smods.add(sNames.includes("量化") ? "m_quant" : "m_ckpt");
      } else if (cn === "可用性/进度" || cn === "性能/资源") {
        smods.add(isSrv ? "m_kv" : "m_loss");
      }
    });
    c._symMods = [...smods];
    c._causeMods = [...new Set(sNames.map((s) => SUB_TO_MOD[s]).filter(Boolean))];
    faultRowIndices.push(idx);
  });

  let localTab = "entries";
  let localFilterQ = "";

  function activeMod(s) {
    return MOD_MAP[s.cmod] ? s.cmod : "m_parse";
  }

  function activeMode(s) {
    const mid = activeMod(s);
    if (MOD_MAP[mid].row === 3) return "fwd";
    return s.cmode === "fwd" ? "fwd" : "back";
  }

  function parsePruners(s) {
    return new Set((s.cprn || "").split(",").map((x) => x.trim()).filter(Boolean));
  }

  function getPrunedSubs(s) {
    const out = new Set();
    if (activeMode(s) === "back") {
      const pset = parsePruners(s);
      PRUNER_RULES.forEach((r) => {
        if (pset.has(r.id)) r.subs.forEach((sub) => out.add(sub));
      });
    }
    return out;
  }

  function filterSliceRows(s) {
    const stg = s.cstg || "";
    const stkSet = s.cstk && stackGroupById[s.cstk] ? stackGroupById[s.cstk] : null;
    const vis = s.cvis || "";
    const prunedSubs = getPrunedSubs(s);

    return faultRowIndices.filter((idx) => {
      const c = docs[idx].causal;
      if (stg && !c.stages.includes(stg)) return false;
      if (stkSet && !c.stacks.some((st) => stkSet.has(st))) return false;
      if (vis && c.vis !== vis) return false;
      if (prunedSubs.size > 0 && c.subs.length > 0 && c.subs.every((sub) => prunedSubs.has(sub))) {
        return false;
      }
      return true;
    });
  }

  function computeScopedRows(s, baseRows = null) {
    const rows = baseRows || filterSliceRows(s);
    const mid = activeMod(s);
    const mode = activeMode(s);
    let scoped = mode === "back"
      ? rows.filter((idx) => docs[idx].causal._symMods.includes(mid))
      : rows.filter((idx) => docs[idx].causal._causeMods.includes(mid));
    if (s.csub) {
      scoped = scoped.filter((idx) => docs[idx].causal.subs.includes(s.csub));
    }
    return scoped;
  }

  function getScopedRowSet(s) {
    return new Set(computeScopedRows(s));
  }

  function describeScope(s) {
    const m = MOD_MAP[activeMod(s)];
    const mode = activeMode(s);
    const pCnt = parsePruners(s).size;
    const parts = [mode === "back" ? `在【${m.name}】暴露` : `模块【${m.name}】根因`];
    if (s.csub) parts.push(`锁定 ${s.csub}`);
    if (s.cstg) parts.push(`阶段:${s.cstg}`);
    if (s.cstk) parts.push(`栈:${s.cstk}`);
    if (s.cvis) parts.push(s.cvis);
    if (pCnt && mode === "back") parts.push(`排除${pCnt}项`);
    return parts.join(" · ");
  }

  function locateEntryState(rowIdx) {
    const c = docs[rowIdx]?.causal;
    if (!c || c.kind === "method") return { view: "map" };
    const sym = c._symMods?.[0];
    const cause = c._causeMods?.[0];
    const sub0 = c.subs?.[0] || "";
    if (sym) {
      return { view: "map", cmode: "", cmod: sym, csub: sub0 };
    }
    return { view: "map", cmode: "fwd", cmod: cause || "b_kern", csub: sub0 };
  }

  function traceGraph(modId, mode, prunedSubs) {
    const activeNodes = new Set([modId]);
    const activeEdges = new Set();
    const dist = { [modId]: 0 };

    if (mode === "back") {
      const queue = [modId];
      while (queue.length) {
        const cur = queue.shift();
        EDGES.forEach((e, idx) => {
          if (e.to === cur) {
            if ((modId === "m_parse" || modId === "m_kv") && (e.from === "m_loss" || e.from === "m_rl")) return;
            const srcMod = MOD_MAP[e.from];
            if (srcMod && srcMod.subs.every((sub) => prunedSubs.has(sub))) return;
            activeEdges.add(idx);
            if (!activeNodes.has(e.from)) {
              activeNodes.add(e.from);
              dist[e.from] = dist[cur] + 1;
              queue.push(e.from);
            }
          }
        });
      }
      ["b_hw", "b_dist", "b_comp", "b_kern"].forEach((bid) => {
        if (!MOD_MAP[bid].subs.every((sub) => prunedSubs.has(sub))) activeNodes.add(bid);
      });
      [12, 13, 14].forEach((i) => activeEdges.add(i));
    } else {
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

  function initControls(container) {
    const stkSel = container.querySelector("#cm-stack");
    stackGroups.forEach((g) => stkSel.append(h("option", { value: g.id }, g.label)));
    const visSel = container.querySelector("#cm-vis");
    VIS_OPTIONS.forEach((v) => visSel.append(h("option", { value: v }, v)));

    container.querySelector("#cm-mode-back").addEventListener("click", () => {
      const s = readState();
      const mid = MOD_MAP[activeMod(s)].row === 3 ? "m_parse" : activeMod(s);
      writeState({ ...s, cmode: "", cmod: mid, csub: "", e: "" }, { push: true });
    });
    container.querySelector("#cm-mode-fwd").addEventListener("click", () => {
      writeState({ ...readState(), cmode: "fwd", csub: "", e: "" }, { push: true });
    });

    container.querySelectorAll(".preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        writeState({ ...readState(), cmode: "", cmod: btn.dataset.preset, csub: "", e: "" }, { push: true });
      });
    });

    container.querySelector("#cm-stage").addEventListener("change", (ev) => {
      writeState({ ...readState(), cstg: ev.target.value, e: "" });
    });
    stkSel.addEventListener("change", (ev) => {
      writeState({ ...readState(), cstk: ev.target.value, e: "" });
    });
    visSel.addEventListener("change", (ev) => {
      writeState({ ...readState(), cvis: ev.target.value, e: "" });
    });
    container.querySelector("#cm-reset").addEventListener("click", () => {
      localFilterQ = "";
      writeState({ ...readState(), cmode: "", cmod: "m_parse", csub: "", cstg: "", cstk: "", cvis: "", cprn: "", e: "" }, { push: true });
    });
  }

  function renderLeftPane(container, s) {
    const mid = activeMod(s);
    const mode = activeMode(s);
    const prunedSubs = getPrunedSubs(s);
    const sliceRows = filterSliceRows(s);

    container.querySelector("#cm-mode-back").classList.toggle("active", mode === "back");
    container.querySelector("#cm-mode-fwd").classList.toggle("active", mode === "fwd");
    container.querySelector("#cm-stage").value = s.cstg || "";
    container.querySelector("#cm-stack").value = s.cstk || "";
    container.querySelector("#cm-vis").value = s.cvis || "";
    container.querySelectorAll(".preset-btn").forEach((b) => {
      b.classList.toggle("active", mode === "back" && mid === b.dataset.preset);
    });

    const modCauseCnt = Object.fromEntries(MODULES.map((m) => [m.id, 0]));
    const modSymCnt = Object.fromEntries(MODULES.map((m) => [m.id, 0]));
    const subCntGlobal = {};
    Object.keys(SUB_TO_MOD).forEach((sub) => { subCntGlobal[sub] = 0; });

    sliceRows.forEach((idx) => {
      const c = docs[idx].causal;
      c._causeMods.forEach((m) => { modCauseCnt[m]++; });
      c._symMods.forEach((m) => { modSymCnt[m]++; });
      c.subs.forEach((sub) => { if (sub in subCntGlobal) subCntGlobal[sub]++; });
    });

    // Check if an individual entry `s.e` is open to overlay its X-ray cause/symptom markers
    let openCauseSet = new Set();
    let openSymSet = new Set();
    if (s.e) {
      const [repo, id] = s.e.split("/");
      const rIdx = index.byKey.get(`${repo}\u0000${id}`);
      const ec = rIdx !== undefined ? docs[rIdx]?.causal : null;
      if (ec && ec.kind !== "method") {
        openCauseSet = new Set(ec._causeMods || []);
        openSymSet = new Set(ec._symMods || []);
      }
    }

    const pedTotal = ["b_hw", "b_dist", "b_comp", "b_kern"].reduce((a, k) => a + modCauseCnt[k], 0);
    container.querySelector("#pedestal-caption").textContent =
      `↑ 底层软硬件承重地基（当前切片共 ${pedTotal.toLocaleString()} 条故障源于此地基；点地基模块看其自身故障库）`;

    const trace = traceGraph(mid, mode, prunedSubs);
    const edgesG = container.querySelector("#svg-edges");
    const nodesG = container.querySelector("#svg-nodes");
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

    MODULES.forEach((m) => {
      const isSel = m.id === mid;
      const allPruned = m.subs.every((sub) => prunedSubs.has(sub));
      const hi = !allPruned && (trace.activeNodes.has(m.id) || openCauseSet.has(m.id) || openSymSet.has(m.id));

      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", "m-node");
      g.setAttribute("opacity", allPruned ? "0.18" : (hi ? "1" : "0.24"));
      g.addEventListener("click", () => {
        const nextMode = m.row === 3 ? "fwd" : mode;
        writeState({ ...readState(), cmode: nextMode === "fwd" ? "fwd" : "", cmod: m.id, csub: "", e: "" }, { push: true });
      });

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", m.x);
      rect.setAttribute("y", m.y);
      rect.setAttribute("width", m.w);
      rect.setAttribute("height", m.h);
      rect.setAttribute("rx", "6");
      rect.setAttribute("fill", "var(--paper)");
      const isEntryFocus = openCauseSet.has(m.id) || openSymSet.has(m.id);
      rect.setAttribute("stroke", isSel || isEntryFocus ? "var(--speculative)" : (hi ? "var(--action)" : "var(--rule)"));
      rect.setAttribute("stroke-width", isSel || isEntryFocus ? "2.6" : (hi ? "1.6" : "1"));
      g.append(rect);

      g.append(
        mkText(m.name, m.x + 8, m.y + 18, null, "700", "var(--ink)"),
        mkText(m.sub, m.x + 8, m.y + 34, null, null, "var(--muted)"),
        mkText(
          m.row === 3 ? `底层根因 ${modCauseCnt[m.id]} 条` : `表象 ${modSymCnt[m.id]} · 根因 ${modCauseCnt[m.id]}`,
          m.x + 8, m.y + 50, null, "600", isSel ? "var(--action)" : "var(--ink)"
        )
      );

      if (openCauseSet.has(m.id) && openSymSet.has(m.id)) {
        g.append(mkText("◆根因&暴露", m.x + m.w - 6, m.y + 18, "end", "700", "var(--speculative)"));
      } else if (openCauseSet.has(m.id)) {
        g.append(mkText("◆条目根因", m.x + m.w - 6, m.y + 18, "end", "700", "var(--speculative)"));
      } else if (openSymSet.has(m.id)) {
        g.append(mkText("◆条目暴露", m.x + m.w - 6, m.y + 18, "end", "700", "var(--speculative)"));
      } else if (allPruned) {
        g.append(mkText("已排除", m.x + m.w - 6, m.y + 18, "end", "600", "var(--muted)"));
      } else if (isSel) {
        g.append(mkText(mode === "back" ? "★观测点" : "★当前模块", m.x + m.w - 6, m.y + 18, "end", "700", "var(--speculative)"));
      } else if (trace.dist[m.id] > 0) {
        g.append(mkText(mode === "back" ? `上游-${trace.dist[m.id]}` : `波及+${trace.dist[m.id]}`, m.x + m.w - 6, m.y + 18, "end", "600", "var(--action)"));
      }

      nodesG.append(g);
    });

    const curMod = MOD_MAP[mid];
    container.querySelector("#wt-badge").textContent = mode === "back"
      ? `当前观测异常位置：${curMod.name}`
      : `当前查看模块：${curMod.name}`;
    const inactiveNames = MODULES.filter((m) => m.row !== 3 && !trace.activeNodes.has(m.id)).map((m) => m.name);
    container.querySelector("#wt-pruned").textContent = mode === "back" && inactiveNames.length
      ? `已按因果方向免查下游：${inactiveNames.join("、")}`
      : `当前切片命中 ${sliceRows.length.toLocaleString()} 条故障`;
    container.querySelector("#wt-steps").textContent = curMod.walkthrough;

    // 26 subsystems strip
    const strip = container.querySelector("#all-subs-strip");
    strip.replaceChildren();
    Object.entries(subCntGlobal)
      .sort((a, b) => b[1] - a[1])
      .forEach(([sname, cnt]) => {
        const isPruned = prunedSubs.has(sname);
        const isSel = s.csub === sname || (mode === "fwd" && curMod.subs.length === 1 && curMod.subs[0] === sname);
        strip.append(h("button", {
          type: "button",
          class: `sub-btn ${isPruned ? "pruned" : ""} ${isSel ? "sel" : ""}`,
          onclick: () => {
            const targetMod = SUB_TO_MOD[sname];
            writeState({
              ...readState(),
              cmode: "fwd",
              cmod: targetMod,
              csub: MOD_MAP[targetMod].subs.length > 1 ? sname : "",
              e: "",
            }, { push: true });
          },
        }, sname, h("b", {}, String(cnt))));
      });

    return { sliceRows, scopedRows: computeScopedRows(s, sliceRows) };
  }

  function renderWorkbenchSheet(sheet, s, { sliceRows, scopedRows }) {
    const mid = activeMod(s);
    const mode = activeMode(s);
    const m = MOD_MAP[mid];
    const pset = parsePruners(s);
    const prunedSubs = getPrunedSubs(s);

    const headerRow = h("div", { class: "wb-head" },
      h("div", {},
        h("div", { class: "insp-title-row" },
          h("span", { class: "insp-tag" }, mode === "back" ? (s.csub ? "锁定真凶路径" : "按表象逆向溯因") : (m.row === 3 ? "底层地基根因库" : "模块自身缺陷库")),
          h("h2", {}, mode === "back"
            ? (s.csub ? `根因【${s.csub}】➔ 在【${m.name}】暴露` : `在【${m.name}】看到异常时`)
            : (s.csub ? `${m.name} · ${s.csub}` : m.name)),
          s.csub ? h("button", {
            type: "button", class: "link-btn",
            onclick: () => writeState({ ...readState(), csub: "" }),
          }, "取消锁定 ×") : null
        ),
        h("p", { class: "insp-desc" }, mode === "back"
          ? `统计表象落在 ${m.carriers.join(" / ")} 的全部案例（含自身缺陷、上游传导及底层地基算错）。`
          : `统计真实修复点（cause.subsystem）归属在 ${m.subs.join("、")} 的全部故障。`)
      ),
      h("button", {
        type: "button",
        class: "primary wb-bridge-btn",
        onclick: () => writeState({ ...readState(), view: "", cscope: "1", e: "", k: 0 }, { push: true }),
      }, `在此范围内检索（${scopedRows.length} 条） →`)
    );

    const nodes = [headerRow];

    // Sub-module filter pills in forward mode
    if (mode === "fwd" && m.subs.length > 1) {
      const baseCause = sliceRows.filter((idx) => docs[idx].causal._causeMods.includes(mid));
      nodes.push(h("div", { class: "sub-pills", style: "margin-bottom:0.75rem" },
        h("button", {
          type: "button", class: `sub-btn ${!s.csub ? "sel" : ""}`,
          onclick: () => writeState({ ...readState(), csub: "" }),
        }, "本框全部", h("b", {}, String(baseCause.length))),
        ...m.subs.map((sub) => {
          const n = baseCause.filter((idx) => docs[idx].causal.subs.includes(sub)).length;
          return h("button", {
            type: "button", class: `sub-btn ${s.csub === sub ? "sel" : ""}`,
            onclick: () => writeState({ ...readState(), csub: s.csub === sub ? "" : sub }),
          }, sub, h("b", {}, String(n)));
        })
      ));
    }

    const silCnt = scopedRows.filter((idx) => docs[idx].causal.vis === "静默错误").length;
    const prbCnt = scopedRows.filter((idx) => (docs[idx].causal.flags & 12) !== 0).length;
    const fixCnt = scopedRows.filter((idx) => (docs[idx].causal.flags & 16) !== 0).length;
    const baseSymRows = mode === "back" ? sliceRows.filter((idx) => docs[idx].causal._symMods.includes(mid)) : [];
    const mysCnt = mode === "back" ? baseSymRows.filter((idx) => (docs[idx].causal.flags & 64) !== 0).length : 0;

    nodes.push(h("div", { class: "kpi-row" },
      h("div", { class: "kpi-box" },
        h("span", {}, mode === "back" ? "暴露案例 (未定悬案)" : "自身根因故障数"),
        h("strong", {}, mode === "back" ? `${scopedRows.length} (${mysCnt})` : String(scopedRows.length))),
      h("div", { class: "kpi-box" },
        h("span", {}, "静默无报错占比"),
        h("strong", {}, `${scopedRows.length ? Math.round((silCnt / scopedRows.length) * 100) : 0}%`)),
      h("div", { class: "kpi-box" },
        h("span", {}, mode === "back" ? "已有检测/签名" : "已记录修复方案"),
        h("strong", {}, `${scopedRows.length ? Math.round(((mode === "back" ? prbCnt : fixCnt) / scopedRows.length) * 100) : 0}%`))
    ));

    // Probe & detector methods
    const methods = [];
    const seenM = new Set();
    m.carriers.forEach((cn) => {
      (methodsByCarrier[cn] || []).forEach((md) => {
        if (!seenM.has(md.id)) { seenM.add(md.id); methods.push(md); }
      });
    });
    nodes.push(h("div", { class: "probe-card" },
      h("div", { class: "probe-card-head" },
        h("span", {}, mode === "back" ? "二分插桩检查建议（先在哪一刀切开上下游）" : "该模块隔离验证建议"),
        methods.length ? h("span", { style: "font-family:var(--mono);color:var(--muted)" }, `检测工具 ${methods.length} 个`) : null),
      h("div", {}, m.probe),
      methods.length ? h("div", { class: "method-chips" }, methods.map((md) =>
        h("button", {
          type: "button",
          class: "chip-link",
          title: md.title,
          onclick: () => writeState({ ...readState(), e: `${md.repo}/${md.id}` }, { push: true }),
        }, `${md.id} ${md.name}`)
      )) : null
    ));

    if (mode === "back") {
      // Environment pruners
      nodes.push(h("div", { class: "pruner-sec" },
        h("div", { style: "display:flex;justify-content:space-between;align-items:center;font-size:var(--step--1)" },
          h("strong", {}, "排除已确认无关的环境条件（勾选实时重算嫌疑占比）"),
          pset.size ? h("button", {
            type: "button", class: "link-btn", style: "padding:0",
            onclick: () => writeState({ ...readState(), cprn: "" }),
          }, "清空排除项") : null),
        h("div", { class: "pruner-pills" }, PRUNER_RULES.map((r) => {
          const on = pset.has(r.id);
          return h("button", {
            type: "button",
            class: `prn-pill ${on ? "on" : ""}`,
            onclick: () => {
              const next = new Set(pset);
              if (on) next.delete(r.id); else next.add(r.id);
              writeState({ ...readState(), cprn: [...next].join(",") });
            },
          }, `${on ? "✓ " : ""}${r.label}`);
        }))
      ));

      // Root-cause ranking bars
      const subHits = {};
      let totHits = 0;
      baseSymRows.forEach((idx) => {
        docs[idx].causal.subs.forEach((sname) => {
          if (prunedSubs.has(sname)) return;
          subHits[sname] = (subHits[sname] || 0) + 1;
          totHits++;
        });
      });
      const barNodes = Object.entries(subHits).sort((a, b) => b[1] - a[1]).slice(0, 9).map(([sname, cnt]) => {
        const pct = totHits ? ((cnt / totHits) * 100).toFixed(1) : "0.0";
        const srcMod = MOD_MAP[SUB_TO_MOD[sname]];
        const role = srcMod?.id === mid ? "自身" : (srcMod?.row === 3 ? "地基" : "上游");
        return h("button", {
          type: "button",
          class: `bar-row ${s.csub === sname ? "pinned" : ""}`,
          onclick: () => writeState({ ...readState(), csub: s.csub === sname ? "" : sname }),
        },
          h("span", {}, sname, h("span", { class: "role-dot" }, `·${role}`)),
          h("span", { class: "bar-track" }, h("span", { class: "bar-fill", style: `width:${Math.max(4, Number(pct))}%` })),
          h("span", { class: "bar-num" }, `${cnt} (${pct}%)`)
        );
      });
      nodes.push(h("div", { class: "bars-sec" },
        h("div", { style: "display:flex;justify-content:space-between;font-size:var(--step--1);margin-bottom:0.35rem" },
          h("strong", {}, `真凶模块排行榜（在 ${baseSymRows.length} 条暴露案例中谁是真正根因）`),
          h("span", { style: "color:var(--muted)" }, "点任一行锁定")),
        ...(barNodes.length ? barNodes : [h("p", { class: "panel-sub" }, "当前排除条件下无匹配根因。")])
      ));
    } else {
      // Forward blast radius manifestation bars
      const symHits = {};
      scopedRows.forEach((idx) => {
        docs[idx].causal._symMods.forEach((sm) => { symHits[sm] = (symHits[sm] || 0) + 1; });
      });
      const maxS = Math.max(...Object.values(symHits), 1);
      const barNodes = Object.entries(symHits).sort((a, b) => b[1] - a[1]).map(([smId, cnt]) =>
        h("button", {
          type: "button",
          class: "bar-row",
          onclick: () => writeState({
            ...readState(),
            cmode: "",
            cmod: smId,
            csub: m.subs.length === 1 ? m.subs[0] : s.csub,
          }, { push: true }),
        },
          h("span", {}, MOD_MAP[smId].name),
          h("span", { class: "bar-track" }, h("span", { class: "bar-fill", style: `width:${Math.max(5, Math.round((cnt / maxS) * 100))}%` })),
          h("span", { class: "bar-num" }, `${cnt} 条暴露`)
        )
      );
      if (barNodes.length) {
        nodes.push(h("div", { class: "bars-sec" },
          h("div", { style: "font-size:var(--step--1);margin-bottom:0.35rem" },
            h("strong", {}, "该模块出错后，异常表象最终暴露在哪些模块")),
          ...barNodes
        ));
      }
    }

    // Bottom drilldown tabs (Entries / Patterns / Mysteries)
    const mysteries = sliceRows.filter((idx) => (docs[idx].causal.flags & 64) !== 0);
    const patMap = new Map();
    scopedRows.forEach((idx) => {
      const p = docs[idx].pattern;
      if (p) patMap.set(p, { id: p, title: docs[idx].pattern_title || "待定", cnt: (patMap.get(p)?.cnt || 0) + 1 });
    });

    const listBody = h("div", { class: "entry-list" });
    const searchInput = h("input", {
      type: "search",
      class: "mini-search",
      value: localFilterQ,
      placeholder: "页内快筛编号 / 标题…",
      oninput: (ev) => {
        localFilterQ = ev.target.value;
        fillListBody();
      },
    });

    const tabBtns = h("div", { class: "tab-btns" });
    const rebuildTabBar = () => {
      tabBtns.replaceChildren(
        ...[
          ["entries", `候选案例 (${scopedRows.length})`],
          ["patterns", `常见模式族 (${patMap.size})`],
          ["mysteries", `待定悬案 (${mysteries.length})`],
        ].map(([tid, lbl]) => h("button", {
          type: "button",
          class: localTab === tid ? "active" : "",
          onclick: () => { localTab = tid; rebuildTabBar(); fillListBody(); },
        }, lbl))
      );
    };

    const renderCandidateHit = (idx) => {
      const d = docs[idx];
      const c = d.causal;
      const key = `${d.repo}/${d.id}`;
      return h("button", {
        type: "button",
        class: "hit",
        onclick: () => writeState({ ...readState(), e: key }, { push: true }),
      },
        h("span", { class: "hit-head" },
          h("span", { class: "acc" }, d.id),
          h("span", { class: "snake" }, d.name),
          stampFor(d),
          (c.flags & 2) ? h("span", { class: "badge-mini badge-warn" }, "INF→训练挂载") : null
        ),
        d.title ? h("p", { class: "hit-title" }, d.title) : null,
        h("p", { class: "hit-summary" }, summaryLines(d.summary)),
        h("p", { class: "facts" },
          h("span", {}, "根因", h("b", {}, c.subs.length ? c.subs.join("+") : "待定")),
          h("span", {}, "传播链", h("b", {}, c.effs.join(" → ") || "待定")),
          d.pattern ? h("span", {}, "模式", h("b", {}, d.pattern)) : null
        )
      );
    };

    function fillListBody() {
      const q = localFilterQ.trim().toLowerCase();
      listBody.replaceChildren();
      if (localTab === "entries") {
        let arr = scopedRows;
        if (q) arr = arr.filter((i) => docs[i].id.toLowerCase().includes(q) || docs[i].name.toLowerCase().includes(q) || (docs[i].title || "").toLowerCase().includes(q));
        arr.slice(0, 40).forEach((i) => listBody.append(renderCandidateHit(i)));
        if (!arr.length) listBody.append(h("p", { class: "panel-sub" }, "无匹配案例。"));
        else if (arr.length > 40) {
          listBody.append(h("button", {
            type: "button", class: "more",
            onclick: () => writeState({ ...readState(), view: "", cscope: "1", q: localFilterQ.trim(), e: "", k: 0 }, { push: true }),
          }, `共 ${arr.length} 条，转到检索视图跑 BM25 / 语义排序 →`));
        }
        return;
      }
      if (localTab === "patterns") {
        let pats = [...patMap.values()].sort((a, b) => b.cnt - a.cnt);
        if (q) pats = pats.filter((p) => p.id.toLowerCase().includes(q) || p.title.toLowerCase().includes(q));
        pats.slice(0, 50).forEach((p) => {
          listBody.append(h("div", { class: "entry-card", style: "display:flex;justify-content:space-between;align-items:center;gap:0.5rem" },
            h("div", {}, h("code", {}, p.id), " ", p.title),
            h("button", {
              type: "button", class: "link-btn",
              onclick: () => writeState({ ...readState(), view: "", pattern: p.id, e: "", k: 0 }, { push: true }),
            }, `只看这一族 (${p.cnt}) →`)
          ));
        });
        if (!pats.length) listBody.append(h("p", { class: "panel-sub" }, "当前节点下无聚合模式族。"));
        return;
      }
      let mys = mysteries;
      if (q) mys = mys.filter((i) => docs[i].id.toLowerCase().includes(q) || (docs[i].title || "").toLowerCase().includes(q));
      mys.forEach((i) => listBody.append(renderCandidateHit(i)));
      if (!mys.length) listBody.append(h("p", { class: "panel-sub" }, "当前切片下无待定根因悬案。"));
    }

    rebuildTabBar();
    fillListBody();
    nodes.push(h("div", { class: "wb-tabs-wrap" }, h("div", { class: "tabs-head" }, tabBtns, searchInput), listBody));

    sheet.replaceChildren(...nodes);
  }

  return {
    initControls,
    renderLeftPane,
    renderWorkbenchSheet,
    getScopedRowSet,
    describeScope,
    locateEntryState,
    activeMod,
    MOD_MAP,
  };
}
