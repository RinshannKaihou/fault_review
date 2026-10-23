// Retrieval core shared by the browser (app.js) and the Node eval (web/eval/rank.mjs), so the
// gated code is the shipped code. Ports fault_rag/search.py — filters, BM25, RRF, lookup,
// related — with one deliberate change: the tokenizer is bigram-only (see web/build_site.py).
// Option names stay snake_case to match the Python signature and the eval case filters.

export const RRF_K = 60;
const LATIN_RE = /[a-z0-9][a-z0-9_.\-\/#]*/g;
const CJK_RUN_RE = /[一-鿿]+/g;
const ID_SHAPED = /^(?:(?:\d+|[A-Z][A-Z0-9-]*)\.\d+|(?:INF|TRN)-\d+|K\d+\.\d+)$/;

// Byte-identical to build_site.tokenize (checked by web/eval/check_web.py).
export function tokenize(text) {
  text = text.toLowerCase();
  const toks = text.match(LATIN_RE) || [];
  for (const run of text.match(CJK_RUN_RE) || []) {
    if (run.length === 1) toks.push(run);
    else for (let i = 0; i < run.length - 1; i++) toks.push(run.slice(i, i + 2));
  }
  return toks;
}

const lstrip = (s, chars) => {
  let k = 0;
  while (k < s.length && chars.includes(s[k])) k++;
  return s.slice(k);
};

// Old Markdown categories: accept NUM / cat1 / 1 / KV1 / kv1 / neg_rejected forms.
function legacyCategoryMatches(ec, category) {
  const cat = lstrip(category.toLowerCase(), "cat");
  ec = ec.toLowerCase();
  return ec === category.toLowerCase() || ec === "cat" + cat || ec === "kv" + cat ||
    lstrip(ec, "catkv") === cat;
}

function categoryMatches(e, category) {
  if (e.category === category || e.category.toLowerCase() === category.toLowerCase()) return true;
  const legacy = e.legacy_category || (e.neg || e.legacy_ids == null ? e.category : null);
  return !!legacy && legacyCategoryMatches(legacy, category);
}

function passes(e, f) {
  if (e.entry_type === "pointer" && !f.include_pointers) return false;
  if (f.repo && e.repo !== f.repo) return false;
  if (f.category && !categoryMatches(e, f.category)) return false;
  if (f.confidence && (e.confidence || "") !== f.confidence) return false;
  if (f.pattern && (e.pattern || "") !== f.pattern) return false;
  if (f.neg != null && !!e.neg !== f.neg) return false;
  return true;
}

// {row: rank} over eligible rows, best score = rank 0; ties keep row order (stable sort).
function ranks(scores, eligible) {
  const order = [...eligible].sort((a, b) => scores[b] - scores[a]);
  return new Map(order.map((i, r) => [i, r]));
}

const positiveRanks = (scores, eligible) => ranks(scores, eligible.filter((i) => scores[i] > 0));

// Fuse rank maps; Map insertion order reproduces the Python dict order that breaks score ties.
function rrfScores(...rankings) {
  const fused = new Map();
  for (const ranking of rankings) {
    for (const [i, r] of ranking) fused.set(i, (fused.get(i) ?? 0) + 1 / (RRF_K + r));
  }
  return fused;
}

export class Index {
  // docs: data/docs.json rows; bm25: data/bm25.json; vectors: Float32Array (n * dim) or null.
  constructor(docs, bm25, vectors = null, dim = 0) {
    this.docs = docs;
    this.bm25 = { ...bm25, postings: new Map(Object.entries(bm25.postings)) };
    this.vectors = vectors;
    this.dim = dim;
    this.byId = new Map();
    this.byLegacy = new Map();
    this.byName = new Map();
    this.byKey = new Map();
    const add = (m, k, i) => (m.has(k) ? m.get(k).push(i) : m.set(k, [i]));
    docs.forEach((e, i) => {
      add(this.byId, e.id, i);
      for (const old of e.legacy_ids || []) add(this.byLegacy, old, i);
      add(this.byName, e.name, i);
      for (const alias of e.aliases || []) add(this.byName, alias, i);
      this.byKey.set(`${e.repo}\u0000${e.id}`, i);
    });
  }

  bm25Scores(tokens) {
    const { k1, b, n, avgdl, dl, postings } = this.bm25;
    const N = Math.max(1, n);
    const out = new Float64Array(n);
    for (const t of new Set(tokens)) {
      const p = postings.get(t);
      if (!p) continue;
      const df = p.length / 2;
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
      for (let j = 0; j < p.length; j += 2) {
        const i = p[j], f = p[j + 1];
        const denom = f + k1 * (1 - b + b * dl[i] / avgdl);
        out[i] += idf * f * (k1 + 1) / denom;
      }
    }
    return out;
  }

  denseScores(qv, eligible) {
    const { vectors: V, dim } = this;
    const out = new Float64Array(this.docs.length);
    for (const i of eligible) {
      let s = 0;
      for (let d = 0, o = i * dim; d < dim; d++) s += V[o + d] * qv[d];
      out[i] = s;
    }
    return out;
  }

  // Rows passing the filters, in catalog order: the browse view, and search's candidate set.
  filter({ repo, category, confidence, include_neg = true, include_pointers = false,
           pattern } = {}) {
    const f = { repo, category, confidence, pattern, include_pointers,
                neg: include_neg ? null : false };
    const rows = [];
    this.docs.forEach((e, i) => passes(e, f) && rows.push(i));
    return rows;
  }

  // mode: "bm25" | "dense" | "hybrid"; dense and hybrid need the query vector qv.
  search(query, { topk = 8, mode = "bm25", ...filters } = {}, qv = null) {
    if (mode !== "bm25" && !qv) throw new Error(`mode ${mode} needs a query vector`);
    const eligible = this.filter(filters);
    if (!eligible.length) return [];

    let denseR = new Map(), bm25R = new Map();
    if (mode === "hybrid" || mode === "dense") denseR = ranks(this.denseScores(qv, eligible), eligible);
    if (mode === "hybrid" || mode === "bm25") bm25R = positiveRanks(this.bm25Scores(tokenize(query)), eligible);
    const fused = rrfScores(denseR, bm25R);
    const order = [...fused.keys()].sort((a, b) => fused.get(b) - fused.get(a));

    return order.slice(0, topk).map((i, n) => {
      const e = this.docs[i];
      return {
        rank: n + 1,
        score: Math.round(fused.get(i) * 1e6) / 1e6,
        dense_rank: denseR.get(i) ?? null,
        bm25_rank: bm25R.get(i) ?? null,
        rerank_rank: null,          // the cross-encoder only runs in the CLI
        row: i,
        id: e.id, name: e.name, title: e.title, summary: e.summary,
        legacy_ids: e.legacy_ids || [], repo: e.repo, category: e.category,
        legacy_category: e.legacy_category, confidence: e.confidence, status: e.status,
        pattern: e.pattern, neg: e.neg, entry_type: e.entry_type,
        name_ambiguous: e.name_ambiguous, same_name_ids: e.same_name_ids,
        file: e.file, line: e.line, links: e.links.slice(0, 6),
      };
    });
  }

  // Exact id, old Markdown id (incl. merged ones such as 1.70 -> INF-0044), name / alias, then
  // name substring for non-ID queries. Returns row numbers.
  lookup(q) {
    if (this.byId.has(q)) return this.byId.get(q);
    if (this.byLegacy.has(q)) return this.byLegacy.get(q);
    if (this.byName.has(q)) return this.byName.get(q);
    if (ID_SHAPED.test(q)) return [];     // an unknown catalog ID must not partially match another
    const ql = q.toLowerCase();
    const rows = [];
    for (let i = 0; i < this.docs.length && rows.length < 10; i++) {
      const e = this.docs[i];
      if (e.name.toLowerCase().includes(ql) || e.id.toLowerCase().includes(ql)) rows.push(i);
    }
    return rows;
  }

  // Only the exact tiers of lookup (id, old id, name/alias): the page shows these as an
  // "ID match" above the search results, where a name-substring hit would be noise.
  exactLookup(q) {
    return this.byId.get(q) || this.byLegacy.get(q) || this.byName.get(q) || [];
  }

  isIdShaped(q) {
    return ID_SHAPED.test(q);
  }

  // Walk cross-reference links from the lookup hits: [{row, link_kind}], seeds excluded.
  related(q, depth = 1) {
    return this.relatedFrom(this.lookup(q), depth);
  }

  relatedFrom(seeds, depth = 1) {
    const seen = new Set(seeds);
    let frontier = [...seeds];
    const out = [];
    for (let d = 0; d < Math.max(1, depth); d++) {
      const next = [];
      for (const s of frontier) {
        for (const link of this.docs[s].links) {
          const i = this.byKey.get(`${link.repo}\u0000${link.id}`);
          if (i === undefined || seen.has(i)) continue;
          seen.add(i);
          out.push({ row: i, link_kind: link.kind });
          next.push(i);
        }
      }
      frontier = next;
    }
    return out;
  }
}
