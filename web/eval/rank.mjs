#!/usr/bin/env node
// Rank eval cases with the site's own retrieval code on a built site, for check_web.py.
//   node rank.mjs <dist>   stdin  {"tier": "bm25"|"hybrid", "cases": [{query, filters}], "texts": [..]}
//                          stdout {"rankings": [[[repo, id], ...], ...], "tokens": [[...], ...]}
// "texts" are only tokenized (tokenizer parity check against build_site.tokenize).
import fs from "node:fs";
import path from "node:path";
import { Index, tokenize } from "../src/search.js";
import { createEmbedder } from "../src/semantic.js";
import { DEVICE, transformers } from "../node_env.mjs";

const dist = process.argv[2];
const read = (f) => fs.readFileSync(path.join(dist, "data", f));
const { tier, cases, texts = [] } = JSON.parse(fs.readFileSync(0, "utf8"));
const meta = JSON.parse(read("meta.json"));

let vectors = null, embedder = null;
if (tier === "hybrid") {
  if (!meta.semantic) throw new Error("site was built --no-vectors; no hybrid tier");
  const buf = read("vectors.f32");
  vectors = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  embedder = await createEmbedder(await transformers(), meta.semantic, { device: DEVICE });
}
const index = new Index(JSON.parse(read("docs.json")), JSON.parse(read("bm25.json")),
                        vectors, meta.semantic?.dim ?? 0);

const rankings = [];
for (const c of cases) {
  const qv = embedder ? await embedder.query(c.query) : null;
  const rows = index.search(c.query, { topk: 10, mode: tier, ...c.filters }, qv);
  rankings.push(rows.map((r) => [r.repo, r.id]));
}
process.stdout.write(JSON.stringify({ rankings, tokens: texts.map(tokenize) }));
