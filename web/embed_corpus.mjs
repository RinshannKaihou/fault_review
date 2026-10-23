#!/usr/bin/env node
// Embed catalog passages for the semantic tier: node embed_corpus.mjs <passages.json> <out.f32>
// Called by build_site.py. Output: row-major float32, one L2-normalised vector per entry.
//
// One passage per forward: the int8 model quantizes activations per tensor, so in a batch every
// vector depends on its neighbours and padding (cos ~0.998 vs alone). Batch 1 makes each vector a
// function of its own text only — adding an entry no longer perturbs every later vector — matches
// the query side, and lets the work shard across single-threaded wasm workers without changing a
// bit of the output. FAULT_RAG_WEB_EMBED_WORKERS overrides the worker count.
import fs from "node:fs";
import os from "node:os";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { createEmbedder } from "./src/semantic.js";
import { DEVICE, modelConfig, transformers } from "./node_env.mjs";

if (isMainThread) {
  const [, , inPath, outPath] = process.argv;
  const passages = JSON.parse(fs.readFileSync(inPath, "utf8"));
  const k = Number(process.env.FAULT_RAG_WEB_EMBED_WORKERS) || Math.min(os.availableParallelism(), 16);
  const t0 = Date.now();
  let done = 0, dim = 0;
  const rows = new Array(passages.length);
  await Promise.all(Array.from({ length: k }, (_, w) => new Promise((resolve, reject) => {
    const idx = passages.map((_, i) => i).filter((i) => i % k === w);
    const worker = new Worker(new URL(import.meta.url), { workerData: { texts: idx.map((i) => passages[i]) } });
    worker.on("message", ({ n, vec }) => {
      rows[idx[n]] = vec;
      dim = vec.length;
      if (++done % 200 === 0) process.stderr.write(`embed ${done}/${passages.length}\r`);
    });
    worker.on("error", reject);
    worker.on("exit", (code) => (code ? reject(new Error(`embed worker exited ${code}`)) : resolve()));
  })));
  const all = new Float32Array(passages.length * dim);
  rows.forEach((v, i) => all.set(v, i * dim));
  fs.writeFileSync(outPath, Buffer.from(all.buffer));
  console.error(`embedded ${passages.length} passages (dim ${dim}, ${k} workers) in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
} else {
  const embedder = await createEmbedder(await transformers(), modelConfig(), { device: DEVICE });
  for (let n = 0; n < workerData.texts.length; n++) {
    const vec = await embedder.passages([workerData.texts[n]]);
    parentPort.postMessage({ n, vec }, [vec.buffer]);
  }
}
