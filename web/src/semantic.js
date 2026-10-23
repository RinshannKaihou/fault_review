// Semantic tier: multilingual-e5-small (int8 ONNX, pinned revision) through transformers.js.
// The browser (app.js) and Node (embed_corpus.mjs, eval/rank.mjs) share this code path; only the
// transformers.js build differs (vendored web bundle vs the npm package), so corpus vectors, the
// gated query vectors and the shipped query vectors come from the same model file.

// Browser only: the vendored CPU wasm next to the page (no CDN), models from remoteHost.
// GitHub Pages cannot send the COOP/COEP headers multi-threaded wasm needs, so it runs on one
// thread there; a query is a single short forward, which one thread handles.
export function configureWeb(lib, remoteHost) {
  const abs = (p) => new URL(p, document.baseURI).href;
  lib.env.allowLocalModels = false;
  lib.env.remoteHost = remoteHost;
  lib.env.backends.onnx.wasm.wasmPaths = {
    mjs: abs("vendor/ort-wasm-simd-threaded.mjs"), wasm: abs("vendor/ort-wasm-simd-threaded.wasm"),
  };
  lib.env.backends.onnx.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  return lib;
}

// device: leave unset in the browser (wasm); Node passes node_env.DEVICE.
export async function createEmbedder(lib, cfg, { progress_callback, device } = {}) {
  // Pin the revision in the URL template too: transformers.js 4.3.1 resolves the pipeline's file
  // list and sizes without passing `revision`, so those requests would otherwise hit `main`.
  lib.env.remotePathTemplate = `{model}/resolve/${cfg.revision}/`;
  const extractor = await lib.pipeline("feature-extraction", cfg.model, {
    dtype: cfg.dtype, revision: cfg.revision, progress_callback, device,
  });
  // Mean pooling + L2 norm; the pipeline truncates to the tokenizer's 512-token limit.
  const run = async (texts, prefix) => {
    const t = await extractor(texts.map((s) => prefix + s), { pooling: cfg.pooling, normalize: true });
    return t.data;           // Float32Array, texts.length * dim
  };
  return {
    query: (q) => run([q], cfg.query_prefix),
    passages: (texts) => run(texts, cfg.passage_prefix),
  };
}
