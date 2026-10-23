// transformers.js for Node, running on the browser's runtime: onnxruntime-web's wasm (the same
// binary the site vendors), one thread. Native onnxruntime-node uses different int8 kernels — its
// query vectors are only cos ~0.999 to the browser's, enough to reorder near-ties, and they vary
// across x86 CPUs. The wasm path is bit-identical to Chromium (checked by eval/browser_parity.py)
// and across machines, so corpus vectors, the gate and users' browsers all see the same numbers.
//   FAULT_RAG_WEB_MODEL_DIR=<dir>  load <dir>/Xenova/multilingual-e5-small/... offline
//   HF_ENDPOINT=<url>              fetch through a mirror (e.g. https://hf-mirror.com)
// Otherwise files come from huggingface.co and are cached under node_modules (CI caches it).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const WEB = path.dirname(fileURLToPath(import.meta.url));
// With an injected runtime transformers.js accepts only "auto" (no EP list: ORT's default, wasm).
export const DEVICE = "auto";

let loaded = null;

// Call before anything imports @huggingface/transformers: the library binds its runtime on first
// import, and would otherwise pick onnxruntime-node.
export function transformers() {
  loaded ??= (async () => {
    const dist = path.join(WEB, "node_modules", "onnxruntime-web", "dist") + path.sep;
    const ort = await import(pathToFileURL(dist + "ort.node.min.mjs").href);
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { wasm: dist + "ort-wasm-simd-threaded.wasm", mjs: dist + "ort-wasm-simd-threaded.mjs" };
    globalThis[Symbol.for("onnxruntime")] = ort;
    const lib = await import("@huggingface/transformers");
    lib.env.useWasmCache = false;     // its pre-loader fetch()es wasmPaths; Node can't for file paths
    if (process.env.FAULT_RAG_WEB_MODEL_DIR) {
      lib.env.localModelPath = process.env.FAULT_RAG_WEB_MODEL_DIR;
      lib.env.allowRemoteModels = false;
    } else if (process.env.HF_ENDPOINT) {
      lib.env.remoteHost = process.env.HF_ENDPOINT.replace(/\/?$/, "/");
    }
    return lib;
  })();
  return loaded;
}

export const modelConfig = () =>
  JSON.parse(fs.readFileSync(path.join(WEB, "src", "model.json"), "utf8"));
