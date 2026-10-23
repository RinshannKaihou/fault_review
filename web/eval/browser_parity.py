#!/usr/bin/env python3
"""Shipped-vs-gated check: rank every eval case in headless Chromium (the page's own search.js,
semantic.js and vendored onnxruntime-web wasm) and compare with rank.mjs in Node (onnxruntime-node),
which is what check_web.py gates. Not part of CI (needs Playwright + Chromium); run it after a
transformers.js / onnxruntime / model bump.

  python3 web/eval/browser_parity.py                 # model from huggingface.co (or HF_ENDPOINT)
  FAULT_RAG_WEB_MODEL_DIR=<dir> python3 web/eval/browser_parity.py   # offline, same layout as Node
"""

import argparse
import functools
import http.server
import json
import os
import re
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
ROOT = os.path.dirname(WEB)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from check_web import SUITES, rank_with_js           # noqa: E402
from eval.run_eval import evaluate_cases, load_cases, resolve_legacy_ids   # noqa: E402

MODEL_DIR = os.environ.get("FAULT_RAG_WEB_MODEL_DIR")


class Handler(http.server.SimpleHTTPRequestHandler):
    """dist/ at /, and (offline mode) <MODEL_DIR>/<model>/<file> at /_models/<model>/resolve/<rev>/<file>
    with Range support, which transformers.js uses to probe file sizes."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        m = re.match(r"^/_models/(.+?)/resolve/[^/]+/(.+)$", self.path)
        if not m:
            return super().do_GET()
        path = os.path.join(MODEL_DIR, m.group(1), m.group(2))
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        start, end = 0, size - 1
        rng = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range") or "")
        if rng:
            start, end = int(rng.group(1)), int(rng.group(2) or size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                self.wfile.write(chunk)
                left -= len(chunk)


PAGE_SCRIPT = """async ({cases, host}) => {
  const u = (p) => new URL(p, location.href).href;
  const { Index } = await import(u('search.js'));
  const { configureWeb, createEmbedder } = await import(u('semantic.js'));
  const lib = configureWeb(await import(u('vendor/transformers.min.js')), host);
  const get = async (p) => { const r = await fetch(u(p)); if (!r.ok) throw new Error(p); return r; };
  const meta = await (await get('data/meta.json')).json();
  const docs = await (await get('data/docs.json')).json();
  const bm25 = await (await get('data/bm25.json')).json();
  const vec = new Float32Array(await (await get('data/vectors.f32')).arrayBuffer());
  const index = new Index(docs, bm25, vec, meta.semantic.dim);
  const emb = await createEmbedder(lib, meta.semantic);
  const out = [];
  for (const c of cases) {
    const qv = await emb.query(c.query);
    out.push(index.search(c.query, { topk: 10, mode: 'hybrid', ...c.filters }, qv).map((r) => [r.repo, r.id]));
  }
  return out;
}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dist", default=os.path.join(WEB, "dist"))
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    with open(os.path.join(args.dist, "data", "docs.json"), encoding="utf-8") as f:
        docs = json.load(f)
    keys = {(d["repo"], d["id"]) for d in docs}
    legacy = {(d["repo"], o): (d["repo"], d["id"]) for d in docs for o in d["legacy_ids"] or []}

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=args.dist))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/"
    host = (base + "_models/" if MODEL_DIR else
            os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/")

    failed = False
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(base)
        for name, fname, _, _ in SUITES:
            cases = resolve_legacy_ids(load_cases(os.path.join(ROOT, "eval", fname)), keys, legacy)
            node = rank_with_js(args.dist, "hybrid", cases, [])["rankings"]
            web = page.evaluate(PAGE_SCRIPT, {"cases": [{"query": c["query"], "filters": c["filters"]}
                                                        for c in cases], "host": host})
            diff = [c["case_id"] for c, a, b in zip(cases, node, web) if a != b]
            agg = {}
            for label, ranked in (("node", node), ("browser", web)):
                lookup = {(c["query"], json.dumps(c["filters"], sort_keys=True)): r for c, r in zip(cases, ranked)}
                rep = evaluate_cases(cases, search_fn=lambda q, topk, mode, **f: [
                    {"repo": r, "id": i} for r, i in lookup[(q, json.dumps(f, sort_keys=True))][:topk]])
                agg[label] = rep["aggregate"]
            same = agg["node"] == agg["browser"]
            failed |= not same
            print(f"{name:8s} top-10 identical {len(cases) - len(diff)}/{len(cases)}"
                  f"{'' if not diff else ' (differ: ' + ', '.join(diff) + ')'}; metrics "
                  f"{'identical' if same else 'DIFFER: node ' + str(agg['node']) + ' browser ' + str(agg['browser'])}")
        browser.close()
    server.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
