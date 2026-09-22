# fault-rag

Retrieval over the lab's two living fault catalogs, built for **agent consumption**
(interactive CLI agents such as Kimi Code / Claude Code call it via Bash).

| Source (read-only) | Entries |
|---|---|
| `/workspace/inference_error_review/FAULT_MASTER_REFERENCE.zh.md` | 1171 catalog entries + appendix C/D negative chunks |
| `/workspace/training_error_review/catalog/FAULT_MASTER_REFERENCE.zh.md` | 217 catalog entries |
| `/workspace/training_error_review/catalog/rejected.md`, `blindspots.md` | negative-doc chunks |

**Retrieval**: hybrid — Okapi BM25 (jieba CJK + code tokens, inline implementation)
+ dense `BAAI/bge-m3` embeddings (CPU, via `transformers.AutoModel`), fused with
Reciprocal Rank Fusion (k=60). One catalog entry = one document; no chunking.

## Quick start

```bash
# one-time: model + index (model download needs the lab proxy)
https_proxy=http://127.0.0.1:7890 python3 scripts/download_model.py
./fault-rag reindex            # incremental; --full to rebuild from scratch

./fault-rag search "loss spike 但无 NaN" --topk 5
./fault-rag search "GRPO advantage std zero" --repo training --category RL-ADV
./fault-rag lookup HW.01
./fault-rag related HW.05 --depth 1
./fault-rag stats
```

All output is JSON on stdout. Every result carries `id`, `name`, `repo`,
`category`, `confidence`, `entry_type`, and `file` + `line` — open the source
entry directly at that line for the full text. Canonical-name collisions also
carry `name_ambiguous` and `same_name_ids`; cite the ID, never the name alone.

## Commands

| Command | Purpose |
|---|---|
| `search <query>` | hybrid retrieval. Filters: `--repo inference\|training`, `--category NUM\|cat1\|KV1`, `--confidence verified\|documented\|speculative`, `--trigger yes\|partial\|no`, `--mode hybrid\|dense\|bm25`, `--no-neg`, `--include-pointers`, `--full` |
| `lookup <id\|name>` | exact id / snake_name / alias; substring fallback |
| `related <id\|name>` | walk cross-reference graph (`var:`, `交叉：`, id mentions) |
| `reindex [--full]` | re-parse catalogs, re-embed only entries whose derived embedding text changed |
| `stats` / `taxonomy` | counts by repo / category / confidence |

## Design notes

- **Entry grammar** (validated against each catalog's own stated count):
  `### <ID> \`<snake_name>\`` where ID is `N.N` / `KN.N` (inference) or
  `PREFIX.NN` (training). Trailing notes（`（arxiv）`、`⚑§9`）and dual-name
  headers such as `### 11.5 <name_a> / <name_b>` are handled; the second name is an alias.
- **Stable categories** come from the entry ID grammar (`9.27 → cat9`,
  `K9.4 → KV9`, `RL-ADV.02 → RL-ADV`), not from nested section headings.
- **Negative docs** (`neg: true`) are indexed deliberately: they stop an agent
  from re-proposing already-rejected mechanisms.
- **Pointers are not faults**: explicit pointer/meta rows use
  `entry_type: pointer`, are excluded from search by default, and remain
  available through exact `lookup` or explicit `--include-pointers`.
- **Duplicate canonical names remain separate facts**. Lookup returns every
  source ID and marks the ambiguity instead of silently choosing or renaming one.
- **Cross-reference graph** is extracted from `var:` coverage, `交叉：` pointers,
  and exact mentions of known ids/names in entry text (no version-number false
  positives).
- **Incrementality**: embeddings are cached per `(repo, id)` and recomputed only
  when the exact derived embedding text changes (`embed_hash`). `data/` is fully derived — delete and
  `reindex --full` anytime.
- **CPU-only by design**: torch is the CPU build and the lab GPUs are busy;
  full build embeds 1.4k docs in ~10 min once. Query cost: ~2s warm
  (in-process), ~6-8s for a cold CLI call (model load dominates).

## Layout

```
fault-rag            CLI entry point (bash wrapper, works from any cwd)
fault_rag/           parse.py · embed.py · index.py · search.py · cli.py
data/                entries.jsonl · embeddings.npy · bm25.pkl · meta.json (derived)
models/bge-m3/       embedding model (2.3 GB, gitignored)
scripts/download_model.py
tests/smoke.sh       CLI check + query battery (tests/smoke.py)
eval/                36 source-labelled queries + Recall/MRR/nDCG gate
```

## Retrieval quality gate

```bash
python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json
```

The release gate reports aggregate and tag-sliced Recall@5, MRR@10, and
nDCG@10. Labels are verified from source entries rather than generated from the
current ranking. See `eval/README.md` for schema and thresholds.

## Updating

The catalogs are maintained by other agents (daily scans merge into the masters).
Run `./fault-rag reindex` to pick up changes — only new/changed entries are
re-embedded. This repo never writes into the source catalogs.

## Troubleshooting

- `index not built yet` → `./fault-rag reindex`
- model missing → `https_proxy=http://127.0.0.1:7890 python3 scripts/download_model.py`
- huggingface.co unreachable → the lab proxy `127.0.0.1:7890` (mihomo) is required
