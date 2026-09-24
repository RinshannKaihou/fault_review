# fault-rag

Retrieval over the lab's two living fault catalogs, built for **agent consumption**
(interactive CLI agents such as Kimi Code / Claude Code call it via Bash).

| Vendored snapshot (read-only) | Contents |
|---|---|
| `catalogs/inference/FAULT_MASTER_REFERENCE.zh.md` | inference catalog entries, pointer rows, and appendix C/D negative chunks |
| `catalogs/training/FAULT_MASTER_REFERENCE.zh.md` | training catalog entries |
| `catalogs/training/rejected.md`, `blindspots.md` | negative-doc chunks |

Snapshots are refreshed from the live catalog repos
(`/workspace/inference_error_review`, `/workspace/training_error_review`) by
`scripts/sync_catalogs.sh` **only after maintainer review** (normally a weekly,
manually requested release). Daily source-catalog scans or Git pushes do not
automatically update this snapshot or its local index. The live repos are never
written from here. Run `./fault-rag stats` for current indexed counts rather
than relying on a count in this README.

**Retrieval**: hybrid — Okapi BM25 (jieba CJK + code tokens, inline implementation)
+ dense `BAAI/bge-m3` embeddings (CPU, via `transformers.AutoModel`), fused with
Reciprocal Rank Fusion (k=60). Each primary catalog entry is one record;
appendix/rejected/blindspot text also contributes negative-knowledge chunks.

## Quick start

```bash
# one-time after clone: model + index
python3 scripts/download_model.py
./fault-rag reindex --full     # builds data/ from the vendored catalogs/
                               # later runs: plain `reindex` is incremental

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

Ranks, timings and `file:line` citations change when catalogs or indexes are
rebuilt; use the live CLI output, not a copied example, for citations.

## Commands

| Command | Purpose |
|---|---|
| `search <query>` | hybrid retrieval. Filters: `--repo inference\|training`, `--category NUM\|cat1\|KV1`, `--confidence verified\|documented\|speculative`, `--trigger yes\|partial\|no`, `--mode hybrid\|dense\|bm25`, `--no-neg`, `--include-pointers`, `--full` |
| `lookup <id\|name>` | exact id / snake_name / alias; substring fallback for names, **not** for missing ID-shaped queries |
| `related <id\|name>` | navigate source cross-references (`var:`, `交叉：`, id mentions); not proof of a shared cause |
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
  positives). `related` only means that the source links the records; its
  `link_kind` preserves the source assertion. In particular, `var` is a stronger
  author claim that still requires mechanism-level review; the source's
  `6.254` relation is pending that review.
- **Retired IDs are not redirects**: after the catalog merged `1.70 → 1.44`
  and `11.30 → 11.24`, `lookup 1.70` / `lookup 11.30` returns no record.
  Consult the source catalog's correction note for the canonical IDs. Missing
  ID-shaped queries never fall back to an unrelated partial-ID hit.
- **Dated follow-up fields** remain visible through exact `lookup` but are not
  automatically part of the BM25 or embedding inputs. A verified correction
  that changes today's answer should be incorporated into canonical fields;
  the notes remain provenance, not an automatic ranking boost.
- **Incrementality**: embeddings are cached per `(repo, id)` and recomputed only
  when the exact derived embedding text changes (`embed_hash`). `data/` is fully derived — delete and
  `reindex --full` anytime.
- **CPU embeddings by design**: the embedding model runs on CPU; build/query
  times vary with machine load, corpus size, and whether the model is cached.

## Layout

```
fault-rag            CLI entry point (bash wrapper, works from any cwd)
fault_rag/           parse.py · embed.py · index.py · search.py · cli.py
catalogs/            vendored catalog snapshots (tracked; synced from the live repos)
data/                entries.jsonl · embeddings.npy · bm25.pkl · meta.json (derived, gitignored)
models/bge-m3/       embedding model (gitignored)
scripts/download_model.py · sync_catalogs.sh
tests/smoke.sh       CLI check + query battery (tests/smoke.py)
eval/                curated source-labelled queries + Recall/MRR/nDCG gate
```

## Retrieval quality gate

```bash
python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json
```

The current 36-query gate reports aggregate and tag-sliced Recall@5, MRR@10,
and nDCG@10. Labels are verified from source entries rather than generated from
the current ranking. Passing it does **not** establish correctness for newly
added IDs, retired-ID mappings, links, or real-world query wording: review the
release delta and add source-checked cases/targeted tests. See `eval/README.md`
for schema and thresholds.

## Updating

Daily scans update the two live catalog repos; they **do not** trigger RAG
publication. The maintainer normally requests a manual, Agent-assisted review
about once a week (or an earlier manual release for an important correction):

1. Compare upstream commits and the four vendored inputs. Review new/changed
   mechanisms, rejected evidence, retired IDs, cross-references and new retrieval
   cases against the underlying sources. Record known unresolved claims rather
   than treating a passing structural/evaluation gate as semantic approval.
2. In an **isolated copy** with the current local index, run
   `scripts/sync_catalogs.sh` and targeted lookup/search checks. The staged
   index stores absolute snapshot paths: do **not** copy its generated `data/`
   back into the live checkout.
3. With no concurrent readers, run `./scripts/sync_catalogs.sh` in the live
   checkout; it copies snapshots, incrementally rebuilds the **local** index,
   then runs the release gate. Run the tests, verify `./fault-rag stats`, key
   lookups/negative results and a second no-change `./fault-rag reindex`
   (expect zero new embeddings). The script is **not atomic**: if it fails,
   do not publish; restore or repair the snapshot/index and verify again.
4. Inspect the diff and stage only intended vendored files under `catalogs/`
   (plus any reviewed code/tests). Commit and push **after** all checks pass;
   verify the remote ref. `data/` and `models/` are ignored, so another machine
   must run `./fault-rag reindex` after pulling a new snapshot.

Only added records or records whose **derived embedding input** changed are
re-embedded. Override the live locations with `FAULT_RAG_SRC_INFERENCE` /
`FAULT_RAG_SRC_TRAINING` if they move. This repo never writes into the source
catalogs.

## Troubleshooting

- `index not built yet` → `./fault-rag reindex` (reads the vendored `catalogs/`)
- live catalog repos moved → set `FAULT_RAG_SRC_INFERENCE` / `FAULT_RAG_SRC_TRAINING` for `scripts/sync_catalogs.sh`
- model missing → `python3 scripts/download_model.py`
