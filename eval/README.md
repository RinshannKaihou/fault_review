# fault-rag retrieval evaluation

Two labelled sets gate the agent-facing search API:

- `cases.jsonl`: 36 source-verified queries spanning inference, training, negative
  knowledge, Chinese/English mixes, pointer opt-in, and the known duplicate-name family.
- `indy_cases.jsonl`: the 63-case independent blind set (2026-09-23). Queries were written
  by separate generators that read only the source entries, de-contaminated against entry
  ids/names and long n-gram overlap, then blind-reviewed. Its negative cases carry both the
  original `neg-docs` tag and `negative`. Tracked and gated since 2026-10-14; it is no
  longer blind to future changes, so do not tune against individual cases.

A third set is measured but not gated:

- `heldout_cases.jsonl`: 50 colloquial queries (2026-10-22), one per randomly sampled entry
  (seed 20261022; 40 inference, 10 training) that is not a gold in either gated set. The
  writer model (qwen3.8-max) saw only the entry's observable fields (effect detail,
  behaviour, trigger surface) and was told not to copy distinctive phrases or identifiers.
  It was the fresh check for the reranker; it has now informed one decision, so the next
  change that needs an untouched check needs a new set.

## Label discipline

Each row contains:

- `case_id`: stable unique identifier;
- `query`: user-style retrieval query;
- `mode`: `hybrid`, `dense`, or `bm25`;
- `filters`: arguments passed to `fault_rag.search.search`;
- `relevant`: one or more `{repo, id}` labels with current ids (`INF-0044`, `TRN-0001`,
  `NEG-INF-C6`). The labels were first written with the old Markdown ids (`1.44`, `HW.01`) and
  rewritten one-to-one through the snapshot's `legacy_ids` when those ids were retired
  (2026-10-26); an old id is now an unknown entry and fails validation;
- `tags`: reporting slices;
- `rationale`: why the source entry is relevant.

Labels must be chosen by reading the source entry, not by copying the current top-k
output. The evaluator fails on duplicate case IDs, missing labels, unknown entry IDs,
or unsupported filters.

## Run

From the repository root:

```bash
# Full labelled evaluation; writes a report only if absolute quality gates pass.
python3 eval/run_eval.py \
  --cases eval/cases.jsonl \
  --output eval/baseline.json

# Release/regression gate against the checked-in baselines (both must pass;
# scripts/sync_catalogs.sh runs both).
python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json
python3 eval/run_eval.py --cases eval/indy_cases.jsonl --check eval/baseline_indy.json --per-case

# Observation only.
python3 eval/run_eval.py --cases eval/heldout_cases.jsonl --no-gate --min-cases 50
```

The model is loaded once per run. Output is JSON on stdout; gate failures are also
written to stderr and return exit status 1.

`--no-gate` only measures and writes the report (for ad-hoc case files).

## Metrics and gates

Reported per case and per slice: Recall@5, Recall@10, MRR@10, nDCG@10.

- aggregate Recall@5 >= 0.85
- aggregate MRR@10 >= 0.70
- aggregate nDCG@10 >= 0.75
- negative-slice Recall@5 >= 0.75
- aggregate Recall@5 and Recall@10 may not drop at all below the baseline
  (`baseline.json` / `baseline_indy.json`); MRR@10 and nDCG@10 may regress by at most 0.02
  (metrics missing from an older baseline are skipped)

If a catalog sync legitimately moves a gold entry (e.g. a new near-duplicate entry),
review the case, then regenerate the baseline with `--output` and record why in
ACCEPTANCE.md. The baselines are reviewed quality artifacts and are tracked. Ad-hoc reports belong
under `eval/results/`, which is ignored by Git; acceptance results worth keeping (e.g.
the 2026-10-13 switch to the YAML catalogs) are recorded in [ACCEPTANCE.md](ACCEPTANCE.md).
