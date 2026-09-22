# fault-rag retrieval evaluation

`cases.jsonl` is a source-verified relevance set for the agent-facing search API.
It contains 36 queries spanning inference, training, negative knowledge, Chinese/English
mixes, pointer opt-in, and the known duplicate-name family.

## Label discipline

Each row contains:

- `case_id`: stable unique identifier;
- `query`: user-style retrieval query;
- `mode`: `hybrid`, `dense`, or `bm25`;
- `filters`: arguments passed to `fault_rag.search.search`;
- `relevant`: one or more `{repo, id}` labels;
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

# Release/regression gate against the checked-in baseline.
python3 eval/run_eval.py \
  --cases eval/cases.jsonl \
  --check eval/baseline.json
```

The model is loaded once per run. Output is JSON on stdout; gate failures are also
written to stderr and return exit status 1.

## Metrics and gates

- aggregate Recall@5 >= 0.85
- aggregate MRR@10 >= 0.70
- aggregate nDCG@10 >= 0.75
- negative-slice Recall@5 >= 0.75
- no aggregate metric may regress by more than 0.02 from `baseline.json`

`baseline.json` is a reviewed quality artifact and is tracked. Ad-hoc reports belong
under `eval/results/`, which is ignored by Git.
