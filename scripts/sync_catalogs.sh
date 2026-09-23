#!/usr/bin/env bash
# Sync vendored catalog snapshots from the live catalog repos, rebuild the
# index, and run the retrieval quality gate. Maintainer-side only: requires
# read access to the live repos. Does NOT commit — git actions stay manual.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Live source locations; override via env when the catalog repos move.
SRC_INF="${FAULT_RAG_SRC_INFERENCE:-/workspace/inference_error_review/FAULT_MASTER_REFERENCE.zh.md}"
SRC_TR_DIR="${FAULT_RAG_SRC_TRAINING:-/workspace/training_error_review/catalog}"

for f in "$SRC_INF" "$SRC_TR_DIR/FAULT_MASTER_REFERENCE.zh.md" \
         "$SRC_TR_DIR/rejected.md" "$SRC_TR_DIR/blindspots.md"; do
    [ -r "$f" ] || { echo "sync_catalogs: cannot read $f" >&2; exit 1; }
done

install -m 0644 "$SRC_INF" catalogs/inference/FAULT_MASTER_REFERENCE.zh.md
install -m 0644 "$SRC_TR_DIR/FAULT_MASTER_REFERENCE.zh.md" catalogs/training/FAULT_MASTER_REFERENCE.zh.md
install -m 0644 "$SRC_TR_DIR/rejected.md"  catalogs/training/rejected.md
install -m 0644 "$SRC_TR_DIR/blindspots.md" catalogs/training/blindspots.md
echo "snapshots updated under catalogs/"

./fault-rag reindex
python3 eval/run_eval.py --cases eval/cases.jsonl --check eval/baseline.json

echo
echo "quality gate passed. Next steps:"
echo "  git add catalogs/ && git commit -m \"sync catalogs $(date +%F)\" && git push"
