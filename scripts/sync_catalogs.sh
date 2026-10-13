#!/usr/bin/env bash
# Sync vendored catalog snapshots from the live catalog repos, rebuild the index,
# and run the retrieval quality gate. Maintainer-side only: requires read access
# to the live repos. Does NOT commit — git actions stay manual.
#
# Since the 2026-10-13 cut-over the fault entries are the upstream structured
# catalogs (catalog/entries/*.yaml + catalog/id_registry.tsv). Negative docs are
# unchanged: training rejected.md / blindspots.md, and appendix C/D of the frozen
# inference Markdown master (no longer updated upstream).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Live repo roots; override via env when the catalog repos move.
SRC_INF="${FAULT_RAG_SRC_INFERENCE_REPO:-/workspace/inference_error_review}"
SRC_TR="${FAULT_RAG_SRC_TRAINING_REPO:-/workspace/training_error_review}"

for f in "$SRC_INF/catalog/entries" "$SRC_INF/catalog/id_registry.tsv" "$SRC_INF/FAULT_MASTER_REFERENCE.zh.md" \
         "$SRC_TR/catalog/entries" "$SRC_TR/catalog/id_registry.tsv" \
         "$SRC_TR/catalog/rejected.md" "$SRC_TR/catalog/blindspots.md"; do
    [ -r "$f" ] || { echo "sync_catalogs: cannot read $f" >&2; exit 1; }
done

sync_entries() {   # src_repo dst_dir
    local tmp
    tmp="$(mktemp -d)"
    cp -a "$1/catalog/entries/." "$tmp/"
    rm -rf "$2/entries"
    mkdir -p "$2"
    mv "$tmp" "$2/entries"
    chmod 0755 "$2/entries"
    find "$2/entries" -type f -exec chmod 0644 {} +
    install -m 0644 "$1/catalog/id_registry.tsv" "$2/id_registry.tsv"
}
sync_entries "$SRC_INF" catalogs/inference
sync_entries "$SRC_TR" catalogs/training
install -m 0644 "$SRC_INF/FAULT_MASTER_REFERENCE.zh.md" catalogs/inference/FAULT_MASTER_REFERENCE.zh.md
install -m 0644 "$SRC_TR/catalog/rejected.md"  catalogs/training/rejected.md
install -m 0644 "$SRC_TR/catalog/blindspots.md" catalogs/training/blindspots.md

{
    echo "# upstream commits of this snapshot (written by scripts/sync_catalogs.sh)"
    for r in "$SRC_INF" "$SRC_TR"; do
        echo "$(basename "$r") $(git -C "$r" rev-parse HEAD) $(git -C "$r" log -1 --format=%cI) entries=$(ls "$r/catalog/entries" | grep -c '\.yaml$')"
    done
} > catalogs/SOURCES.txt
echo "snapshots updated under catalogs/:"
cat catalogs/SOURCES.txt

./fault-rag reindex

# Acceptance gate: the 36-case bench and the 63-case independent set; Recall@5/@10
# may not drop below either baseline (see eval/README.md, eval/ACCEPTANCE.md).
mkdir -p eval/results
for set in "cases baseline bench" "indy_cases baseline_indy indy"; do
    read -r cases base name <<< "$set"
    python3 eval/run_eval.py --cases "eval/$cases.jsonl" --check "eval/$base.json" \
        --output "eval/results/sync_$name.json" > /dev/null
    python3 - "eval/$base.json" "eval/results/sync_$name.json" "$name" <<'EOF'
import json, sys
b, r = (json.load(open(p))["aggregate"] for p in sys.argv[1:3])
print(f"{sys.argv[3]:5s}", "  ".join(f"{k} {b.get(k, float('nan')):.4f}->{r[k]:.4f}"
                                     for k in ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")))
EOF
done

echo
echo "quality gate passed. Next steps:"
echo "  git add catalogs/ && git commit -m \"sync catalogs $(date +%F)\" && git push"
