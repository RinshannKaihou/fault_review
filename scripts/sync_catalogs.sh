#!/usr/bin/env bash
# Sync vendored catalog snapshots from the live catalog repos, rebuild the index,
# and run BOTH the CLI retrieval gate (when neural models are installed) and the
# Web static-site BM25 gate. Maintainer-side only; does NOT commit.
#
# Usage:
#   scripts/sync_catalogs.sh                  # full sync: CLI reindex + CLI gate + Web gate
#   scripts/sync_catalogs.sh --lexical-only   # model-free sync: lexical CLI index + Web BM25 gate
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEXICAL_ONLY=0
ALLOW_PENDING=0
for arg in "$@"; do
    case "$arg" in
        --lexical-only) LEXICAL_ONLY=1 ;;
        --allow-pending) ALLOW_PENDING=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

DEFAULT_INF="/workspace/inference_error_review"
[ -d "$DEFAULT_INF" ] || DEFAULT_INF="$(cd "$ROOT/.." && pwd)/inference_error_review"
DEFAULT_TR="/workspace/training_error_review"
[ -d "$DEFAULT_TR" ] || DEFAULT_TR="$(cd "$ROOT/.." && pwd)/training_error_review"

SRC_INF="${FAULT_RAG_SRC_INFERENCE_REPO:-$DEFAULT_INF}"
SRC_TR="${FAULT_RAG_SRC_TRAINING_REPO:-$DEFAULT_TR}"

for f in "$SRC_INF/catalog/entries" "$SRC_INF/catalog/id_registry.tsv" "$SRC_INF/FAULT_MASTER_REFERENCE.zh.md" \
         "$SRC_TR/catalog/entries" "$SRC_TR/catalog/id_registry.tsv" \
         "$SRC_TR/catalog/rejected.md" "$SRC_TR/catalog/blindspots.md"; do
    [ -r "$f" ] || { echo "sync_catalogs: cannot read $f" >&2; exit 1; }
done

if [ "$ALLOW_PENDING" -eq 0 ]; then
    pending="$(python3 - "$SRC_INF/catalog/entries" "$SRC_TR/catalog/entries" <<'EOF'
import os, sys, yaml
n = 0
for d in sys.argv[1:]:
    for fn in os.listdir(d):
        if fn.endswith(".yaml"):
            if (yaml.safe_load(open(os.path.join(d, fn), encoding="utf-8")) or {}).get("status") == "pending_audit":
                n += 1
print(n)
EOF
)"
    if [ "$pending" -gt 0 ]; then
        echo "sync_catalogs: refusing to sync while $pending upstream entries have status=pending_audit (wait for daily audit or pass --allow-pending)" >&2
        exit 1
    fi
fi

sync_dir_atomic() {   # src_dir dst_dir
    local tmp
    tmp="$(mktemp -d "${2}.tmp.XXXXXX")"
    cp -a "$1/." "$tmp/"
    chmod 0755 "$tmp"
    find "$tmp" -type f -exec chmod 0644 {} +
    rm -rf "$2"
    mv "$tmp" "$2"
}

mkdir -p catalogs/inference catalogs/training
sync_dir_atomic "$SRC_INF/catalog/entries" catalogs/inference/entries
sync_dir_atomic "$SRC_TR/catalog/entries"  catalogs/training/entries
[ -d "$SRC_INF/catalog/patterns" ] && sync_dir_atomic "$SRC_INF/catalog/patterns" catalogs/inference/patterns
[ -d "$SRC_TR/catalog/patterns" ]  && sync_dir_atomic "$SRC_TR/catalog/patterns"  catalogs/training/patterns

install -m 0644 "$SRC_INF/catalog/id_registry.tsv"         catalogs/inference/id_registry.tsv
install -m 0644 "$SRC_TR/catalog/id_registry.tsv"          catalogs/training/id_registry.tsv
install -m 0644 "$SRC_INF/FAULT_MASTER_REFERENCE.zh.md"    catalogs/inference/FAULT_MASTER_REFERENCE.zh.md
[ -r "$SRC_INF/catalog/rejected.md" ]   && install -m 0644 "$SRC_INF/catalog/rejected.md"   catalogs/inference/rejected.md
[ -r "$SRC_INF/catalog/blindspots.md" ] && install -m 0644 "$SRC_INF/catalog/blindspots.md" catalogs/inference/blindspots.md
install -m 0644 "$SRC_TR/catalog/rejected.md"              catalogs/training/rejected.md
install -m 0644 "$SRC_TR/catalog/blindspots.md"            catalogs/training/blindspots.md

{
    echo "# upstream commits of this snapshot (written by scripts/sync_catalogs.sh)"
    for r in "$SRC_INF" "$SRC_TR"; do
        active="$(python3 - "$r/catalog/entries" <<'EOF'
import os, sys, yaml
print(sum(
    (yaml.safe_load(open(os.path.join(sys.argv[1], fn), encoding="utf-8")) or {}).get("status") not in ("merged", "retired")
    for fn in os.listdir(sys.argv[1]) if fn.endswith(".yaml")
))
EOF
)"
        echo "$(basename "$r") $(git -C "$r" rev-parse HEAD) $(git -C "$r" log -1 --format=%cI) entries=$active"
    done
} > catalogs/SOURCES.txt
echo "snapshots updated under catalogs/:"
cat catalogs/SOURCES.txt

if [ "$LEXICAL_ONLY" -eq 1 ]; then
    ./fault-rag reindex --lexical > /dev/null
else
    ./fault-rag reindex > /dev/null
    mkdir -p eval/results
    for set in "cases baseline bench" "indy_cases baseline_indy indy --per-case"; do
        read -r cases base name flags <<< "$set"
        python3 eval/run_eval.py --cases "eval/$cases.jsonl" --check "eval/$base.json" $flags \
            --output "eval/results/sync_$name.json" > /dev/null
        python3 - "eval/$base.json" "eval/results/sync_$name.json" "$name" <<'EOF'
import json, sys
b, r = (json.load(open(p))["aggregate"] for p in sys.argv[1:3])
print(f"cli/{sys.argv[3]:5s}", "  ".join(f"{k} {b.get(k, float('nan')):.4f}->{r[k]:.4f}"
                                         for k in ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")))
EOF
    done
fi

# Always run the Web static-site quality gate so CI Pages failures are caught locally before git push.
WEB_TMP="$(mktemp -d "${TMPDIR:-/tmp}/fault_rag_web_gate.XXXXXX")"
trap 'rm -rf "$WEB_TMP"' EXIT
if [ -d "$ROOT/web/node_modules/@huggingface/transformers" ] && [ "$LEXICAL_ONLY" -eq 0 ]; then
    python3 web/build_site.py --out "$WEB_TMP/dist"
    python3 web/eval/check_web.py --dist "$WEB_TMP/dist"
else
    python3 web/build_site.py --no-vectors --out "$WEB_TMP/dist"
    python3 web/eval/check_web.py --dist "$WEB_TMP/dist" --tier bm25
fi

echo
echo "quality gate passed. Next steps:"
echo "  git add catalogs/ && git commit -m \"sync catalogs $(date +%F)\" && git push"
