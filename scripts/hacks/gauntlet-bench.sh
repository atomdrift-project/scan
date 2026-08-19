#!/bin/sh
# Per-sample gauntlet benchmark: replicates gauntlet's ascan invocation model —
# one atomscan process per sample, sequential — over a dataset directory.
# Usage: gauntlet-bench.sh <binary> <dataset-dir> <out-dir> [file-list]
# Emits <out-dir>/times.tsv (file, wall_s, maxrss_mib, exit) and per-sample
# stdout JSON under <out-dir>/json/. Prints total wall + peak RSS at the end.
set -eu

BIN=$1
DATASET=$2
OUT=$3
LIST=${4:-}

SNAP=/private/tmp/claude-501/-Users-t-dev-atomdrift-scan/e6ef4f79-beda-4e4a-bf24-4f5010d7032c/scratchpad/traits-snapshot
export SCAN_NO_ANALYSIS_CACHE=1
export SCAN_FETCH=none
export CLEAVE_TRAITS_DIR=$SNAP
export SCAN_NO_UPDATE=1
export RUST_LOG=scan=info,cleave=error

mkdir -p "$OUT/json" "$OUT/err"
: > "$OUT/times.tsv"

if [ -n "$LIST" ]; then
    FILES=$(cat "$LIST")
else
    FILES=$(cd "$DATASET" && ls | LC_ALL=C sort)
fi

total_start=$(date +%s)
echo "$FILES" | while IFS= read -r f; do
    [ -f "$DATASET/$f" ] || continue
    safe=$(printf '%s' "$f" | tr '/ ' '__')
    status=0
    /usr/bin/time -l "$BIN" --format json --show all fs "$DATASET/$f" \
        > "$OUT/json/$safe.json" 2> "$OUT/err/$safe.err" || status=$?
    wall=$(awk '/ real /{print $1; exit}' "$OUT/err/$safe.err")
    rss=$(awk '/maximum resident set size/{printf "%.1f", $1/1048576; exit}' "$OUT/err/$safe.err")
    printf '%s\t%s\t%s\t%s\n' "$f" "$wall" "$rss" "$status" >> "$OUT/times.tsv"
done
total_end=$(date +%s)

echo "=== total wall: $((total_end - total_start)) s over $(grep -c . "$OUT/times.tsv") samples ==="
awk -F'\t' '{w+=$2; if ($3>m) m=$3} END {printf "sum(per-sample wall)=%.1fs  peak RSS=%.1f MiB\n", w, m}' "$OUT/times.tsv"
