#!/bin/sh
# Interleaved full-corpus A/B: for each sample run both binaries back-to-back,
# alternating which goes first, so thermal/load drift hits both legs equally.
# Usage: gauntlet-ab.sh <binA> <binB> <dataset-dir> <out-dir>
set -u

BIN_A=$1
BIN_B=$2
DATASET=$3
OUT=$4

SNAP=/private/tmp/claude-501/-Users-t-dev-atomdrift-scan/e6ef4f79-beda-4e4a-bf24-4f5010d7032c/scratchpad/traits-snapshot
export SCAN_NO_ANALYSIS_CACHE=1
export SCAN_FETCH=none
export CLEAVE_TRAITS_DIR=$SNAP
export SCAN_NO_UPDATE=1
export RUST_LOG=scan=warn,cleave=error

mkdir -p "$OUT/jsonA" "$OUT/jsonB" "$OUT/err"
: > "$OUT/times.tsv"

i=0
cd "$DATASET" || exit 1
ls | LC_ALL=C sort | while IFS= read -r f; do
    [ -f "$DATASET/$f" ] || continue
    i=$((i+1))
    safe=$(printf '%s' "$f" | tr '/ ' '__')
    run_leg() {
        leg=$1; bin=$2
        /usr/bin/time -l "$bin" --format json --show all fs "$DATASET/$f" \
            > "$OUT/json$leg/$safe.json" 2> "$OUT/err/$safe.$leg.err" || true
        awk -v w="" '/ real /{print $1; exit}' "$OUT/err/$safe.$leg.err"
    }
    rss_of() {
        awk '/maximum resident set size/{printf "%.0f", $1/1048576; exit}' "$OUT/err/$safe.$1.err"
    }
    if [ $((i % 2)) -eq 1 ]; then
        wa=$(run_leg A "$BIN_A"); wb=$(run_leg B "$BIN_B")
    else
        wb=$(run_leg B "$BIN_B"); wa=$(run_leg A "$BIN_A")
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$f" "$wa" "$(rss_of A)" "$wb" "$(rss_of B)" >> "$OUT/times.tsv"
done

awk -F'\t' '{a+=$2; b+=$4; if($3>ma)ma=$3; if($5>mb)mb=$5}
    END{printf "TOTAL A=%.0fs B=%.0fs  delta=%.1f%%  peakRSS A=%.0f B=%.0f MiB\n", a, b, 100*(b-a)/a, ma, mb}' "$OUT/times.tsv"
