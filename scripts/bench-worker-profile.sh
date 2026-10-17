#!/bin/sh
# Samply-profile the worker over realworld-small via the mock hopper.
# Usage: profile-worker.sh OUT_PROFILE [BINARY]
# Assumes fletch caches are warm. Mirrors the Makefile worker-benchmark env.
set -eu

out=${1:-/tmp/litmus-worker-profile.json.gz}
binary=${2:-/home/t/scan/out/atomscan.bench}
dataset=${DATASET:-$HOME/data/benchmark/realworld-small}
hopper_out=$(mktemp)

/home/t/scan/target/release/scan-bench-hopper --dataset "$dataset" --port 0 \
    --order fifo >"$hopper_out" 2>/tmp/scan-bench-hopper-prof.err &
hp=$!
trap 'kill $hp 2>/dev/null' EXIT INT TERM
port=
for _ in $(seq 1 100); do
    port=$(sed -n 's/^PORT=//p' "$hopper_out")
    [ -n "$port" ] && break
    sleep 0.1
done
[ -n "$port" ] || { echo "error: mock hopper did not start" >&2; exit 1; }
echo "hopper on port $port"

# Bypass persisted analysis results while retaining compiled runtime caches, as
# a long-lived worker does after startup.
SCAN_NO_ANALYSIS_CACHE=1 SCAN_FETCH=all SCAN_NO_UPDATE_CHECK=1 \
SCAN_HEARTBEAT_SECS=5 \
samply record --save-only -o "$out" -- \
    "$binary" worker \
    --url "http://127.0.0.1:$port" \
    --data-dir "$dataset" \
    --exit-if-empty --nice 0 --no-update --no-validate \
    --workers 4 \
    2>&1 | tee /tmp/litmus-worker-profile.log
echo "profile: $out"
