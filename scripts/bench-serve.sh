#!/bin/sh
# Benchmark atomscan's long-lived HTTP server over a local dataset.
#
# The caller selects policy through the environment. The Makefile target sets
# SCAN_FETCH=all and disables all analysis-result caches while deliberately
# retaining fletch's warmed download cache.
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "usage: $0 BINARY DATASET OUTPUT_DIR [PORT] [WORKERS]" >&2
    exit 2
fi

binary=$1
dataset=$2
output=$3
port=${4:-49997}
workers=${5:-20}
responses="$output/responses"
server_log="$output/server.log"
url="http://127.0.0.1:$port"

case "$output" in
    "" | / | . | ..)
        echo "error: refusing unsafe output directory: $output" >&2
        exit 2
        ;;
esac
if [ -e "$output" ] || [ -L "$output" ]; then
    echo "error: output path already exists: $output" >&2
    exit 2
fi
if [ ! -x "$binary" ]; then
    echo "error: binary is not executable: $binary" >&2
    exit 2
fi
if [ ! -d "$dataset" ]; then
    echo "error: dataset is not a directory: $dataset" >&2
    exit 2
fi
for command in awk curl find python3 sort xargs; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "error: required command not found: $command" >&2
        exit 2
    }
done

mkdir -p "$responses"

server_pid=
cleanup() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

started_ns=$(date +%s%N)
RUST_LOG=${RUST_LOG:-scan=info} \
    "$binary" serve \
    --bind "127.0.0.1:$port" \
    --workers "$workers" \
    --max-rss-gb -1 \
    --analysis-timeout 0 \
    >"$server_log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 1200); do
    if curl -fsS "$url/_/health" >"$output/health.json" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "error: server exited during startup" >&2
        sed -n '1,200p' "$server_log" >&2
        exit 1
    fi
    sleep 0.1
done
if [ "$ready" -ne 1 ]; then
    echo "error: server did not become healthy" >&2
    exit 1
fi
ready_ns=$(date +%s%N)

export responses url
batch_started_ns=$(date +%s%N)
find "$dataset" -type f -print0 |
    sort -z |
    xargs -0 -r -P "$workers" -n 1 sh -c '
        file=$1
        name=$(basename "$file")
        body="$responses/$name.json"
        code=$(
            curl -sS \
                -o "$body.tmp" \
                -w "%{http_code}" \
                -F "file=@$file" \
                "$url/analyze"
        ) || {
            printf "%s\n" "000" >"$body.status"
            exit 1
        }
        mv "$body.tmp" "$body"
        printf "%s\n" "$code" >"$body.status"
    ' sh
batch_finished_ns=$(date +%s%N)

# Let response bodies and request-local state drop before recording retained RSS.
sleep 1
status_count=$(find "$responses" -type f -name '*.status' | wc -l)
bad_count=$(
    find "$responses" -type f -name '*.status' -exec awk '
        $0 != "200" { bad++ }
        END { print bad + 0 }
    ' {} + |
        awk '{ total += $1 } END { print total + 0 }'
)
file_count=$(find "$dataset" -type f | wc -l)

vmhwm_kib=$(awk '/^VmHWM:/ { print $2 }' "/proc/$server_pid/status")
vmrss_kib=$(awk '/^VmRSS:/ { print $2 }' "/proc/$server_pid/status")
ready_s=$(awk -v a="$started_ns" -v b="$ready_ns" 'BEGIN { printf "%.3f", (b-a)/1000000000 }')
batch_s=$(awk -v a="$batch_started_ns" -v b="$batch_finished_ns" 'BEGIN { printf "%.3f", (b-a)/1000000000 }')
vmhwm_mib=$(awk -v kib="$vmhwm_kib" 'BEGIN { printf "%.1f", kib/1024 }')
vmrss_mib=$(awk -v kib="$vmrss_kib" 'BEGIN { printf "%.1f", kib/1024 }')

scripts/detect-fingerprint.py "$responses" >"$output/fingerprint.json"
cat >"$output/summary.json" <<EOF
{
  "ready_s": $ready_s,
  "batch_s": $batch_s,
  "vmhwm_mib": $vmhwm_mib,
  "vmrss_after_mib": $vmrss_mib,
  "files": $file_count,
  "responses": $status_count,
  "non_200": $bad_count
}
EOF
cat "$output/summary.json"

if [ "$status_count" -ne "$file_count" ] || [ "$bad_count" -ne 0 ]; then
    echo "error: incomplete batch; inspect $responses and $server_log" >&2
    exit 1
fi
