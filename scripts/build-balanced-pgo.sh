#!/bin/sh
# Build a host-specific scanner trained on cache-disabled analysis.
# PGO_TRAIN_SECONDS bounds instrumentation-heavy training (optional).
# PGO_USE_EXISTING=1 rebuilds using OUTPUT_DIR/atomscan.profdata.
# Usage: LLVM_PROFDATA=/path/to/llvm-profdata sh scripts/build-balanced-pgo.sh CORPUS OUTPUT_DIR [CARGO_CONFIG]
# PATH should select the Rizin build used for evaluation. Analysis scheduling
# variables are inherited. Build and training logs are retained in OUTPUT_DIR.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
corpus=${1:?usage: build-balanced-pgo.sh CORPUS OUTPUT_DIR [CARGO_CONFIG]}
output_dir=${2:?usage: build-balanced-pgo.sh CORPUS OUTPUT_DIR [CARGO_CONFIG]}
config=${3:-}
[ -e "$corpus" ] || { echo "Corpus not found: $corpus" >&2; exit 1; }
llvm_profdata=${LLVM_PROFDATA:-llvm-profdata}
command -v "$llvm_profdata" >/dev/null 2>&1 || { echo "Set LLVM_PROFDATA to the llvm-profdata matching rustc's LLVM version" >&2; exit 1; }
mkdir -p "$output_dir"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd)
if [ "${PGO_USE_EXISTING:-0}" = 1 ]; then
    [ -f "$output_dir/atomscan.profdata" ] || { echo "No existing profile: $output_dir/atomscan.profdata" >&2; exit 1; }
else
    [ ! -e "$output_dir/atomscan.profdata" ] || { echo "Use a fresh output directory, or PGO_USE_EXISTING=1: $output_dir" >&2; exit 1; }
fi
mkdir -p "$output_dir/raw"
export CARGO_TARGET_DIR="$output_dir/target"
base_rustflags="${RUSTFLAGS:-} -C target-cpu=native"
export CFLAGS="${CFLAGS:-} -march=native -mtune=native"
export CXXFLAGS="${CXXFLAGS:-} -march=native -mtune=native"
set --
if [ -n "$config" ]; then set -- --config "$config"; fi
rustc -vV > "$output_dir/toolchain.txt"
"$llvm_profdata" --version >> "$output_dir/toolchain.txt"
if [ "${PGO_USE_EXISTING:-0}" != 1 ]; then
printf '%s\n' "Building instrumented scanner; log: $output_dir/generate.log"
RUSTFLAGS="$base_rustflags -C profile-generate=$output_dir/raw" \
    cargo build --manifest-path "$repo_dir/Cargo.toml" --profile profiling \
    --bin atomscan -j "${PGO_BUILD_JOBS:-12}" "$@" > "$output_dir/generate.log" 2>&1
cp "$CARGO_TARGET_DIR/profiling/atomscan" "$output_dir/atomscan-instrumented"
# Build scripts can also be instrumented. Keep only actual scan counters.
python3 - "$output_dir/raw" <<'PY'
from pathlib import Path
import sys
for path in Path(sys.argv[1]).glob('*.profraw'):
    path.unlink()
PY
printf '%s\n' "Training on corpus: $corpus (time budget: ${PGO_TRAIN_SECONDS:-unlimited})"
train() {
    set --
    if [ -n "${PGO_TRAIN_SECONDS:-}" ]; then
        set -- --interrupt-after "$PGO_TRAIN_SECONDS"
    fi
    LLVM_PROFILE_FILE="$output_dir/raw/%p-%m.profraw" \
        python3 "$repo_dir/scripts/bench-balanced.py" "$corpus" \
        --binary "$output_dir/atomscan-instrumented" --output "$output_dir/training" "$@" \
        > "$output_dir/training.measurement"
}
if ! train; then
    # A deliberately interrupted training run may still provide useful profile
    # counters. It remains a failed benchmark; crashes and other failures abort.
    python3 - "$output_dir/training.time.json" <<'CHECK'
import json
import sys
with open(sys.argv[1]) as source:
    result = json.load(source)
if not result.get('interrupted') or result['exit_code'] not in (0, 1, 2, 130):
    raise SystemExit('Training failed without a controlled interruption')
print('Using interrupted training counters; this is not a valid benchmark result')
CHECK
fi
"$llvm_profdata" merge -o "$output_dir/atomscan.profdata" "$output_dir/raw/"*.profraw
fi
printf '%s\n' "Building optimized scanner; log: $output_dir/use.log"
RUSTFLAGS="$base_rustflags -C profile-use=$output_dir/atomscan.profdata" \
    cargo build --manifest-path "$repo_dir/Cargo.toml" --profile profiling \
    --bin atomscan -j "${PGO_BUILD_JOBS:-12}" "$@" > "$output_dir/use.log" 2>&1
cp "$CARGO_TARGET_DIR/profiling/atomscan" "$output_dir/atomscan"
printf '%s\n' "Built: $output_dir/atomscan (requires this host's CPU features)"
