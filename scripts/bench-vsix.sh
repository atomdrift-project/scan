#!/bin/sh
# Linux preset for complete VSIX scans, including native helpers and all members.
# Usage: sh scripts/bench-vsix.sh CORPUS SCANNER RIZIN OUTPUT_PREFIX
# Override any scheduling variable below to compare configurations. The default
# requires all five fresh-process runs to complete within 47 seconds.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
corpus=${1:?usage: bench-vsix.sh CORPUS SCANNER RIZIN OUTPUT_PREFIX}
scanner=${2:?usage: bench-vsix.sh CORPUS SCANNER RIZIN OUTPUT_PREFIX}
rizin=${3:?usage: bench-vsix.sh CORPUS SCANNER RIZIN OUTPUT_PREFIX}
output=${4:?usage: bench-vsix.sh CORPUS SCANNER RIZIN OUTPUT_PREFIX}
export CLEAVE_NATIVE_MEMBER_THREADS=${CLEAVE_NATIVE_MEMBER_THREADS:-32}
export CLEAVE_SOURCE_LEAF_THREADS=${CLEAVE_SOURCE_LEAF_THREADS:-32}
export CLEAVE_SOURCE_LEAF_MIN_KB=${CLEAVE_SOURCE_LEAF_MIN_KB:-3072}
export CLEAVE_SOURCE_EARLY_DISPATCH=${CLEAVE_SOURCE_EARLY_DISPATCH:-1}
export CLEAVE_AST_LOOP_GATE=${CLEAVE_AST_LOOP_GATE:-1}
export CLEAVE_SOURCE_FLIGHT_WAIT=${CLEAVE_SOURCE_FLIGHT_WAIT:-1}
export CLEAVE_ANALYSIS_MEMO_MB=${CLEAVE_ANALYSIS_MEMO_MB:-4096}
export CLEAVE_MEMBER_WINDOW_COUNT=${CLEAVE_MEMBER_WINDOW_COUNT:-8192}
export CLEAVE_NATIVE_FACTS_THREADS=${CLEAVE_NATIVE_FACTS_THREADS:-4}
export CLEAVE_NATIVE_FACTS_ISOLATED=${CLEAVE_NATIVE_FACTS_ISOLATED:-1}
export CLEAVE_RAYON_THREADS=${CLEAVE_RAYON_THREADS:-96}
export CLEAVE_AST_BATCH_CPU_MS=${CLEAVE_AST_BATCH_CPU_MS:-5000}
export CLEAVE_AST_BATCH_GUARDED=${CLEAVE_AST_BATCH_GUARDED:-1}
export CLEAVE_MEMBER_PRESSURE_FLOOR_MB=${CLEAVE_MEMBER_PRESSURE_FLOOR_MB:-8192}
export CLEAVE_PRIORITY_HEAVY_MEMBERS=${CLEAVE_PRIORITY_HEAVY_MEMBERS:-1}
export CLEAVE_MEMGATE_EXPANSION=${CLEAVE_MEMGATE_EXPANSION:-32}
export _RJEM_MALLOC_CONF=${_RJEM_MALLOC_CONF:-retain:true,dirty_decay_ms:60000,muzzy_decay_ms:60000}
export RUST_LOG=${RUST_LOG:-warn}
exec python3 "$repo_dir/scripts/bench-balanced.py" "$corpus" \
    --binary "$scanner" --rizin "$rizin" --native-cpus "${VSIX_NATIVE_CPUS:-0,1}" \
    --output "$output" --runs "${VSIX_RUNS:-5}" --limit "${VSIX_LIMIT:-47}"
