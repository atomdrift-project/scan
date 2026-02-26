#!/bin/bash
# Batch feature extraction with crash recovery
# Usage: batch-extract.sh <input_dir> <output_prefix> <label> <batch_size> <workers> <limit> [options]
# Options: --require-suspicious, --extension <ext>

set -o pipefail

INPUT_DIR="$1"
OUTPUT_PREFIX="$2"
LABEL="$3"
BATCH_SIZE="${4:-1024}"
WORKERS="${5:-4}"
LIMIT="${6:-0}"
shift 6

# Parse optional flags
EXTRA_FLAGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --require-suspicious)
            EXTRA_FLAGS="$EXTRA_FLAGS --require-suspicious"
            shift
            ;;
        --extension)
            EXTRA_FLAGS="$EXTRA_FLAGS --extension $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_PREFIX" || -z "$LABEL" ]]; then
    echo "Usage: $0 <input_dir> <output_prefix> <label> [batch_size] [workers] [limit] [--require-suspicious] [--extension ext]"
    exit 1
fi

# Get script directory for locating divine binary
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIVINE_BIN="$SCRIPT_DIR/../target/release/divine"

if [[ ! -x "$DIVINE_BIN" ]]; then
    echo "Error: divine binary not found at $DIVINE_BIN"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PREFIX")"

# Collect all files
echo "  Collecting files from $INPUT_DIR..."
TEMP_FILE_LIST=$(mktemp)
find "$INPUT_DIR" -type f ! -name "*.json" ! -name "*.txt" ! -name ".*" 2>/dev/null > "$TEMP_FILE_LIST"

TOTAL_FILES=$(wc -l < "$TEMP_FILE_LIST" | tr -d ' ')
echo "  Found $TOTAL_FILES files"

# Shuffle and apply limit if specified
if [[ "$LIMIT" -gt 0 && "$TOTAL_FILES" -gt "$LIMIT" ]]; then
    echo "  Randomly sampling $LIMIT files"
    TEMP_LIMITED=$(mktemp)
    sort -R "$TEMP_FILE_LIST" | head -n "$LIMIT" > "$TEMP_LIMITED"
    mv "$TEMP_LIMITED" "$TEMP_FILE_LIST"
    TOTAL_FILES="$LIMIT"
fi

# Calculate number of batches
NUM_BATCHES=$(( (TOTAL_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "  Processing in $NUM_BATCHES batches of $BATCH_SIZE files each"

# Process each batch
SUCCEEDED=0
FAILED=0
FAILED_BATCHES=""

for ((i=0; i<NUM_BATCHES; i++)); do
    BATCH_NUM=$((i + 1))
    SKIP=$((i * BATCH_SIZE))
    OUTPUT_FILE="${OUTPUT_PREFIX}_batch_$(printf '%04d' $BATCH_NUM).json"

    # Skip if output already exists (resume support)
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo "  [Batch $BATCH_NUM/$NUM_BATCHES] Skipping (already exists): $OUTPUT_FILE"
        SUCCEEDED=$((SUCCEEDED + 1))
        continue
    fi

    echo "  [Batch $BATCH_NUM/$NUM_BATCHES] Processing files $((SKIP + 1))-$((SKIP + BATCH_SIZE))..."

    # Extract batch of files and save file list
    BATCH_FILE_LIST="${OUTPUT_PREFIX}_batch_$(printf '%04d' $BATCH_NUM).files"
    tail -n "+$((SKIP + 1))" "$TEMP_FILE_LIST" | head -n "$BATCH_SIZE" > "$BATCH_FILE_LIST"

    # Run extraction (capture exit code)
    if cat "$BATCH_FILE_LIST" | "$DIVINE_BIN" extract - -o "$OUTPUT_FILE" -l "$LABEL" -w "$WORKERS" $EXTRA_FLAGS 2>&1; then
        SUCCEEDED=$((SUCCEEDED + 1))
        echo "  [Batch $BATCH_NUM/$NUM_BATCHES] Success: $OUTPUT_FILE"
        # Remove file list on success (keep output JSON only)
        rm -f "$BATCH_FILE_LIST"
    else
        EXIT_CODE=$?
        FAILED=$((FAILED + 1))
        FAILED_BATCHES="$FAILED_BATCHES $BATCH_NUM"
        echo "  [Batch $BATCH_NUM/$NUM_BATCHES] FAILED (exit $EXIT_CODE)"
        echo "  File list saved: $BATCH_FILE_LIST"
    fi
done

rm -f "$TEMP_FILE_LIST"

echo ""
echo "  Batch extraction complete:"
echo "    Succeeded: $SUCCEEDED/$NUM_BATCHES batches"
echo "    Failed:    $FAILED/$NUM_BATCHES batches"

if [[ -n "$FAILED_BATCHES" ]]; then
    echo "    Failed batches:$FAILED_BATCHES"
    echo ""
    echo "  To debug failed batches, run with DIVINE_DEBUG=1:"
    echo "    DIVINE_DEBUG=1 cat <batch>.files | ./target/release/divine extract - -o /tmp/debug.json -l $LABEL -w 1"
fi

# Exit with error if all batches failed
if [[ "$SUCCEEDED" -eq 0 ]]; then
    exit 1
fi

exit 0
