#!/bin/sh
# Load the apko-built ascan image and assert it works end-to-end:
#  1. ascan --version reports the expected version
#  2. ascan --help runs cleanly
#  3. A scan against a known-clean binary succeeds with the bundled models
#  4. Same scan with bogus SCAN_MODELS_DIR errors out (proves models
#     are actually being consulted, not silently skipped)
#
# Atomdrift Scan exit codes (per project README): 0=clean, 1=hostile,
# 2=suspicious, 3=errors. We accept 0/1/2 from a real scan (the target
# is /usr/bin/cleave which any of those could legitimately apply to);
# 3 means the scan itself failed and that's a real bug.

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../../.." && pwd)
OUT_DIR="$REPO_ROOT/out/wolfi"
IMAGE_TAR="$OUT_DIR/ascan.tar"
IMAGE_REF="ascan:smoke"
EXPECTED_VERSION="${EXPECTED_VERSION:-1.2.1}"
VM_NAME="ascan-wolfi"

[ -f "$IMAGE_TAR" ] || { echo "error: $IMAGE_TAR not found; run wolfi-build first" >&2; exit 1; }

os=$(uname -s)
case "$os" in
  Darwin) NERDCTL="limactl shell --workdir / $VM_NAME nerdctl" ; IMAGE_IN_VM="/work/out/ascan.tar" ;;
  Linux)
    if command -v nerdctl >/dev/null 2>&1; then NERDCTL="nerdctl";
    elif command -v docker >/dev/null 2>&1; then NERDCTL="docker";
    elif command -v podman >/dev/null 2>&1; then NERDCTL="podman";
    else echo "error: no container runtime" >&2; exit 1; fi
    IMAGE_IN_VM="$IMAGE_TAR"
    ;;
  *) echo "error: unsupported OS '$os'" >&2; exit 1 ;;
esac

echo "==> loading image"
loaded=$($NERDCTL load -i "$IMAGE_IN_VM" 2>&1) || { echo "$loaded" >&2; exit 1; }
loaded_ref=$(echo "$loaded" | awk '/Loaded image/ {print $NF}' | tail -1)
[ -n "$loaded_ref" ] || { echo "error: could not parse loaded image ref from: $loaded" >&2; exit 1; }
$NERDCTL tag "$loaded_ref" "$IMAGE_REF" >/dev/null

fail=0
report() {
  status=$1; shift
  if [ "$status" -eq 0 ]; then
    echo "  ok    $*"
  else
    echo "  FAIL  $*" >&2
    fail=1
  fi
}

echo "==> 1/4 ascan --version reports $EXPECTED_VERSION"
out=$($NERDCTL run --rm "$IMAGE_REF" --version 2>&1 || true)
echo "$out" | grep -q "$EXPECTED_VERSION"
report $? "saw version: $out"

echo "==> 2/4 ascan --help runs"
$NERDCTL run --rm "$IMAGE_REF" --help >/dev/null 2>&1
report $? "help exits 0"

echo "==> 3/4 scan with bundled models succeeds (any verdict, not error)"
# `|| rc=$?` so set -e doesn't abort on ascan's 1/2 verdict exits.
rc=0
$NERDCTL run --rm "$IMAGE_REF" /usr/bin/cleave >/tmp/ascan-smoke.out 2>&1 || rc=$?
# 0=clean, 1=hostile, 2=suspicious all mean the scan worked. 3+ = errors.
if [ "$rc" -ge 3 ]; then
  echo "  FAIL  scan returned error exit $rc; output: $(head -20 /tmp/ascan-smoke.out)" >&2
  fail=1
else
  report 0 "scan of /usr/bin/cleave returned verdict exit=$rc"
fi

echo "==> 4/4 scan with bogus SCAN_MODELS_DIR errors out"
out=$($NERDCTL run --rm \
  --env SCAN_MODELS_DIR=/nonexistent-models \
  "$IMAGE_REF" /usr/bin/cleave 2>&1 || true)
if echo "$out" | grep -qi 'SCAN_MODELS_DIR\|models\|nonexistent'; then
  report 0 "rejected bogus models dir"
else
  echo "  FAIL  bogus models dir did not produce a models-related error; output: $out" >&2
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "==> smoke ok"
else
  echo "==> smoke FAILED" >&2
  exit 1
fi
