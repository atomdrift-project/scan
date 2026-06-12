#!/bin/sh
# Idempotently ensure the ascan-wolfi Lima VM is up and ready to build.
# Mirrors cleave/packaging/wolfi/scripts/bootstrap-lima.sh but with
# different mount paths and a smaller VM (no fat LTO in scan).

set -eu

VM_NAME="ascan-wolfi"
HERE=$(cd "$(dirname "$0")" && pwd)
PKG_DIR=$(cd "$HERE/.." && pwd)
REPO_ROOT=$(cd "$PKG_DIR/../.." && pwd)
# Source mount = PARENT of scan, so cleave (sibling) is also visible.
SRC_DIR=$(cd "$REPO_ROOT/.." && pwd)
OUT_DIR="$REPO_ROOT/out/wolfi"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/ascan-wolfi"
LIMA_TEMPLATE="$PKG_DIR/lima.yaml"
APKO_IMAGE="${APKO_IMAGE:-cgr.dev/chainguard/apko:latest}"
MELANGE_IMAGE="${MELANGE_IMAGE:-cgr.dev/chainguard/melange:latest}"

mkdir -p "$OUT_DIR" "$CACHE_DIR"

# Sibling cleave repo is required for the [patch] override and for
# bundling the cleave apk into ascan's image.
[ -d "$SRC_DIR/cleave" ] || {
  echo "error: sibling 'cleave' repo not found at $SRC_DIR/cleave" >&2
  echo "       ascan's wolfi build needs cleave checked out alongside it." >&2
  exit 1
}

os=$(uname -s)
case "$os" in
  Darwin)
    command -v limactl >/dev/null 2>&1 || {
      echo "error: limactl not found. Install: brew install lima" >&2
      exit 1
    }

    if ! limactl list --format '{{.Name}}' 2>/dev/null | grep -qx "$VM_NAME"; then
      echo "==> creating Lima VM '$VM_NAME' (this takes a few minutes the first time)"
      rendered=$(mktemp -t ascan-wolfi-lima.XXXXXX)
      trap 'rm -f "$rendered"' EXIT
      sed \
        -e "s|__SRC_DIR__|$SRC_DIR|g" \
        -e "s|__OUT_DIR__|$OUT_DIR|g" \
        -e "s|__CACHE_DIR__|$CACHE_DIR|g" \
        "$LIMA_TEMPLATE" > "$rendered"
      limactl create --name "$VM_NAME" --tty=false "$rendered"
    fi

    status=$(limactl list --format '{{.Status}}' "$VM_NAME" 2>/dev/null || echo "Unknown")
    if [ "$status" != "Running" ]; then
      echo "==> starting VM '$VM_NAME'"
      limactl start "$VM_NAME"
    fi

    NERDCTL="limactl shell --workdir / $VM_NAME nerdctl"
    ;;
  Linux)
    if command -v nerdctl >/dev/null 2>&1; then
      NERDCTL="nerdctl"
    elif command -v docker >/dev/null 2>&1; then
      NERDCTL="docker"
    elif command -v podman >/dev/null 2>&1; then
      NERDCTL="podman"
    else
      echo "error: need nerdctl, docker, or podman in PATH" >&2
      exit 1
    fi
    ;;
  *)
    echo "error: unsupported OS '$os' (need Darwin or Linux)" >&2
    exit 1
    ;;
esac

for img in "$MELANGE_IMAGE" "$APKO_IMAGE"; do
  if ! $NERDCTL image inspect "$img" >/dev/null 2>&1; then
    echo "==> pulling $img"
    $NERDCTL pull "$img"
  fi
done

echo "==> bootstrap ok"
