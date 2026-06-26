#!/bin/sh
# Build (multi-arch) and publish the scan OCI image to a registry,
# then sign it keyless via cosign + Google OIDC.
#
# Default target: docker.io/atomdrift/scan:<version> and :latest
# Override with REGISTRY=, ORG=, ARCHS=.
#
# Prerequisites (script will check):
#   - cosign on PATH (brew install cosign)
#   - lima VM logged into the target registry — `make docker-login`
#   - Google account available for OIDC browser flow (or
#     COSIGN_IDENTITY_TOKEN set to a service-account JWT for CI)
#
# Set DRY_RUN=1 to build everything but skip the actual push + sign.

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
PKG_DIR=$(cd "$HERE/.." && pwd)
REPO_ROOT=$(cd "$PKG_DIR/../.." && pwd)
OUT_DIR="$REPO_ROOT/out/wolfi"
VM_NAME="scan-wolfi"

REGISTRY="${REGISTRY:-docker.io}"
ORG="${ORG:-atomdrift}"
ARCHS="${ARCHS:-aarch64,x86_64}"
APKO_IMAGE="${APKO_IMAGE:-cgr.dev/chainguard/apko:latest}"
OIDC_ISSUER="${OIDC_ISSUER:-https://accounts.google.com}"
DRY_RUN="${DRY_RUN:-}"

IMAGE="$REGISTRY/$ORG/scan"

# Read the version from the upstream-shape yaml; that's the source of truth.
VERSION=$(awk '/^  version:/ { gsub(/[" ]/, "", $2); print $2; exit }' "$PKG_DIR/melange.yaml")
[ -n "$VERSION" ] || { echo "error: could not parse version from $PKG_DIR/melange.yaml" >&2; exit 1; }

# Prereqs.
command -v cosign >/dev/null 2>&1 || {
  echo "error: cosign not found (install: brew install cosign)" >&2
  exit 1
}
case "$(uname -s)" in
  Darwin)
    command -v limactl >/dev/null 2>&1 || {
      echo "error: limactl not found (install: brew install lima)" >&2; exit 1; }
    limactl list --format '{{.Status}}' "$VM_NAME" 2>/dev/null | grep -qx Running || {
      echo "error: lima VM '$VM_NAME' is not running — run 'make wolfi-bootstrap' first" >&2
      exit 1
    }
    NERDCTL="limactl shell --workdir / $VM_NAME nerdctl"
    SRC_IN_RT="/work/out/source"
    OUT_IN_RT="/work/out"
    CACHE_IN_RT="/work/cache"
    ;;
  Linux)
    if command -v nerdctl >/dev/null 2>&1; then NERDCTL="nerdctl";
    elif command -v docker >/dev/null 2>&1; then NERDCTL="docker";
    elif command -v podman >/dev/null 2>&1; then NERDCTL="podman";
    else echo "error: no container runtime" >&2; exit 1; fi
    SRC_IN_RT="$OUT_DIR/source"
    OUT_IN_RT="$OUT_DIR"
    CACHE_IN_RT="${XDG_CACHE_HOME:-$HOME/.cache}/scan-wolfi"
    ;;
  *) echo "error: unsupported OS" >&2; exit 1 ;;
esac

# Step 1: ensure apks for all target arches exist. build.sh's per-component
# stamps skip work whose inputs are unchanged, so on a clean repo this is
# typically a no-op past the first run.
echo "==> ensuring apks for ARCHS=$ARCHS"
WOLFI_ARCH="$ARCHS" "$HERE/build.sh"

# Step 2: publish a multi-arch OCI image to the registry.
# apko publish builds + pushes in one step; the result is a manifest list
# (one platform manifest per arch) tagged at $IMAGE:$VERSION + :latest.
TAGS="$IMAGE:$VERSION $IMAGE:latest"

if [ -n "$DRY_RUN" ]; then
  echo "==> DRY_RUN=1: would run apko publish for $TAGS (--arch $ARCHS)"
else
  echo "==> publishing $TAGS (--arch $ARCHS) — if push fails, run: make docker-login REGISTRY=$REGISTRY"
  # shellcheck disable=SC2086  # $TAGS expansion is intentional (space-separated positional args)
  $NERDCTL run --rm \
    -v "$SRC_IN_RT:/src:ro" \
    -v "$OUT_IN_RT:/out" \
    -v "$CACHE_IN_RT:/cache" \
    -w /out \
    "$APKO_IMAGE" publish \
      /src/scan/packaging/wolfi/apko.yaml \
      $TAGS \
      --arch "$ARCHS" \
      --keyring-append /out/melange.rsa.pub \
      --cache-dir /cache/apko
fi

# Step 3: sign each tag keyless via Google OIDC. cosign opens a browser
# for the OIDC flow unless COSIGN_IDENTITY_TOKEN is set (CI path).
# The signature is stored alongside the image in the registry; the
# transparency record goes to Rekor (public-good instance).
COSIGN_FLAGS="--yes --oidc-issuer=$OIDC_ISSUER"

for tag in $TAGS; do
  if [ -n "$DRY_RUN" ]; then
    echo "==> DRY_RUN=1: would run cosign sign $COSIGN_FLAGS $tag"
  else
    echo "==> signing $tag"
    # shellcheck disable=SC2086  # COSIGN_FLAGS is intentionally word-split
    cosign sign $COSIGN_FLAGS "$tag"
  fi
done

echo ""
echo "==> done"
echo "    image:     $IMAGE:$VERSION"
echo "    also:      $IMAGE:latest"
echo "    arches:    $ARCHS"
echo "    signature: cosign + $OIDC_ISSUER (Rekor)"
echo ""
echo "verify with:"
echo "  cosign verify \\"
echo "    --certificate-identity-regexp '.*' \\"
echo "    --certificate-oidc-issuer $OIDC_ISSUER \\"
echo "    $IMAGE:$VERSION"
