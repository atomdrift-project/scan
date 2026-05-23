#!/bin/sh
# Build litmus + litmus-models apks via melange, then assemble an OCI
# image via apko. Idempotent: skips work when source + yaml hashes match
# the on-disk stamp.
#
# Local layout in $STAGE_DIR:
#   cleave/      — cleave repo (with _filefacts vendored, Cargo.toml patched);
#                  same staging recipe as cleave/packaging/wolfi/scripts/build.sh
#   litmus/      — litmus repo, Cargo.toml extended with a [patch] block that
#                  overrides the upstream cleave git dep with `../cleave`.
#
# melange's --source-dir is $STAGE_DIR, so /home/build inside the chroot
# has both subdirs; the build pipeline does `cd litmus && cargo build`.
# The upstream-shape melange.yaml has the same `cargo build` line without
# the cd; build.sh substitutes a `cd litmus` over a marker comment.

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
PKG_DIR=$(cd "$HERE/.." && pwd)
REPO_ROOT=$(cd "$PKG_DIR/../.." && pwd)
SIBLING_DIR=$(cd "$REPO_ROOT/.." && pwd)
CLEAVE_REPO="$SIBLING_DIR/cleave"
FILEFACTS_REPO="$SIBLING_DIR/filefacts"
AZOTH_REPO="$SIBLING_DIR/azoth"
OUT_DIR="$REPO_ROOT/out/wolfi"
STAGE_DIR="$OUT_DIR/source"
STAMP="$OUT_DIR/.build.stamp"
VM_NAME="litmus-wolfi"
APKO_IMAGE="${APKO_IMAGE:-cgr.dev/chainguard/apko:latest}"
MELANGE_IMAGE="${MELANGE_IMAGE:-cgr.dev/chainguard/melange:latest}"

[ -d "$CLEAVE_REPO" ]    || { echo "error: missing $CLEAVE_REPO" >&2; exit 1; }
[ -d "$FILEFACTS_REPO" ] || { echo "error: missing $FILEFACTS_REPO" >&2; exit 1; }
[ -d "$AZOTH_REPO" ]     || { echo "error: missing $AZOTH_REPO (needed for litmus-models subpackage)" >&2; exit 1; }

# Default to multi-arch (aarch64,x86_64) so the apks we produce match
# what the publish step needs. For fast single-arch local iteration,
# set WOLFI_ARCH=$(uname -m).
WOLFI_ARCH="${WOLFI_ARCH:-aarch64,x86_64}"
# Pick a host-compatible arch for the local apko smoke-test image.
host_arch=$(uname -m)
case "$host_arch" in
  arm64|aarch64) HOST_WOLFI_ARCH=aarch64 ;;
  x86_64|amd64)  HOST_WOLFI_ARCH=x86_64 ;;
  *) echo "error: unsupported host arch '$host_arch'" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR/packages" "$STAGE_DIR"

# Per-component hashes. Each melange step has its own stamp so we
# only re-run the work whose inputs actually changed:
#   cleave_hash:  cleave src + filefacts src + cleave's melange.yaml.
#                 Invalidates the cleave + cleave-traits apks.
#   litmus_hash:  litmus src + azoth + litmus's melange.yaml + cleave_hash.
#                 Invalidates the litmus + litmus-models apks.
#   image_hash:   everything above + apko.yaml + arch.
#                 Invalidates the OCI tarball.
hash_tree() {
  # shellcheck disable=SC2068  # find prunes intentionally split
  ( cd "$1" && shift && find $@ -type f \
        -not -path '*/target/*' -not -path '*/.git/*' -not -name '*.swp' -print0 2>/dev/null \
      | sort -z | xargs -0 shasum -a 256 2>/dev/null )
}

cleave_hash=$( {
    hash_tree "$CLEAVE_REPO"    src crates cleave-macros Cargo.toml Cargo.lock
    hash_tree "$FILEFACTS_REPO" .
    shasum -a 256 "$CLEAVE_REPO/packaging/wolfi/melange.yaml" 2>/dev/null
    echo "arch=$WOLFI_ARCH"
  } | shasum -a 256 | awk '{print $1}')

litmus_hash=$( {
    echo "cleave=$cleave_hash"
    hash_tree "$REPO_ROOT"  src Cargo.toml Cargo.lock
    hash_tree "$AZOTH_REPO" .
    shasum -a 256 "$PKG_DIR/melange.yaml" 2>/dev/null
    echo "arch=$WOLFI_ARCH"
  } | shasum -a 256 | awk '{print $1}')

image_hash=$( {
    echo "litmus=$litmus_hash"
    shasum -a 256 "$PKG_DIR/apko.yaml" 2>/dev/null
  } | shasum -a 256 | awk '{print $1}')

CLEAVE_STAMP="$OUT_DIR/.cleave.stamp"
LITMUS_STAMP="$OUT_DIR/.litmus.stamp"

if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$image_hash" ] && [ -f "$OUT_DIR/litmus.tar" ]; then
  echo "==> up to date (image hash $image_hash); skipping build"
  echo "    force rebuild: rm $STAMP"
  exit 0
fi

# Generate the local-build variant of melange.yaml:
#   1. Strip the upstream git-checkout step.
#   2. Substitute `cd litmus` for the # LOCAL_BUILD_CD_HERE marker.
#   3. Drop --locked from cargo (the appended [patch] block makes the
#      lock want to refresh; --locked forbids it).
LOCAL_YAML="$OUT_DIR/melange.local.yaml"
awk '
  /# LOCAL_BUILD_STRIP_BEGIN/ { skip = 1; next }
  /# LOCAL_BUILD_STRIP_END/   { skip = 0; next }
  skip { next }
  /# LOCAL_BUILD_CD_HERE/    { sub(/# LOCAL_BUILD_CD_HERE.*/,    "cd litmus");                       print; next }
  /# LOCAL_BUILD_AZOTH_HERE/ { sub(/# LOCAL_BUILD_AZOTH_HERE.*/, "cp -a /home/build/azoth/. models-src/"); print; next }
  { print }
' "$PKG_DIR/melange.yaml" \
  | sed 's/ --locked//g' \
  > "$LOCAL_YAML.new" && mv -f "$LOCAL_YAML.new" "$LOCAL_YAML"

# Same strip + --locked-drop treatment for cleave's melange.yaml — we
# invoke it below to build the cleave + cleave-traits apks. Without
# stripping git-checkout, melange would overlay the upstream cleave tag
# on top of our staged tree, producing a mixed-version cleave that
# refuses to compile (E0761 on pdf.rs vs pdf/mod.rs etc.).
CLEAVE_LOCAL_YAML="$OUT_DIR/cleave-melange.local.yaml"
awk '
  /# LOCAL_BUILD_STRIP_BEGIN/ { skip = 1; next }
  /# LOCAL_BUILD_STRIP_END/   { skip = 0; next }
  !skip
' "$CLEAVE_REPO/packaging/wolfi/melange.yaml" \
  | sed 's/ --locked//g' \
  > "$CLEAVE_LOCAL_YAML.new" && mv -f "$CLEAVE_LOCAL_YAML.new" "$CLEAVE_LOCAL_YAML"

# Stage cleave + filefacts + litmus into a clean tree. Same recipe as
# cleave's packaging for the cleave/_filefacts vendoring; litmus gets a
# `[patch]` block appended that points at the staged cleave.
echo "==> staging source into $STAGE_DIR"
command -v rsync >/dev/null 2>&1 || { echo "error: rsync required" >&2; exit 1; }
EXCLUDES="--exclude=target/ --exclude=.git/ --exclude=out/ --exclude=node_modules/ --exclude=.DS_Store"

# shellcheck disable=SC2086
rsync -a --delete $EXCLUDES "$CLEAVE_REPO/"    "$STAGE_DIR/cleave/"
# shellcheck disable=SC2086
rsync -a --delete $EXCLUDES "$FILEFACTS_REPO/" "$STAGE_DIR/cleave/_filefacts/"
# shellcheck disable=SC2086
rsync -a --delete $EXCLUDES "$REPO_ROOT/"      "$STAGE_DIR/litmus/"
# shellcheck disable=SC2086
rsync -a --delete $EXCLUDES "$AZOTH_REPO/"     "$STAGE_DIR/azoth/"

# Rewrite cleave's Cargo.toml (filefacts → _filefacts; exclude from workspace).
awk '
  /^members = \[/ && !ws_excluded {
    print
    print "exclude = [\"_filefacts\"]"
    ws_excluded = 1
    next
  }
  /^filefacts = \{ path = "\.\.\/filefacts" \}/ {
    print "filefacts = { path = \"_filefacts\" }"
    next
  }
  { print }
' "$STAGE_DIR/cleave/Cargo.toml" > "$STAGE_DIR/cleave/Cargo.toml.new"
mv -f "$STAGE_DIR/cleave/Cargo.toml.new" "$STAGE_DIR/cleave/Cargo.toml"
grep -q '^filefacts = { path = "_filefacts" }$' "$STAGE_DIR/cleave/Cargo.toml" \
  || { echo "error: cleave Cargo.toml filefacts rewrite failed" >&2; exit 1; }

# Append [patch] to litmus's Cargo.toml so cargo uses the staged cleave
# instead of fetching from codeberg. Idempotent: only append if missing.
if ! grep -q '^\[patch\."https://codeberg.org/atomdrift/cleave.git"\]' "$STAGE_DIR/litmus/Cargo.toml"; then
  cat >> "$STAGE_DIR/litmus/Cargo.toml" <<'PATCH'

# Appended by packaging/wolfi/scripts/build.sh — points cargo at the
# locally-staged cleave tree (which has _filefacts vendored).
[patch."https://codeberg.org/atomdrift/cleave.git"]
cleave = { path = "../cleave" }
PATCH
fi

os=$(uname -s)
case "$os" in
  Darwin)
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
    SRC_IN_RT="$STAGE_DIR"
    OUT_IN_RT="$OUT_DIR"
    CACHE_IN_RT="${XDG_CACHE_HOME:-$HOME/.cache}/litmus-wolfi"
    mkdir -p "$CACHE_IN_RT"
    ;;
  *) echo "error: unsupported OS '$os'" >&2; exit 1 ;;
esac

if [ ! -f "$OUT_DIR/melange.rsa" ]; then
  echo "==> generating local melange signing key"
  $NERDCTL run --rm -v "$OUT_IN_RT:/out" -w /out \
    "$MELANGE_IMAGE" keygen melange.rsa
fi

# Skip the cleave melange step if our cleave_hash matches the stamp AND
# both apks are still on disk. apks names embed the version, so a
# version bump invalidates via the hash (the yaml change).
have_cleave_apks() {
  ls "$OUT_DIR/packages/$WOLFI_ARCH/cleave-"*.apk        >/dev/null 2>&1 && \
  ls "$OUT_DIR/packages/$WOLFI_ARCH/cleave-traits-"*.apk >/dev/null 2>&1
}
if [ -f "$CLEAVE_STAMP" ] && [ "$(cat "$CLEAVE_STAMP")" = "$cleave_hash" ] && have_cleave_apks; then
  echo "==> cleave apks up to date (hash $cleave_hash); skipping cleave melange"
else
  echo "==> building cleave + cleave-traits ($WOLFI_ARCH)"
  $NERDCTL run --rm \
    -v "$SRC_IN_RT:/src:ro" \
    -v "$OUT_IN_RT:/out" \
    -v "$CACHE_IN_RT:/cache" \
    -w /out \
    --privileged \
    "$MELANGE_IMAGE" build \
      /out/cleave-melange.local.yaml \
      --arch "$WOLFI_ARCH" \
      --source-dir /src/cleave \
      --cache-dir /cache/melange \
      --signing-key /out/melange.rsa \
      --out-dir /out/packages
  echo "$cleave_hash" > "$CLEAVE_STAMP.new" && mv -f "$CLEAVE_STAMP.new" "$CLEAVE_STAMP"
fi

have_litmus_apks() {
  ls "$OUT_DIR/packages/$WOLFI_ARCH/litmus-"[0-9]*.apk   >/dev/null 2>&1 && \
  ls "$OUT_DIR/packages/$WOLFI_ARCH/litmus-models-"*.apk >/dev/null 2>&1
}
if [ -f "$LITMUS_STAMP" ] && [ "$(cat "$LITMUS_STAMP")" = "$litmus_hash" ] && have_litmus_apks; then
  echo "==> litmus apks up to date (hash $litmus_hash); skipping litmus melange"
else
  echo "==> building litmus + litmus-models ($WOLFI_ARCH)"
  $NERDCTL run --rm \
    -v "$SRC_IN_RT:/src:ro" \
    -v "$OUT_IN_RT:/out" \
    -v "$CACHE_IN_RT:/cache" \
    -w /out \
    --privileged \
    "$MELANGE_IMAGE" build \
      /out/melange.local.yaml \
      --arch "$WOLFI_ARCH" \
      --source-dir /src \
      --cache-dir /cache/melange \
      --signing-key /out/melange.rsa \
      --out-dir /out/packages
  echo "$litmus_hash" > "$LITMUS_STAMP.new" && mv -f "$LITMUS_STAMP.new" "$LITMUS_STAMP"
fi

cp -f "$OUT_DIR/melange.rsa.pub" "$OUT_DIR/packages/melange.rsa.pub"

echo "==> assembling OCI image with apko (local smoke-test image, $HOST_WOLFI_ARCH only)"
# The local litmus.tar is used by smoke-test.sh which loads it into the
# host container runtime — only the host arch matters here. Multi-arch
# publishing happens via scripts/publish.sh.
TAR_TMP="$OUT_DIR/litmus.tar.new"
rm -f "$TAR_TMP"
$NERDCTL run --rm \
  -v "$SRC_IN_RT:/src:ro" \
  -v "$OUT_IN_RT:/out" \
  -v "$CACHE_IN_RT:/cache" \
  -w /out \
  "$APKO_IMAGE" build \
    /src/litmus/packaging/wolfi/apko.yaml \
    litmus:latest \
    /out/litmus.tar.new \
    --arch "$HOST_WOLFI_ARCH" \
    --keyring-append /out/melange.rsa.pub \
    --cache-dir /cache/apko

mv -f "$TAR_TMP" "$OUT_DIR/litmus.tar"
echo "$image_hash" > "$STAMP.new" && mv -f "$STAMP.new" "$STAMP"

echo "==> done: $OUT_DIR/litmus.tar"
