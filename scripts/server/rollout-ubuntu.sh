#!/bin/sh
# rollout-ubuntu.sh - Deploy litmus on Ubuntu
# Runs on the local machine. Re-run to update.
# Must be invoked from the repository root.
#
# sudo is required only for package installation.
#
# The service runs as the current user, kept alive by a cron watchdog (restarts within 1 min of a crash).
# Models and traits are cloned automatically by litmus on first start.

set -ex

BINARY=litmus
BIN_DIR="$HOME/bin"
LOG="$HOME/.local/share/litmus/litmus.log"
BIND="${BIND:-0.0.0.0:49999}"
ALLOW_CIDR="${ALLOW_CIDR:-10.0.0.0/8}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

[ -f Makefile ] || die "run this script from the repository root"
[ -r /etc/os-release ] || die "/etc/os-release not found"
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "this rollout is for Ubuntu"

command -v rizin >/dev/null 2>&1 || die "rizin not found — build and install it from https://rizin.re before running this script"

pkgs_needed=""
for pkg in git pkg-config build-essential clang lld ca-certificates \
           p7zip-full upx-ucl innoextract unrar cron; do
    dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed" \
        || pkgs_needed="$pkgs_needed $pkg"
done

if [ -n "$pkgs_needed" ]; then
    command -v sudo >/dev/null 2>&1 || die "sudo not found"
    log "Installing missing packages:$pkgs_needed"
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    # shellcheck disable=SC2086
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $pkgs_needed
else
    log "All packages already installed"
fi

log "Ensuring Rust toolchain"
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path \
        || die "rustup install failed"
fi
. "$HOME/.cargo/env"

log "Building"
if command -v sccache >/dev/null 2>&1; then
    RUSTC_WRAPPER=sccache RUSTFLAGS="-C link-arg=-fuse-ld=lld" cargo build --release || die "build failed"
else
    RUSTFLAGS="-C link-arg=-fuse-ld=lld" cargo build --release || die "build failed"
fi

mkdir -p "$BIN_DIR" "$(dirname "$LOG")"

log "Installing binary"
restart_needed=0
if ! cmp -s "target/release/$BINARY" "$BIN_DIR/$BINARY" 2>/dev/null; then
    install -m 755 "target/release/$BINARY" "$BIN_DIR/$BINARY"
    restart_needed=1
fi

log "Installing cron entry"
cron_cmd="* * * * * pgrep -x $BINARY >/dev/null 2>&1 || { nohup $BIN_DIR/$BINARY -u serve --bind $BIND --allow-cidr $ALLOW_CIDR < /dev/null >> $LOG 2>&1 & }"
if ! crontab -l 2>/dev/null | grep -qF "$cron_cmd"; then
    (crontab -l 2>/dev/null | grep -v "litmus.*serve" || true; echo "$cron_cmd") | crontab -
fi

if [ "$restart_needed" -eq 1 ]; then
    log "Restarting litmus"
    pkill -x "$BINARY" 2>/dev/null || true
    sleep 1
    nohup "$BIN_DIR/$BINARY" -u serve --bind "$BIND" --allow-cidr "$ALLOW_CIDR" < /dev/null >> "$LOG" 2>&1 &
else
    log "Binary unchanged, skipping restart"
fi

log "Deployment complete"
