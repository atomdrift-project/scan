#!/bin/sh
# worker-alpine.sh - Deploy litmus worker on Alpine Linux
# Usage: ./worker-alpine.sh <url>
# Runs on the local machine. Re-run to update.
# Must be invoked from the repository root.
#
# doas is required only for package management. Add to /etc/doas.conf:
#   permit nopass <youruser> as root cmd apk
#   permit nopass <youruser> as root cmd tee

set -ex

URL="$1"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

# Optional: cap concurrent analysis slots (--workers). Unset = worker auto.
WORKERS="${WORKERS:-}"
worker_args="worker --url $URL"
[ -n "$WORKERS" ] && worker_args="$worker_args --workers $WORKERS"

BINARY=litmus
BIN_DIR="$HOME/bin"
LOG="$HOME/.local/share/litmus/litmus-worker.log"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

log "Enabling testing repository"
grep -q 'edge/testing' /etc/apk/repositories 2>/dev/null || \
    echo 'https://dl-cdn.alpinelinux.org/alpine/edge/testing' | doas tee -a /etc/apk/repositories

log "Installing dependencies"
pkgs_needed=""
for pkg in rustup git 7zip upx rizin innoextract gcc g++ musl-dev; do
    apk info -e "$pkg" >/dev/null 2>&1 || pkgs_needed="$pkgs_needed $pkg"
done
if [ -n "$pkgs_needed" ]; then
    # shellcheck disable=SC2086
    doas apk add --no-cache $pkgs_needed
else
    log "All packages already installed"
fi

log "Updating Rust toolchain"
if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
    /usr/bin/rustup-init -y --no-modify-path || die "rustup-init failed"
fi
"$HOME/.cargo/bin/rustup" update stable || die "rustup update failed"
. "$HOME/.cargo/env"

log "Building"
cargo build --release || die "build failed"

mkdir -p "$BIN_DIR" "$(dirname "$LOG")"

log "Installing binary"
restart_needed=0
if ! cmp -s "target/release/$BINARY" "$BIN_DIR/$BINARY" 2>/dev/null; then
    install -m 755 "target/release/$BINARY" "$BIN_DIR/$BINARY"
    restart_needed=1
fi

log "Installing cron entry"
cron_cmd="* * * * * pgrep -af 'litmus worker' >/dev/null 2>&1 || nohup $BIN_DIR/$BINARY $worker_args < /dev/null >> $LOG 2>&1 &"
(crontab -l 2>/dev/null | grep -v "litmus worker" || true; echo "$cron_cmd") | crontab -

if [ "$restart_needed" -eq 1 ]; then
    log "Restarting litmus worker"
    pkill -f "litmus worker" 2>/dev/null || true
    sleep 1
    nohup "$BIN_DIR/$BINARY" $worker_args < /dev/null >> "$LOG" 2>&1 &
else
    log "Binary unchanged, skipping restart"
fi

log "Deployment complete"
