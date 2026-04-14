#!/bin/sh
# worker-openbsd.sh - Deploy litmus worker on OpenBSD
# Usage: ./worker-openbsd.sh <url>
# Runs on the local machine. Re-run to update.
# Must be invoked from the repository root.
#
# doas is required only for package installation. Add to /etc/doas.conf:
#   permit nopass <youruser> as root cmd pkg_add

set -ex

URL="$1"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

BINARY=litmus
BIN_DIR="$HOME/bin"
LOG="$HOME/.local/share/litmus/litmus-worker.log"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

log "Installing dependencies"
missing=""
for p in rust git p7zip rizin innoextract unrar; do
    pkg_info -q | grep -q "^${p}-" || missing="$missing $p"
done
[ -n "$missing" ] && doas pkg_add -I $missing || true

log "Building"
ulimit -d "$(ulimit -Hd)"
cargo build --release || die "build failed"

mkdir -p "$BIN_DIR" "$(dirname "$LOG")"

log "Installing binary"
restart_needed=0
if ! cmp -s "target/release/$BINARY" "$BIN_DIR/$BINARY" 2>/dev/null; then
    install -m 755 "target/release/$BINARY" "$BIN_DIR/$BINARY"
    restart_needed=1
fi

log "Installing cron entry"
cron_cmd="* * * * * pgrep -af 'litmus worker' >/dev/null 2>&1 || { ulimit -d \$(ulimit -Hd); nohup $BIN_DIR/$BINARY worker --url $URL < /dev/null >> $LOG 2>&1 & }"
(crontab -l 2>/dev/null | grep -v "litmus worker" || true; echo "$cron_cmd") | crontab -

if [ "$restart_needed" -eq 1 ]; then
    log "Restarting litmus worker"
    pkill -f "litmus worker" 2>/dev/null || true
    sleep 1
    nohup "$BIN_DIR/$BINARY" worker --url "$URL" < /dev/null >> "$LOG" 2>&1 &
else
    log "Binary unchanged, skipping restart"
fi

log "Deployment complete"
