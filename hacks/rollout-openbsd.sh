#!/bin/sh
# rollout-openbsd.sh - Deploy litmus on OpenBSD
# Runs on the local machine. Re-run to update.
# Must be invoked from the repository root.
#
# doas is required only for package installation. Add to /etc/doas.conf:
#   permit nopass <youruser> as root cmd pkg_add
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

log "Installing dependencies"
doas pkg_add -I rust git p7zip rizin innoextract

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
cron_cmd="* * * * * pgrep -x $BINARY >/dev/null 2>&1 || nohup $BIN_DIR/$BINARY -u serve --bind $BIND --allow-cidr $ALLOW_CIDR >> $LOG 2>&1 &"
(crontab -l 2>/dev/null | grep -v "litmus.*serve"; echo "$cron_cmd") | crontab -

if [ "$restart_needed" -eq 1 ]; then
    log "Restarting litmus"
    pkill -x "$BINARY" 2>/dev/null || true
    sleep 1
    nohup "$BIN_DIR/$BINARY" -u serve --bind "$BIND" --allow-cidr "$ALLOW_CIDR" >> "$LOG" 2>&1 &
else
    log "Binary unchanged, skipping restart"
fi

log "Deployment complete"
