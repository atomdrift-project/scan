#!/bin/sh
# uninstall-bastille.sh - Remove Atomdrift Scan worker persistence from a Bastille jail
# Usage: ./uninstall-bastille.sh [run-jail]
set -ex

RUN="${1:-scan}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

doas bastille cmd "$RUN" true || die "run jail '$RUN' not accessible"

log "Disabling and stopping scan-worker service"
doas bastille sysrc "$RUN" scan_worker_enable=NO || true
doas bastille service "$RUN" scan-worker stop 2>/dev/null || true
doas bastille cmd "$RUN" pkill -9 -F /var/run/scan_worker.pid 2>/dev/null || true

log "Removing rc.d script"
doas bastille cmd "$RUN" rm -f /usr/local/etc/rc.d/scan-worker

log "Removing any legacy ascan-worker service (pre-rename install)"
doas bastille sysrc "$RUN" ascan_worker_enable=NO || true
doas bastille service "$RUN" ascan-worker stop 2>/dev/null || true
doas bastille cmd "$RUN" pkill -9 -F /var/run/ascan_worker.pid 2>/dev/null || true
doas bastille cmd "$RUN" rm -f /usr/local/etc/rc.d/ascan-worker

log "Uninstall complete"
