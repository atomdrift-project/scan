#!/bin/sh
# uninstall-bastille.sh - Remove Atomdrift Scan server persistence from a Bastille jail
# Usage: ./uninstall-bastille.sh [run-jail]
set -ex

RUN="${1:-ascan}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

doas bastille cmd "$RUN" true || die "run jail '$RUN' not accessible"

log "Disabling and stopping ascan service"
doas bastille sysrc "$RUN" ascan_enable=NO || true
doas bastille service "$RUN" ascan stop 2>/dev/null || true
doas bastille cmd "$RUN" pkill -9 -F /var/run/ascan.pid 2>/dev/null || true

log "Removing rc.d script"
doas bastille cmd "$RUN" rm -f /usr/local/etc/rc.d/ascan

log "Uninstall complete"
