#!/bin/sh
# uninstall-bastille.sh - Remove litmus worker persistence from a Bastille jail
# Usage: ./uninstall-bastille.sh [run-jail]
set -ex

RUN="${1:-litmus}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

doas bastille cmd "$RUN" true || die "run jail '$RUN' not accessible"

log "Disabling and stopping litmus-worker service"
doas bastille sysrc "$RUN" litmus_worker_enable=NO || true
doas bastille service "$RUN" litmus-worker stop 2>/dev/null || true
doas bastille cmd "$RUN" pkill -9 -x litmus 2>/dev/null || true

log "Removing rc.d script"
doas bastille cmd "$RUN" rm -f /usr/local/etc/rc.d/litmus-worker

log "Uninstall complete"
