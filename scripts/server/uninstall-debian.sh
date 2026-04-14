#!/bin/sh
# uninstall-debian.sh - Remove litmus server persistence on a Debian host via SSH
# Usage: ./uninstall-debian.sh [run-host]
set -ex

RUN="${1:-litmus}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

rssh() { ssh -o BatchMode=yes "$RUN" "$@"; }

rssh true || die "run host '$RUN' not accessible"

log "Stopping and disabling litmus service on $RUN"
rssh "sudo systemctl stop litmus.service 2>/dev/null || true && \
      sudo systemctl disable litmus.service 2>/dev/null || true && \
      sudo rm -f /etc/systemd/system/litmus.service && \
      sudo systemctl daemon-reload"

log "Uninstall complete"
