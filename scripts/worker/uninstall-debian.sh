#!/bin/sh
# uninstall-debian.sh - Remove litmus worker persistence on a Debian host via SSH
# Usage: ./uninstall-debian.sh [run-host]
set -ex

RUN="${1:-litmus}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

rssh() { ssh -o BatchMode=yes "$RUN" "$@"; }

rssh true || die "run host '$RUN' not accessible"

log "Stopping and disabling litmus-worker service on $RUN"
rssh "sudo systemctl stop litmus-worker.service 2>/dev/null || true && \
      sudo systemctl disable litmus-worker.service 2>/dev/null || true && \
      sudo rm -f /etc/systemd/system/litmus-worker.service && \
      sudo systemctl daemon-reload"

log "Uninstall complete"
