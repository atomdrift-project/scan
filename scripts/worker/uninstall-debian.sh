#!/bin/sh
# uninstall-debian.sh - Remove Atomdrift Scan worker persistence on a Debian host via SSH
# Usage: ./uninstall-debian.sh [run-host]
set -ex

RUN="${1:-scan}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

rssh() { ssh -o BatchMode=yes "$RUN" "$@"; }

rssh true || die "run host '$RUN' not accessible"

log "Stopping and disabling scan-worker service on $RUN"
rssh "sudo systemctl stop scan-worker.service 2>/dev/null || true && \
      sudo systemctl disable scan-worker.service 2>/dev/null || true && \
      sudo rm -f /etc/systemd/system/scan-worker.service && \
      sudo systemctl daemon-reload"

log "Removing any legacy ascan-worker service (pre-rename installs) on $RUN"
rssh "sudo systemctl stop ascan-worker.service 2>/dev/null || true && \
      sudo systemctl disable ascan-worker.service 2>/dev/null || true && \
      sudo rm -f /etc/systemd/system/ascan-worker.service && \
      sudo systemctl daemon-reload"

log "Uninstall complete"
