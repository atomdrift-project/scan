#!/bin/sh
# uninstall-debian.sh - Remove Atomdrift Scan worker persistence on a Debian host via SSH
# Usage: ./uninstall-debian.sh [run-host]
set -ex

RUN="${1:-scan}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

rssh() { ssh -o BatchMode=yes "$RUN" "$@"; }

rssh true || die "run host '$RUN' not accessible"

# Privilege escalation runs on the remote host: prefer doas, fall back to sudo.
RSUDO=$(rssh 'if command -v doas >/dev/null 2>&1; then echo doas; elif command -v sudo >/dev/null 2>&1; then echo sudo; fi')
[ -n "$RSUDO" ] || die "need doas or sudo on run host '$RUN'"

log "Stopping and disabling scan-worker service on $RUN"
rssh "$RSUDO systemctl stop scan-worker.service 2>/dev/null || true && \
      $RSUDO systemctl disable scan-worker.service 2>/dev/null || true && \
      $RSUDO rm -f /etc/systemd/system/scan-worker.service && \
      $RSUDO systemctl daemon-reload"

log "Removing any legacy ascan-worker service (pre-rename installs) on $RUN"
rssh "$RSUDO systemctl stop ascan-worker.service 2>/dev/null || true && \
      $RSUDO systemctl disable ascan-worker.service 2>/dev/null || true && \
      $RSUDO rm -f /etc/systemd/system/ascan-worker.service && \
      $RSUDO systemctl daemon-reload"

log "Uninstall complete"
