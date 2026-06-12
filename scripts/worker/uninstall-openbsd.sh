#!/bin/sh
# uninstall-openbsd.sh - Remove Atomdrift Scan worker persistence on OpenBSD
set -ex

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

log "Removing cron entry"
(crontab -l 2>/dev/null | grep -v "ascan worker" || true) | crontab -

log "Killing any remaining processes"
pkill -x ascan 2>/dev/null || true

log "Uninstall complete"
