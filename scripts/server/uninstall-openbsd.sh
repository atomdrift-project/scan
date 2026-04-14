#!/bin/sh
# uninstall-openbsd.sh - Remove litmus server persistence on OpenBSD
set -ex

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

log "Removing cron entry"
(crontab -l 2>/dev/null | grep -v "litmus.*serve" || true) | crontab -

log "Killing any remaining processes"
pkill -f "litmus.*serve" 2>/dev/null || true

log "Uninstall complete"
