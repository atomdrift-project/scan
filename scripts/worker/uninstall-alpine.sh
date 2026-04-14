#!/bin/sh
# uninstall-alpine.sh - Remove litmus worker persistence on Alpine Linux
set -ex

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

log "Removing cron entry"
(crontab -l 2>/dev/null | grep -v "litmus worker" || true) | crontab -

log "Killing any remaining processes"
pkill -x litmus 2>/dev/null || true

log "Uninstall complete"
