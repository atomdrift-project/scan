#!/bin/sh
# uninstall-alpine.sh - Remove Atomdrift Scan worker persistence on Alpine Linux
set -ex

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

log "Removing cron entry"
(crontab -l 2>/dev/null | grep -v "atomscan worker" || true) | crontab -

log "Killing any remaining processes"
pkill -x scan 2>/dev/null || true
pkill -x ascan 2>/dev/null || true  # legacy pre-rename binary

log "Uninstall complete"
