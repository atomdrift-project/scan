#!/bin/sh
# uninstall-ubuntu.sh - Remove litmus server persistence on Ubuntu
set -ex

BINARY=litmus

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

log "Removing cron entry"
(crontab -l 2>/dev/null | grep -v "litmus.*serve" || true) | crontab -

log "Killing any remaining processes"
pkill -f "$BINARY.*serve" 2>/dev/null || true

log "Uninstall complete"
