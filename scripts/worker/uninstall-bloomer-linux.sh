#!/bin/sh
# uninstall-bloomer-linux.sh - Stop and remove the scan-bloomer timer + service.
# Leaves the `scan` user, the source checkouts, the bloom checkout, and all
# credentials under /var/lib/atomdrift in place (remove manually for a fresh state).
set -eu

SERVICE_NAME=scan-bloomer
TIMER_FILE=/etc/systemd/system/${SERVICE_NAME}.timer
SERVICE_FILE=/etc/systemd/system/${SERVICE_NAME}.service

log() { printf '==> %s\n' "$*"; }

command -v systemctl >/dev/null 2>&1 || { log "systemctl not found; nothing to do"; exit 0; }

removed=0
if [ -f "$TIMER_FILE" ]; then
    log "Stopping and disabling ${SERVICE_NAME}.timer"
    sudo systemctl disable --now "${SERVICE_NAME}.timer" 2>/dev/null || true
    log "Removing ${TIMER_FILE}"
    sudo rm -f "$TIMER_FILE"
    removed=1
fi

# Stop a cycle that happens to be mid-run, then drop the service unit.
if [ -f "$SERVICE_FILE" ]; then
    sudo systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
    log "Removing ${SERVICE_FILE}"
    sudo rm -f "$SERVICE_FILE"
    removed=1
fi

[ "$removed" -eq 1 ] && sudo systemctl daemon-reload

log "Uninstall complete"
log "Note: user 'scan' and /var/lib/atomdrift/{scan,scan-src,bloom} left intact (remove manually for a fresh state)."
