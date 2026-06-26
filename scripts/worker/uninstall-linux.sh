#!/bin/sh
# uninstall-linux.sh - Stop and remove the scan-worker systemd service.
# Also clears any legacy cron entry from the previous cron-based deploy.
set -eu

SERVICE_NAME=scan-worker
UNIT_FILE=/etc/systemd/system/${SERVICE_NAME}.service

log() { printf '==> %s\n' "$*"; }

if command -v systemctl >/dev/null 2>&1 && [ -f "$UNIT_FILE" ]; then
    log "Stopping and disabling ${SERVICE_NAME}"
    sudo systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
    log "Removing ${UNIT_FILE}"
    sudo rm -f "$UNIT_FILE"
    sudo systemctl daemon-reload
fi

LEGACY_UNIT=/etc/systemd/system/ascan-worker.service
if command -v systemctl >/dev/null 2>&1 && [ -f "$LEGACY_UNIT" ]; then
    log "Removing legacy ascan-worker service (pre-rename install)"
    sudo systemctl disable --now ascan-worker.service 2>/dev/null || true
    sudo rm -f "$LEGACY_UNIT"
    sudo systemctl daemon-reload
fi

if crontab -l 2>/dev/null | grep -q "scan worker"; then
    log "Removing legacy cron entry"
    (crontab -l 2>/dev/null | grep -v "scan worker" || true) | crontab -
fi

log "Killing any remaining scan worker processes"
sudo pkill -f "scan worker" 2>/dev/null || true
pkill -u "$(id -u)" -f "scan worker" 2>/dev/null || true

log "Uninstall complete"
log "Note: service user 'scan' and /var/lib/atomdrift/scan left intact (remove manually for a fresh state)."
