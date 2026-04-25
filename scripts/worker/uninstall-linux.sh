#!/bin/sh
# uninstall-linux.sh - Stop and remove the litmus-worker systemd service.
# Also clears any legacy cron entry from the previous cron-based deploy.
set -eu

SERVICE_NAME=litmus-worker
UNIT_FILE=/etc/systemd/system/${SERVICE_NAME}.service

log() { printf '==> %s\n' "$*"; }

if command -v systemctl >/dev/null 2>&1 && [ -f "$UNIT_FILE" ]; then
    log "Stopping and disabling ${SERVICE_NAME}"
    sudo systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
    log "Removing ${UNIT_FILE}"
    sudo rm -f "$UNIT_FILE"
    sudo systemctl daemon-reload
fi

if crontab -l 2>/dev/null | grep -q "litmus worker"; then
    log "Removing legacy cron entry"
    (crontab -l 2>/dev/null | grep -v "litmus worker" || true) | crontab -
fi

log "Killing any remaining litmus worker processes"
sudo pkill -f "litmus worker" 2>/dev/null || true
pkill -u "$(id -u)" -f "litmus worker" 2>/dev/null || true

log "Uninstall complete"
log "Note: service user 'litmus' and /var/lib/litmus left intact (remove manually for a fresh state)."
