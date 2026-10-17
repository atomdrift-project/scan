#!/bin/sh
# uninstall-linux.sh - Stop and remove the scan-worker systemd service.
# Also clears any legacy cron entry from the previous cron-based deploy.
set -eu

SERVICE_NAME=scan-worker
UNIT_FILE=/etc/systemd/system/${SERVICE_NAME}.service

log() { printf '==> %s\n' "$*"; }

# Privilege escalation: prefer doas, fall back to sudo.
if   command -v doas >/dev/null 2>&1; then SUDO=doas
elif command -v sudo >/dev/null 2>&1; then SUDO=sudo
else echo "error: need doas or sudo" >&2; exit 1
fi

if command -v systemctl >/dev/null 2>&1 && [ -f "$UNIT_FILE" ]; then
    log "Stopping and disabling ${SERVICE_NAME}"
    $SUDO systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
    log "Removing ${UNIT_FILE}"
    $SUDO rm -f "$UNIT_FILE"
    $SUDO systemctl daemon-reload
fi

LEGACY_UNIT=/etc/systemd/system/ascan-worker.service
if command -v systemctl >/dev/null 2>&1 && [ -f "$LEGACY_UNIT" ]; then
    log "Removing legacy ascan-worker service (pre-rename install)"
    $SUDO systemctl disable --now ascan-worker.service 2>/dev/null || true
    $SUDO rm -f "$LEGACY_UNIT"
    $SUDO systemctl daemon-reload
fi

if crontab -l 2>/dev/null | grep -q "scan worker"; then
    log "Removing legacy cron entry"
    (crontab -l 2>/dev/null | grep -v "scan worker" || true) | crontab -
fi

log "Killing any remaining atomscan worker processes"
$SUDO pkill -f "atomscan worker" 2>/dev/null || true
pkill -u "$(id -u)" -f "atomscan worker" 2>/dev/null || true

log "Uninstall complete"
log "Note: service user 'scan' and /var/lib/atomdrift/scan left intact (remove manually for a fresh state)."
