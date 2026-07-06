#!/bin/sh
# uninstall-bloomer-linux.sh - Stop and remove the scan-bloomer timer + service.
# Leaves the `bloom` user, the source checkouts, the bloom checkout, and all
# credentials under /var/lib/bloom in place (remove manually for a fresh state).
set -eu

SERVICE_NAME=scan-bloomer
TIMER_FILE=/etc/systemd/system/${SERVICE_NAME}.timer
SERVICE_FILE=/etc/systemd/system/${SERVICE_NAME}.service

log() { printf '==> %s\n' "$*"; }

command -v systemctl >/dev/null 2>&1 || { log "systemctl not found; nothing to do"; exit 0; }

# Privilege escalation: prefer doas, fall back to sudo.
if   command -v doas >/dev/null 2>&1; then SUDO=doas
elif command -v sudo >/dev/null 2>&1; then SUDO=sudo
else echo "error: need doas or sudo" >&2; exit 1
fi

removed=0
if [ -f "$TIMER_FILE" ]; then
    log "Stopping and disabling ${SERVICE_NAME}.timer"
    $SUDO systemctl disable --now "${SERVICE_NAME}.timer" 2>/dev/null || true
    log "Removing ${TIMER_FILE}"
    $SUDO rm -f "$TIMER_FILE"
    removed=1
fi

# Stop a cycle that happens to be mid-run, then drop the service unit.
if [ -f "$SERVICE_FILE" ]; then
    $SUDO systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
    log "Removing ${SERVICE_FILE}"
    $SUDO rm -f "$SERVICE_FILE"
    removed=1
fi

[ "$removed" -eq 1 ] && $SUDO systemctl daemon-reload

log "Uninstall complete"
log "Note: user 'bloom' and /var/lib/bloom left intact (remove manually for a fresh state)."
