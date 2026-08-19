#!/bin/sh
# uninstall-linux.sh - Stop and remove the scan systemd service (atomscan serve).
set -eu

SERVICE_NAME=scan
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

log "Killing any remaining atomscan serve processes"
# ExecStart is `atomscan -u serve ...`; also catch a bare `atomscan serve`.
$SUDO pkill -f "atomscan -u serve" 2>/dev/null || true
$SUDO pkill -f "atomscan serve" 2>/dev/null || true

log "Uninstall complete"
log "Note: service user 'scan' and /var/lib/atomdrift/scan left intact (remove manually for a fresh state)."
