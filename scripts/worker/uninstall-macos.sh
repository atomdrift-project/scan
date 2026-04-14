#!/bin/sh
# uninstall-macos.sh - Remove litmus worker persistence on macOS
set -ex

LABEL=com.atomdrift.litmus-worker
PLIST=/Library/LaunchDaemons/com.atomdrift.litmus-worker.plist
BINARY=litmus

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

log "Stopping launchd service"
if sudo launchctl print "system/$LABEL" >/dev/null 2>&1; then
    sudo launchctl bootout "system/$LABEL" || true
fi

log "Removing plist"
sudo rm -f "$PLIST"

log "Killing any remaining processes"
sudo pkill -x "$BINARY" 2>/dev/null || true

log "Uninstall complete"
