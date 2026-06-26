#!/bin/sh
# uninstall-macos.sh - Remove Atomdrift Scan worker persistence on macOS
set -ex

LABEL=com.atomdrift.scan-worker
PLIST=/Library/LaunchDaemons/com.atomdrift.scan-worker.plist
BINARY=scan

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

log "Stopping launchd service"
if sudo launchctl print "system/$LABEL" >/dev/null 2>&1; then
    sudo launchctl bootout "system/$LABEL" || true
fi

log "Removing plist"
sudo rm -f "$PLIST"

log "Removing any legacy ascan-worker launchd service (pre-rename install)"
LEGACY_LABEL=com.atomdrift.ascan-worker
if sudo launchctl print "system/$LEGACY_LABEL" >/dev/null 2>&1; then
    sudo launchctl bootout "system/$LEGACY_LABEL" || true
fi
sudo rm -f /Library/LaunchDaemons/com.atomdrift.ascan-worker.plist

log "Killing any remaining processes"
sudo pkill -x "$BINARY" 2>/dev/null || true
sudo pkill -x ascan 2>/dev/null || true

log "Uninstall complete"
