#!/bin/sh
# uninstall-macos.sh - Remove Atomdrift Scan worker persistence on macOS
set -ex

LABEL=com.atomdrift.scan-worker
PLIST=/Library/LaunchDaemons/com.atomdrift.scan-worker.plist
BINARY=atomscan

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

# Privilege escalation: prefer doas, fall back to sudo.
if   command -v doas >/dev/null 2>&1; then SUDO=doas
elif command -v sudo >/dev/null 2>&1; then SUDO=sudo
else die "need doas or sudo"
fi

log "Stopping launchd service"
if $SUDO launchctl print "system/$LABEL" >/dev/null 2>&1; then
    $SUDO launchctl bootout "system/$LABEL" || true
fi

log "Removing plist"
$SUDO rm -f "$PLIST"

log "Removing any legacy ascan-worker launchd service (pre-rename install)"
LEGACY_LABEL=com.atomdrift.ascan-worker
if $SUDO launchctl print "system/$LEGACY_LABEL" >/dev/null 2>&1; then
    $SUDO launchctl bootout "system/$LEGACY_LABEL" || true
fi
$SUDO rm -f /Library/LaunchDaemons/com.atomdrift.ascan-worker.plist

log "Killing any remaining processes"
$SUDO pkill -x "$BINARY" 2>/dev/null || true
$SUDO pkill -x ascan 2>/dev/null || true

log "Uninstall complete"
