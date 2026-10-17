#!/bin/sh
# uninstall-freebsd.sh - Remove the native Atomdrift Scan worker rc.d service on FreeBSD.
#
# Stops and disables the service and removes the rc.d script. Leaves the
# `scan` user, its home, and the installed binary intact (remove manually
# for a fully clean state).
set -eu

die() { echo "error: $*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

if command -v doas >/dev/null 2>&1; then
	SUDO=doas
elif command -v sudo >/dev/null 2>&1; then
	SUDO=sudo
else
	die "need doas or sudo"
fi

log "Disabling and stopping scan-worker service"
$SUDO sysrc scan_worker_enable=NO >/dev/null 2>&1 || true
$SUDO service scan-worker stop 2>/dev/null || true
# Belt-and-suspenders: reap any worker the stop left behind. Target the child
# by name, not `-F /var/run/scan_worker.pid` — that pidfile is the daemon(8)
# supervisor, and SIGKILLing it leaves its atomscan child orphaned and running.
$SUDO pkill -9 -x atomscan 2>/dev/null || true

log "Removing rc.d script"
$SUDO rm -f /usr/local/etc/rc.d/scan-worker

log "Removing any legacy ascan-worker service (pre-rename install)"
$SUDO sysrc ascan_worker_enable=NO >/dev/null 2>&1 || true
$SUDO service ascan-worker stop 2>/dev/null || true
$SUDO pkill -9 -F /var/run/ascan_worker.pid 2>/dev/null || true
$SUDO rm -f /usr/local/etc/rc.d/ascan-worker

log "Note: service user 'scan', ~scan, and /usr/local/bin/atomscan left intact."
log "Uninstall complete"
