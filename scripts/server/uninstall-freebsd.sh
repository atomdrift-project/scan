#!/bin/sh
# uninstall-freebsd.sh - Remove the native Atomdrift Scan server rc.d service.
#
# Stops and disables the service and removes the rc.d script. Leaves the `scan`
# user, its home, and the installed binary intact (remove manually for a fully
# clean state). The Cloudflare Tunnel connector, if one was deployed, is
# stopped too — it fronts a server that is going away.
set -eu

SERVICE_NAME=scan
TUNNEL_SERVICE=scan_tunnel

die() { echo "error: $*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

if command -v doas >/dev/null 2>&1; then
	SUDO=doas
elif command -v sudo >/dev/null 2>&1; then
	SUDO=sudo
else
	die "need doas or sudo"
fi

log "Disabling and stopping ${SERVICE_NAME} service"
$SUDO sysrc "${SERVICE_NAME}_enable=NO" >/dev/null 2>&1 || true
# The rc.d stop is bounded and kills the daemon(8) supervisor together with the
# server it spawned. Deliberately no `pkill -x atomscan` fallback: on a host
# that also runs a worker that would take the worker down with it.
$SUDO service "${SERVICE_NAME}" stop 2>/dev/null || true

log "Removing rc.d script"
$SUDO rm -f "/usr/local/etc/rc.d/${SERVICE_NAME}"

if [ -f "/usr/local/etc/rc.d/${TUNNEL_SERVICE}" ]; then
	log "Disabling and stopping ${TUNNEL_SERVICE}"
	$SUDO sysrc "${TUNNEL_SERVICE}_enable=NO" >/dev/null 2>&1 || true
	$SUDO service "${TUNNEL_SERVICE}" stop 2>/dev/null || true
	$SUDO rm -f "/usr/local/etc/rc.d/${TUNNEL_SERVICE}"
fi

log "Note: service user 'scan', ~scan, and /usr/local/bin/atomscan left intact."
log "Uninstall complete"
