#!/bin/sh
# stop-worker.sh - Forcefully stop the Atomdrift Scan worker on this host.
#
# Hardcore-reliable shutdown for redeploys. The cardinal rule on every platform:
# tell the supervisor to stop FIRST so it can't respawn the process the instant
# we kill it (systemd Restart=, launchd KeepAlive, ...), THEN escalate
# SIGTERM -> SIGKILL against any lingering `ascan worker` process.
#
# This is a *stop*, not an uninstall: the unit/plist/rc.d entry is left
# installed and enabled so the subsequent `make deploy-worker` brings it back.
# Idempotent and quiet when the worker is already stopped; exits non-zero only
# if a process refuses to die after SIGKILL (a D-state hang worth surfacing).
#
# Supported: Linux/systemd, macOS/launchd, FreeBSD/rc.d. Other systems fall
# back to the generic SIGTERM -> SIGKILL escalation.
set -u

SERVICE_NAME=ascan-worker
BINARY=ascan
# Match the running worker, not this script or the make/ssh wrapper invoking it
# ("make stop-worker"/"deploy-worker" contain no space, so they never match).
PATTERN='ascan worker'
LAUNCHD_LABEL=com.atomdrift.ascan-worker
FREEBSD_PIDFILE=/var/run/ascan_worker.pid

log() { printf '==> %s\n' "$*"; }

# Privilege escalation: doas (FreeBSD-preferred) or sudo; empty if already root
# or neither exists (commands then run unprivileged and best-effort).
if [ "$(id -u)" = "0" ]; then
	SUDO=""
elif command -v doas >/dev/null 2>&1; then
	SUDO=doas
elif command -v sudo >/dev/null 2>&1; then
	SUDO=sudo
else
	SUDO=""
fi

alive() { pgrep -f "$PATTERN" >/dev/null 2>&1; }

# --- 1. Ask the platform supervisor to stop (prevents respawn) --------------

case "$(uname -s)" in
Linux)
	if command -v systemctl >/dev/null 2>&1 &&
		systemctl list-unit-files "${SERVICE_NAME}.service" >/dev/null 2>&1; then
		log "Stopping ${SERVICE_NAME}.service (systemd)"
		$SUDO systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
	fi
	;;
Darwin)
	# launchd KeepAlive respawns the process unless the job is booted out, so
	# a plain kill is not enough — unload it first. deploy re-bootstraps it.
	if $SUDO launchctl print "system/$LAUNCHD_LABEL" >/dev/null 2>&1; then
		log "Booting out system/$LAUNCHD_LABEL (launchd)"
		$SUDO launchctl bootout "system/$LAUNCHD_LABEL" 2>/dev/null || true
	fi
	;;
FreeBSD)
	if command -v service >/dev/null 2>&1; then
		log "Stopping ${SERVICE_NAME} (rc.d)"
		$SUDO service "$SERVICE_NAME" stop 2>/dev/null || true
	fi
	# rc.d records the pid; kill it directly in case `service stop` was a no-op.
	$SUDO pkill -TERM -F "$FREEBSD_PIDFILE" 2>/dev/null || true
	;;
*)
	log "Unknown platform $(uname -s) — relying on generic process escalation"
	;;
esac

# --- 2. Escalate against any survivor regardless of platform ----------------

if alive; then
	log "Sending SIGTERM to lingering '$PATTERN' process(es)"
	$SUDO pkill -TERM -f "$PATTERN" 2>/dev/null || true

	# Give it up to 10s to flush and exit cleanly.
	i=0
	while [ "$i" -lt 10 ] && alive; do
		sleep 1
		i=$((i + 1))
	done

	if alive; then
		log "Still alive after ${i}s — sending SIGKILL (-9)"
		$SUDO pkill -KILL -f "$PATTERN" 2>/dev/null || true
		# macOS pkill -f can miss a short argv; -x on the bare binary catches it.
		$SUDO pkill -KILL -x "$BINARY" 2>/dev/null || true
		sleep 1
	fi
fi

if alive; then
	log "ERROR: '$PATTERN' process(es) survived SIGKILL:"
	# pgrep -a is Linux-only; fall back to ps on hosts without it.
	# shellcheck disable=SC2009
	pgrep -af "$PATTERN" 2>/dev/null || ps -ax -o pid,command 2>/dev/null | grep "$PATTERN" | grep -v grep || true
	exit 1
fi

log "Worker stopped"
