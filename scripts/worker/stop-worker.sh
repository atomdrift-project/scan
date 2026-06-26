#!/bin/sh
# stop-worker.sh - Forcefully stop the Atomdrift Scan worker on this host.
#
# Hardcore-reliable shutdown for redeploys, built around two facts:
#
#   1. A supervisor stop (systemctl stop / launchctl bootout / service stop)
#      WAITS for the process to exit, so a wedged worker hangs it indefinitely.
#   2. But killing the process while the supervisor is still active makes it
#      respawn (systemd Restart=, launchd KeepAlive).
#
# So we issue the supervisor stop in the BACKGROUND — that registers the stop
# intent (unit -> deactivating, KeepAlive job dropped), which is what disables
# respawn — then SIGTERM -> SIGKILL the process ourselves so it dies now, then
# bounded-reap the backgrounded stop so it can never hang us.
#
# This is a *stop*, not an uninstall: the unit/plist/rc.d entry stays installed
# and enabled so the subsequent `make deploy-worker` brings it back. Idempotent
# and quiet when already stopped; exits non-zero only if a process survives
# SIGKILL (an unkillable D-state hang worth surfacing).
#
# Supported: Linux/systemd, macOS/launchd, FreeBSD/rc.d. Other systems fall
# back to the generic SIGTERM -> SIGKILL escalation.
set -u

SERVICE_NAME=scan-worker
BINARY=scan
# Match the running worker, not this script or the make/ssh wrapper invoking it
# ("make stop-worker"/"deploy-worker" contain no space, so they never match).
PATTERN='scan worker'
LAUNCHD_LABEL=com.atomdrift.scan-worker

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

# --- 1. Tell the supervisor to stop, in the background ----------------------
#
# Backgrounding is the whole point: the stop intent is registered (disabling
# respawn) without us blocking on the supervisor's own wait-for-exit. We reap
# this pid under a deadline in step 3.

SUPERVISOR_PID=""

case "$(uname -s)" in
Linux)
	if command -v systemctl >/dev/null 2>&1 &&
		systemctl list-unit-files "${SERVICE_NAME}.service" >/dev/null 2>&1; then
		log "Stopping ${SERVICE_NAME}.service (systemd, backgrounded)"
		$SUDO systemctl stop "${SERVICE_NAME}.service" >/dev/null 2>&1 &
		SUPERVISOR_PID=$!
	fi
	;;
Darwin)
	# bootout drops the KeepAlive job so launchd won't respawn what we kill.
	if $SUDO launchctl print "system/$LAUNCHD_LABEL" >/dev/null 2>&1; then
		log "Booting out system/$LAUNCHD_LABEL (launchd, backgrounded)"
		$SUDO launchctl bootout "system/$LAUNCHD_LABEL" >/dev/null 2>&1 &
		SUPERVISOR_PID=$!
	fi
	;;
FreeBSD)
	if command -v service >/dev/null 2>&1; then
		log "Stopping ${SERVICE_NAME} (rc.d, backgrounded)"
		$SUDO service "$SERVICE_NAME" stop >/dev/null 2>&1 &
		SUPERVISOR_PID=$!
	fi
	;;
*)
	log "Unknown platform $(uname -s) — relying on generic process escalation"
	;;
esac

# --- 2. Kill the process ourselves: SIGTERM grace, then SIGKILL -------------
#
# Covers the no-supervisor case (we send the first SIGTERM) and the wedged case
# (the supervisor's own SIGTERM never landed it). Either way we don't wait on
# the supervisor to do it.

if alive; then
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

# --- 3. Reap the backgrounded supervisor stop under a deadline --------------
#
# With the process now dead, the stop should complete near-instantly. If it
# doesn't (a wedged supervisor), kill the stop client too — the stop intent was
# already registered, so respawn stays disabled regardless.

if [ -n "$SUPERVISOR_PID" ]; then
	i=0
	while [ "$i" -lt 5 ] && kill -0 "$SUPERVISOR_PID" 2>/dev/null; do
		sleep 1
		i=$((i + 1))
	done
	if kill -0 "$SUPERVISOR_PID" 2>/dev/null; then
		log "Supervisor stop did not return in ${i}s — abandoning it (process already killed)"
		kill -9 "$SUPERVISOR_PID" 2>/dev/null || true
	fi
	wait "$SUPERVISOR_PID" 2>/dev/null || true
fi

# --- 4. Verify ---------------------------------------------------------------

if alive; then
	log "ERROR: '$PATTERN' process(es) survived SIGKILL:"
	# pgrep -a is Linux-only; fall back to ps on hosts without it.
	# shellcheck disable=SC2009
	pgrep -af "$PATTERN" 2>/dev/null || ps -ax -o pid,command 2>/dev/null | grep "$PATTERN" | grep -v grep || true
	exit 1
fi

log "Worker stopped"
