#!/bin/sh
# worker-freebsd.sh - Install Atomdrift Scan worker as a native FreeBSD rc.d service.
#
# Local install for a FreeBSD host (no jail). Builds as the invoking user,
# installs the binary, creates an unprivileged `scan` service user, and runs
# the worker under rc.d via daemon(8). This mirrors the systemd deploy
# (worker-linux.sh): one host, one supervised service. For a jailed deploy
# use `make deploy-jail-worker` (scripts/worker/worker-bastille.sh) instead.
#
# Re-runnable: idempotent. The binary is reinstalled and the service restarted
# only when the binary or the rc.d script actually changed on disk.
#
# Usage: ./worker-freebsd.sh <url>
#
# doas or sudo is required for package installation, user creation, and
# service management.
#
# Environment overrides:
#   DATA_DIR            local sample root (--data-dir)           (default: unset)
#   WORKERS             concurrency (--workers)                  (default: worker auto)
#   MAX_RSS_GB          pause threshold (--max-rss-gb)            (default: 0 = auto)
#   LLM                 OpenAI-compatible LLM endpoint (SCAN_LLM) (default: http://10.9.8.149:8000/v1)

set -eu

URL="${1:-}"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

DATA_DIR="${DATA_DIR:-}"
WORKERS="${WORKERS:-}"
MAX_RSS_GB="${MAX_RSS_GB:-0}"
LLM="${LLM:-http://10.9.8.149:8000/v1}"

BINARY=atomscan
BIN_PATH=/usr/local/bin/${BINARY}
RCD_FILE=/usr/local/etc/rc.d/scan-worker

die() { echo "error: $*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
# shellcheck source=scripts/worker/lib/freebsd-rcd.sh
. "$SCRIPT_DIR/lib/freebsd-rcd.sh"

TMP_RCD=""
trap '[ -n "$TMP_RCD" ] && rm -f "$TMP_RCD"' EXIT

# --- Preconditions -----------------------------------------------------------

[ -f Makefile ]              || die "run from the repository root"
[ "$(uname -s)" = "FreeBSD" ] || die "this script is for FreeBSD; use deploy-jail-worker for jails"

if command -v doas >/dev/null 2>&1; then
	SUDO=doas
elif command -v sudo >/dev/null 2>&1; then
	SUDO=sudo
else
	die "need doas or sudo for privileged steps"
fi

# --- Packages ---------------------------------------------------------------
# Build deps (rust, git, pkgconf, mold) and runtime deps (7-zip, upx, rizin,
# innoextract) on one host. pkg install is idempotent, but checking first
# keeps a no-op re-run from touching the package DB.

missing=""
for pkg in rust git pkgconf mold 7-zip upx rizin innoextract; do
	pkg info -e "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"
done
if [ -n "$missing" ]; then
	log "Installing packages:$missing"
	# shellcheck disable=SC2086
	$SUDO pkg install -y $missing
else
	log "All packages already installed"
fi

# --- Build (as the invoking user) ------------------------------------------
# Raise the data-segment limit to its hard cap; FreeBSD's default datasize
# limit can starve a release build of rustc.
# FreeBSD /bin/sh exposes the data-segment hard limit through ulimit -Hd.
# shellcheck disable=SC3045
ulimit -d "$(ulimit -Hd)" 2>/dev/null || true

log "Building"
RUSTFLAGS="-C link-arg=-fuse-ld=mold" cargo build --release || die "build failed"
[ -x "target/release/${BINARY}" ] || die "build did not produce target/release/${BINARY}"

# --- Service user -----------------------------------------------------------
# A home directory is required: the worker resolves cleave's traits/models
# under the service user's data dir, and daemon(8) -u sets HOME from it. The
# shell is /bin/sh (not nologin) so the `su -l scan` model refresh below can
# run; the account has no password, so it is not remotely loginable. This
# mirrors the run-jail user in worker-bastille.sh.

if ! id -u scan >/dev/null 2>&1; then
	log "Creating service user 'scan'"
	$SUDO pw useradd scan -m -s /bin/sh -c "Atomdrift Scan Worker"
fi

# --- Binary -----------------------------------------------------------------

binary_changed=0
if $SUDO cmp -s "target/release/${BINARY}" "${BIN_PATH}" 2>/dev/null; then
	log "Binary unchanged"
else
	log "Installing ${BIN_PATH}"
	# install(1) writes-then-renames; safe over a running exe (the kernel
	# pins the inode of the running process).
	$SUDO install -m 0755 -o root -g wheel "target/release/${BINARY}" "${BIN_PATH}"
	binary_changed=1
fi

# --- Models and traits ------------------------------------------------------
# Populate the scan user's data dir so the first worker start is not racing
# a clone. Idempotent: update-rules pulls when a checkout already exists.

log "Refreshing models and traits"
$SUDO su -l scan -c "${BIN_PATH} update-rules" || die "update-rules failed"

# --- rc.d service -----------------------------------------------------------

worker_args=$(scan_worker_args "$URL" "$WORKERS" "$DATA_DIR" "$MAX_RSS_GB")

TMP_RCD=$(mktemp -t scan-worker.rcd.XXXXXX)
scan_rcd_script "${BIN_PATH}" "$worker_args" "$LLM" >"$TMP_RCD"

rcd_changed=0
if $SUDO cmp -s "$TMP_RCD" "$RCD_FILE" 2>/dev/null; then
	log "rc.d script unchanged"
else
	log "Writing ${RCD_FILE}"
	$SUDO install -m 0755 -o root -g wheel "$TMP_RCD" "$RCD_FILE"
	rcd_changed=1
fi

# --- Activate ---------------------------------------------------------------

# sysrc is idempotent; enable so the service also comes back across reboots.
$SUDO sysrc scan_worker_enable=YES >/dev/null

if [ "$binary_changed" -eq 1 ] || [ "$rcd_changed" -eq 1 ]; then
	log "Restarting scan-worker"
	# The rc.d stop is now bounded: it SIGTERMs the worker, waits out a short
	# drain, then SIGKILLs the whole daemon(8) tree — so a busy or wedged worker
	# can't stall the redeploy the way rc.subr's default (wait-forever) stop did,
	# and no lingering child is left to kill -9 by hand.
	$SUDO service scan-worker stop || true
	$SUDO service scan-worker start || die "service failed to start; see /var/log/scan-worker.log"
else
	log "No changes; ensuring scan-worker is running"
	$SUDO service scan-worker status >/dev/null 2>&1 || $SUDO service scan-worker start
fi

$SUDO service scan-worker status || true
log "Deployment complete"
