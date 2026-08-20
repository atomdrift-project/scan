#!/bin/sh
# worker-bastille.sh - Deploy Atomdrift Scan worker using separate build and run jails
# Usage: ./worker-bastille.sh [build-jail] [run-jail] <hopper-url>
#
# Environment overrides:
#   WORKERS            concurrency (--workers)                   (default: worker auto)
#   LLM                OpenAI-compatible LLM endpoint (SCAN_LLM)
#   HOPPER_TOKEN_FILE  hopper API token to install in the run jail
#                                                                (default: ~/.tok/hopper)

set -ex
# FreeBSD /bin/sh supports pipefail; this deploy script only runs on FreeBSD.
# shellcheck disable=SC3040
set -o pipefail

BUILD="${1:-build}"
RUN="${2:-litworker}"
URL="$3"
[ -n "$URL" ] || { echo "error: hopper URL required as third argument" >&2; exit 1; }

# Optional: cap concurrent analysis slots (--workers). Unset = worker auto.
WORKERS="${WORKERS:-}"
# LLM second-opinion pass: endpoint (exported as SCAN_LLM) + interpret gate.
LLM="${LLM:-http://10.9.8.149:8000/v1}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

# Shared rc.d service definition (also used by the native worker-freebsd.sh).
SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
# shellcheck source=scripts/worker/lib/freebsd-rcd.sh
. "$SCRIPT_DIR/lib/freebsd-rcd.sh"
worker_args=$(scan_worker_args "$URL" "$WORKERS")

install_missing_build_packages() {
    set --
    for pkg in rust git pkgconf mold gmake; do
        if ! doas bastille cmd "$BUILD" pkg info -e "$pkg" >/dev/null 2>&1; then
            set -- "$@" "$pkg"
        fi
    done
    if [ "$#" -gt 0 ]; then
        doas bastille pkg "$BUILD" install -y "$@"
    fi
}

doas bastille cmd "$BUILD" true || die "build jail '$BUILD' not accessible"
doas bastille cmd "$RUN" true || die "run jail '$RUN' not accessible"

log "Ensuring build user exists"
doas bastille cmd "$BUILD" id -u scan >/dev/null 2>&1 || \
    doas bastille cmd "$BUILD" pw useradd scan -m -s /bin/sh -c "Atomdrift Scan Build"

log "Installing build dependencies"
install_missing_build_packages

log "Syncing source to build jail (preserving target/)"
doas bastille cmd "$BUILD" su -l scan -c "mkdir -p ~/scan"
tar -cf - --exclude=./target --exclude=./out --exclude=./.git . \
    | doas bastille cmd "$BUILD" su -l scan -c "tar -xf - -C ~/scan"

log "Killing any stale cargo processes in build jail"
doas bastille cmd "$BUILD" su -l scan -c "killall cargo 2>/dev/null || true"

log "Building tarball"
doas bastille cmd "$BUILD" su -l scan -c "cd ~/scan && RUSTFLAGS='-C link-arg=-fuse-ld=mold' gmake tarball" \
    || die "build failed in build jail"

log "Running tests"
doas bastille cmd "$BUILD" su -l scan -c "cd ~/scan && RUSTFLAGS='-C link-arg=-fuse-ld=mold' cargo test --release -- --nocapture" \
    || die "tests failed in build jail"

log "Transferring tarball to run jail"
BASTILLE_DIR="/usr/local/bastille/jails"
doas cp "$BASTILLE_DIR/$BUILD/root/home/scan/scan/out/atomscan.tgz" \
       "$BASTILLE_DIR/$RUN/root/tmp/atomscan.tgz"

log "Ensuring run user exists"
doas bastille cmd "$RUN" id -u scan >/dev/null 2>&1 || \
    doas bastille cmd "$RUN" pw useradd scan -m -s /bin/sh -c "Atomdrift Scan Worker"

log "Installing runtime dependencies"
doas bastille pkg "$RUN" install -y git 7-zip upx rizin innoextract

# --- Hopper API token --------------------------------------------------------
#
# Hopper requires `Authorization: Bearer <token>` on every API route, so a
# worker without this file cannot claim work. Copied from the deploying host's
# ~/.tok/hopper into the run jail's service account home, where the worker
# reads it — daemon(8) -u sets HOME from the passwd entry. Never an argument or
# an rc.conf value: argv is visible in ps(1) and rc.conf is world-readable.
HOPPER_TOKEN_SRC="${HOPPER_TOKEN_FILE:-${HOME}/.tok/hopper}"
HOPPER_TOKEN_DST="$BASTILLE_DIR/$RUN/root/home/scan/.tok/hopper"
if [ -s "$HOPPER_TOKEN_SRC" ]; then
    doas bastille cmd "$RUN" install -d -m 0700 -o scan -g scan /home/scan/.tok
    doas install -m 0600 "$HOPPER_TOKEN_SRC" "$HOPPER_TOKEN_DST"
    doas bastille cmd "$RUN" chown scan:scan /home/scan/.tok/hopper
    log "Installed hopper API token at $RUN:/home/scan/.tok/hopper"
elif ! doas test -s "$HOPPER_TOKEN_DST"; then
    # Not fatal: a hopper deployed without --token-file needs no client token.
    log "WARNING: no hopper API token at $HOPPER_TOKEN_SRC; this worker cannot claim work from an authenticated hopper"
fi

log "Installing binary"
doas bastille cmd "$RUN" tar -xzf /tmp/atomscan.tgz -C /usr/local/bin
doas bastille cmd "$RUN" rm -f /tmp/atomscan.tgz

log "Refreshing models and traits in run jail"
doas bastille cmd "$RUN" su -l scan -c "atomscan update-rules" \
    || die "update-rules failed in run jail"

log "Creating rc.d service"
doas bastille cmd "$RUN" mkdir -p /usr/local/etc/rc.d
scan_rcd_script /usr/local/bin/atomscan "$worker_args" "$LLM" \
    | doas bastille cmd "$RUN" tee /usr/local/etc/rc.d/scan-worker >/dev/null
doas bastille cmd "$RUN" chmod 755 /usr/local/etc/rc.d/scan-worker

log "Enabling and restarting scan-worker service"
doas bastille sysrc "$RUN" scan_worker_enable=YES
# The rc.d installed just above has a bounded stop (SIGTERM -> short drain ->
# SIGKILL the daemon(8) tree), so this can't wedge on a busy worker and won't
# orphan the child — no separate force-kill needed.
doas bastille service "$RUN" scan-worker stop || true
doas bastille service "$RUN" scan-worker start

log "Deployment complete"
