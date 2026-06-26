#!/bin/sh
# worker-bastille.sh - Deploy Atomdrift Scan worker using separate build and run jails
# Usage: ./worker-bastille.sh [build-jail] [run-jail] <hopper-url>

set -ex
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
. "$SCRIPT_DIR/lib/freebsd-rcd.sh"
worker_args=$(scan_worker_args "$URL" "$WORKERS")

install_missing_build_packages() {
    missing=""
    for pkg in rust git pkgconf mold gmake; do
        if ! doas bastille cmd "$BUILD" pkg info -e "$pkg" >/dev/null 2>&1; then
            missing="$missing $pkg"
        fi
    done
    if [ -n "$missing" ]; then
        doas bastille pkg "$BUILD" install -y $missing
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
doas cp "$BASTILLE_DIR/$BUILD/root/home/scan/scan/out/scan.tgz" \
       "$BASTILLE_DIR/$RUN/root/tmp/scan.tgz"

log "Ensuring run user exists"
doas bastille cmd "$RUN" id -u scan >/dev/null 2>&1 || \
    doas bastille cmd "$RUN" pw useradd scan -m -s /bin/sh -c "Atomdrift Scan Worker"

log "Installing runtime dependencies"
doas bastille pkg "$RUN" install -y git 7-zip upx rizin innoextract

log "Extracting tarball"
doas bastille cmd "$RUN" rm -rf /usr/local/share/atomdrift/scan
doas bastille cmd "$RUN" mkdir -p /usr/local/share/atomdrift/scan
doas bastille cmd "$RUN" tar -xzf /tmp/scan.tgz -C /usr/local/share/atomdrift/scan
doas bastille cmd "$RUN" rm -f /tmp/scan.tgz
doas bastille cmd "$RUN" ln -sf /usr/local/share/atomdrift/scan/scan /usr/local/bin/scan

log "Refreshing models and traits in run jail"
doas bastille cmd "$RUN" su -l scan -c "scan update-rules" \
    || die "update-rules failed in run jail"

log "Creating rc.d service"
doas bastille cmd "$RUN" mkdir -p /usr/local/etc/rc.d
scan_rcd_script /usr/local/share/atomdrift/scan/scan "$worker_args" "$LLM" \
    | doas bastille cmd "$RUN" tee /usr/local/etc/rc.d/scan-worker >/dev/null
doas bastille cmd "$RUN" chmod 755 /usr/local/etc/rc.d/scan-worker

log "Enabling and restarting scan-worker service"
doas bastille sysrc "$RUN" scan_worker_enable=YES
doas bastille service "$RUN" scan-worker stop 2>/dev/null || true
doas bastille cmd "$RUN" pkill -9 -F /var/run/scan_worker.pid 2>/dev/null || true
doas bastille service "$RUN" scan-worker start

log "Deployment complete"
