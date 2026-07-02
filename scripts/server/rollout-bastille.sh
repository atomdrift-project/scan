#!/bin/sh
# rollout-bastille.sh - Deploy Atomdrift Scan using separate build and run jails
# Usage: ./rollout-bastille.sh [build-jail] [run-jail]

set -ex
set -o pipefail

BUILD="${1:-build}"
RUN="${2:-scan}"

die() {
    echo "error: $*" >&2
    exit 1
}

log() {
    echo "==> $*"
}

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

# Verify jails are accessible
doas bastille cmd "$BUILD" true || die "build jail '$BUILD' not accessible"
doas bastille cmd "$RUN" true || die "run jail '$RUN' not accessible"

# --- Build jail setup ---

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

# --- Transfer tarball via jail filesystem ---

log "Transferring tarball to run jail"
BASTILLE_DIR="/usr/local/bastille/jails"
doas cp "$BASTILLE_DIR/$BUILD/root/home/scan/scan/out/atomscan.tgz" \
       "$BASTILLE_DIR/$RUN/root/tmp/atomscan.tgz"

# --- Run jail setup ---

log "Ensuring run user exists"
doas bastille cmd "$RUN" id -u scan >/dev/null 2>&1 || \
    doas bastille cmd "$RUN" pw useradd scan -m -s /bin/sh -c "Atomdrift Scan Service"

log "Installing runtime dependencies"
doas bastille pkg "$RUN" install -y git 7-zip upx rizin innoextract

log "Installing binary"
doas bastille cmd "$RUN" tar -xzf /tmp/atomscan.tgz -C /usr/local/bin
doas bastille cmd "$RUN" rm -f /tmp/atomscan.tgz

log "Refreshing models and traits in run jail"
doas bastille cmd "$RUN" su -l scan -c "atomscan update-rules" \
    || die "update-rules failed in run jail"

log "Creating rc.d service"
doas bastille cmd "$RUN" mkdir -p /usr/local/etc/rc.d
doas bastille cmd "$RUN" tee /usr/local/etc/rc.d/scan >/dev/null <<'EOF'
#!/bin/sh

# PROVIDE: scan
# REQUIRE: LOGIN DAEMON NETWORKING
# KEYWORD: shutdown

. /etc/rc.subr

name="scan"
rcvar="scan_enable"

load_rc_config $name

: ${scan_enable:="NO"}
: ${scan_logfile:="/var/log/scan.log"}

pidfile="/var/run/${name}.pid"
command="/usr/sbin/daemon"
command_args="-c -f -P ${pidfile} -r -o ${scan_logfile} -u scan /usr/local/bin/atomscan -u serve --bind 0.0.0.0:49999 --allow-cidr 10.0.0.0/8"

run_rc_command "$1"
EOF

doas bastille cmd "$RUN" chmod 755 /usr/local/etc/rc.d/scan

log "Enabling and restarting scan service"
doas bastille sysrc "$RUN" scan_enable=YES
doas bastille service "$RUN" scan stop 2>/dev/null || true
doas bastille cmd "$RUN" pkill -9 -F /var/run/scan.pid 2>/dev/null || true
doas bastille service "$RUN" scan start

log "Deployment complete"
