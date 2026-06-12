#!/bin/sh
# rollout-bastille.sh - Deploy Atomdrift Scan using separate build and run jails
# Usage: ./rollout-bastille.sh [build-jail] [run-jail]

set -ex
set -o pipefail

BUILD="${1:-build}"
RUN="${2:-ascan}"

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
doas bastille cmd "$BUILD" id -u ascan >/dev/null 2>&1 || \
    doas bastille cmd "$BUILD" pw useradd ascan -m -s /bin/sh -c "Atomdrift Scan Build"

log "Installing build dependencies"
install_missing_build_packages

log "Syncing source to build jail (preserving target/)"
doas bastille cmd "$BUILD" su -l ascan -c "mkdir -p ~/scan"
tar -cf - --exclude=./target --exclude=./out --exclude=./.git . \
    | doas bastille cmd "$BUILD" su -l ascan -c "tar -xf - -C ~/scan"

log "Killing any stale cargo processes in build jail"
doas bastille cmd "$BUILD" su -l ascan -c "killall cargo 2>/dev/null || true"

log "Building tarball"
doas bastille cmd "$BUILD" su -l ascan -c "cd ~/scan && RUSTFLAGS='-C link-arg=-fuse-ld=mold' gmake tarball" \
    || die "build failed in build jail"

log "Running tests"
doas bastille cmd "$BUILD" su -l ascan -c "cd ~/scan && RUSTFLAGS='-C link-arg=-fuse-ld=mold' cargo test --release -- --nocapture" \
    || die "tests failed in build jail"

# --- Transfer tarball via jail filesystem ---

log "Transferring tarball to run jail"
BASTILLE_DIR="/usr/local/bastille/jails"
doas cp "$BASTILLE_DIR/$BUILD/root/home/ascan/scan/out/ascan.tgz" \
       "$BASTILLE_DIR/$RUN/root/tmp/ascan.tgz"

# --- Run jail setup ---

log "Ensuring run user exists"
doas bastille cmd "$RUN" id -u ascan >/dev/null 2>&1 || \
    doas bastille cmd "$RUN" pw useradd ascan -m -s /bin/sh -c "Atomdrift Scan Service"

log "Installing runtime dependencies"
doas bastille pkg "$RUN" install -y git 7-zip upx rizin innoextract

log "Extracting tarball"
doas bastille cmd "$RUN" rm -rf /usr/local/share/atomdrift/scan
doas bastille cmd "$RUN" mkdir -p /usr/local/share/atomdrift/scan
doas bastille cmd "$RUN" tar -xzf /tmp/ascan.tgz -C /usr/local/share/atomdrift/scan
doas bastille cmd "$RUN" rm -f /tmp/ascan.tgz
doas bastille cmd "$RUN" ln -sf /usr/local/share/atomdrift/scan/ascan /usr/local/bin/ascan

log "Refreshing models and traits in run jail"
doas bastille cmd "$RUN" su -l ascan -c "ascan update-rules" \
    || die "update-rules failed in run jail"

log "Creating rc.d service"
doas bastille cmd "$RUN" mkdir -p /usr/local/etc/rc.d
doas bastille cmd "$RUN" tee /usr/local/etc/rc.d/ascan >/dev/null <<'EOF'
#!/bin/sh

# PROVIDE: ascan
# REQUIRE: LOGIN DAEMON NETWORKING
# KEYWORD: shutdown

. /etc/rc.subr

name="ascan"
rcvar="ascan_enable"

load_rc_config $name

: ${ascan_enable:="NO"}
: ${ascan_logfile:="/var/log/ascan.log"}

pidfile="/var/run/${name}.pid"
command="/usr/sbin/daemon"
command_args="-c -f -P ${pidfile} -r -o ${ascan_logfile} -u ascan /usr/local/share/atomdrift/scan/ascan -u serve --bind 0.0.0.0:49999 --allow-cidr 10.0.0.0/8"

run_rc_command "$1"
EOF

doas bastille cmd "$RUN" chmod 755 /usr/local/etc/rc.d/ascan

log "Enabling and restarting ascan service"
doas bastille sysrc "$RUN" ascan_enable=YES
doas bastille service "$RUN" ascan stop 2>/dev/null || true
doas bastille cmd "$RUN" pkill -9 -F /var/run/ascan.pid 2>/dev/null || true
doas bastille service "$RUN" ascan start

log "Deployment complete"
