#!/bin/sh
# rollout-bastille.sh - Deploy litmus using separate build and run jails
# Usage: ./rollout-bastille.sh <build-jail> <run-jail>

set -e

BUILD="$1"
RUN="$2"

die() {
    echo "error: $*" >&2
    exit 1
}

log() {
    echo "==> $*"
}

[ -z "$BUILD" ] || [ -z "$RUN" ] && die "usage: $0 <build-jail> <run-jail>"

# Verify jails are accessible
doas bastille cmd "$BUILD" true || die "build jail '$BUILD' not accessible"
doas bastille cmd "$RUN" true || die "run jail '$RUN' not accessible"

# --- Build jail setup ---

log "Ensuring build user exists"
doas bastille cmd "$BUILD" id -u litmus >/dev/null 2>&1 || \
    doas bastille cmd "$BUILD" pw useradd litmus -m -s /bin/sh -c "Litmus Build"

log "Installing build dependencies"
doas bastille pkg "$BUILD" install -y rust sccache git pkgconf onnxruntime

log "Syncing source to build jail (preserving target/)"
doas bastille cmd "$BUILD" mkdir -p /home/litmus/litmus
tar -cf - --exclude=./target --exclude=./out --exclude=./.git . \
    | doas bastille cmd "$BUILD" tar -xf - -C /home/litmus/litmus
doas bastille cmd "$BUILD" chown -R litmus:litmus /home/litmus/litmus

log "Building tarball"
doas bastille cmd "$BUILD" su -l litmus -c "cd ~/litmus && RUSTC_WRAPPER=sccache make tarball"

# --- Transfer tarball via jail filesystem ---

log "Transferring tarball to run jail"
BASTILLE_DIR="/usr/local/bastille/jails"
doas cp "$BASTILLE_DIR/$BUILD/root/home/litmus/litmus/out/litmus.tgz" \
       "$BASTILLE_DIR/$RUN/root/tmp/litmus.tgz"

# --- Run jail setup ---

log "Ensuring run user exists"
doas bastille cmd "$RUN" id -u litmus >/dev/null 2>&1 || \
    doas bastille cmd "$RUN" pw useradd litmus -m -s /bin/sh -c "Litmus Service"

log "Installing runtime dependencies"
doas bastille pkg "$RUN" install -y git onnxruntime

log "Extracting tarball"
doas bastille cmd "$RUN" rm -rf /usr/local/share/litmus
doas bastille cmd "$RUN" mkdir -p /usr/local/share/litmus
doas bastille cmd "$RUN" tar -xzf /tmp/litmus.tgz -C /usr/local/share/litmus
doas bastille cmd "$RUN" rm -f /tmp/litmus.tgz
doas bastille cmd "$RUN" ln -sf /usr/local/share/litmus/litmus /usr/local/bin/litmus

log "Creating rc.d service"
doas bastille cmd "$RUN" mkdir -p /usr/local/etc/rc.d
doas bastille cmd "$RUN" tee /usr/local/etc/rc.d/litmus >/dev/null <<'EOF'
#!/bin/sh

# PROVIDE: litmus
# REQUIRE: LOGIN DAEMON NETWORKING
# KEYWORD: shutdown

. /etc/rc.subr

name="litmus"
rcvar="litmus_enable"

load_rc_config $name

: ${litmus_enable:="NO"}
: ${litmus_logfile:="/var/log/litmus.log"}

pidfile="/var/run/${name}.pid"
command="/usr/sbin/daemon"
command_args="-c -f -P ${pidfile} -r -o ${litmus_logfile} -u litmus /usr/local/share/litmus/litmus --verbose serve --bind 0.0.0.0:8081"

run_rc_command "$1"
EOF

doas bastille cmd "$RUN" chmod 755 /usr/local/etc/rc.d/litmus

log "Enabling and restarting litmus service"
doas bastille sysrc "$RUN" litmus_enable=YES
doas bastille service "$RUN" litmus stop 2>/dev/null || true
doas bastille service "$RUN" litmus start

log "Deployment complete"
