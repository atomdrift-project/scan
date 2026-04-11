#!/bin/sh
# rollout-openbsd.sh - Deploy litmus to OpenBSD nodes using separate build and run hosts
# Usage: ./rollout-openbsd.sh [build-host] [run-host]
#
# build-host / run-host are SSH targets (e.g. "user@host" or an ssh_config alias).
# The same host may be passed for both. Remote user must have passwordless doas.

set -e

BUILD="${1:-build}"
RUN="${2:-litmus}"

die() {
    echo "error: $*" >&2
    exit 1
}

log() {
    echo "==> $*"
}

bssh() {
    ssh -o BatchMode=yes "$BUILD" "$@"
}

rssh() {
    ssh -o BatchMode=yes "$RUN" "$@"
}

# Verify hosts are reachable
bssh true || die "build host '$BUILD' not accessible"
rssh true || die "run host '$RUN' not accessible"

# --- Build host setup ---

log "Installing build dependencies on $BUILD"
bssh "doas pkg_add -I rust git sccache-- || true"

log "Ensuring build user exists on $BUILD"
bssh "id -u _litmusbld >/dev/null 2>&1 || \
      doas useradd -m -d /home/_litmusbld -s /bin/ksh -c 'Litmus Build' _litmusbld"

log "Syncing source to build host (excluding target/, out/, .git)"
bssh "doas install -d -o _litmusbld -g _litmusbld /home/_litmusbld/litmus"
tar -cf - --exclude=./target --exclude=./out --exclude=./.git . \
    | bssh "doas -u _litmusbld tar -xf - -C /home/_litmusbld/litmus"

log "Building tarball on $BUILD"
bssh "doas -u _litmusbld ksh -c 'cd ~/litmus && RUSTC_WRAPPER=sccache gmake tarball || make tarball'" \
    || die "build failed on build host"

log "Upgrading rules on $BUILD"
bssh "doas -u _litmusbld ksh -c 'cd ~/litmus && ./target/release/litmus update-rules'" \
    || die "update-rules failed on build host"

log "Running tests on $BUILD"
bssh "doas -u _litmusbld ksh -c 'cd ~/litmus && RUSTC_WRAPPER=sccache cargo test --release -- --nocapture'" \
    || die "tests failed on build host"

# --- Transfer tarball via SSH pipe ---

log "Transferring tarball from $BUILD to $RUN"
bssh "doas cat /home/_litmusbld/litmus/out/litmus.tgz" \
    | rssh "doas tee /tmp/litmus.tgz >/dev/null"

# --- Run host setup ---

log "Ensuring service user exists on $RUN"
rssh "id -u _litmus >/dev/null 2>&1 || \
      doas useradd -d /var/empty -s /sbin/nologin -c 'Litmus Service' _litmus"

log "Installing runtime dependencies on $RUN"
rssh "doas pkg_add -I git || true"

log "Extracting tarball on $RUN"
rssh "doas rm -rf /usr/local/share/litmus && \
      doas mkdir -p /usr/local/share/litmus && \
      doas tar -xzf /tmp/litmus.tgz -C /usr/local/share/litmus && \
      doas rm -f /tmp/litmus.tgz && \
      doas ln -sf /usr/local/share/litmus/litmus /usr/local/bin/litmus"

log "Installing rc.d script on $RUN"
rssh "doas tee /etc/rc.d/litmus >/dev/null" <<'EOF'
#!/bin/ksh
#
# litmus - Litmus server

daemon="/usr/local/share/litmus/litmus"
daemon_flags="--verbose -u serve --bind 0.0.0.0:8880"
daemon_user="_litmus"
daemon_logger="daemon.info"

. /etc/rc.d/rc.subr

rc_bg=YES
rc_reload=NO

rc_cmd $1
EOF

rssh "doas chmod 555 /etc/rc.d/litmus"

log "Enabling and restarting litmus service on $RUN"
rssh "doas rcctl enable litmus && \
      doas rcctl set litmus status on && \
      doas rcctl restart litmus"

log "Service status:"
rssh "doas rcctl check litmus" || true

log "Deployment complete"
