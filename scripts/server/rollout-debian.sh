#!/bin/sh
# rollout-debian.sh - Deploy litmus to Debian nodes using separate build and run hosts
# Usage: ./rollout-debian.sh [build-host] [run-host]
#
# build-host / run-host are SSH targets (e.g. "user@host" or an ssh_config alias).
# The same host may be passed for both. Remote user must have passwordless sudo.

set -ex

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
bssh "sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cargo rustc git pkg-config build-essential clang lld sccache ca-certificates"

log "Ensuring build user exists on $BUILD"
bssh "id -u litmus >/dev/null 2>&1 || sudo useradd -m -s /bin/sh -c 'Litmus Build' litmus"

log "Syncing source to build host (excluding target/, out/, .git)"
bssh "sudo -u litmus mkdir -p /home/litmus/litmus"
tar -cf - --exclude=./target --exclude=./out --exclude=./.git . \
    | bssh "sudo -u litmus tar -xf - -C /home/litmus/litmus"

log "Building tarball on $BUILD"
bssh "sudo -u litmus sh -c 'cd ~/litmus && RUSTC_WRAPPER=sccache RUSTFLAGS=\"-C link-arg=-fuse-ld=lld\" make tarball'" \
    || die "build failed on build host"

log "Upgrading rules on $BUILD"
bssh "sudo -u litmus sh -c 'cd ~/litmus && ./target/release/litmus update-rules'" \
    || die "update-rules failed on build host"

log "Running tests on $BUILD"
bssh "sudo -u litmus sh -c 'cd ~/litmus && RUSTC_WRAPPER=sccache RUSTFLAGS=\"-C link-arg=-fuse-ld=lld\" cargo test --release -- --nocapture'" \
    || die "tests failed on build host"

# --- Transfer tarball via SSH pipe ---

log "Transferring tarball from $BUILD to $RUN"
bssh "sudo cat /home/litmus/litmus/out/litmus.tgz" \
    | rssh "sudo tee /tmp/litmus.tgz >/dev/null"

# --- Run host setup ---

log "Ensuring run user exists on $RUN"
rssh "id -u litmus >/dev/null 2>&1 || sudo useradd -r -s /usr/sbin/nologin -d /nonexistent -c 'Litmus Service' litmus"

log "Installing runtime dependencies on $RUN"
rssh "sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git ca-certificates p7zip-full upx rizin innoextract unrar"

log "Extracting tarball on $RUN"
rssh "sudo rm -rf /usr/local/share/litmus && \
      sudo mkdir -p /usr/local/share/litmus && \
      sudo tar -xzf /tmp/litmus.tgz -C /usr/local/share/litmus && \
      sudo rm -f /tmp/litmus.tgz && \
      sudo ln -sf /usr/local/share/litmus/litmus /usr/local/bin/litmus"

log "Installing systemd unit on $RUN"
rssh "sudo tee /etc/systemd/system/litmus.service >/dev/null" <<'EOF'
[Unit]
Description=Litmus server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=litmus
Group=litmus
ExecStart=/usr/local/share/litmus/litmus -u serve --bind 0.0.0.0:49999
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/litmus.log
StandardError=append:/var/log/litmus.log

# Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
LockPersonality=true

[Install]
WantedBy=multi-user.target
EOF

log "Preparing log file on $RUN"
rssh "sudo touch /var/log/litmus.log && sudo chown litmus:litmus /var/log/litmus.log"

log "Enabling and restarting litmus service on $RUN"
rssh "sudo systemctl daemon-reload && \
      sudo systemctl enable litmus.service && \
      sudo systemctl restart litmus.service"

log "Service status:"
rssh "sudo systemctl --no-pager --full status litmus.service" || true

log "Deployment complete"
