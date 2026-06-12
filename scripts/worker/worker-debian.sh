#!/bin/sh
# worker-debian.sh - Deploy Atomdrift Scan worker to Debian nodes via SSH
# Usage: ./worker-debian.sh [build-host] [run-host] <url>
#
# build-host / run-host are SSH targets (e.g. "user@host" or an ssh_config alias).
# The same host may be passed for both. Remote user must have passwordless sudo.

set -ex

BUILD="${1:-build}"
RUN="${2:-ascan}"
URL="$3"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

# Optional: cap concurrent analysis slots (--workers). Unset = worker auto.
WORKERS="${WORKERS:-}"
worker_args="worker --url $URL"
[ -n "$WORKERS" ] && worker_args="$worker_args --workers $WORKERS"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

bssh() { ssh -o BatchMode=yes "$BUILD" "$@"; }
rssh() { ssh -o BatchMode=yes "$RUN" "$@"; }

bssh true || die "build host '$BUILD' not accessible"
rssh true || die "run host '$RUN' not accessible"

log "Installing build dependencies on $BUILD"
bssh "sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cargo rustc git pkg-config build-essential clang lld sccache ca-certificates"

log "Ensuring build user exists on $BUILD"
bssh "id -u ascan >/dev/null 2>&1 || sudo useradd -m -s /bin/sh -c 'Atomdrift Scan Build' ascan"

log "Syncing source to build host (excluding target/, out/, .git)"
bssh "sudo -u ascan mkdir -p /home/ascan/scan"
tar -cf - --exclude=./target --exclude=./out --exclude=./.git . \
    | bssh "sudo -u ascan tar -xf - -C /home/ascan/scan"

log "Building tarball on $BUILD"
bssh "sudo -u ascan sh -c 'cd ~/scan && RUSTC_WRAPPER=sccache RUSTFLAGS=\"-C link-arg=-fuse-ld=lld\" make tarball'" \
    || die "build failed on build host"

log "Running tests on $BUILD"
bssh "sudo -u ascan sh -c 'cd ~/scan && RUSTC_WRAPPER=sccache RUSTFLAGS=\"-C link-arg=-fuse-ld=lld\" cargo test --release -- --nocapture'" \
    || die "tests failed on build host"

log "Transferring tarball from $BUILD to $RUN"
bssh "sudo cat /home/ascan/scan/out/ascan.tgz" \
    | rssh "sudo tee /tmp/ascan.tgz >/dev/null"

log "Ensuring run user exists on $RUN"
rssh "id -u ascan >/dev/null 2>&1 || sudo useradd -r -s /usr/sbin/nologin -d /nonexistent -c 'Atomdrift Scan Worker' ascan"

log "Installing runtime dependencies on $RUN"
rssh "sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git ca-certificates p7zip-full upx rizin innoextract"

log "Extracting tarball on $RUN"
rssh "sudo rm -rf /usr/local/share/atomdrift/scan && \
      sudo mkdir -p /usr/local/share/atomdrift/scan && \
      sudo tar -xzf /tmp/ascan.tgz -C /usr/local/share/atomdrift/scan && \
      sudo rm -f /tmp/ascan.tgz && \
      sudo ln -sf /usr/local/share/atomdrift/scan/ascan /usr/local/bin/ascan"

log "Installing systemd unit on $RUN"
rssh "sudo tee /etc/systemd/system/ascan-worker.service >/dev/null" <<EOF
[Unit]
Description=Atomdrift Scan worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ascan
Group=ascan
ExecStart=/usr/local/share/atomdrift/scan/ascan $worker_args
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/ascan-worker.log
StandardError=append:/var/log/ascan-worker.log

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
rssh "sudo touch /var/log/ascan-worker.log && sudo chown ascan:ascan /var/log/ascan-worker.log"

log "Enabling and restarting ascan-worker service on $RUN"
rssh "sudo systemctl daemon-reload && \
      sudo systemctl enable ascan-worker.service && \
      sudo systemctl restart ascan-worker.service"

log "Service status:"
rssh "sudo systemctl --no-pager --full status ascan-worker.service" || true

log "Deployment complete"
