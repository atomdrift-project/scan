#!/bin/sh
# worker-debian.sh - Deploy Atomdrift Scan worker to Debian nodes via SSH
# Usage: ./worker-debian.sh [build-host] [run-host] <url>
#
# build-host / run-host are SSH targets (e.g. "user@host" or an ssh_config alias).
# The same host may be passed for both. Remote user must have passwordless sudo.

set -ex

BUILD="${1:-build}"
RUN="${2:-scan}"
URL="$3"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

# Optional: cap concurrent analysis slots (--workers). Unset = worker auto.
WORKERS="${WORKERS:-}"
# LLM second-opinion pass: endpoint (exported as SCAN_LLM) + interpret gate.
LLM="${LLM:-http://10.9.8.149:8000/v1}"
worker_args="worker --url $URL --interpret"
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
bssh "id -u scan >/dev/null 2>&1 || sudo useradd -m -s /bin/sh -c 'Atomdrift Scan Build' scan"

log "Syncing source to build host (excluding target/, out/, .git)"
bssh "sudo -u scan mkdir -p /home/scan/scan"
tar -cf - --exclude=./target --exclude=./out --exclude=./.git . \
    | bssh "sudo -u scan tar -xf - -C /home/scan/scan"

log "Building tarball on $BUILD"
bssh "sudo -u scan sh -c 'cd ~/scan && RUSTC_WRAPPER=sccache RUSTFLAGS=\"-C link-arg=-fuse-ld=lld\" make tarball'" \
    || die "build failed on build host"

log "Running tests on $BUILD"
bssh "sudo -u scan sh -c 'cd ~/scan && RUSTC_WRAPPER=sccache RUSTFLAGS=\"-C link-arg=-fuse-ld=lld\" cargo test --release -- --nocapture'" \
    || die "tests failed on build host"

log "Transferring tarball from $BUILD to $RUN"
bssh "sudo cat /home/scan/scan/out/scan.tgz" \
    | rssh "sudo tee /tmp/scan.tgz >/dev/null"

log "Ensuring run user exists on $RUN"
rssh "id -u scan >/dev/null 2>&1 || sudo useradd -r -s /usr/sbin/nologin -d /nonexistent -c 'Atomdrift Scan Worker' scan"

log "Installing runtime dependencies on $RUN"
rssh "sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git ca-certificates p7zip-full upx rizin innoextract"

log "Extracting tarball on $RUN"
rssh "sudo rm -rf /usr/local/share/atomdrift/scan && \
      sudo mkdir -p /usr/local/share/atomdrift/scan && \
      sudo tar -xzf /tmp/scan.tgz -C /usr/local/share/atomdrift/scan && \
      sudo rm -f /tmp/scan.tgz && \
      sudo ln -sf /usr/local/share/atomdrift/scan/scan /usr/local/bin/scan"

log "Installing systemd unit on $RUN"
rssh "sudo tee /etc/systemd/system/scan-worker.service >/dev/null" <<EOF
[Unit]
Description=Atomdrift Scan worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=scan
Group=scan
ExecStart=/usr/local/share/atomdrift/scan/scan $worker_args
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/scan-worker.log
StandardError=append:/var/log/scan-worker.log

# OpenAI-compatible endpoint for the --interpret LLM second-opinion pass.
Environment=SCAN_LLM=$LLM

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
rssh "sudo touch /var/log/scan-worker.log && sudo chown scan:scan /var/log/scan-worker.log"

log "Enabling and restarting scan-worker service on $RUN"
rssh "sudo systemctl daemon-reload && \
      sudo systemctl enable scan-worker.service && \
      sudo systemctl restart scan-worker.service"

log "Service status:"
rssh "sudo systemctl --no-pager --full status scan-worker.service" || true

log "Deployment complete"
