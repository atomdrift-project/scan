#!/bin/sh
# worker-debian.sh - Deploy Atomdrift Scan worker to Debian nodes via SSH
# Usage: ./worker-debian.sh [build-host] [run-host] <url>
#
# build-host / run-host are SSH targets (e.g. "user@host" or an ssh_config alias).
# The same host may be passed for both. Remote user must have passwordless doas or sudo.
#
# Environment overrides:
#   WORKERS            concurrency (--workers)                   (default: worker auto)
#   LLM                OpenAI-compatible LLM endpoint (SCAN_LLM)
#   HOPPER_TOKEN_FILE  hopper API token to install on the run host
#   LLM_TOKEN_FILE     bearer token for the LLM endpoint, which requires one
#                                                                (default: ~/.tok/llm)
#                                                                (default: ~/.tok/hopper)

set -ex

BUILD="${1:-build}"
RUN="${2:-scan}"
URL="$3"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

# Optional: cap concurrent analysis slots (--workers). Unset = worker auto.
WORKERS="${WORKERS:-}"
# LLM second-opinion pass: endpoint (exported as SCAN_LLM) + interpret gate.
# No default here on purpose: the site's failover chain is defined once, in
# the Makefile (LLM ?=), which exports it to every deploy script. Unset leaves
# atomscan's own default.
LLM="${LLM:-}"
worker_args="worker --url $URL --interpret"
[ -n "$WORKERS" ] && worker_args="$worker_args --workers $WORKERS"
# The service account's home on the run host. The unit runs with
# ProtectHome=true, so operator secrets live here rather than under /home, and
# the worker reads ~/.tok/hopper out of it.
STATE_HOME=/var/lib/atomdrift/scan

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

bssh() { ssh -o BatchMode=yes "$BUILD" "$@"; }
rssh() { ssh -o BatchMode=yes "$RUN" "$@"; }

bssh true || die "build host '$BUILD' not accessible"
rssh true || die "run host '$RUN' not accessible"

# Privilege escalation runs on the remote hosts, so pick each host's tool there:
# prefer doas, fall back to sudo.
remote_sudo='if command -v doas >/dev/null 2>&1; then echo doas; elif command -v sudo >/dev/null 2>&1; then echo sudo; fi'
BSUDO=$(bssh "$remote_sudo"); [ -n "$BSUDO" ] || die "need doas or sudo on build host '$BUILD'"
RSUDO=$(rssh "$remote_sudo"); [ -n "$RSUDO" ] || die "need doas or sudo on run host '$RUN'"

# useradd lives in /usr/sbin on Debian, a directory absent from a non-root
# PATH; doas (unlike sudo) forwards the caller's PATH unchanged. Prefixed to
# the useradd calls below so the remote shell expands $PATH on the far host.
# shellcheck disable=SC2016  # $PATH is expanded remotely, not here.
SBIN_PATH='env PATH=/usr/local/sbin:/usr/sbin:/sbin:$PATH'

log "Installing build dependencies on $BUILD"
bssh "$BSUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
      $BSUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cargo rustc git pkg-config build-essential clang lld sccache ca-certificates"

log "Ensuring build user exists on $BUILD"
bssh "id -u scan >/dev/null 2>&1 || $BSUDO $SBIN_PATH useradd -m -s /bin/sh -c 'Atomdrift Scan Build' scan"

log "Syncing source to build host (excluding target/, out/, .git)"
bssh "$BSUDO -u scan mkdir -p /home/scan/scan"
tar -cf - --exclude=./target --exclude=./out --exclude=./.git . \
    | bssh "$BSUDO -u scan tar -xf - -C /home/scan/scan"

log "Building tarball on $BUILD"
bssh "$BSUDO -u scan sh -c 'cd ~/scan && RUSTC_WRAPPER=sccache RUSTFLAGS=\"-C link-arg=-fuse-ld=lld\" make tarball'" \
    || die "build failed on build host"

log "Running tests on $BUILD"
bssh "$BSUDO -u scan sh -c 'cd ~/scan && RUSTC_WRAPPER=sccache RUSTFLAGS=\"-C link-arg=-fuse-ld=lld\" cargo test --release -- --nocapture'" \
    || die "tests failed on build host"

log "Transferring tarball from $BUILD to $RUN"
bssh "$BSUDO cat /home/scan/scan/out/atomscan.tgz" \
    | rssh "$RSUDO tee /tmp/atomscan.tgz >/dev/null"

log "Ensuring run user exists on $RUN"
rssh "id -u scan >/dev/null 2>&1 || $RSUDO $SBIN_PATH useradd -r -s /usr/sbin/nologin -d $STATE_HOME -M -c 'Atomdrift Scan Worker' scan"
rssh "$RSUDO install -d -m 0750 -o scan -g scan $STATE_HOME && \
      $RSUDO install -d -m 0700 -o scan -g scan $STATE_HOME/.tok"

# --- Hopper API token --------------------------------------------------------
#
# Hopper requires `Authorization: Bearer <token>` on every API route, so a
# worker without this file cannot claim work. Streamed from the deploying
# host's ~/.tok/hopper into the run host's service account home over the SSH
# channel — on stdin, never on argv or in a unit file, both of which are
# world-readable on the run host.
HOPPER_TOKEN_SRC="${HOPPER_TOKEN_FILE:-${HOME}/.tok/hopper}"
if [ -s "$HOPPER_TOKEN_SRC" ]; then
    rssh "$RSUDO sh -c 'umask 077; cat > $STATE_HOME/.tok/hopper && \
          chown scan:scan $STATE_HOME/.tok/hopper'" < "$HOPPER_TOKEN_SRC"
    log "Installed hopper API token at $RUN:$STATE_HOME/.tok/hopper"
elif ! rssh "$RSUDO test -s $STATE_HOME/.tok/hopper"; then
    # Not fatal: a hopper deployed without --token-file needs no client token.
    log "WARNING: no hopper API token at $HOPPER_TOKEN_SRC; this worker cannot claim work from an authenticated hopper"
fi

# --- LLM endpoint token ------------------------------------------------------
#
# Our vLLM endpoint requires `Authorization: Bearer <token>`; the worker reads
# it from $HOME/.tok/llm. Streamed over the SSH channel on stdin,
# never on argv or in a unit file, both world-readable on the run host.
#
# Not fatal when absent: every interpret call is refused with 401 and the
# verdict falls back to ML alone. That is silent at runtime, so warn here.
LLM_TOKEN_SRC="${LLM_TOKEN_FILE:-${HOME}/.tok/llm}"
if [ -s "$LLM_TOKEN_SRC" ]; then
    rssh "$RSUDO sh -c 'umask 077; cat > $STATE_HOME/.tok/llm && \
          chown scan:scan $STATE_HOME/.tok/llm'" < "$LLM_TOKEN_SRC"
    log "Installed LLM endpoint token at $RUN:$STATE_HOME/.tok/llm"
elif ! rssh "$RSUDO test -s $STATE_HOME/.tok/llm"; then
    log "WARNING: no LLM token at $LLM_TOKEN_SRC; $LLM will refuse the second-opinion pass with 401"
fi

log "Installing runtime dependencies on $RUN"
rssh "$RSUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
      $RSUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git ca-certificates p7zip-full upx rizin innoextract"

log "Installing binary on $RUN"
rssh "$RSUDO tar -xzf /tmp/atomscan.tgz -C /usr/local/bin && \
      $RSUDO rm -f /tmp/atomscan.tgz"

log "Installing systemd unit on $RUN"
rssh "$RSUDO tee /etc/systemd/system/scan-worker.service >/dev/null" <<EOF
[Unit]
Description=Atomdrift Scan worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=scan
Group=scan
ExecStart=/usr/local/bin/atomscan $worker_args
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/scan-worker.log
StandardError=append:/var/log/scan-worker.log

# OpenAI-compatible endpoint for the --interpret LLM second-opinion pass.
Environment=SCAN_LLM=$LLM
# ProtectHome=true hides the account's real home, and the token is a file, not
# an Environment= value — unit files are world-readable in /etc/systemd/system.
Environment=HOME=$STATE_HOME

# Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$STATE_HOME
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
rssh "$RSUDO touch /var/log/scan-worker.log && $RSUDO chown scan:scan /var/log/scan-worker.log"

log "Enabling and restarting scan-worker service on $RUN"
rssh "$RSUDO systemctl daemon-reload && \
      $RSUDO systemctl enable scan-worker.service && \
      $RSUDO systemctl restart scan-worker.service"

log "Service status:"
rssh "$RSUDO systemctl --no-pager --full status scan-worker.service" || true

log "Deployment complete"
