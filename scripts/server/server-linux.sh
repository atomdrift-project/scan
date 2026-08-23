#!/bin/sh
# server-linux.sh - Install Atomdrift Scan serve as a hardened systemd service.
#
# Local install for any systemd-equipped Linux. Packages are installed via the
# host's native manager — apt-get (Debian, Ubuntu, Mint, Pop!_OS, ...),
# dnf/yum (Fedora, RHEL, Rocky, Alma, CentOS), zypper (openSUSE, SLE),
# pacman (Arch, CachyOS, EndeavourOS, Manjaro, ...) or xbps (Void).
# Re-runnable: idempotent. The unit is daemon-reloaded and the service is
# restarted only when the binary or unit file actually changed on disk.
#
# This is the Linux counterpart of `make deploy` on FreeBSD (Bastille jails +
# rc.d). One host, one supervised `atomscan serve`. Invoked by `make deploy`
# / `make deploy-server` on Linux.
#
# Usage: ./server-linux.sh
#
# The API requires a bearer token. The token is read from ~/.tok/scan on this
# host — generated on first deploy if absent — and installed into the service
# account's own ~/.tok/scan, which the unit reads at startup. It never passes
# through the command line, the environment, or the unit file. Clients send it
# as `Authorization: Bearer <token>`; only /_/health is exempt. Rotate by
# editing ~/.tok/scan and redeploying.
#
# Environment overrides:
#   TOKEN_SRC   token file to install (empty disables authentication)
#                                                                (default: ~/.tok/scan)
#   BIND        listen address (--bind)                          (default: 127.0.0.1:49999,
#                                                                 i.e. reachable only through
#                                                                 a local tunnel or proxy)
#   ALLOW_CIDR  extra CIDR allow-list (--allow-cidr); empty skips the flag
#                                                                (default: 10.0.0.0/8)
#   WORKERS     concurrency (--workers)                          (default: server auto)
#   ALLOWED_DIRS  comma-separated /analyze-path roots            (default: unset)
#   HOPPER      hopper base URL (--hopper / SCAN_HOPPER)         (default: unset)
#   HOPPER_READ hopper replica for lookups (--hopper-read); reads prefer it and
#               fall back to HOPPER, writes always go to HOPPER  (default: unset)
#   HOPPER_TOKEN_FILE  hopper API token, installed whenever the file exists
#                      (HOPPER need not be set)               (default: ~/.tok/hopper)
#   MAX_RSS_GB  pause threshold (--max-rss-gb)                   (default: -1 = off; systemd MemoryMax handles OOM)
#   MEMORY_MAX  systemd MemoryMax= (e.g. 16G, 80%, infinity)     (default: 80%)
#   LLM / LLM_URL  OpenAI-compatible LLM endpoint or named target (SCAN_LLM)
#                                                                (default: http://10.9.8.149:8000/v1;
#                                                                 `openrouter` → https://openrouter.ai/api/v1)
#   LLM_MODEL      pinned model (SCAN_LLM_MODEL); required for OpenRouter
#   SCAN_LLM_KEY   OpenRouter key if ~/.tok/openrouter is absent
#   CLOUDFLARED    Cloudflare Tunnel: "auto" installs and supervises cloudflared
#                  only when CF_TUNNEL_TOKEN is passed or a token from an
#                  earlier deploy is on disk, so a host reached over the LAN
#                  needs no extra flags. 1 requires it, 0 skips it even with a
#                  token present.                                (default: auto)
#   CF_TUNNEL_TOKEN       tunnel token; needed once, then stored
#   CF_TUNNEL_TOKEN_FILE  where it is stored
#                                          (default: /etc/atomdrift/scan/cloudflared-token)

set -eu

SERVICE_USER=scan
SERVICE_NAME=scan
BINARY=atomscan
BIN_PATH=/usr/local/bin/${BINARY}
STATE_HOME=/var/lib/atomdrift/scan
UNIT_FILE=/etc/systemd/system/${SERVICE_NAME}.service

# `BIND:-` / `MEMORY_MAX:-` treat empty as unset. `ALLOW_CIDR-` / `TOKEN_SRC-`
# (no colon) keep an explicit empty, so operators can disable the CIDR flag with
# ALLOW_CIDR= and — deliberately, on a host they trust — authentication with
# TOKEN_SRC=.
#
# BIND defaults to loopback: the intended exposure is a Cloudflare tunnel (or
# another local proxy) terminating on this host. Set BIND=0.0.0.0:49999 to
# listen on every interface, and pair it with ALLOW_CIDR.
BIND="${BIND:-127.0.0.1:49999}"
ALLOW_CIDR="${ALLOW_CIDR-10.0.0.0/8}"
TOKEN_SRC="${TOKEN_SRC-${HOME}/.tok/scan}"
WORKERS="${WORKERS:-}"
ALLOWED_DIRS="${ALLOWED_DIRS:-}"
HOPPER="${HOPPER:-}"
HOPPER_READ="${HOPPER_READ:-}"
MAX_RSS_GB="${MAX_RSS_GB:--1}"
MEMORY_MAX="${MEMORY_MAX:-80%}"
# Memory the kernel will not reclaim from the server under host-wide pressure.
# Best-effort (MemoryLow, not MemoryMin) so it cannot deadlock the host.
MEMORY_LOW="${MEMORY_LOW:-50%}"
# Scheduling priority. The server is the reason these boxes exist, so it wins
# every CPU and disk contest against anything else on the host.
NICE="${NICE:--20}"
# LLM_URL is an alias for LLM (SCAN_LLM): `local`, `openrouter`, or a base URL.
if [ -z "${LLM:-}" ] && [ -n "${LLM_URL:-}" ]; then
    LLM=$LLM_URL
fi
LLM="${LLM:-http://10.9.8.149:8000/v1}"
LLM_MODEL="${LLM_MODEL:-${SCAN_LLM_MODEL:-}}"
CLOUDFLARED="${CLOUDFLARED:-auto}"
CF_TUNNEL_TOKEN_FILE="${CF_TUNNEL_TOKEN_FILE:-/etc/atomdrift/scan/cloudflared-token}"

die() { echo "error: $*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

TMP_UNIT=""
trap '[ -n "$TMP_UNIT" ] && rm -f "$TMP_UNIT"' EXIT

# --- Preconditions -----------------------------------------------------------

[ -f Makefile ]                      || die "run from the repository root"
[ "$(uname -s)" = "Linux" ]          || die "this script is for Linux"
command -v systemctl >/dev/null 2>&1 || die "systemctl not found (systemd required)"
command -v rizin     >/dev/null 2>&1 || die "rizin not found — install from https://rizin.re first"

# Privilege escalation: prefer doas, fall back to sudo.
if   command -v doas >/dev/null 2>&1; then SUDO=doas
elif command -v sudo >/dev/null 2>&1; then SUDO=sudo
else die "need doas or sudo"
fi

# systemd's StateDirectory= setup rejects a directory that is reached through
# a symlink (for example /var/lib/atomdrift -> /data/atomdrift). Resolve the
# deployment path as root, since the target may not be traversable by the
# invoking user, and use the physical path in the unit below. On hosts without
# the relocation this remains /var/lib/atomdrift/scan.
#
# -m, not -f: the directory is created further down, so on a first deploy the
# path does not exist yet and -f would fail on the missing parent. -m still
# resolves symlinks in the components that do exist.
RESOLVED_STATE_HOME=$($SUDO readlink -m -- "${STATE_HOME}") \
    || die "cannot resolve state directory ${STATE_HOME}"
[ -n "${RESOLVED_STATE_HOME}" ] || die "resolved state directory is empty"
STATE_HOME=${RESOLVED_STATE_HOME}
log "Using state directory: ${STATE_HOME}"

# --- Packages ---------------------------------------------------------------
#
# Detect the host package manager, then install two groups:
#   core  — build toolchain; the build cannot proceed without these.
#   extra — unpacking helpers (7z, upx, innoextract) used during analysis.
#           Installed best-effort: names and availability drift across distros,
#           and a missing unpacker only degrades the server (it skips that
#           archive format) rather than blocking the build.
# Package names differ per distro, so each manager carries its own spelling.
# This block mirrors scripts/worker/worker-linux.sh.

if   command -v apt-get       >/dev/null 2>&1; then PKG=apt
elif command -v dnf           >/dev/null 2>&1; then PKG=dnf
elif command -v yum           >/dev/null 2>&1; then PKG=yum
elif command -v zypper        >/dev/null 2>&1; then PKG=zypper
elif command -v pacman        >/dev/null 2>&1; then PKG=pacman
elif command -v xbps-install  >/dev/null 2>&1; then PKG=xbps
else die "no supported package manager (need apt-get, dnf, yum, zypper, pacman, or xbps)"
fi
log "Using package manager: $PKG"

case "$PKG" in
    apt)    core="git pkg-config build-essential clang lld ca-certificates"
            extra="p7zip-full upx-ucl innoextract" ;;
    dnf)    core="git pkgconf-pkg-config gcc gcc-c++ make clang lld ca-certificates"
            extra="p7zip p7zip-plugins upx innoextract" ;;
    yum)    core="git pkgconfig gcc gcc-c++ make clang lld ca-certificates"
            extra="p7zip p7zip-plugins upx innoextract" ;;
    zypper) core="git pkg-config gcc gcc-c++ make clang lld ca-certificates"
            extra="p7zip upx innoextract" ;;
    pacman) core="git pkgconf base-devel clang lld ca-certificates"
            extra="p7zip upx innoextract" ;;
    xbps)   core="git pkgconf base-devel clang lld ca-certificates"
            extra="p7zip upx innoextract" ;;
esac

# Is a package already installed? Lets us skip a metadata sync on warm re-runs.
pkg_have() {
    case "$PKG" in
        apt)              dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed" ;;
        dnf|yum|zypper)   rpm -q "$1"        >/dev/null 2>&1 ;;
        pacman)           pacman -Qi "$1"    >/dev/null 2>&1 ;;
        xbps)             xbps-query "$1"    >/dev/null 2>&1 ;;
    esac
}

SYNCED=0
pkg_sync() {
    [ "$SYNCED" -eq 1 ] && return 0
    case "$PKG" in
        apt)    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq ;;
        # dnf/yum/zypper refresh metadata on demand; pacman/xbps sync via -y/-S
        # in pkg_install below. Nothing extra to do here.
        *)      : ;;
    esac
    SYNCED=1
}

pkg_install() {
    case "$PKG" in
        apt)    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@" ;;
        dnf)    $SUDO dnf install -y --setopt=install_weak_deps=False "$@" ;;
        yum)    $SUDO yum install -y "$@" ;;
        zypper) $SUDO zypper --non-interactive install --no-recommends "$@" ;;
        # --needed is idempotent; -y syncs the DB so a stale mirror snapshot
        # doesn't cause spurious 'target not found' failures.
        pacman) $SUDO pacman -Sy --needed --noconfirm "$@" ;;
        xbps)   $SUDO xbps-install -Sy "$@" ;;
    esac
}

needed=""
for pkg in $core; do pkg_have "$pkg" || needed="$needed $pkg"; done
if [ -n "$needed" ]; then
    log "Installing core build packages:$needed"
    pkg_sync
    # shellcheck disable=SC2086
    pkg_install $needed || die "failed to install core build packages:$needed"
else
    log "Core build packages already installed"
fi

for pkg in $extra; do
    pkg_have "$pkg" && continue
    pkg_sync
    log "Installing analysis helper: $pkg"
    pkg_install "$pkg" >/dev/null 2>&1 || log "  note: '$pkg' unavailable via $PKG, skipping"
done

# The build proceeds regardless, but warn loudly about any unpacker we lack so
# silently-skipped sample formats are visible in the deploy log.
command -v 7z >/dev/null 2>&1 || command -v 7za >/dev/null 2>&1 || command -v 7zz >/dev/null 2>&1 \
    || log "warning: no 7z/7za/7zz on PATH — packed-archive samples won't unpack"
command -v upx         >/dev/null 2>&1 || log "warning: upx not found — UPX-packed samples won't unpack"
command -v innoextract >/dev/null 2>&1 || log "warning: innoextract not found — Inno Setup installers won't unpack"

# --- Rust toolchain ---------------------------------------------------------

if ! command -v cargo >/dev/null 2>&1; then
    log "Installing Rust toolchain"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path \
        || die "rustup install failed"
    # rustup creates this file at a runtime-dependent home path.
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi

# --- Build (as the invoking user) ------------------------------------------

log "Building"
if command -v sccache >/dev/null 2>&1; then
    RUSTC_WRAPPER=sccache RUSTFLAGS="-C link-arg=-fuse-ld=lld" cargo build --release \
        || die "build failed"
else
    RUSTFLAGS="-C link-arg=-fuse-ld=lld" cargo build --release || die "build failed"
fi

[ -x "target/release/${BINARY}" ] || die "build did not produce target/release/${BINARY}"

# --- Service user + state dir ----------------------------------------------

if ! getent passwd "${SERVICE_USER}" >/dev/null; then
    log "Creating service user '${SERVICE_USER}'"
    $SUDO useradd --system --home-dir "${STATE_HOME}" --no-create-home \
                 --shell /usr/sbin/nologin \
                 --comment "Atomdrift Scan server" "${SERVICE_USER}"
fi

# Pre-create state dir so an early failure doesn't leave us without one. The
# unit uses the canonical path with ReadWritePaths= below; this also avoids
# systemd's StateDirectory= symlink handling, which fails before ExecStart.
$SUDO install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${STATE_HOME}"

# The unit runs as `scan` with ProtectHome=true and HOME under the canonical
# state directory, so operator secrets are copied into the service account's
# own ~/.tok.
$SUDO install -d -m 0700 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${STATE_HOME}/.tok"

# --- API token --------------------------------------------------------------
#
# Installed as a file, never as an argument or an Environment= line: argv is
# world-readable through ps(1), and unit files are world-readable in
# /etc/systemd/system. Redeploying with no source token keeps the installed
# one, so a redeploy can never silently drop authentication.
#
# The token is never held in a shell variable — only paths are — so it cannot
# leak through a trace or an error message.
TOKEN_DST="${STATE_HOME}/.tok/scan"
token_changed=0
if [ -n "${TOKEN_SRC}" ]; then
    if [ ! -s "${TOKEN_SRC}" ] && ! $SUDO test -s "${TOKEN_DST}"; then
        (umask 077; mkdir -p "$(dirname "${TOKEN_SRC}")")
        (umask 077; { head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'; echo; } \
            > "${TOKEN_SRC}")
        [ -s "${TOKEN_SRC}" ] || die "failed to generate a token at ${TOKEN_SRC}"
        log "Generated an API token at ${TOKEN_SRC}"
        log "  clients: curl -H \"Authorization: Bearer \$(cat ${TOKEN_SRC})\" ..."
    fi
    if [ -s "${TOKEN_SRC}" ]; then
        # cmp compares paths, not contents on the command line. A changed token
        # must force a restart below: the server reads it once, at startup, so
        # installing a rotated token without one leaves the old token live.
        $SUDO cmp -s "${TOKEN_SRC}" "${TOKEN_DST}" 2>/dev/null || token_changed=1
        $SUDO install -m 0600 -o "${SERVICE_USER}" -g "${SERVICE_USER}" \
            "${TOKEN_SRC}" "${TOKEN_DST}"
    fi
    $SUDO test -s "${TOKEN_DST}" || die "no API token at ${TOKEN_DST}"
    [ "$token_changed" -eq 1 ] && log "API token installed at ${TOKEN_DST}"
else
    log "TOKEN_SRC is empty — deploying an UNAUTHENTICATED server"
fi

# --- Hopper API token --------------------------------------------------------
#
# Distinct from TOKEN_SRC above: that one authenticates *clients of this
# server*, this one authenticates *this server to hopper*. Hopper requires
# `Authorization: Bearer <token>` on every API route and does not exempt
# loopback, so without it every result renewal is rejected with 401. Installed
# as a file for the same reason as the others: argv is world-readable through
# ps(1), and unit files are world-readable in /etc/systemd/system.
#
# Installed whenever the operator has one, NOT only when HOPPER is set, so
# adding HOPPER= to a later deploy needs nothing else in place. The file is
# inert while --hopper is off. Matches rollout-bastille.sh.
hopper_token_src="${HOPPER_TOKEN_FILE:-${HOME}/.tok/hopper}"
hopper_token_dst="${STATE_HOME}/.tok/hopper"
if [ -s "$hopper_token_src" ]; then
    $SUDO cmp -s "$hopper_token_src" "$hopper_token_dst" 2>/dev/null || token_changed=1
    $SUDO install -m 0600 -o "${SERVICE_USER}" -g "${SERVICE_USER}" \
        "$hopper_token_src" "$hopper_token_dst"
    log "Installed hopper API token at ${hopper_token_dst}"
elif [ -n "${HOPPER}" ] && ! $SUDO test -s "$hopper_token_dst"; then
    # Only worth a warning when there is a hopper to talk to.
    log "WARNING: no hopper API token at ${hopper_token_src}; result renewal on ${HOPPER} will be rejected"
fi

# OpenRouter: copy the operator key into the service home as well.
openrouter_target() {
    case "$LLM" in
        openrouter|https://openrouter.ai/*|http://openrouter.ai/*) return 0 ;;
        *) return 1 ;;
    esac
}
if openrouter_target; then
    [ -n "$LLM_MODEL" ] || die "OpenRouter deploy requires LLM_MODEL= (e.g. qwen/qwen3.8-27b)"
    dst="${STATE_HOME}/.tok/openrouter"
    src="${HOME}/.tok/openrouter"
    if [ -n "${SCAN_LLM_KEY:-}" ]; then
        tmp=$(mktemp)
        printf '%s\n' "$SCAN_LLM_KEY" > "$tmp"
        $SUDO install -m 0600 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "$tmp" "$dst"
        rm -f "$tmp"
    elif [ -s "$src" ]; then
        $SUDO install -m 0600 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "$src" "$dst"
    else
        die "OpenRouter deploy needs a key in $src (or SCAN_LLM_KEY)"
    fi
    log "Installed OpenRouter token at ${dst}"
fi

# --- Binary -----------------------------------------------------------------

binary_changed=0
if $SUDO cmp -s "target/release/${BINARY}" "${BIN_PATH}" 2>/dev/null; then
    log "Binary unchanged"
else
    log "Installing ${BIN_PATH}"
    # install(1) writes-then-renames; safe over a running exe (the kernel pins
    # the inode of the running process).
    $SUDO install -m 0755 -o root -g root "target/release/${BINARY}" "${BIN_PATH}"
    binary_changed=1
fi

# --- Compose ExecStart ------------------------------------------------------

# The server refreshes models and traits at startup, installing into the
# already-created physical state directory. -u forces the refresh even when
# the local copy looks current, matching the FreeBSD rc.d invocation.
exec_args="-u serve --bind ${BIND} --traits-dir ${STATE_HOME}/traits --max-rss-gb ${MAX_RSS_GB}"
exec_args="${exec_args} --interpret"
if [ -n "${ALLOW_CIDR}" ]; then
    exec_args="${exec_args} --allow-cidr ${ALLOW_CIDR}"
fi
if [ -n "${WORKERS}" ]; then
    exec_args="${exec_args} --workers ${WORKERS}"
fi
if [ -n "${ALLOWED_DIRS}" ]; then
    exec_args="${exec_args} --allowed-dirs ${ALLOWED_DIRS}"
fi
if [ -n "${HOPPER}" ]; then
    exec_args="${exec_args} --hopper ${HOPPER}"
fi
# A lookup this worker's index cannot answer is deferred to the corpus, which
# is a read and belongs on a replica rather than on the machine taking every
# write. Set both to get failover: reads prefer HOPPER_READ and fall back to
# HOPPER when it cannot be reached. Set HOPPER alone and reads go wherever
# writes do, which a replica running the write relay handles fine — it just
# has nowhere to fall back to.
if [ -n "${HOPPER_READ}" ]; then
    exec_args="${exec_args} --hopper-read ${HOPPER_READ}"
fi
# Pass the path, never the token. atomscan refuses to start if the file is
# missing or empty, so a lost token fails loudly instead of opening the API.
if [ -n "${TOKEN_SRC}" ]; then
    exec_args="${exec_args} --token-file ${STATE_HOME}/.tok/scan"
fi

# --- Unit -------------------------------------------------------------------

TMP_UNIT=$(mktemp -t scan.service.XXXXXX)
LLM_MODEL_LINE=""
if [ -n "$LLM_MODEL" ]; then
    LLM_MODEL_LINE="Environment=SCAN_LLM_MODEL=${LLM_MODEL}"
fi

cat >"$TMP_UNIT" <<EOF
[Unit]
Description=Atomdrift Scan HTTP classification server
Documentation=https://github.com/atomdrift-project/scan
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}

# The state directory is canonicalized during deployment. StateDirectory=
# cannot be used here because systemd rejects symlinked paths while preparing
# the service; ReadWritePaths= grants the same writable exception to
# ProtectSystem=strict without requiring /var/lib to be the backing mount.
ReadWritePaths=${STATE_HOME}

WorkingDirectory=${STATE_HOME}
ExecStart=${BIN_PATH} ${exec_args}
Restart=always
RestartSec=10s
TimeoutStopSec=30s

Environment=HOME=${STATE_HOME}
# OpenAI-compatible endpoint for the --interpret LLM second-opinion pass.
# Named target 'openrouter' is resolved by the binary.
Environment=SCAN_LLM=${LLM}
${LLM_MODEL_LINE}

# Resource caps. Under systemd we disable the server's in-process RSS
# throttling (--max-rss-gb=-1) and let MemoryMax do the enforcement: a
# stuck/leaking server is killed and Restart=always brings it back, instead
# of looping on 503-from-RSS. Override MAX_RSS_GB at install time to
# re-enable in-process throttling.
MemoryMax=${MEMORY_MAX}
MemoryLow=${MEMORY_LOW}
TasksMax=4096

# OOM priority. Under host-wide memory pressure the kernel picks its victim by
# oom_score_adj; -900 puts the server behind almost everything else but still
# ahead of sshd, whose listener sets itself to -1000. -1000 is deliberately not
# used here: it would make the process ineligible for the OOM killer entirely,
# so hitting MemoryMax= above would wedge the cgroup instead of restarting it.
# systemd-oomd is told to look elsewhere first for the same reason.
OOMScoreAdjust=-900
ManagedOOMPreference=avoid

# Scheduling priority. Nice= is applied by systemd before it drops to
# ${SERVICE_USER}, so no CAP_SYS_NICE is needed in the (empty) bounding set,
# and analysis children (rizin, cleave) inherit it. CPUWeight/IOWeight are the
# cgroup-v2 shares: 10000 is the maximum, ~100x the default 100, so under
# contention the server gets essentially all of the CPU and disk bandwidth.
# Realtime scheduling (SCHED_FIFO/RR) is deliberately not used: a CPU-bound
# analysis at realtime priority can starve sshd and lock the box out.
Nice=${NICE}
CPUWeight=10000
StartupCPUWeight=10000
IOWeight=10000
StartupIOWeight=10000
IOSchedulingClass=best-effort
IOSchedulingPriority=0
# A killed analysis subprocess (rizin OOM, etc.) must not bring the server down.
OOMPolicy=continue

# Filesystem isolation.
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
PrivateDevices=true
PrivateMounts=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectKernelLogs=true
ProtectControlGroups=true
ProtectClock=true
ProtectHostname=true
ProtectProc=invisible
ProcSubset=pid
UMask=0077

# Process hardening. MemoryDenyWriteExecute is intentionally omitted because
# rizin/cleave map analysed binaries with PROT_EXEC.
NoNewPrivileges=true
RestrictSUIDSGID=true
RestrictRealtime=true
RestrictNamespaces=true
LockPersonality=true
SystemCallArchitectures=native
SystemCallFilter=@system-service
CapabilityBoundingSet=
AmbientCapabilities=
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

# Logging.
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

unit_changed=0
if $SUDO cmp -s "$TMP_UNIT" "$UNIT_FILE" 2>/dev/null; then
    log "Unit unchanged"
else
    log "Writing ${UNIT_FILE}"
    $SUDO install -m 0644 -o root -g root "$TMP_UNIT" "$UNIT_FILE"
    unit_changed=1
fi

# --- Activate ---------------------------------------------------------------

[ "$unit_changed" -eq 1 ] && $SUDO systemctl daemon-reload

# enable --now is idempotent and starts the service on first deploy.
$SUDO systemctl enable --now "${SERVICE_NAME}.service" >/dev/null

if [ "$binary_changed" -eq 1 ] || [ "$unit_changed" -eq 1 ] || [ "$token_changed" -eq 1 ]; then
    log "Restarting ${SERVICE_NAME}"
    if ! $SUDO systemctl restart "${SERVICE_NAME}.service"; then
        $SUDO systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
        die "service failed to start; see: journalctl -u ${SERVICE_NAME} -n 50"
    fi
else
    log "No changes; leaving service running"
fi

$SUDO systemctl --no-pager --full status "${SERVICE_NAME}.service" || true

# --- Cloudflare Tunnel (optional) -------------------------------------------
#
# Started only after the server is up: a connector that advertises an origin
# which is not yet serving hands Cloudflare a 502 window on every deploy.
case "$CLOUDFLARED" in
    0|no|NO) want_tunnel=0 ;;
    auto)
        want_tunnel=0
        if [ -n "${CF_TUNNEL_TOKEN:-}" ] || $SUDO test -s "${CF_TUNNEL_TOKEN_FILE}"; then
            want_tunnel=1
        fi
        ;;
    *) want_tunnel=1 ;;
esac

if [ "$want_tunnel" -eq 1 ]; then
    log "Deploying Cloudflare Tunnel"
    CF_TUNNEL_TOKEN_FILE="${CF_TUNNEL_TOKEN_FILE}" \
        ./scripts/server/cloudflared-linux.sh "http://127.0.0.1:${BIND##*:}"
else
    log "Skipping Cloudflare Tunnel (CLOUDFLARED=${CLOUDFLARED})"
fi

BASE="http://127.0.0.1:${BIND##*:}"
# Every route except /_/health wants the bearer token, so fold it into the
# examples rather than printing a header the reader has to paste in by hand.
if [ -n "${TOKEN_SRC}" ]; then
    auth="-H \"Authorization: Bearer \$(cat ${TOKEN_SRC})\""
else
    auth=""
fi

log "Deployment complete"
log "Health:  curl -sS ${BASE}/_/health"
log "SHA256:  curl -sS ${auth} '${BASE}/lookup?sha256=<64-hex-digest>'"
log "PURL:    curl -sS ${auth} '${BASE}/lookup?purl=pkg%3Anpm%2Fleft-pad%401.3.0'"
log "         A stored verdict is a 200 {sha,lvl,eng,why,hits,bloom}; nothing"
log "         stored is a 404 {\"error\":\"unknown sample\",\"bloom\":…}, where"
log "         bloom is skip|known-bad|conflicted|unknown from the published"
log "         filters. Lookups never analyze — send bytes or a PURL for that."
log "Analyze: curl -sS ${auth} -H 'Content-Type: application/json' \\"
log "              -d '{\"purl\":\"pkg:npm/left-pad@1.3.0\"}' ${BASE}/analyze-purl"
log "         curl -sS ${auth} -F file=@sample.bin ${BASE}/analyze"
if [ "$want_tunnel" -eq 1 ]; then
    log "Tunnel: systemctl status scan-tunnel"
fi
