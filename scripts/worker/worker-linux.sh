#!/bin/sh
# worker-linux.sh - Install Atomdrift Scan worker as a hardened systemd service.
#
# Local install for any systemd-equipped Linux. Packages are installed via the
# host's native manager — apt-get (Debian, Ubuntu, Mint, Pop!_OS, ...),
# dnf/yum (Fedora, RHEL, Rocky, Alma, CentOS), zypper (openSUSE, SLE),
# pacman (Arch, CachyOS, EndeavourOS, Manjaro, ...) or xbps (Void).
# Re-runnable: idempotent. The unit is daemon-reloaded and the service is
# restarted only when the binary or unit file actually changed on disk.
#
# Usage: ./worker-linux.sh <url>
#
# Environment overrides:
#   DATA_DIR    local sample dir shared with hopper           (default: unset → download)
#   WORKERS     concurrency (--workers)                        (default: worker auto)
#   MAX_RSS_GB  pause threshold (--max-rss-gb)                 (default: -1 = off; systemd MemoryMax handles OOM)
#   MEMORY_MAX  systemd MemoryMax= (e.g. 16G, 80%, infinity)     (default: 80%)
#   LLM         OpenAI-compatible LLM endpoint (SCAN_LLM)      (default: http://10.9.8.149:8000/v1)

set -eu

URL="${1:-}"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

SERVICE_USER=scan
SERVICE_NAME=scan-worker
BINARY=atomscan
BIN_PATH=/usr/local/bin/${BINARY}
STATE_HOME=/var/lib/atomdrift/scan
UNIT_FILE=/etc/systemd/system/${SERVICE_NAME}.service

DATA_DIR="${DATA_DIR:-}"
WORKERS="${WORKERS:-}"
MAX_RSS_GB="${MAX_RSS_GB:--1}"
MEMORY_MAX="${MEMORY_MAX:-80%}"
LLM="${LLM:-http://10.9.8.149:8000/v1}"

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

# --- Packages ---------------------------------------------------------------
#
# Detect the host package manager, then install two groups:
#   core  — build toolchain; the build cannot proceed without these.
#   extra — unpacking helpers (7z, upx, innoextract) used during analysis.
#           Installed best-effort: names and availability drift across distros,
#           and a missing unpacker only degrades the worker (it skips that
#           archive format) rather than blocking the build.
# Package names differ per distro, so each manager carries its own spelling.

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
                 --comment "Atomdrift Scan worker" "${SERVICE_USER}"
fi

# Pre-create state dir so an early failure doesn't leave us without one;
# systemd re-asserts ownership/mode on each start via StateDirectory=.
$SUDO install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${STATE_HOME}"

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

# %S is a systemd specifier that expands to /var/lib at unit-load time, so
# --traits-dir resolves to /var/lib/atomdrift/scan/traits inside the namespace.
exec_args="worker --url ${URL} --traits-dir %S/atomdrift/scan/traits --max-rss-gb ${MAX_RSS_GB}"
exec_args="${exec_args} --interpret"
if [ -n "${WORKERS}" ];  then exec_args="${exec_args} --workers ${WORKERS}";   fi
if [ -n "${DATA_DIR}" ]; then exec_args="${exec_args} --data-dir ${DATA_DIR}"; fi

# --- Unit -------------------------------------------------------------------

TMP_UNIT=$(mktemp -t scan-worker.service.XXXXXX)

cat >"$TMP_UNIT" <<EOF
[Unit]
Description=Atomdrift Scan worker (analyses samples claimed from hopper)
Documentation=https://github.com/atomdrift-project/scan
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}

# One writable namespace under /var/lib for all atomdrift tools: the worker's
# HOME is %S/atomdrift/scan, and cleave (and any future tool) caches alongside
# it under %S/atomdrift/<tool>. Declaring the parent keeps the unit stable as
# tools are added.
StateDirectory=atomdrift
StateDirectoryMode=0750

WorkingDirectory=%S/atomdrift/scan
ExecStart=${BIN_PATH} ${exec_args}
Restart=always
RestartSec=10s
TimeoutStopSec=30s

Environment=HOME=%S/atomdrift/scan
# OpenAI-compatible endpoint for the --interpret LLM second-opinion pass.
Environment=SCAN_LLM=${LLM}

# Resource caps. Under systemd we disable the worker's in-process RSS
# throttling (--max-rss-gb=-1) and let MemoryMax do the enforcement: a
# stuck/leaking worker is killed and Restart=always brings it back, instead
# of looping on "memory pressure: pausing" warnings. Override MAX_RSS_GB
# at install time to re-enable in-process throttling.
MemoryMax=${MEMORY_MAX}
TasksMax=4096
# A killed analysis subprocess (rizin OOM, etc.) must not bring the worker down.
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

# --- Migrate from the cron-based deploy ------------------------------------

if crontab -l 2>/dev/null | grep -q "scan worker"; then
    log "Removing legacy cron entry from $(id -un)'s crontab"
    (crontab -l 2>/dev/null | grep -v "scan worker" || true) | crontab -
    log "Stopping any user-owned 'scan worker' processes from the cron era"
    pkill -u "$(id -u)" -f "scan worker" 2>/dev/null || true
fi

# --- Activate ---------------------------------------------------------------

[ "$unit_changed" -eq 1 ] && $SUDO systemctl daemon-reload

# enable --now is idempotent and starts the service on first deploy.
$SUDO systemctl enable --now "${SERVICE_NAME}.service" >/dev/null

if [ "$binary_changed" -eq 1 ] || [ "$unit_changed" -eq 1 ]; then
    log "Restarting ${SERVICE_NAME}"
    if ! $SUDO systemctl restart "${SERVICE_NAME}.service"; then
        $SUDO systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
        die "service failed to start; see: journalctl -u ${SERVICE_NAME} -n 50"
    fi
else
    log "No changes; leaving service running"
fi

$SUDO systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
log "Deployment complete"
