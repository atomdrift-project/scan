#!/bin/sh
# worker-linux.sh - Install Atomdrift Scan worker as a hardened systemd service.
#
# Local install for any systemd-equipped Linux with apt-get (Debian, Ubuntu,
# Mint, Pop!_OS, ...) or pacman (Arch, CachyOS, EndeavourOS, Manjaro, ...).
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

set -eu

URL="${1:-}"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

SERVICE_USER=ascan
SERVICE_NAME=ascan-worker
BINARY=ascan
BIN_PATH=/usr/local/bin/${BINARY}
STATE_HOME=/var/lib/atomdrift/scan
UNIT_FILE=/etc/systemd/system/${SERVICE_NAME}.service

DATA_DIR="${DATA_DIR:-}"
WORKERS="${WORKERS:-}"
MAX_RSS_GB="${MAX_RSS_GB:--1}"
MEMORY_MAX="${MEMORY_MAX:-80%}"

die() { echo "error: $*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

TMP_UNIT=""
trap '[ -n "$TMP_UNIT" ] && rm -f "$TMP_UNIT"' EXIT

# --- Preconditions -----------------------------------------------------------

[ -f Makefile ]                      || die "run from the repository root"
[ "$(uname -s)" = "Linux" ]          || die "this script is for Linux"
command -v systemctl >/dev/null 2>&1 || die "systemctl not found (systemd required)"
command -v sudo      >/dev/null 2>&1 || die "sudo required"
command -v rizin     >/dev/null 2>&1 || die "rizin not found — install from https://rizin.re first"

# --- Packages (apt or pacman) -----------------------------------------------

if command -v apt-get >/dev/null 2>&1; then
    pkgs_needed=""
    for pkg in git pkg-config build-essential clang lld ca-certificates \
               p7zip-full upx-ucl innoextract; do
        dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed" \
            || pkgs_needed="$pkgs_needed $pkg"
    done
    if [ -n "$pkgs_needed" ]; then
        log "Installing missing apt packages:$pkgs_needed"
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
        # shellcheck disable=SC2086
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $pkgs_needed
    else
        log "All apt packages already installed"
    fi
elif command -v pacman >/dev/null 2>&1; then
    # pacman -S --needed is idempotent; -y syncs the package DB so a stale
    # mirror snapshot doesn't cause spurious 'target not found' failures.
    log "Ensuring pacman packages"
    sudo pacman -Sy --needed --noconfirm \
        git pkgconf base-devel clang lld p7zip upx innoextract ca-certificates
else
    die "no supported package manager found (need apt-get or pacman)"
fi

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
    sudo useradd --system --home-dir "${STATE_HOME}" --no-create-home \
                 --shell /usr/sbin/nologin \
                 --comment "Atomdrift Scan worker" "${SERVICE_USER}"
fi

# Pre-create state dir so an early failure doesn't leave us without one;
# systemd re-asserts ownership/mode on each start via StateDirectory=.
sudo install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${STATE_HOME}"

# --- Binary -----------------------------------------------------------------

binary_changed=0
if sudo cmp -s "target/release/${BINARY}" "${BIN_PATH}" 2>/dev/null; then
    log "Binary unchanged"
else
    log "Installing ${BIN_PATH}"
    # install(1) writes-then-renames; safe over a running exe (the kernel pins
    # the inode of the running process).
    sudo install -m 0755 -o root -g root "target/release/${BINARY}" "${BIN_PATH}"
    binary_changed=1
fi

# --- Compose ExecStart ------------------------------------------------------

# %S is a systemd specifier that expands to /var/lib at unit-load time, so
# --traits-dir resolves to /var/lib/atomdrift/scan/traits inside the namespace.
exec_args="worker --url ${URL} --traits-dir %S/atomdrift/scan/traits --max-rss-gb ${MAX_RSS_GB}"
if [ -n "${WORKERS}" ];  then exec_args="${exec_args} --workers ${WORKERS}";   fi
if [ -n "${DATA_DIR}" ]; then exec_args="${exec_args} --data-dir ${DATA_DIR}"; fi

# --- Unit -------------------------------------------------------------------

TMP_UNIT=$(mktemp -t ascan-worker.service.XXXXXX)

cat >"$TMP_UNIT" <<EOF
[Unit]
Description=Atomdrift Scan worker (analyses samples claimed from hopper)
Documentation=https://codeberg.org/atomdrift/scan
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}

StateDirectory=ascan
StateDirectoryMode=0750

WorkingDirectory=%S/atomdrift/scan
ExecStart=${BIN_PATH} ${exec_args}
Restart=always
RestartSec=10s
TimeoutStopSec=30s

Environment=HOME=%S/atomdrift/scan

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
if sudo cmp -s "$TMP_UNIT" "$UNIT_FILE" 2>/dev/null; then
    log "Unit unchanged"
else
    log "Writing ${UNIT_FILE}"
    sudo install -m 0644 -o root -g root "$TMP_UNIT" "$UNIT_FILE"
    unit_changed=1
fi

# --- Migrate from the cron-based deploy ------------------------------------

if crontab -l 2>/dev/null | grep -q "ascan worker"; then
    log "Removing legacy cron entry from $(id -un)'s crontab"
    (crontab -l 2>/dev/null | grep -v "ascan worker" || true) | crontab -
    log "Stopping any user-owned 'ascan worker' processes from the cron era"
    pkill -u "$(id -u)" -f "ascan worker" 2>/dev/null || true
fi

# --- Activate ---------------------------------------------------------------

[ "$unit_changed" -eq 1 ] && sudo systemctl daemon-reload

# enable --now is idempotent and starts the service on first deploy.
sudo systemctl enable --now "${SERVICE_NAME}.service" >/dev/null

if [ "$binary_changed" -eq 1 ] || [ "$unit_changed" -eq 1 ]; then
    log "Restarting ${SERVICE_NAME}"
    if ! sudo systemctl restart "${SERVICE_NAME}.service"; then
        sudo systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
        die "service failed to start; see: journalctl -u ${SERVICE_NAME} -n 50"
    fi
else
    log "No changes; leaving service running"
fi

sudo systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
log "Deployment complete"
