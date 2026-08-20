#!/bin/sh
# cloudflared-linux.sh - Install and supervise a Cloudflare Tunnel fronting the
# Atomdrift Scan server on a systemd host.
#
# The tunnel and its ingress rules are configured in the Cloudflare dashboard
# (Zero Trust -> Networks -> Tunnels); this script only installs the connector
# and points it at the token issued there. Ingress should target the server's
# --bind address, which server-linux.sh keeps on loopback for exactly this
# reason.
#
# First deployment:
#   CF_TUNNEL_TOKEN='...' make deploy-server
#
# Later deployments reuse the stored token, so CF_TUNNEL_TOKEN is only needed
# again when the tunnel is rotated.
#
# The token is written to a root-only file and handed to the connector through
# a systemd credential, so it never reaches argv, the unit file, or the
# service environment — all three of which are readable by any local user.

set -eu

TOKEN_FILE="${CF_TUNNEL_TOKEN_FILE:-/etc/atomdrift/scan/cloudflared-token}"
ORIGIN_URL="${1:-http://127.0.0.1:49999}"

# Deliberately not "cloudflared": Cloudflare's own packaging and
# `cloudflared service install` both claim that unit name. This one belongs to
# the scan deploy, and neither should silently overwrite the other.
SERVICE_NAME=scan-tunnel
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

TOKEN=${CF_TUNNEL_TOKEN:-}
unset CF_TUNNEL_TOKEN

TMP_UNIT=""
TMP_SECRET=""
cleanup() {
    [ -z "$TMP_UNIT" ]   || rm -f "$TMP_UNIT"
    [ -z "$TMP_SECRET" ] || rm -f "$TMP_SECRET"
}
trap cleanup EXIT HUP INT TERM

die() { echo "error: $*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

[ "$(uname -s)" = "Linux" ]          || die "this script is for Linux"
command -v systemctl >/dev/null 2>&1 || die "systemctl not found (systemd required)"

if   command -v doas >/dev/null 2>&1; then SUDO=doas
elif command -v sudo >/dev/null 2>&1; then SUDO=sudo
else die "need doas or sudo"
fi

# A token on the command line wins; otherwise an already-installed token keeps
# the tunnel running across deploys. With neither, there is nothing to connect.
if [ -z "$TOKEN" ] && ! $SUDO test -s "$TOKEN_FILE"; then
    die "no Cloudflare Tunnel token; rerun with CF_TUNNEL_TOKEN set"
fi

case "$TOKEN" in
    *[[:cntrl:]]*) die "CF_TUNNEL_TOKEN must not contain control characters" ;;
esac

if ! command -v cloudflared >/dev/null 2>&1; then
    log "Installing cloudflared"
    if   command -v apt-get >/dev/null 2>&1; then
        $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq
        $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y cloudflared
    elif command -v dnf     >/dev/null 2>&1; then $SUDO dnf install -y cloudflared
    elif command -v yum     >/dev/null 2>&1; then $SUDO yum install -y cloudflared
    elif command -v zypper  >/dev/null 2>&1; then $SUDO zypper --non-interactive install cloudflared
    elif command -v pacman  >/dev/null 2>&1; then $SUDO pacman -Sy --needed --noconfirm cloudflared
    elif command -v xbps-install >/dev/null 2>&1; then $SUDO xbps-install -Sy cloudflared
    else
        die "cloudflared is not installed and no supported package manager was found"
    fi
fi
command -v cloudflared >/dev/null 2>&1 \
    || die "cloudflared installation failed; configure Cloudflare's package repository and retry"
CLOUDFLARED_BIN=$(command -v cloudflared)
case "$CLOUDFLARED_BIN" in
    /usr/bin/cloudflared|/usr/local/bin/cloudflared) ;;
    *) die "cloudflared must be installed at /usr/bin or /usr/local/bin (found $CLOUDFLARED_BIN)" ;;
esac

# --token-file landed in 2025.4.0. Older connectors only accept --token, which
# would put the secret in the process arguments.
VERSION=$("$CLOUDFLARED_BIN" --version | awk '$1 == "cloudflared" && $2 == "version" {print $3; exit}')
YEAR=${VERSION%%.*}
REST=${VERSION#*.}
MONTH=${REST%%.*}
case "$YEAR:$MONTH" in
    *[!0-9:]*|:*|*:) die "could not parse cloudflared version: $VERSION" ;;
esac
if [ "$YEAR" -lt 2025 ] || { [ "$YEAR" -eq 2025 ] && [ "$MONTH" -lt 4 ]; }; then
    die "cloudflared 2025.4.0 or newer is required for --token-file support (have $VERSION)"
fi

# Some distro packages are dynamically linked, so the no-execute sandbox below
# has to keep the system library roots executable alongside the binary itself.
EXEC_PATHS=$CLOUDFLARED_BIN
for libdir in /usr/lib /usr/lib64 /lib /lib64; do
    [ -e "$libdir" ] && EXEC_PATHS="$EXEC_PATHS $libdir"
done

# Restart only on a real change. A connector that is already serving is the one
# thing in this deploy that does not have to blink, and every bounce is a
# window where the edge has no origin at all.
changed=0

if [ -n "$TOKEN" ]; then
    TMP_SECRET=$(mktemp -t scan.cftoken.XXXXXX)
    chmod 0600 "$TMP_SECRET"
    printf '%s\n' "$TOKEN" >"$TMP_SECRET"
    unset TOKEN
    if $SUDO cmp -s "$TMP_SECRET" "$TOKEN_FILE" 2>/dev/null; then
        log "Cloudflare Tunnel token unchanged"
    else
        log "Installing the supplied Cloudflare Tunnel token"
        $SUDO install -d -m 0755 -o root -g root "$(dirname "$TOKEN_FILE")"
        $SUDO install -m 0600 -o root -g root "$TMP_SECRET" "$TOKEN_FILE"
        changed=1
    fi
    rm -f "$TMP_SECRET"
    TMP_SECRET=""
else
    log "Keeping the existing Cloudflare Tunnel token"
fi
$SUDO chown root:root "$TOKEN_FILE"
$SUDO chmod 0600 "$TOKEN_FILE"

TMP_UNIT=$(mktemp -t scan-tunnel.service.XXXXXX)
cat >"$TMP_UNIT" <<EOF
[Unit]
Description=Cloudflare Tunnel for Atomdrift Scan
Documentation=https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/
Wants=network-online.target scan.service
After=network-online.target scan.service
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=notify
NotifyAccess=main
# The token is supplied as a credential rather than an argument or an
# Environment= line: argv is world-readable through ps(1), and unit files are
# world-readable in /etc/systemd/system.
DynamicUser=yes
ExecStart=${CLOUDFLARED_BIN} tunnel --no-autoupdate run --token-file %d/tunnel-token
LoadCredential=tunnel-token:${TOKEN_FILE}
Environment=HOME=/run/${SERVICE_NAME}
RuntimeDirectory=${SERVICE_NAME}
RuntimeDirectoryMode=0700

Restart=on-failure
RestartSec=5
TimeoutStartSec=60
TimeoutStopSec=15
KillSignal=SIGTERM
UMask=0077
TasksMax=256

# The connector needs outbound TCP/UDP, access to the loopback origin, and its
# own notify socket. It needs no privilege and mutates nothing on the host.
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
SystemCallArchitectures=native
SystemCallFilter=@system-service
SystemCallErrorNumber=EPERM

ProtectSystem=strict
ProtectHome=yes
NoExecPaths=/
ExecPaths=${EXEC_PATHS}
PrivateTmp=yes
PrivateDevices=yes
DevicePolicy=closed
PrivateIPC=yes
PrivateMounts=yes
ProtectControlGroups=yes
ProtectKernelModules=yes
ProtectKernelTunables=yes
ProtectKernelLogs=yes
ProtectClock=yes
ProtectHostname=yes
ProtectProc=invisible
ProcSubset=pid
RestrictNamespaces=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
LockPersonality=yes
MemoryDenyWriteExecute=yes
KeyringMode=private
RemoveIPC=yes

[Install]
WantedBy=multi-user.target
EOF

if $SUDO cmp -s "$TMP_UNIT" "$UNIT_FILE" 2>/dev/null; then
    log "${SERVICE_NAME}.service unchanged"
else
    log "Writing ${UNIT_FILE}"
    $SUDO install -m 0644 -o root -g root "$TMP_UNIT" "$UNIT_FILE"
    $SUDO systemctl daemon-reload
    changed=1
fi

$SUDO systemctl enable "${SERVICE_NAME}.service" >/dev/null

if ! $SUDO systemctl is-active --quiet "${SERVICE_NAME}.service"; then
    log "Starting ${SERVICE_NAME}"
elif [ "$changed" -eq 1 ]; then
    log "Restarting ${SERVICE_NAME}"
else
    log "${SERVICE_NAME} already running and unchanged"
    exit 0
fi

if ! $SUDO systemctl restart "${SERVICE_NAME}.service"; then
    $SUDO systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
    $SUDO journalctl -u "${SERVICE_NAME}" -n 50 --no-pager || true
    die "${SERVICE_NAME} failed to start"
fi

# Type=notify means systemd only reports the unit active once cloudflared has
# registered with the edge, so a green unit here is a connected tunnel rather
# than a process that will retry in the background forever.
$SUDO systemctl is-active --quiet "${SERVICE_NAME}.service" \
    || die "${SERVICE_NAME} did not remain active"

log "Cloudflare Tunnel connected (origin: ${ORIGIN_URL})"
