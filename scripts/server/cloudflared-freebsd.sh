#!/bin/sh
# cloudflared-freebsd.sh - Install and supervise a Cloudflare Tunnel fronting the
# Atomdrift Scan server on a native FreeBSD host (no jail).
#
# The tunnel and its ingress rules are configured in the Cloudflare dashboard
# (Zero Trust -> Networks -> Tunnels); this script only installs the connector
# and points it at the token issued there. Ingress should target the server's
# --bind address, which server-freebsd.sh keeps on loopback for exactly this
# reason.
#
# First deployment:
#   CF_TUNNEL_TOKEN='...' make deploy-server
#
# Later deployments reuse the stored token, so CF_TUNNEL_TOKEN is only needed
# again when the tunnel is rotated.
#
# The token is written to a root-owned file the connector's own account can
# read, and handed to cloudflared with --token-file, so it never reaches argv
# or rc.conf — both of which are readable by any local user.
#
# Usage: ./cloudflared-freebsd.sh [origin-url]

set -eu

TOKEN_FILE="${CF_TUNNEL_TOKEN_FILE:-/usr/local/etc/atomdrift/cloudflared-token}"
ORIGIN_URL="${1:-http://127.0.0.1:49999}"

# Deliberately not named "cloudflared": net/cloudflared ships its own
# rc.d/cloudflared, which runs the connector as root off a config file and is
# rewritten by every pkg upgrade. This is a separate service so neither one
# silently overwrites the other.
SERVICE_NAME=scan_tunnel
RCD_FILE=/usr/local/etc/rc.d/${SERVICE_NAME}
TUNNEL_USER=cloudflared
TUNNEL_HOME=/var/db/cloudflared
TUNNEL_LOG=/var/log/${SERVICE_NAME}.log

# The token must never reach a trace, argv, or rc.conf. Capture it here and
# unset it, then move it into place through the filesystem.
TOKEN=${CF_TUNNEL_TOKEN:-}
unset CF_TUNNEL_TOKEN

TMP_RCD=""
TMP_SECRET=""
cleanup() {
	[ -z "$TMP_RCD" ]    || rm -f "$TMP_RCD"
	[ -z "$TMP_SECRET" ] || rm -f "$TMP_SECRET"
}
trap cleanup EXIT HUP INT TERM

die() { echo "error: $*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

[ "$(uname -s)" = "FreeBSD" ] || die "this script is for FreeBSD"

if command -v doas >/dev/null 2>&1; then
	SUDO=doas
elif command -v sudo >/dev/null 2>&1; then
	SUDO=sudo
else
	die "need doas or sudo"
fi

# A token on the command line wins; otherwise an already-installed token keeps
# the tunnel running across deploys. With neither, there is nothing to connect.
if [ -z "$TOKEN" ] && ! $SUDO test -s "$TOKEN_FILE"; then
	die "no Cloudflare Tunnel token; rerun with CF_TUNNEL_TOKEN set"
fi

case "$TOKEN" in
	*[[:cntrl:]]*) die "CF_TUNNEL_TOKEN must not contain control characters" ;;
esac

if ! pkg info -e cloudflared >/dev/null 2>&1; then
	log "Installing cloudflared"
	$SUDO pkg install -y cloudflared
fi
CLOUDFLARED_BIN=/usr/local/bin/cloudflared
[ -x "$CLOUDFLARED_BIN" ] || die "cloudflared is not installed at $CLOUDFLARED_BIN"

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

log "Ensuring cloudflared service user exists"
id -u "$TUNNEL_USER" >/dev/null 2>&1 || \
	$SUDO pw useradd "$TUNNEL_USER" -d "$TUNNEL_HOME" -s /usr/sbin/nologin \
		-c "Cloudflare Tunnel"
$SUDO install -d -m 0700 -o "$TUNNEL_USER" -g "$TUNNEL_USER" "$TUNNEL_HOME"
$SUDO install -d -m 0755 -o root -g wheel "$(dirname "$TOKEN_FILE")"

# Restart only on a real change: a connector that is already serving is the one
# thing here that does not have to blink, and every bounce is a window where
# the edge has no origin at all.
changed=0

if [ -n "$TOKEN" ]; then
	TMP_SECRET=$(mktemp -t scan.cftoken.XXXXXX)
	chmod 0600 "$TMP_SECRET"
	printf '%s\n' "$TOKEN" >"$TMP_SECRET"
	TOKEN=""
	if $SUDO cmp -s "$TMP_SECRET" "$TOKEN_FILE" 2>/dev/null; then
		log "Cloudflare Tunnel token unchanged"
	else
		log "Installing the supplied Cloudflare Tunnel token"
		$SUDO install -m 0640 -o root -g "$TUNNEL_USER" "$TMP_SECRET" "$TOKEN_FILE"
		changed=1
	fi
	rm -f "$TMP_SECRET"
	TMP_SECRET=""
else
	log "Keeping the existing Cloudflare Tunnel token"
fi
# root:cloudflared 0640 — writable only by root, readable by the connector
# after daemon(8) drops privileges, opaque to everyone else on the host.
$SUDO chown "root:$TUNNEL_USER" "$TOKEN_FILE"
$SUDO chmod 0640 "$TOKEN_FILE"

TMP_RCD=$(mktemp -t scan_tunnel.rcd.XXXXXX)
cat >"$TMP_RCD" <<EOF
#!/bin/sh

# PROVIDE: $SERVICE_NAME
# REQUIRE: LOGIN DAEMON NETWORKING scan
# KEYWORD: shutdown

. /etc/rc.subr

name="$SERVICE_NAME"
rcvar="${SERVICE_NAME}_enable"

load_rc_config \$name

: \${${SERVICE_NAME}_enable:="NO"}
: \${${SERVICE_NAME}_user:="$TUNNEL_USER"}
: \${${SERVICE_NAME}_token_file:="$TOKEN_FILE"}
: \${${SERVICE_NAME}_logfile:="$TUNNEL_LOG"}

pidfile="/var/run/\${name}.pid"
command="/usr/sbin/daemon"
# The token is read from a file rather than passed as an argument so it never
# appears in ps(1) output. -r -R 5 supervises the connector: a crash or a lost
# edge connection is restarted, paced 5s apart.
command_args="-c -f -r -R 5 -P \${pidfile} -o \${${SERVICE_NAME}_logfile} -u \${${SERVICE_NAME}_user} /usr/bin/env HOME=$TUNNEL_HOME $CLOUDFLARED_BIN tunnel --no-autoupdate run --token-file \${${SERVICE_NAME}_token_file}"

run_rc_command "\$1"
EOF

if $SUDO cmp -s "$TMP_RCD" "$RCD_FILE" 2>/dev/null; then
	log "rc.d/$SERVICE_NAME unchanged"
else
	log "Writing $RCD_FILE"
	$SUDO install -m 0755 -o root -g wheel "$TMP_RCD" "$RCD_FILE"
	changed=1
fi

$SUDO sysrc "${SERVICE_NAME}_enable=YES" >/dev/null
# The port's own connector would otherwise come up alongside ours at boot, as
# root, against a config file nothing here maintains.
$SUDO sysrc cloudflared_enable=NO >/dev/null

$SUDO sh -c "[ -e '$TUNNEL_LOG' ] || install -m 0640 -o $TUNNEL_USER -g wheel /dev/null '$TUNNEL_LOG'"

# daemon(8) appends to the log, so a "Registered" line from an earlier deploy is
# still sitting in it. Remember where this run starts and read only past it.
LOG_OFFSET=$($SUDO stat -f %z "$TUNNEL_LOG" 2>/dev/null || echo 0)

if ! $SUDO service "$SERVICE_NAME" status >/dev/null 2>&1; then
	log "Starting $SERVICE_NAME"
	$SUDO service "$SERVICE_NAME" start
elif [ "$changed" -eq 1 ]; then
	log "Restarting $SERVICE_NAME"
	$SUDO service "$SERVICE_NAME" restart
else
	log "$SERVICE_NAME already running and unchanged"
	exit 0
fi

# A connector that cannot reach Cloudflare retries forever in the background,
# so a silent start says nothing about whether the tunnel is actually serving.
log "Waiting for the tunnel to register a connection"
registered=0
for _ in $(jot 30 1); do
	if $SUDO tail -c "+$((LOG_OFFSET + 1))" "$TUNNEL_LOG" 2>/dev/null \
		| grep -q "Registered tunnel connection"; then
		registered=1
		break
	fi
	sleep 1
done
if [ "$registered" -ne 1 ]; then
	$SUDO tail -n 50 "$TUNNEL_LOG" >&2 || true
	die "cloudflared did not register a tunnel connection within 30 seconds"
fi

log "Cloudflare Tunnel connected (origin: $ORIGIN_URL)"
