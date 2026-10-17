#!/bin/sh
# rollout-bastille.sh - Deploy Atomdrift Scan using separate build and run jails
# Usage: ./rollout-bastille.sh [build-jail] [run-jail]
#
# Prerequisites:
#   - "interpret" must have an entry in /etc/hosts on this host, pointing at the
#     OpenAI-compatible LLM endpoint (vLLM) that --interpret sends samples to.
#     The entry is copied into the run jail, which has no resolver of its own.
#   - The endpoint requires a bearer token. LLM_TOKEN_FILE (default ~/.tok/llm)
#     is installed into the jail whenever it exists; without it the jail serves
#     on ML alone, since every interpret call is refused with 401.
#
# Hopper (optional):
#   HOPPER        hopper base URL. When set, the server renews every analyzed
#                 result on <HOPPER>/api/result (`serve --hopper`); stored in
#                 the jail's rc.conf as scan_hopper.        (default: unset)
#   HOPPER_TOKEN_FILE  hopper API token, installed into the jail whenever the
#                 file exists — HOPPER need not be set, so turning renewal on
#                 later needs nothing else.                 (default: ~/.tok/hopper)
#   IDLE          analysis slots the embedded idle worker may spend on hopper
#                 queue work (`serve --idle-worker-slots`); 0 disables
#                 background claiming entirely. Stored in the jail's rc.conf as
#                 scan_idle_slots; capped at half the slots by the server, and
#                 inert without HOPPER.       (default: unset = half the slots)
#
# Cloudflare Tunnel (optional):
#   CLOUDFLARED   "auto" installs and supervises a connector in the run jail
#                 only when CF_TUNNEL_TOKEN is passed or a token from an
#                 earlier deploy is already in the jail. 1 requires it, 0 skips
#                 it even with a token present.                 (default: auto)
#   CF_TUNNEL_TOKEN  tunnel token; needed once, then stored in the jail at
#                    /usr/local/etc/atomdrift/cloudflared-token

set -ex
# FreeBSD /bin/sh supports pipefail; this deploy script only runs on FreeBSD.
# shellcheck disable=SC3040
set -o pipefail

BUILD="${1:-build}"
RUN="${2:-scan}"
CLOUDFLARED="${CLOUDFLARED:-auto}"
# Not a secret, so unlike the tokens it may live in the jail's rc.conf; the
# rc.d below reads it as $scan_hopper.
HOPPER="${HOPPER:-}"
# Also not a secret, so it too lives in the jail's rc.conf, read below as
# $scan_idle_slots. Empty means "unset" — the server's own default applies —
# while IDLE=0 is a real value that must reach --idle-worker-slots as "0".
IDLE="${IDLE:-}"

# The tunnel token must never reach the `set -x` trace, argv, or the jail's
# rc.conf. Capture it here, unset it, and move it into the jail through the
# filesystem — the same discipline the API token below uses.
set +x
TUNNEL_TOKEN=${CF_TUNNEL_TOKEN:-}
unset CF_TUNNEL_TOKEN
set -x

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
# shellcheck source=scripts/server/lib/freebsd-rcd.sh
. "$SCRIPT_DIR/lib/freebsd-rcd.sh"

TUNNEL_SERVICE=scan_tunnel
TUNNEL_TOKEN_JAIL_PATH=/usr/local/etc/atomdrift/cloudflared-token
TUNNEL_USER=cloudflared
TUNNEL_HOME=/var/db/cloudflared
TUNNEL_LOG=/var/log/${TUNNEL_SERVICE}.log

TUNNEL_TMP=""
cleanup() {
    [ -z "$TUNNEL_TMP" ] || rm -f "$TUNNEL_TMP"
}
trap cleanup EXIT HUP INT TERM

die() {
    echo "error: $*" >&2
    exit 1
}

log() {
    echo "==> $*"
}

# Printed at every exit path so a deploy always ends with the two lookups
# people actually reach for. Commands are host-side: the jail has no route of
# its own here, and the token file is only readable by root/scan inside it.
usage_hints() {
    # The whole command runs inside the jail: the host has no route to the
    # jail's loopback, and the token file is only readable in there.
    _j="doas bastille cmd $RUN sh -c"
    _h="-H \"Authorization: Bearer \$(cat /home/scan/.tok/scan)\""
    _u="http://127.0.0.1:49999"
    log "Health:  $_j 'curl -sS $_u/_/health'"
    log "SHA256:  $_j 'curl -sS $_h \"$_u/lookup?sha256=<64-hex-digest>\"'"
    log "PURL:    $_j 'curl -sS $_h \"$_u/lookup?purl=pkg%3Anpm%2Fleft-pad%401.3.0\"'"
    log "         A stored verdict is a 200 {sha,lvl,eng,why,hits,bloom};"
    log "         nothing stored is a 404 {error:unknown sample,bloom:…}, where"
    log "         bloom is skip|known-bad|conflicted|unknown from the published"
    log "         filters. Lookups never analyze — send bytes or a PURL."
    log "Analyze: $_j 'curl -sS $_h -H \"Content-Type: application/json\""
    log "               -d \"{\\\"purl\\\":\\\"pkg:npm/left-pad@1.3.0\\\"}\" $_u/analyze-purl'"
    log "         $_j 'curl -sS $_h -F file=@sample.bin $_u/analyze'"
}

install_missing_build_packages() {
    set --
    for pkg in rust git pkgconf mold gmake; do
        if ! doas bastille cmd "$BUILD" pkg info -e "$pkg" >/dev/null 2>&1; then
            set -- "$@" "$pkg"
        fi
    done
    if [ "$#" -gt 0 ]; then
        doas bastille pkg "$BUILD" install -y "$@"
    fi
}

case "$IDLE" in
    '') ;;
    *[!0-9]*) die "IDLE must be a non-negative integer (got '$IDLE')" ;;
esac

# Verify jails are accessible
doas bastille cmd "$BUILD" true || die "build jail '$BUILD' not accessible"
doas bastille cmd "$RUN" true || die "run jail '$RUN' not accessible"

# --- Build jail setup ---

log "Ensuring build user exists"
doas bastille cmd "$BUILD" id -u scan >/dev/null 2>&1 || \
    doas bastille cmd "$BUILD" pw useradd scan -m -s /bin/sh -c "Atomdrift Scan Build"

log "Installing build dependencies"
install_missing_build_packages

log "Syncing source to build jail (preserving target/)"
doas bastille cmd "$BUILD" su -l scan -c "mkdir -p ~/scan"
tar -cf - --exclude=./target --exclude=./out --exclude=./.git . \
    | doas bastille cmd "$BUILD" su -l scan -c "tar -xf - -C ~/scan"

log "Killing any stale cargo processes in build jail"
doas bastille cmd "$BUILD" su -l scan -c "killall cargo 2>/dev/null || true"

log "Building tarball"
doas bastille cmd "$BUILD" su -l scan -c "cd ~/scan && RUSTFLAGS='-C link-arg=-fuse-ld=mold' gmake tarball" \
    || die "build failed in build jail"

log "Running tests"
doas bastille cmd "$BUILD" su -l scan -c "cd ~/scan && RUSTFLAGS='-C link-arg=-fuse-ld=mold' cargo test --release -- --nocapture" \
    || die "tests failed in build jail"

# --- Transfer tarball via jail filesystem ---

log "Transferring tarball to run jail"
BASTILLE_DIR="/usr/local/bastille/jails"
doas cp "$BASTILLE_DIR/$BUILD/root/home/scan/scan/out/atomscan.tgz" \
       "$BASTILLE_DIR/$RUN/root/tmp/atomscan.tgz"

# --- Run jail setup ---

log "Ensuring run user exists"
doas bastille cmd "$RUN" id -u scan >/dev/null 2>&1 || \
    doas bastille cmd "$RUN" pw useradd scan -m -s /bin/sh -c "Atomdrift Scan Service"

log "Installing runtime dependencies"
doas bastille pkg "$RUN" install -y git 7-zip upx rizin innoextract

log "Installing binary"
doas bastille cmd "$RUN" tar -xzf /tmp/atomscan.tgz -C /usr/local/bin
doas bastille cmd "$RUN" rm -f /tmp/atomscan.tgz

log "Refreshing models and traits in run jail"
doas bastille cmd "$RUN" su -l scan -c "atomscan update-rules" \
    || die "update-rules failed in run jail"

# --- Resolve the interpret (LLM) hostname ---
# --interpret posts to an OpenAI-compatible endpoint at http://interpret:8000/v1.
# The run jail has no resolver, so "interpret" only resolves via its /etc/hosts;
# copy the deploy host's entry in. Missing is non-fatal: atomscan still serves,
# and the LLM second opinion is simply skipped when the endpoint is unreachable.
INTERPRET_HOST="${INTERPRET_HOST:-interpret}"
INTERPRET_LINE=$(awk -v h="$INTERPRET_HOST" '$0 ~ "[[:space:]]" h "([[:space:]]|$)" {print; exit}' /etc/hosts)
if [ -z "$INTERPRET_LINE" ]; then
    log "WARNING: $INTERPRET_HOST not found in /etc/hosts — --interpret will have no endpoint to reach"
else
    log "Using $INTERPRET_HOST: $INTERPRET_LINE"
    if ! doas bastille cmd "$RUN" awk -v h="$INTERPRET_HOST" '$0 ~ "[[:space:]]" h "([[:space:]]|$)" {found=1} END{exit !found}' /etc/hosts 2>/dev/null; then
        doas bastille cmd "$RUN" sh -c "echo '$INTERPRET_LINE' >> /etc/hosts"
        log "Added $INTERPRET_HOST to jail /etc/hosts"
    fi
fi

# --- API token --------------------------------------------------------------
#
# The API requires `Authorization: Bearer <token>` on every route except
# /_/health. The token lives in a file: never an argument (argv is visible in
# ps(1)) and never an environment variable. It reaches the jail through the
# filesystem rather than as a command argument, so the `set -x` trace above
# cannot echo it — and it is never held in a shell variable, for the same
# reason. Generated on this host on first deploy; rotate by editing
# $TOKEN_SRC and redeploying.
TOKEN_SRC="${TOKEN_SRC:-${HOME}/.tok/scan}"
TOKEN_DST="$BASTILLE_DIR/$RUN/root/home/scan/.tok/scan"

log "Installing API token"
if [ ! -s "$TOKEN_SRC" ] && ! doas test -s "$TOKEN_DST"; then
    (umask 077; mkdir -p "$(dirname "$TOKEN_SRC")")
    (umask 077; { head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'; echo; } > "$TOKEN_SRC")
    [ -s "$TOKEN_SRC" ] || die "failed to generate a token at $TOKEN_SRC"
    log "Generated an API token at $TOKEN_SRC"
    log "  clients: curl -H \"Authorization: Bearer \$(cat $TOKEN_SRC)\" ..."
fi
doas bastille cmd "$RUN" install -d -m 0700 -o scan -g scan /home/scan/.tok
# No source token means a redeploy of a host that already has one: keep it,
# rather than silently dropping authentication.
if [ -s "$TOKEN_SRC" ]; then
    doas install -m 0600 "$TOKEN_SRC" "$TOKEN_DST"
    doas bastille cmd "$RUN" chown scan:scan /home/scan/.tok/scan
    doas bastille cmd "$RUN" chmod 0600 /home/scan/.tok/scan
fi
doas test -s "$TOKEN_DST" || die "no API token at $TOKEN_DST"

# --- Hopper API token --------------------------------------------------------
#
# Distinct from the API token above: that one authenticates *clients of this
# server*, this one authenticates *this server to hopper*. Hopper requires
# `Authorization: Bearer <token>` on every API route and does not exempt
# loopback, so without it every result renewal is rejected with 401. Moved
# through the filesystem, never as an argument, so the `set -x` trace above
# cannot echo it.
#
# Installed whenever the operator has one, NOT only when HOPPER is set: the URL
# lives in the jail's rc.conf and can be switched on later with `sysrc
# scan_hopper=`, and a server that gains --hopper without the token 401s on
# every renewal. The file is inert while --hopper is off.
HOPPER_TOKEN_SRC="${HOPPER_TOKEN_FILE:-${HOME}/.tok/hopper}"
HOPPER_TOKEN_DST="$BASTILLE_DIR/$RUN/root/home/scan/.tok/hopper"
if [ -s "$HOPPER_TOKEN_SRC" ]; then
    log "Installing hopper API token"
    doas install -m 0600 "$HOPPER_TOKEN_SRC" "$HOPPER_TOKEN_DST"
    doas bastille cmd "$RUN" chown scan:scan /home/scan/.tok/hopper
    doas bastille cmd "$RUN" chmod 0600 /home/scan/.tok/hopper
elif [ -n "$HOPPER" ] && ! doas test -s "$HOPPER_TOKEN_DST"; then
    # Only worth a warning when there is a hopper to talk to.
    log "WARNING: no hopper API token at $HOPPER_TOKEN_SRC; result renewal on $HOPPER will be rejected"
fi

# --- LLM endpoint token ------------------------------------------------------
#
# The interpret endpoint requires `Authorization: Bearer <token>`; atomscan
# reads it from $HOME/.tok/llm inside the jail. Moved through the filesystem
# like the tokens above, so the `set -x` trace cannot echo it.
#
# Installed whenever the operator has one. Without it the jail still serves —
# the second-opinion pass just 401s and every verdict falls back to ML alone,
# which is silent at runtime, so warn here where it is actionable.
LLM_TOKEN_SRC="${LLM_TOKEN_FILE:-${HOME}/.tok/llm}"
LLM_TOKEN_DST="$BASTILLE_DIR/$RUN/root/home/scan/.tok/llm"
if [ -s "$LLM_TOKEN_SRC" ]; then
    log "Installing LLM endpoint token"
    doas install -m 0600 "$LLM_TOKEN_SRC" "$LLM_TOKEN_DST"
    doas bastille cmd "$RUN" chown scan:scan /home/scan/.tok/llm
    doas bastille cmd "$RUN" chmod 0600 /home/scan/.tok/llm
elif ! doas test -s "$LLM_TOKEN_DST"; then
    log "WARNING: no LLM token at $LLM_TOKEN_SRC; the interpret endpoint will reject the second-opinion pass with 401"
fi

log "Creating rc.d service"
doas bastille cmd "$RUN" mkdir -p /usr/local/etc/rc.d

# The service definition is shared with the native host deploy
# (scripts/server/server-freebsd.sh) through lib/freebsd-rcd.sh, so daemon(8)
# supervision, jemalloc tuning, scheduling priority and the bounded stop stay
# identical in the jail and on a bare host.
#
# 0.0.0.0 is the jail's own address space, so binding every interface here
# exposes only the jail; the CIDR allow-list is what actually gates callers.
# The LLM endpoint is the "interpret" entry copied into the jail's /etc/hosts
# above — the jail has no resolver of its own. --hopper and
# --idle-worker-slots are not baked in: they come from the jail's rc.conf,
# written with sysrc below.
scan_server_rcd_script /usr/local/bin/atomscan \
    "$(scan_server_args "0.0.0.0:49999" "10.0.0.0/8" "/home/scan/.tok/scan")" \
    "http://interpret:8000/v1" "" "/home/scan" \
    | doas bastille cmd "$RUN" tee /usr/local/etc/rc.d/scan >/dev/null

doas bastille cmd "$RUN" chmod 755 /usr/local/etc/rc.d/scan

log "Enabling and restarting scan service"
doas bastille sysrc "$RUN" scan_enable=YES
# Written unconditionally, empty when HOPPER is not set, so dropping the
# variable actually turns renewal off instead of silently keeping the previous
# target — and rc.conf shows that it is off on purpose. Empty rather than
# `sysrc -x`: the rc.d treats both the same, and this needs no flag forwarded
# through `bastille sysrc`.
doas bastille sysrc "$RUN" scan_hopper="$HOPPER"
# Same reasoning: written unconditionally, empty when IDLE is unset, so
# dropping the variable restores the server default instead of silently keeping
# the previous cap.
doas bastille sysrc "$RUN" scan_idle_slots="$IDLE"
doas bastille service "$RUN" scan stop 2>/dev/null || true
doas bastille cmd "$RUN" pkill -9 -F /var/run/scan.pid 2>/dev/null || true
doas bastille service "$RUN" scan start

# --- Cloudflare Tunnel (optional) -------------------------------------------
#
# The tunnel and its ingress rules are configured in the Cloudflare dashboard;
# this only installs the connector and points it at the token issued there.
# Ingress should target http://127.0.0.1:49999 inside this jail.
#
# Deployed after scan is serving: a connector that advertises an origin which
# is not yet up hands Cloudflare a 502 window on every deploy.
BASTILLE_TOKEN_DST="$BASTILLE_DIR/$RUN/root${TUNNEL_TOKEN_JAIL_PATH}"

set +x
want_tunnel=0
case "$CLOUDFLARED" in
    0|no|NO) ;;
    auto)
        if [ -n "$TUNNEL_TOKEN" ] || doas test -s "$BASTILLE_TOKEN_DST"; then
            want_tunnel=1
        fi
        ;;
    *) want_tunnel=1 ;;
esac
set -x

if [ "$want_tunnel" -eq 0 ]; then
    log "Skipping Cloudflare Tunnel (CLOUDFLARED=$CLOUDFLARED)"
    log "Deployment complete"
    usage_hints
    exit 0
fi

set +x
if [ -z "$TUNNEL_TOKEN" ] && ! doas test -s "$BASTILLE_TOKEN_DST"; then
    set -x
    die "no Cloudflare Tunnel token; rerun with CF_TUNNEL_TOKEN set"
fi
case "$TUNNEL_TOKEN" in
    *[[:cntrl:]]*)
        set -x
        die "CF_TUNNEL_TOKEN must not contain control characters"
        ;;
esac
set -x

if ! doas bastille cmd "$RUN" pkg info -e cloudflared >/dev/null 2>&1; then
    log "Installing cloudflared in run jail"
    doas bastille pkg "$RUN" install -y cloudflared
fi

# --token-file landed in 2025.4.0. Older connectors only accept --token, which
# would put the secret in the process arguments.
TUNNEL_VERSION=$(doas bastille cmd "$RUN" /usr/local/bin/cloudflared --version \
    | awk '$1 == "cloudflared" && $2 == "version" {print $3; exit}')
TUNNEL_YEAR=${TUNNEL_VERSION%%.*}
TUNNEL_REST=${TUNNEL_VERSION#*.}
TUNNEL_MONTH=${TUNNEL_REST%%.*}
case "$TUNNEL_YEAR:$TUNNEL_MONTH" in
    *[!0-9:]*|:*|*:) die "could not parse cloudflared version: $TUNNEL_VERSION" ;;
esac
if [ "$TUNNEL_YEAR" -lt 2025 ] \
    || { [ "$TUNNEL_YEAR" -eq 2025 ] && [ "$TUNNEL_MONTH" -lt 4 ]; }; then
    die "cloudflared 2025.4.0 or newer is required for --token-file support (have $TUNNEL_VERSION)"
fi

log "Ensuring cloudflared service user exists in run jail"
doas bastille cmd "$RUN" id -u "$TUNNEL_USER" >/dev/null 2>&1 || \
    doas bastille cmd "$RUN" pw useradd "$TUNNEL_USER" -d "$TUNNEL_HOME" \
        -s /usr/sbin/nologin -c "Cloudflare Tunnel"
doas bastille cmd "$RUN" install -d -m 0700 -o "$TUNNEL_USER" -g "$TUNNEL_USER" "$TUNNEL_HOME"
doas bastille cmd "$RUN" install -d -m 0755 -o root -g wheel \
    "$(dirname "$TUNNEL_TOKEN_JAIL_PATH")"

# Restart only on a real change: a connector that is already serving is the one
# thing here that does not have to blink.
tunnel_changed=0

log "Installing Cloudflare Tunnel token"
set +x
if [ -n "$TUNNEL_TOKEN" ]; then
    TUNNEL_TMP=$(mktemp -t scan.cftoken.XXXXXX)
    chmod 0600 "$TUNNEL_TMP"
    printf '%s\n' "$TUNNEL_TOKEN" >"$TUNNEL_TMP"
    TUNNEL_TOKEN=""
    if doas cmp -s "$TUNNEL_TMP" "$BASTILLE_TOKEN_DST" 2>/dev/null; then
        set -x
        log "Cloudflare Tunnel token unchanged"
    else
        doas install -m 0600 "$TUNNEL_TMP" "$BASTILLE_TOKEN_DST"
        set -x
        log "Cloudflare Tunnel token updated"
        tunnel_changed=1
    fi
    rm -f "$TUNNEL_TMP"
    TUNNEL_TMP=""
else
    set -x
    log "Keeping the existing Cloudflare Tunnel token"
fi
# root:cloudflared 0640 — writable only by root, readable by the connector
# after daemon(8) drops privileges, opaque to everyone else in the jail.
doas bastille cmd "$RUN" chown "root:$TUNNEL_USER" "$TUNNEL_TOKEN_JAIL_PATH"
doas bastille cmd "$RUN" chmod 0640 "$TUNNEL_TOKEN_JAIL_PATH"

# Deliberately not named "cloudflared": net/cloudflared ships its own
# rc.d/cloudflared, which runs the connector as root off a config file and is
# rewritten by every pkg upgrade. This is a separate service so neither one
# silently overwrites the other.
log "Staging rc.d/$TUNNEL_SERVICE"
doas bastille cmd "$RUN" tee "/usr/local/etc/rc.d/$TUNNEL_SERVICE.new" >/dev/null <<EOF
#!/bin/sh

# PROVIDE: $TUNNEL_SERVICE
# REQUIRE: LOGIN DAEMON NETWORKING scan
# KEYWORD: shutdown

. /etc/rc.subr

name="$TUNNEL_SERVICE"
rcvar="${TUNNEL_SERVICE}_enable"

load_rc_config \$name

: \${${TUNNEL_SERVICE}_enable:="NO"}
# Deliberately NOT named ${TUNNEL_SERVICE}_user: rc.subr treats \${name}_user as
# a magic knob and wraps the whole command in su(1), so daemon(8) itself would
# run as the connector's account and fail to write the root-owned pidfile in
# /var/run. daemon(8) -u below does the privilege drop, after the pidfile.
: \${${TUNNEL_SERVICE}_run_user:="$TUNNEL_USER"}
: \${${TUNNEL_SERVICE}_token_file:="$TUNNEL_TOKEN_JAIL_PATH"}
: \${${TUNNEL_SERVICE}_logfile:="$TUNNEL_LOG"}

pidfile="/var/run/\${name}.pid"
command="/usr/sbin/daemon"
# The token is read from a file rather than passed as an argument so it never
# appears in ps(1) output.
command_args="-c -f -r -R 5 -P \${pidfile} -o \${${TUNNEL_SERVICE}_logfile} -u \${${TUNNEL_SERVICE}_run_user} /usr/bin/env HOME=$TUNNEL_HOME /usr/local/bin/cloudflared tunnel --no-autoupdate run --token-file \${${TUNNEL_SERVICE}_token_file}"

run_rc_command "\$1"
EOF

if doas bastille cmd "$RUN" cmp -s "/usr/local/etc/rc.d/$TUNNEL_SERVICE.new" \
    "/usr/local/etc/rc.d/$TUNNEL_SERVICE" 2>/dev/null; then
    log "rc.d/$TUNNEL_SERVICE unchanged"
    doas bastille cmd "$RUN" rm -f "/usr/local/etc/rc.d/$TUNNEL_SERVICE.new"
else
    log "rc.d/$TUNNEL_SERVICE changed"
    doas bastille cmd "$RUN" mv -f "/usr/local/etc/rc.d/$TUNNEL_SERVICE.new" \
        "/usr/local/etc/rc.d/$TUNNEL_SERVICE"
    doas bastille cmd "$RUN" chmod 755 "/usr/local/etc/rc.d/$TUNNEL_SERVICE"
    tunnel_changed=1
fi
doas bastille sysrc "$RUN" "${TUNNEL_SERVICE}_enable=YES"
# The port's own connector would otherwise come up alongside ours at boot, as
# root, against a config file nothing here maintains.
doas bastille sysrc "$RUN" cloudflared_enable=NO

doas bastille cmd "$RUN" sh -c "[ -e '$TUNNEL_LOG' ] || install -m 0640 -o $TUNNEL_USER -g wheel /dev/null '$TUNNEL_LOG'"

# daemon(8) appends to the log, so a "Registered" line from an earlier deploy
# is still sitting in it. Remember where this run starts and read only past it.
TUNNEL_LOG_OFFSET=$(doas bastille cmd "$RUN" stat -f %z "$TUNNEL_LOG" 2>/dev/null || echo 0)

if ! doas bastille service "$RUN" "$TUNNEL_SERVICE" status >/dev/null 2>&1; then
    log "Starting $TUNNEL_SERVICE"
    doas bastille service "$RUN" "$TUNNEL_SERVICE" start
elif [ "$tunnel_changed" -eq 1 ]; then
    log "Restarting $TUNNEL_SERVICE"
    doas bastille service "$RUN" "$TUNNEL_SERVICE" restart
else
    log "$TUNNEL_SERVICE already running and unchanged"
    log "Deployment complete"
    usage_hints
    exit 0
fi

# A connector that cannot reach Cloudflare retries forever in the background,
# so a silent start says nothing about whether the tunnel is actually serving.
log "Waiting for the tunnel to register a connection"
tunnel_registered=0
for _ in $(jot 30 1); do
    if doas bastille cmd "$RUN" tail -c "+$((TUNNEL_LOG_OFFSET + 1))" "$TUNNEL_LOG" 2>/dev/null \
        | grep -q "Registered tunnel connection"; then
        tunnel_registered=1
        break
    fi
    sleep 1
done
if [ "$tunnel_registered" -ne 1 ]; then
    doas bastille cmd "$RUN" tail -n 50 "$TUNNEL_LOG" >&2 || true
    die "cloudflared did not register a tunnel connection within 30 seconds"
fi
log "Cloudflare Tunnel connected (origin: http://127.0.0.1:49999)"

log "Deployment complete"
usage_hints
