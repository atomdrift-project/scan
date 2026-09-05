#!/bin/sh
# server-freebsd.sh - Install Atomdrift Scan serve as a native FreeBSD rc.d service.
#
# Local install for a FreeBSD host (no jail). Builds as the invoking user,
# installs the binary, creates an unprivileged `scan` service user, and runs
# `atomscan serve` under rc.d via daemon(8). This is the FreeBSD counterpart of
# scripts/server/server-linux.sh (systemd) and the server counterpart of
# scripts/worker/worker-freebsd.sh: one host, one supervised service. For a
# jailed deploy use `make deploy-jail` (scripts/server/rollout-bastille.sh).
#
# Re-runnable: idempotent. The binary is reinstalled and the service restarted
# only when the binary, the rc.d script, or a token actually changed on disk.
#
# Usage: ./server-freebsd.sh
#
# doas or sudo is required for package installation, user creation, and service
# management.
#
# The API requires a bearer token. The token is read from ~/.tok/scan on this
# host — generated on first deploy if absent — and installed into the service
# account's own ~/.tok/scan, which the server reads at startup. It never passes
# through the command line, the environment, or rc.conf. Clients send it as
# `Authorization: Bearer <token>`; only /_/health is exempt. Rotate by editing
# ~/.tok/scan and redeploying.
#
# Environment overrides:
#   TOKEN_SRC   token file to install (empty disables authentication)
#                                                                (default: ~/.tok/scan)
#   BIND        listen address (--bind)                          (default: unset = atomscan's
#                                                                 own, 127.0.0.1:49999, i.e.
#                                                                 reachable only through a
#                                                                 local tunnel or proxy)
#   ALLOW_CIDR  extra CIDR allow-list (--allow-cidr); empty skips the flag
#                                                                (default: 10.0.0.0/8)
#   WORKERS     concurrency (--workers)                          (default: server auto)
#   IDLE        analysis slots the embedded idle worker may spend on hopper
#               queue work (--idle-worker-slots); 0 disables background
#               claiming entirely. Capped at half of WORKERS by the server, and
#               inert without HOPPER. Stored in rc.conf as scan_idle_slots.
#                                                                (default: server auto = half of WORKERS)
#   ALLOWED_DIRS  comma-separated /analyze-path roots            (default: unset)
#   HOPPER      hopper base URL, or several comma-separated in preference
#               order: put the replica first and the primary behind it, and a
#               replica outage costs a retry rather than a lost verdict.
#               Stored in rc.conf as scan_hopper.                (default: unset)
#   HOPPER_TOKEN_FILE  hopper API token, installed whenever the file exists
#                      (HOPPER need not be set)                  (default: ~/.tok/hopper)
#   MAX_RSS_GB  pause threshold (--max-rss-gb). Unlike the systemd deploy
#               there is no cgroup cap to fall back on here, so in-process
#               throttling stays on at atomscan's default, which auto-resolves
#               to the process memory limit.
#                                                                (default: unset = auto)
#   NICE        scheduling priority, stored in rc.conf as scan_nice
#                                                                (default: rc.d default, -20)
#   LLM / LLM_URL  OpenAI-compatible LLM endpoint or named target — comma-separate
#               several to fail over in order — stored in
#               rc.conf as scan_llm; empty turns the second-opinion pass off
#                                                                (default: the Makefile's LLM,
#                                                                 exported on every deploy;
#                                                                 `openrouter` → https://openrouter.ai/api/v1)
#   LLM_MODEL      pinned model (scan_llm_model). Pairs positionally with a
#                  comma-separated LLM chain; a blank slot leaves that endpoint
#                  on atomscan's own default: the largest model it serves, or
#                  `openrouter/auto` for OpenRouter
#                                                                (default: unset)
#   SCAN_LLM_KEY   LLM bearer token, overriding the file below
#   LLM_TOKEN_FILE LLM endpoint bearer token, installed whenever the file
#                  exists; our vLLM requires one    (default: ~/.tok/llm)
#   CLOUDFLARED    Cloudflare Tunnel: "auto" installs and supervises cloudflared
#                  only when CF_TUNNEL_TOKEN is passed or a token from an
#                  earlier deploy is on disk, so a host reached over the LAN
#                  needs no extra flags. 1 requires it, 0 skips it even with a
#                  token present.                                (default: auto)
#   CF_TUNNEL_TOKEN       tunnel token; needed once, then stored
#   CF_TUNNEL_TOKEN_FILE  where it is stored
#                                    (default: /usr/local/etc/atomdrift/cloudflared-token)

set -eu

SERVICE_USER=scan
SERVICE_NAME=scan
BINARY=atomscan
BIN_PATH=/usr/local/bin/${BINARY}
RCD_FILE=/usr/local/etc/rc.d/${SERVICE_NAME}
LOG_FILE=/var/log/scan.log
PIDFILE=/var/run/${SERVICE_NAME}.pid

# `BIND:-` / `MAX_RSS_GB:-` treat empty as unset. `ALLOW_CIDR-` / `TOKEN_SRC-`
# (no colon) keep an explicit empty, so operators can disable the CIDR flag with
# ALLOW_CIDR= and — deliberately, on a host they trust — authentication with
# TOKEN_SRC=.
#
# Deliberately no default: unset leaves atomscan's own, loopback, so the
# intended exposure is a Cloudflare tunnel (or another local proxy) terminating
# on this host. Set BIND=0.0.0.0:49999 to listen on every interface, and pair
# it with ALLOW_CIDR.
BIND="${BIND:-}"
ALLOW_CIDR="${ALLOW_CIDR-10.0.0.0/8}"
TOKEN_SRC="${TOKEN_SRC-${HOME}/.tok/scan}"
WORKERS="${WORKERS:-}"
# Empty means "unset" here rather than a meaningful value: the server's own
# default (half the slots) applies. IDLE=0 is a real value — background
# claiming off — so it must survive as the string "0" and reach rc.conf.
IDLE="${IDLE:-}"
ALLOWED_DIRS="${ALLOWED_DIRS:-}"
HOPPER="${HOPPER:-}"
# Deliberately no default: unset leaves atomscan's own (auto-resolve), and the
# flag is only passed when an operator names a value.
MAX_RSS_GB="${MAX_RSS_GB:-}"
NICE="${NICE:-}"
# LLM_URL is an alias for LLM (SCAN_LLM): `local`, `openrouter`, or a base URL.
if [ -z "${LLM:-}" ] && [ -n "${LLM_URL:-}" ]; then
	LLM=$LLM_URL
fi
# LLM second-opinion pass: endpoint (exported as SCAN_LLM) + interpret gate.
# No default here on purpose: the site's failover chain is defined once, in
# the Makefile (LLM ?=), which exports it to every deploy script. Unset leaves
# atomscan's own default.
LLM="${LLM:-}"
# Deliberately no default. atomscan picks the model itself — the largest one a
# vLLM/Ollama endpoint reports, `openrouter/auto` for OpenRouter — and only an
# operator's explicit pin is passed through.
LLM_MODEL="${LLM_MODEL:-${SCAN_LLM_MODEL:-}}"
CLOUDFLARED="${CLOUDFLARED:-auto}"
CF_TUNNEL_TOKEN_FILE="${CF_TUNNEL_TOKEN_FILE:-/usr/local/etc/atomdrift/cloudflared-token}"

die() { echo "error: $*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
# shellcheck source=scripts/server/lib/freebsd-rcd.sh
. "$SCRIPT_DIR/lib/freebsd-rcd.sh"

TMP_RCD=""
trap '[ -n "$TMP_RCD" ] && rm -f "$TMP_RCD"' EXIT

# --- Preconditions -----------------------------------------------------------

[ -f Makefile ]               || die "run from the repository root"
[ "$(uname -s)" = "FreeBSD" ] || die "this script is for FreeBSD; use deploy-jail for jails"

case "${IDLE}" in
	'') ;;
	*[!0-9]*) die "IDLE must be a non-negative integer (got '${IDLE}')" ;;
esac

if command -v doas >/dev/null 2>&1; then
	SUDO=doas
elif command -v sudo >/dev/null 2>&1; then
	SUDO=sudo
else
	die "need doas or sudo for privileged steps"
fi

# --- Listening port ----------------------------------------------------------
#
# The listening socket is the one thing this deploy cannot share, and a host
# that already runs some other `atomscan serve` (a promoter's warm server, an
# older hand-rolled service) is exactly where a collision is likely. Left
# unchecked it is silent in both directions: the new server dies with
# "Address already in use" while daemon(8) keeps a supervisor around, and a
# health probe against the port is answered by the *other* process — a deploy
# that reports success and installed nothing.
#
# sockstat only reveals another user's sockets to root
# (security.bsd.see_other_uids), so every query goes through $SUDO; as a plain
# user it returns nothing at all, which would read as "port free".
PORT=${BIND##*:}
BIND_HOST=${BIND%:*}
case "${BIND_HOST}" in
	''|0.0.0.0|::|'[::]'|'*') PROBE_HOST=127.0.0.1 ;;
	*) PROBE_HOST=${BIND_HOST} ;;
esac
BASE="http://${PROBE_HOST}:${PORT}"

# The full sockstat row for whoever is listening on PORT, empty if nobody is.
port_owner_row() {
	$SUDO sockstat -46l -p "${PORT}" 2>/dev/null \
		| awk 'NR > 1 && $3 ~ /^[0-9]+$/ { print; exit }'
}
port_owner_pid() { port_owner_row | awk '{print $3}'; }

# The pid of the atomscan this rc.d service is supervising, if any: daemon(8)
# writes its own pid to the pidfile and forks the server as its only child.
# Empty when the service is down — including when a stale pidfile names a
# supervisor that is gone.
service_child_pid() {
	_scp_sup=$($SUDO cat "${PIDFILE}" 2>/dev/null || true)
	case "${_scp_sup}" in
		''|*[!0-9]*) return 0 ;;
	esac
	$SUDO pgrep -P "${_scp_sup}" 2>/dev/null | head -1
	return 0
}

# Refuse early — before a 15-minute build — when the port belongs to something
# that is not this service.
preflight_owner=$(port_owner_row)
if [ -n "${preflight_owner}" ] \
	&& [ "$(printf '%s' "${preflight_owner}" | awk '{print $3}')" != "$(service_child_pid)" ]; then
	log "port ${PORT} is already in use:"
	log "  ${preflight_owner}"
	die "another process owns ${BIND}; stop it, or deploy with BIND=<addr>:<free-port>"
fi

# --- Packages ---------------------------------------------------------------
# Build deps (rust, git, pkgconf, mold) and runtime deps (7-zip, upx, rizin,
# innoextract) on one host. pkg install is idempotent, but checking first keeps
# a no-op re-run from touching the package DB. Mirrors worker-freebsd.sh.

missing=""
for pkg in rust git pkgconf mold 7-zip upx rizin innoextract; do
	pkg info -e "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"
done
if [ -n "$missing" ]; then
	log "Installing packages:$missing"
	# shellcheck disable=SC2086
	$SUDO pkg install -y $missing
else
	log "All packages already installed"
fi

# --- Build (as the invoking user) ------------------------------------------
# Raise the data-segment limit to its hard cap; FreeBSD's default datasize
# limit can starve a release build of rustc.
# FreeBSD /bin/sh exposes the data-segment hard limit through ulimit -Hd.
# shellcheck disable=SC3045
ulimit -d "$(ulimit -Hd)" 2>/dev/null || true

log "Building"
RUSTFLAGS="-C link-arg=-fuse-ld=mold" cargo build --release || die "build failed"
[ -x "target/release/${BINARY}" ] || die "build did not produce target/release/${BINARY}"

# --- Service user -----------------------------------------------------------
# A home directory is required: the server resolves the hopper token, the
# OpenRouter key, and cleave's traits/models under it. The shell is /bin/sh
# (not nologin) so the `su -l scan` model refresh below can run; the account has
# no password, so it is not remotely loginable.

if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
	log "Creating service user '${SERVICE_USER}'"
	$SUDO pw useradd "${SERVICE_USER}" -m -s /bin/sh -c "Atomdrift Scan Server"
fi

# Honour an existing account's home rather than assuming /home/scan: on a host
# that already runs a worker the account is there, and its home is where the
# tokens and the traits checkout already live.
SERVICE_HOME=$(getent passwd "${SERVICE_USER}" | cut -d: -f6)
[ -n "${SERVICE_HOME}" ] || die "cannot resolve the home directory of '${SERVICE_USER}'"
log "Service home: ${SERVICE_HOME}"
$SUDO install -d -m 0700 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${SERVICE_HOME}/.tok"

# --- API token --------------------------------------------------------------
#
# Installed as a file, never as an argument or an rc.conf line: argv is
# world-readable through ps(1), and rc.conf is world-readable. Redeploying with
# no source token keeps the installed one, so a redeploy can never silently
# drop authentication.
#
# The token is never held in a shell variable — only paths are — so it cannot
# leak through a trace or an error message.
TOKEN_DST="${SERVICE_HOME}/.tok/scan"
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
# loopback, so without it every result renewal is rejected with 401.
#
# Installed whenever the operator has one, NOT only when HOPPER is set: the URL
# lives in rc.conf and can be switched on later with `sysrc scan_hopper=`, and
# a server that gains --hopper without the token 401s on every renewal. The
# file is inert while --hopper is off.
hopper_token_src="${HOPPER_TOKEN_FILE:-${HOME}/.tok/hopper}"
hopper_token_dst="${SERVICE_HOME}/.tok/hopper"
if [ -s "$hopper_token_src" ]; then
	$SUDO cmp -s "$hopper_token_src" "$hopper_token_dst" 2>/dev/null || token_changed=1
	$SUDO install -m 0600 -o "${SERVICE_USER}" -g "${SERVICE_USER}" \
		"$hopper_token_src" "$hopper_token_dst"
	log "Installed hopper API token at ${hopper_token_dst}"
elif [ -n "${HOPPER}" ] && ! $SUDO test -s "$hopper_token_dst"; then
	# Only worth a warning when there is a hopper to talk to.
	log "WARNING: no hopper API token at ${hopper_token_src}; result renewal on ${HOPPER} will be rejected"
fi

# --- OpenRouter key ----------------------------------------------------------
# Only when the LLM target is OpenRouter: copy the operator key into the
# service home, where atomscan finds it through HOME.
# The LLM target may be a comma-separated failover chain, so OpenRouter can sit
# anywhere in it. Anywhere is enough to need its key installed here; only when
# it is the *whole* chain is a missing key fatal, because then there is no
# other endpoint left to grade with. There is no model check to match: an
# unpinned OpenRouter slot is `openrouter/auto` in the binary.
openrouter_target() {
	_or_rest="$LLM"
	while [ -n "$_or_rest" ]; do
		_or_one=${_or_rest%%,*}
		case "$_or_rest" in
			*,*) _or_rest=${_or_rest#*,} ;;
			*)   _or_rest="" ;;
		esac
		case "$(printf '%s' "$_or_one" | tr -d '[:space:]')" in
			openrouter|https://openrouter.ai/*|http://openrouter.ai/*) return 0 ;;
		esac
	done
	return 1
}

# OpenRouter and nothing else — the case where its key is required
# rather than merely useful.
openrouter_only() {
	case "$LLM" in
		*,*) return 1 ;;
	esac
	openrouter_target
}
if openrouter_target; then
	dst="${SERVICE_HOME}/.tok/openrouter"
	src="${HOME}/.tok/openrouter"
	if [ -n "${SCAN_LLM_KEY:-}" ]; then
		tmp=$(mktemp -t scan.orkey.XXXXXX)
		chmod 0600 "$tmp"
		printf '%s\n' "$SCAN_LLM_KEY" > "$tmp"
		$SUDO install -m 0600 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "$tmp" "$dst"
		rm -f "$tmp"
	elif [ -s "$src" ]; then
		$SUDO install -m 0600 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "$src" "$dst"
	else
		if openrouter_only; then
			die "OpenRouter deploy needs a key in $src (or SCAN_LLM_KEY)"
		fi
		log "WARNING: no OpenRouter key in $src (or SCAN_LLM_KEY); that link is dropped from the chain"
		dst=""
	fi
	# Only on a branch that actually installed one.
	if [ -n "$dst" ]; then
		log "Installed OpenRouter token at ${dst}"
	fi
fi

# --- LLM endpoint token ------------------------------------------------------
# Our vLLM endpoint requires `Authorization: Bearer <token>`; atomscan reads it
# from $HOME/.tok/llm, so it has to land in the service home like the others.
# A file rather than argv or an rc.conf variable, for the same reason as the
# tokens above: ps(1) output and rc.conf are both world-readable.
#
# Installed whenever the operator has one, regardless of which endpoint this
# deploy targets: scan_llm is switchable from rc.conf later, and the file is
# inert against an endpoint that wants no key.
llm_token_src="${LLM_TOKEN_FILE:-${HOME}/.tok/llm}"
llm_token_dst="${SERVICE_HOME}/.tok/llm"
if [ -n "${SCAN_LLM_KEY:-}" ] && ! openrouter_target; then
	# An explicit key on the deploy is the operator overriding the file.
	tmp=$(mktemp -t scan.llmkey.XXXXXX)
	chmod 0600 "$tmp"
	printf '%s\n' "$SCAN_LLM_KEY" > "$tmp"
	$SUDO cmp -s "$tmp" "$llm_token_dst" 2>/dev/null || token_changed=1
	$SUDO install -m 0600 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "$tmp" "$llm_token_dst"
	rm -f "$tmp"
	log "Installed LLM endpoint token at ${llm_token_dst}"
elif [ -s "$llm_token_src" ]; then
	$SUDO cmp -s "$llm_token_src" "$llm_token_dst" 2>/dev/null || token_changed=1
	$SUDO install -m 0600 -o "${SERVICE_USER}" -g "${SERVICE_USER}" \
		"$llm_token_src" "$llm_token_dst"
	log "Installed LLM endpoint token at ${llm_token_dst}"
elif [ -n "$LLM" ] && ! openrouter_target && ! $SUDO test -s "$llm_token_dst"; then
	# Not fatal: a missing key only costs the second opinion, and the ML verdict
	# still stands. But it is silent at runtime, so say it loudly here.
	log "WARNING: no LLM token at ${llm_token_src}; ${LLM} will reject the second-opinion pass with 401"
fi

# --- Binary -----------------------------------------------------------------

binary_changed=0
if $SUDO cmp -s "target/release/${BINARY}" "${BIN_PATH}" 2>/dev/null; then
	log "Binary unchanged"
else
	log "Installing ${BIN_PATH}"
	# install(1) writes-then-renames; safe over a running exe (the kernel pins
	# the inode of the running process).
	$SUDO install -m 0755 -o root -g wheel "target/release/${BINARY}" "${BIN_PATH}"
	binary_changed=1
fi

# --- Models and traits ------------------------------------------------------
# Populate the scan user's data dir so the first start is not racing a clone.
# Idempotent: update-rules pulls when a checkout already exists.

log "Refreshing models and traits"
$SUDO su -l "${SERVICE_USER}" -c "${BIN_PATH} update-rules" || die "update-rules failed"

# --- rc.d service -----------------------------------------------------------

token_arg=""
if [ -n "${TOKEN_SRC}" ]; then
	token_arg="${TOKEN_DST}"
fi
serve_args=$(scan_server_args "$BIND" "$ALLOW_CIDR" "$token_arg" \
	"$WORKERS" "$ALLOWED_DIRS" "$MAX_RSS_GB")

TMP_RCD=$(mktemp -t scan.rcd.XXXXXX)
scan_server_rcd_script "${BIN_PATH}" "$serve_args" "$LLM" "$LLM_MODEL" "${SERVICE_HOME}" >"$TMP_RCD"

rcd_changed=0
if $SUDO cmp -s "$TMP_RCD" "$RCD_FILE" 2>/dev/null; then
	log "rc.d script unchanged"
else
	log "Writing ${RCD_FILE}"
	$SUDO install -m 0755 -o root -g wheel "$TMP_RCD" "$RCD_FILE"
	rcd_changed=1
fi

# daemon(8) opens the log as root before dropping privileges, but pre-creating
# it keeps ownership predictable for anyone reading it later. Never truncated:
# an existing log is appended to.
$SUDO sh -c "[ -e '${LOG_FILE}' ] || install -m 0640 -o ${SERVICE_USER} -g wheel /dev/null '${LOG_FILE}'"

# --- rc.conf knobs ----------------------------------------------------------

# sysrc is idempotent; enable so the service also comes back across reboots.
$SUDO sysrc "${SERVICE_NAME}_enable=YES" >/dev/null
# Written unconditionally, empty when unset, so dropping HOPPER=/IDLE= from a
# later deploy actually turns renewal off / restores the server default instead
# of silently keeping the previous value — and rc.conf shows it is off on
# purpose. Compared first so an unchanged value does not force a restart.
conf_changed=0
for pair in "scan_hopper=${HOPPER}" "scan_idle_slots=${IDLE}"; do
	var=${pair%%=*}
	val=${pair#*=}
	if [ "$($SUDO sysrc -n "$var" 2>/dev/null || true)" != "$val" ]; then
		$SUDO sysrc "${var}=${val}" >/dev/null
		conf_changed=1
	fi
done
# NICE is only written when given: an empty scan_nice would break `nice -n`.
# It persists in rc.conf once set, so drop it there to go back to the default.
if [ -n "${NICE}" ] && [ "$($SUDO sysrc -n scan_nice 2>/dev/null || true)" != "${NICE}" ]; then
	$SUDO sysrc "scan_nice=${NICE}" >/dev/null
	conf_changed=1
fi

# --- Activate ---------------------------------------------------------------

# The old server holds the port until it is gone, so a restart has to wait for
# the socket to be released before starting the new one — otherwise the start
# races the stop and loses with "Address already in use".
wait_port_free() {
	_wpf=0
	while [ -n "$(port_owner_row)" ]; do
		if [ "$_wpf" -ge 15 ]; then
			log "port ${PORT} still held by:"
			log "  $(port_owner_row)"
			return 1
		fi
		sleep 1
		_wpf=$((_wpf + 1))
	done
	return 0
}

start_service() {
	wait_port_free || die "cannot start ${SERVICE_NAME}: port ${PORT} is not free"
	$SUDO service "${SERVICE_NAME}" start || die "service failed to start; see ${LOG_FILE}"
}

if [ "$binary_changed" -eq 1 ] || [ "$rcd_changed" -eq 1 ] || [ "$token_changed" -eq 1 ] \
	|| [ "$conf_changed" -eq 1 ]; then
	log "Restarting ${SERVICE_NAME}"
	# The rc.d stop is bounded: SIGTERM, a drain window, then a SIGKILL of the
	# daemon(8) supervisor and its child — so a busy server cannot stall the
	# redeploy the way rc.subr's wait-forever default would.
	$SUDO service "${SERVICE_NAME}" stop || true
	start_service
elif [ -n "$(service_child_pid)" ]; then
	log "No changes; ${SERVICE_NAME} already running"
else
	log "No changes; starting ${SERVICE_NAME}"
	start_service
fi

# --- Health -----------------------------------------------------------------
#
# daemon(8) -r returns success as soon as the supervisor forks, which says
# nothing about whether atomscan bound its port: a bad --bind or an unreadable
# token file is a restart loop that looks like a clean start. Poll until the
# server actually answers.
# curl is a package on FreeBSD; fetch(1) is in base and always there.
if command -v curl >/dev/null 2>&1; then
	http_get() { curl -fsS --max-time 5 "$1"; }
else
	http_get() { fetch -qo - -T 5 "$1"; }
fi

# A 200 from the port is not enough: it proves only that *something* is
# serving there. The answer counts when the socket belongs to the process this
# service supervises.
log "Waiting for ${BASE}/_/health"
healthy=0
for _ in $(jot 60 1); do
	child=$(service_child_pid)
	if [ -n "${child}" ] && [ "${child}" = "$(port_owner_pid)" ] \
		&& http_get "${BASE}/_/health" >/dev/null 2>&1; then
		healthy=1
		break
	fi
	sleep 1
done
if [ "$healthy" -ne 1 ]; then
	owner=$(port_owner_row)
	if [ -n "${owner}" ]; then
		log "port ${PORT} is held by:"
		log "  ${owner}"
	fi
	$SUDO tail -n 50 "${LOG_FILE}" >&2 || true
	die "${SERVICE_NAME} did not answer ${BASE}/_/health within 60s"
fi
log "Server healthy: $(http_get "${BASE}/_/health")"

$SUDO service "${SERVICE_NAME}" status || true

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
		"$SCRIPT_DIR/cloudflared-freebsd.sh" "${BASE}"
else
	log "Skipping Cloudflare Tunnel (CLOUDFLARED=${CLOUDFLARED})"
fi

# Every route except /_/health wants the bearer token, so fold it into the
# examples rather than printing a header the reader has to paste in by hand.
if [ -n "${TOKEN_SRC}" ]; then
	auth="-H \"Authorization: Bearer \$(cat ${TOKEN_SRC})\""
else
	auth=""
fi

log "Deployment complete"
log "Service: service ${SERVICE_NAME} status | tail -f ${LOG_FILE}"
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
	log "Tunnel:  service scan_tunnel status"
fi
