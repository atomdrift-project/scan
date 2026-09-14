#!/bin/sh
# rollout.sh - Redeploy every scan host hopper has seen recently.
#
# The fleet roster is not maintained here: hopper already knows it. Every
# worker and every `atomscan serve` idle worker check in on /api/heartbeat,
# which hopper persists to its `workers` table, so "the hosts that matter" is a
# query rather than a list someone has to remember to edit. Names arrive as
# `hostname[:peer-ip]`, since hopper qualifies duplicates with the peer address.
#
# A server is told from a worker by that roster: `atomscan serve` runs an
# embedded idle worker, which checks in as `<hostname>-idle`, so the suffix is
# what marks the host as a server. A host carrying both rows is treated as a
# server, because that is the more careful of the two paths.
#
# The hopper host goes first and alone. Its redeploy takes seconds where a
# worker's takes minutes, so putting it ahead of them spares every worker a
# second reconnect, and keeps the queue from dropping under whichever workers
# happened to be finishing at that moment. Which host that is comes from $URL
# via /etc/hosts, not from a name written here.
#
# It is also a gate: if the queue does not come back, nothing else is touched.
# Workers claim their work from hopper and file results to it, so redeploying
# them against a dead queue produces a fleet of idle processes and buries the
# failure that caused it under seven concurrent builds.
#
# Workers go next, in parallel batches -- they are interchangeable capacity and
# hopper simply hands their queue work to whoever is left. Servers follow one at
# a time, each one proving itself healthy before the next is touched, because
# they are what callers reach and taking two down at once is an outage.
#
# The health gate is two questions, and a server must answer both:
#
#   1. Its own GET /_/health, asked over the SSH connection already open.
#      `status` must be `ok`, and `uptime_secs` must be small -- a large uptime
#      means the service never actually restarted and we are reading the
#      process we thought we replaced.
#   2. A pinned beamline lookup, from here, over the public edge. `X-Beamline-Pin`
#      names one backend and beamline never substitutes another, so this proves
#      the whole path -- edge, tunnel, server -- came back, and that what came
#      back still returns the right verdict. The pin is a host's `scan-*` alias
#      in /etc/hosts, which is where this fleet already keeps its name map; a
#      server without one is gated on its own health alone.
#
# What each host is told to do is decided ON the host, from the service files
# installed there, rather than from the roster name -- which only reports what
# was running at the last check-in. The make targets pull the repository
# themselves, so there is no separate update step. Five shapes are recognized:
#
#   worker   a scan-worker unit -- stop it, redeploy onto the hopper its own
#            unit names
#   server   a scan unit -- `make deploy`, then the health gate below
#   hopper   a hopper unit beside a ~/hopper checkout -- hopper's own deploy
#            installs hopper AND the scan worker on that box, so the whole host
#            is one `make deploy` over there and the worker is left alone
#   adhoc    macOS with no unit -- the Macs run their workers by hand, so
#            rebuild and relaunch detached, pointed at URL. A launchd plist,
#            if one is ever installed, wins over this.
#   steam    a sneaky-steam user unit -- a Steam Machine, which has neither
#            Rust nor a checkout. It is sent the published static musl binary
#            rather than asked to build one, so it tracks the last release
#            rather than HEAD. See push_binary for why nothing else works.
#
# A host with none of the four is reported as `no-service` and left untouched.
#
# Hosts within a worker batch run concurrently, but authentication does not:
# SSH connection multiplexing splits the two apart. Each batch first opens a
# master connection to its hosts one at a time, so a YubiKey touch or a password
# prompt is faced alone and in order; the deploys then run concurrently over
# those already authenticated sockets and never prompt again.
#
# Usage:
#   ./rollout.sh
#
# Environment overrides:
#   DAYS          how far back a check-in still counts        (default 7)
#   BATCH         workers deployed concurrently, 0 for all    (default 0)
#   SERVER_BATCH  servers one at a time (1) or all at once (0) (default 1)
#   HOSTS         space-separated hosts, skipping discovery   (default: ask hopper)
#   SKIP          space-separated hosts to leave alone        (default: none)
#   PHASES        which phases to run, of `hopper workers servers`
#                 (default: all three; see `make rollout-workers` / `-servers`)
#   URL           hopper endpoint for a worker whose installed unit names none
#   DB            hopper database to read the roster from
#   TIMEOUT       seconds before a wedged deploy is cut loose (default 1800)
#   DRY_RUN       set to any value to print the plan and stop
#   TRACE         set to any value to run the script under `set -x`
#
#   HEALTH_ADDR   where a server answers /_/health            (default 127.0.0.1:49999)
#   HEALTH_WAIT   seconds a server may take to come back      (default 300)
#   HEALTH_UPTIME largest uptime_secs still counted as a restart (default 600)
#   BEAMLINE      beamline base URL, empty to skip the pinned query
#   BEAMLINE_TOKEN  bearer token (default: ~/.tok/beamline)
#   PIN_DOMAIN    domain the scan-* alias is qualified with   (default isotope13.ai)
#   PIN_PURL      the purl the pinned query asks about
#   PIN_SEVERITY  the severity that query must come back with (default benign)
set -u

# TRACE=1 turns on shell tracing. Parallel batches interleave their traces, so
# it reads best with BATCH=1 or a single HOSTS= entry.
[ -z "${TRACE:-}" ] || set -x

DAYS="${DAYS:-7}"
# All of them at once, by default. Batching only ever limited the deploys, not
# the authentication -- each batch still opens its master connections one at a
# time -- so a smaller number bought nothing but a longer wall clock. Set a
# positive BATCH to cap it again, e.g. to spare a busy hopper.
BATCH="${BATCH:-0}"
# Servers stay one at a time unless told otherwise; see Phase 2 for why.
SERVER_BATCH="${SERVER_BATCH:-1}"
HOSTS="${HOSTS:-}"
SKIP="${SKIP:-}"
PHASES="${PHASES:-hopper workers servers}"
URL="${URL:-http://10.9.8.10:8081/}"
DB="${DB:-postgres://hopper@hopper-db/hopper?sslmode=disable}"
TIMEOUT="${TIMEOUT:-1800}"
DRY_RUN="${DRY_RUN:-}"

# Where Git for Windows keeps the POSIX shell. Windows answers SSH with
# cmd.exe, which has neither `sh` nor `true`, so a host that cannot run the
# latter is asked again through this before being called dead.
WIN_SH="${WIN_SH:-C:\\Program Files\\Git\\bin\\bash.exe}"

HEALTH_ADDR="${HEALTH_ADDR:-127.0.0.1:49999}"
HEALTH_WAIT="${HEALTH_WAIT:-300}"
HEALTH_UPTIME="${HEALTH_UPTIME:-600}"
BEAMLINE="${BEAMLINE:-https://api.isotope13.ai}"
BEAMLINE_TOKEN="${BEAMLINE_TOKEN:-}"
PIN_DOMAIN="${PIN_DOMAIN:-isotope13.ai}"
PIN_PURL="${PIN_PURL:-pkg:npm/left-pad@1.3.0}"
PIN_SEVERITY="${PIN_SEVERITY:-benign}"

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

case "$DAYS" in
*[!0-9]* | '') die "DAYS must be a whole number of days, got '$DAYS'" ;;
esac

case "$BATCH" in
*[!0-9]* | '') die "BATCH must be a whole number (0 for all at once), got '$BATCH'" ;;
esac

case "$SERVER_BATCH" in
0 | 1) ;;
*) die "SERVER_BATCH must be 1 (one at a time) or 0 (all at once), got '$SERVER_BATCH'" ;;
esac

# A typo here would otherwise read as "that phase is switched off" and quietly
# skip half the fleet, which is the one mistake this knob must not make.
for phase in $PHASES; do
	case "$phase" in
	hopper | workers | servers) ;;
	*) die "PHASES may name only hopper, workers or servers; got '$phase'" ;;
	esac
done

# wants <phase> — true when this run was asked to touch that phase.
wants() {
	case " $PHASES " in
	*" $1 "*) return 0 ;;
	esac
	return 1
}

[ -n "$BEAMLINE_TOKEN" ] || [ ! -s "$HOME/.tok/beamline" ] ||
	BEAMLINE_TOKEN=$(cat "$HOME/.tok/beamline")

# --- The roster -------------------------------------------------------------
#
# Kept as raw check-in names through discovery: the `-idle` suffix is what says
# which phase a host belongs to, so it cannot be stripped before that decision.

if [ -z "$HOSTS" ]; then
	command -v psql >/dev/null 2>&1 ||
		die "psql not found; install it or pass HOSTS=\"node1 node2\""
	log "Asking hopper which hosts checked in over the last $DAYS days"
	# One row per host, the LATEST one, rather than every row in the window.
	# A host that was a server and is now a worker keeps its old `-idle` row
	# until it ages out, and treating any `-idle` sighting as current puts it
	# in the server phase for days after it stopped being one -- nazgul, whose
	# worker checked in six minutes ago beside a six-day-old `-idle` row.
	# Whichever check-in is newest is what the host is now.
	roster=$(psql "$DB" -Atq -c "
		SELECT DISTINCT ON (host) host || CASE WHEN idle THEN '-idle' ELSE '' END
		FROM (
			SELECT regexp_replace(split_part(name, ':', 1), '-idle\$', '') AS host,
			       split_part(name, ':', 1) LIKE '%-idle'                  AS idle,
			       last_seen
			FROM workers
			WHERE last_seen > now() - interval '$DAYS days'
		) t
		ORDER BY host, last_seen DESC") ||
		die "could not read the worker roster from $DB"
	[ -n "$roster" ] || die "hopper has seen no workers in the last $DAYS days"
else
	roster="$HOSTS"
fi

# The host we are standing on is never rolled from here: redeploying it would
# tear down the SSH endpoint mid-rollout. Compared on the short name so a roster
# entry of `obelisk.lan` matches a local hostname of `obelisk`.
self=$(hostname | cut -d. -f1 | tr '[:upper:]' '[:lower:]')

servers=""
candidates=""
for name in $roster; do
	case "$name" in
	*-idle) host=${name%-idle}; servers="$servers $host" ;;
	*) host="$name" ;;
	esac
	candidates="$candidates $host"
done

# A host is a worker only if nothing claimed it as a server.
workers=""
targets=""
for host in $candidates; do
	case " $targets " in *" $host "*) continue ;; esac
	short=$(echo "$host" | cut -d. -f1 | tr '[:upper:]' '[:lower:]')
	if [ "$short" = "$self" ]; then
		warn "skipping $host: this is the host running the rollout"
		continue
	fi
	case " $SKIP " in
	*" $host "*)
		warn "skipping $host: named in SKIP"
		continue
		;;
	esac
	targets="$targets $host"
	case " $servers " in
	*" $host "*) ;;
	*) workers="$workers $host" ;;
	esac
done

# Rebuild the server list from what survived skipping, in roster order.
kept_servers=""
for host in $targets; do
	case " $servers " in
	*" $host "*) kept_servers="$kept_servers $host" ;;
	esac
done
servers="$kept_servers"

[ -n "$targets" ] || die "every discovered host was skipped; nothing to do"

# The scan-* hosts are rented boxes with no unprivileged operator account; a
# Steam Machine has only `deck`, and the local operator name does not exist
# there at all -- connecting as it is answered by a password prompt for an
# account that cannot log in, which is what "Connection closed by <deck>" was.
ssh_target() {
	case "$1" in
	scan-*) echo "root@$1" ;;
	steamdeck*) echo "deck@$1" ;;
	*) echo "$1" ;;
	esac
}

# hopper_host — the short name of the machine that serves $URL, or empty.
#
# Workers are told their queue as a URL, and /etc/hosts is where this fleet
# already maps an address to a machine -- the same file pin_for reads below,
# and the reason `10.9.8.10  smaug hopper-api forager` is enough to know which
# box that is. Deriving it beats naming smaug here: move hopper and the roster
# follows, because the URL the workers are given is what moved.
hopper_host() {
	h=${URL#*://}
	h=${h%%/*}
	h=${h%%:*}
	case "$h" in
	'') return ;;
	*[!0-9.]*) echo "$h" | cut -d. -f1 | tr '[:upper:]' '[:lower:]'; return ;;
	esac
	awk -v ip="$h" '
		/^[ \t]*#/ { next }
		$1 == ip { print $2; exit }
	' /etc/hosts 2>/dev/null | cut -d. -f1 | tr '[:upper:]' '[:lower:]'
}

# pin_for <host> — the host's scan-* alias in /etc/hosts, which is this fleet's
# existing map from a machine to the name beamline knows it by. Empty when the
# host has none, which is not an error: a server that sits behind no tunnel is
# gated on its own health alone.
pin_for() {
	awk -v want="$1" '
		/^[ \t]*#/ { next }
		{
			for (i = 2; i <= NF; i++) if ($i == want) {
				for (j = 2; j <= NF; j++) if ($j ~ /^scan-/) { print $j; exit }
			}
		}
	' /etc/hosts 2>/dev/null
}

# The queue goes first, alone. Hopper's redeploy is quick and the workers' is
# not, so restarting it ahead of them costs seconds and spares every worker a
# second reconnect -- the same reason deploy-workers-tmux.sh gives hopper a
# head start. Doing it inside the parallel batch would drop the queue under
# whichever workers happened to be finishing at that moment.
hopper=""
hh=$(hopper_host)
if [ -n "$hh" ]; then
	rest=""
	for host in $workers; do
		short=$(echo "$host" | cut -d. -f1 | tr '[:upper:]' '[:lower:]')
		if [ "$short" = "$hh" ] && [ -z "$hopper" ]; then
			hopper="$host"
		else
			rest="$rest $host"
		fi
	done
	workers="$rest"
fi

# Drop the phases this run was not asked for, and rebuild `targets` from what
# survived. The summary and the exit status are both driven by `targets`, so
# leaving a deselected host in it would report the whole other half of the
# fleet as `not-reached` and fail a run that did exactly what it was told.
# Rebuilding here also puts the table in execution order rather than roster
# order, which is how the run actually reads.
wants hopper || hopper=""
wants workers || workers=""
wants servers || servers=""

targets=""
for host in $hopper $workers $servers; do
	targets="$targets $host"
done
[ -n "$targets" ] ||
	die "PHASES='$PHASES' selected none of the discovered hosts"

log "Plan: $(echo "$hopper" | wc -w | tr -d ' ') hopper, $(echo "$workers" | wc -w | tr -d ' ') workers, then $(echo "$servers" | wc -w | tr -d ' ') servers"
if [ -n "$hopper" ]; then
	note "hopper  (first, alone):$hopper — serves $URL"
fi
if [ -n "$workers" ]; then
	if [ "$BATCH" -gt 0 ]; then
		note "workers ($BATCH at a time):$workers"
	else
		note "workers (all at once):$workers"
	fi
fi
for host in $servers; do
	pin=$(pin_for "$host")
	if [ "$SERVER_BATCH" -eq 1 ]; then
		how="one at a time"
	else
		how="ALL AT ONCE"
	fi
	if [ -n "$pin" ]; then
		note "server  ($how): $host — health, then pin $pin.$PIN_DOMAIN"
	else
		note "server  ($how): $host — health only, no scan-* alias in /etc/hosts"
	fi
done
[ -z "$DRY_RUN" ] || { log "DRY_RUN set; stopping before the first connection"; exit 0; }

# --- What each host runs ----------------------------------------------------
#
# Piped to `sh -s` rather than passed as an argument, so the script is written
# once, plainly, with no layer of remote-shell quoting to get wrong.

remote_script() {
	printf "URL_FALLBACK='%s'\n" "$URL"
	cat <<'REMOTE_EOF'
set -u

# A Steam Machine has no Rust and no repository: sneaky-steam supervises a
# finished binary in ~/bin, running it only while nobody is playing. Nothing
# can be built here, so say what this host is and let the caller push one.
if [ -f "$HOME/.config/systemd/user/sneaky-steam.service" ]; then
	echo "rollout: steam (sneaky-steam user unit) — cannot build here, needs a binary"
	exit 92
fi

# Where the checkout lives. Every unix host keeps it in ~/scan; the Windows box
# keeps it at C:\src\scan, which Git for Windows presents as /c/src/scan.
repo=""
for d in "$HOME/scan" /c/src/scan; do
	if [ -f "$d/Makefile" ]; then repo="$d"; break; fi
done
[ -n "$repo" ] || { echo "rollout: no scan checkout on this host"; exit 90; }
cd "$repo" || exit 90

# rustup puts cargo on PATH from the login shell's rc file, which a piped
# `ssh host sh -s` never reads: the build would fail with "cargo not found" on
# a box that compiles fine when you log into it. Put it back where rustup
# installs it, without disturbing a system toolchain that is already ahead.
[ ! -x "$HOME/.cargo/bin/cargo" ] || PATH="$HOME/.cargo/bin:$PATH"
export PATH

# This makefile is GNU make's. FreeBSD's /usr/bin/make is bmake, which does not
# parse it -- it reports a screenful of "Invalid line" and "Fatal errors
# encountered" and stops before doing anything. Ask for gmake by name wherever
# it exists, which on a GNU userland is the same program under another link.
mk=make
command -v gmake >/dev/null 2>&1 && mk=gmake

# Installed services are the ground truth for what this host runs. The roster
# says which phase it belongs to; this says what to run once we are here.
#
# A unit FILE, though, is not proof: nazgul still carries the scan.service it
# was left with after a stint as a server, disabled and inactive beside the
# scan-worker.service it actually runs, and taking the file at its word would
# deploy a server onto a worker-only box. Where a service manager can answer
# the question, ask it; elsewhere the file is written by the deploy script and
# removed by the uninstall script, so its presence is the answer.
unit_live() {
	[ -f "$1" ] || return 1
	case "$1" in
	/etc/systemd/system/*)
		svc=${1##*/}
		systemctl is-enabled "$svc" >/dev/null 2>&1 ||
			systemctl is-active "$svc" >/dev/null 2>&1
		;;
	*) : ;;
	esac
}

worker_unit=""
for f in /etc/systemd/system/scan-worker.service \
	/usr/local/etc/rc.d/scan-worker \
	/Library/LaunchDaemons/com.atomdrift.scan-worker.plist; do
	if unit_live "$f"; then worker_unit="$f"; break; fi
done
server_unit=""
for f in /etc/systemd/system/scan.service /usr/local/etc/rc.d/scan; do
	if unit_live "$f"; then server_unit="$f"; break; fi
done

# Windows keeps no unit file at all: worker-windows.ps1 registers an nssm
# service, so the service manager is the only thing that can be asked. Named
# with the sc.exe suffix because `sc` alone is a builtin in some shells.
if [ -z "$worker_unit" ] && sc.exe query scan-worker >/dev/null 2>&1; then
	worker_unit="service:scan-worker"
fi

# The hopper host is not a scan host that also happens to run hopper: hopper's
# own `make deploy` installs hopper AND the scan worker beside it, from the
# ~/scan checkout, with the slot count and memory cap that box is tuned for.
# Deploying the worker from here would build the same binary a second time and
# then have hopper's deploy overwrite the service anyway, so the whole host is
# one `make deploy` in the other repository.
hopper_unit=""
for f in /etc/systemd/system/hopper.service /usr/local/etc/rc.d/hopper; do
	if unit_live "$f"; then hopper_unit="$f"; break; fi
done
[ -d "$HOME/hopper" ] || hopper_unit=""

# The Macs run their workers by hand rather than under launchd, so a Darwin box
# carrying no unit is not a host with nothing installed -- it is a host whose
# worker is just a process someone started. Every other platform reaching this
# point genuinely has nothing to redeploy.
adhoc=""
[ "$(uname -s)" != Darwin ] || adhoc=yes

if [ -z "$worker_unit" ] && [ -z "$server_unit" ] && [ -z "$hopper_unit" ] && [ -z "$adhoc" ]; then
	echo "rollout: no scan service installed; nothing to redeploy"
	exit 91
fi

rc=0

if [ -n "$hopper_unit" ]; then
	echo "rollout: hopper ($hopper_unit) — deploys hopper and its scan worker together"
	( cd "$HOME/hopper" && git pull --ff-only && "$mk" deploy ) || rc=1
	# hopper's deploy owns the worker on this box; do not redeploy it twice.
	worker_unit=""
fi

if [ -n "$worker_unit" ]; then
	# Redeploy this worker onto the hopper it was provisioned with, read back
	# out of its own unit, so a host pointed at a different queue keeps it. The
	# tag strip covers launchd's plist; the rest tokenizes systemd's ExecStart=
	# and rc.d's command_args= alike, and either `--url X` or `--url=X`.
	case "$worker_unit" in
	service:*)
		# nssm answers in UTF-16, which every tool after this would read as
		# one character followed by a NUL; drop them before tokenizing.
		url=$(nssm get scan-worker AppParameters 2>/dev/null |
			tr -d '\000\r' | tr -s ' \t"' '\n\n\n' |
			awk '/^--url=/ { sub(/^--url=/, ""); print; exit }
			     /^--url$/  { getline; print; exit }')
		;;
	*)
		url=$(sed 's,<[^>]*>, ,g' "$worker_unit" | tr -s ' \t"' '\n\n\n' |
			awk '/^--url=/ { sub(/^--url=/, ""); print; exit }
			     /^--url$/  { getline; print; exit }')
		;;
	esac
	[ -n "$url" ] || url="$URL_FALLBACK"
	echo "rollout: worker ($worker_unit), hopper $url"
	"$mk" stop-worker && "$mk" deploy-worker URL="$url" || rc=1
elif [ -n "$adhoc" ] && [ -z "$hopper_unit" ]; then
	# No unit to stop, no unit to install: rebuild and put the worker back the
	# way it was started, detached from this SSH session. All three descriptors
	# are redirected, not just stdout -- a child holding the pipe open would
	# keep ssh waiting here until the worker exited, which is never.
	echo "rollout: adhoc worker (macOS, no launchd unit), hopper $URL_FALLBACK"
	git stash
	git pull
	if "$mk" kill-scan && "$mk" release; then
		# Started through `make worker`, not by running the binary directly:
		# the LLM failover chain (and the hopper token check) is defined once,
		# in the Makefile, and a worker launched without SCAN_LLM falls back to
		# a loopback endpoint no Mac runs -- which is exactly how this arm
		# failed, dying on "no LLM model available from localhost:8000".
		# `release` has already run, so the dependency is a no-op here.
		nohup "$mk" worker URL="$URL_FALLBACK" \
			</dev/null >>"$HOME/scan-worker.log" 2>&1 &
		# Give it long enough to fail loudly (a missing token, a bad URL)
		# rather than reporting success for a process that already exited.
		sleep 3
		if kill -0 $! 2>/dev/null; then
			echo "rollout: adhoc worker running as pid $!, logging to ~/scan-worker.log"
		else
			echo "rollout: adhoc worker exited immediately; see ~/scan-worker.log"
			tail -20 "$HOME/scan-worker.log"
			rc=1
		fi
	else
		rc=1
	fi
fi

if [ -n "$server_unit" ]; then
	echo "rollout: server ($server_unit)"
	"$mk" deploy || rc=1
fi

exit "$rc"
REMOTE_EOF
}

# --- Connections ------------------------------------------------------------

work=$(mktemp -d) || die "could not create a working directory"

# The multiplexing sockets go in /tmp rather than next to the logs, because a
# unix socket path is capped at 104 bytes and `%C` spends 64 of them on a hash.
# macOS hands mktemp a per-user TMPDIR some 63 characters deep, which overruns
# that on its own: every connection then fails with "ControlPath too long" and
# the whole fleet reads as unreachable in well under a second.
ctl=$(mktemp -d /tmp/scan-rollout.XXXXXX) || die "could not create a socket directory"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$ctl/%C -o ControlPersist=900 \
	-o ConnectTimeout=15 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new"

# Belt and braces: if some other platform's mktemp is verbose enough to put us
# back over the limit, say so now rather than mistranslating it as a dead fleet.
[ "${#ctl}" -le 38 ] ||
	die "socket directory $ctl is too long for a ControlPath; set TMPDIR to something shorter"

# Masters outlive this script by ControlPersist unless they are closed, and a
# rollout should not leave authenticated sockets lying around behind it.
# shellcheck disable=SC2329 # invoked from the trap below
close_masters() {
	for host in $targets; do
		# shellcheck disable=SC2086 # SSH_OPTS is deliberately word-split
		ssh $SSH_OPTS -O exit "$(ssh_target "$host")" 2>/dev/null || true
	done
}
# shellcheck disable=SC2329 # invoked from the trap below
cleanup() {
	close_masters
	rm -rf "$ctl"
}
# Only EXIT cleans up. A signal trap that cleaned up without exiting would tear
# the socket directory out from under a rollout that then kept going: every
# later connection fails with "unix_listener: cannot bind to path ... No such
# file or directory" and the rest of the fleet reads as unreachable. Ctrl-C
# should stop the rollout, and stopping is what runs the cleanup.
trap 'cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# connect <host> — open the multiplexed master, which is the one moment
# authentication happens. Returns non-zero if the host cannot be reached.
connect() {
	target=$(ssh_target "$1")
	printf '    %s ... ' "$target"
	# ssh's own stderr is kept, not discarded: the reason a host is unreachable
	# is the whole value of this line. Throwing it away once turned a local
	# misconfiguration into twelve hosts that all looked dead.
	# The probe asks `uname -s` and judges the ANSWER, not the exit status.
	# cmd.exe reports "'x' is not recognized" on stderr and still exits 0, so
	# a command that merely succeeds proves nothing -- that is how a Windows
	# host passed as a POSIX one and its deploy died on "'sh' is not
	# recognized". A shell that names itself is a shell that can run this.
	# shellcheck disable=SC2086 # SSH_OPTS is deliberately word-split
	if [ -n "$(ssh -t $SSH_OPTS "$target" 'uname -s' 2>"$work/$1.connect" |
		tr -d '\r\n')" ]; then
		echo "sh -s" >"$work/$1.shell"
		echo "connected"
		return 0
	fi
	# Not unreachable, just answering as cmd.exe. Git for Windows ships the
	# shell the rest of this script is written for, so ask for it by path
	# before writing the host off. What answered is remembered here, so the
	# deploy is piped into the same shell the probe passed through.
	# shellcheck disable=SC2086 # SSH_OPTS is deliberately word-split
	if [ -n "$(ssh -t $SSH_OPTS "$target" "\"$WIN_SH\" -c \"uname -s\"" \
		2>>"$work/$1.connect" | tr -d '\r\n')" ]; then
		printf '"%s" -s\n' "$WIN_SH" >"$work/$1.shell"
		echo "connected (git bash)"
		return 0
	fi
	echo "UNREACHABLE"
	sed 's/^/        /' "$work/$1.connect" >&2
	record "$1" UNREACHABLE 0s -- --
	return 1
}

record() {
	printf '%s|%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" "$5" >"$work/$1.status"
}

# push_binary <host> — deploy to a host that cannot compile, by sending it a
# finished binary instead of asking it to build one.
#
# The binary is the published static musl build, not one made here or on
# another node. Every Linux box in this fleet now runs a glibc NEWER than the
# Steam Machine's 2.41 -- gandalf 2.43, nazgul and galadriel 2.44 -- so a
# binary built on any of them refuses to start there with a missing GLIBC_2.4x,
# and nothing in the fleet has docker, podman or cross to build against an
# older one. The musl artifact is static: it has no libc to disagree with.
#
# The cost is that this host tracks the last RELEASE rather than HEAD, which is
# said plainly in the log rather than hidden -- and is still an improvement on
# a deck sitting three releases behind because nothing redeploys it.
push_binary() {
	host="$1"
	target=$(ssh_target "$host")

	command -v gh >/dev/null 2>&1 ||
		{ echo "rollout: gh not installed; cannot fetch a release binary"; return 1; }

	# The newest tag that actually carries the artifact. The newest tag alone
	# is not enough: a release is created before its build matrix finishes, so
	# the current one can legitimately have no assets yet.
	tag=""
	for t in $(gh release list --limit 10 --json tagName -q '.[].tagName'); do
		if gh release view "$t" --json assets \
			-q '.assets[].name' 2>/dev/null |
			grep -q "x86_64-unknown-linux-musl"; then
			tag="$t"
			break
		fi
	done
	[ -n "$tag" ] ||
		{ echo "rollout: no release carries an x86_64-unknown-linux-musl asset"; return 1; }

	dir="$work/$host.push"
	mkdir -p "$dir"
	echo "rollout: fetching $tag x86_64-unknown-linux-musl"
	gh release download "$tag" -p '*x86_64-unknown-linux-musl.tar.gz' -D "$dir" --clobber ||
		{ echo "rollout: could not download the $tag musl artifact"; return 1; }
	tar -xzf "$dir"/*x86_64-unknown-linux-musl.tar.gz -C "$dir" ||
		{ echo "rollout: could not unpack the $tag musl artifact"; return 1; }
	bin=$(find "$dir" -type f -name atomscan -perm -u+x | head -1)
	[ -n "$bin" ] || { echo "rollout: no atomscan binary in the $tag artifact"; return 1; }

	# Land it beside the live one and swap, so a half-copied file is never what
	# the supervisor restarts into. The service is stopped first: sneaky-steam
	# takes its whole cgroup down with it, which is what releases the running
	# binary and lets the replacement take its name.
	# shellcheck disable=SC2086 # SSH_OPTS is deliberately word-split
	scp $SSH_OPTS -q "$bin" "$target:bin/atomscan.new" ||
		{ echo "rollout: could not copy the binary to $target"; return 1; }
	# shellcheck disable=SC2086 # SSH_OPTS is deliberately word-split
	ssh $SSH_OPTS "$target" sh -s <<-'PUSH_EOF'
		set -e
		systemctl --user stop sneaky-steam
		chmod +x "$HOME/bin/atomscan.new"
		mv -f "$HOME/bin/atomscan.new" "$HOME/bin/atomscan"
		systemctl --user start sneaky-steam
		# What it reports is the only proof the file both landed and runs;
		# a binary for the wrong libc fails exactly here.
		echo "rollout: steam now running $("$HOME/bin/atomscan" --version)"
		systemctl --user is-active sneaky-steam
	PUSH_EOF
}

field() { cut -d'|' -f"$2" "$work/$1.status" 2>/dev/null; }

# deploy_host <host> — the remote deploy, bounded by TIMEOUT, with the outcome
# left in a file so a parallel batch can be summarized once it finishes.
deploy_host() {
	host="$1"
	logfile="$work/$host.log"
	start=$(date +%s)

	# shellcheck disable=SC2086 # SSH_OPTS is deliberately word-split
	rsh=$(cat "$work/$host.shell" 2>/dev/null) || rsh="sh -s"
	[ -n "$rsh" ] || rsh="sh -s"
	# shellcheck disable=SC2086 # both SSH_OPTS and $rsh are deliberately split
	remote_script | ssh $SSH_OPTS "$(ssh_target "$host")" $rsh >"$logfile" 2>&1 &
	ssh_pid=$!
	# stdout is closed off deliberately: if this timer is still alive when the
	# rollout ends, an inherited pipe would hold `make rollout | tee` open for
	# the rest of TIMEOUT.
	(
		sleep "$TIMEOUT"
		kill "$ssh_pid" 2>/dev/null
	) >/dev/null 2>&1 &
	timer_pid=$!
	wait "$ssh_pid"
	code=$?
	kill "$timer_pid" 2>/dev/null
	wait "$timer_pid" 2>/dev/null

	elapsed=$(($(date +%s) - start))
	# awk, not sed: BRE alternation is a GNU extension and this runs locally,
	# which is as often as not a BSD userland.
	# Counted once each: an arm that reports progress as well as announcing
	# itself would otherwise show up as "adhoc+adhoc". The guards keep the
	# order the host did things in, which "worker+server" depends on.
	kind=$(awk '/^rollout: worker/ && !w++ { print "worker" }
	            /^rollout: steam/  && !t++ { print "steam" }
	            /^rollout: adhoc/  && !a++ { print "adhoc" }
	            /^rollout: hopper/ && !h++ { print "hopper" }
	            /^rollout: server/ && !v++ { print "server" }' "$logfile" | paste -sd+ -)
	case "$code" in
	0) status="ok" ;;
	90) status="no-repo" ;;
	91) status="no-service" ;;
	92)
		# The host said it cannot build. Send it a finished binary instead,
		# and time that too -- the download is the slow part, not the copy.
		if push_binary "$host" >>"$logfile" 2>&1; then status="ok"; else status="FAILED"; fi
		elapsed=$(($(date +%s) - start))
		;;
	*) if [ "$elapsed" -ge "$TIMEOUT" ]; then status="TIMEOUT"; else status="FAILED"; fi ;;
	esac

	# Say why, here, rather than making the operator wait out the rest of the
	# fleet to find out. Deploys run concurrently, so the whole report is
	# assembled first and written with a single printf: two hosts failing at
	# once then interleave as blocks rather than line by line.
	report=$(printf '==> [%s] %s in %ds\n' "$host" "$status" "$elapsed")
	case "$status" in
	FAILED | TIMEOUT)
		report="$report
$(sed 's/^/    /' "$logfile" | tail -15)
    ---- $logfile"
		;;
	esac
	printf '%s\n' "$report"
	record "$host" "$status" "${elapsed}s" "${kind:---}" --
}

# --- The health gate --------------------------------------------------------

# await_health <host> — poll the server's own /_/health over the SSH connection
# already open until it reports ok. A large uptime_secs is a failure, not a
# pass: it means we are reading the process the deploy was supposed to replace.
await_health() {
	host="$1"
	deadline=$(($(date +%s) + HEALTH_WAIT))
	last="unreachable"
	while :; do
		# shellcheck disable=SC2086,SC2029 # SSH_OPTS is word-split; HEALTH_ADDR
		# is meant to expand here, so the remote is handed a finished command
		body=$(ssh $SSH_OPTS "$(ssh_target "$host")" \
			"curl -sf --max-time 10 http://$HEALTH_ADDR/_/health" 2>/dev/null)
		status=$(printf '%s' "$body" | grep -o '"status"[^,]*' | head -1 | cut -d'"' -f4)
		uptime=$(printf '%s' "$body" | grep -o '"uptime_secs"[^,}]*' | head -1 | tr -dc '0-9')
		[ -n "$status" ] && last="$status"
		if [ "$status" = "ok" ]; then
			if [ -n "$uptime" ] && [ "$uptime" -gt "$HEALTH_UPTIME" ]; then
				warn "$host: healthy but uptime is ${uptime}s — the service did not restart"
				return 1
			fi
			note "$host: /_/health ok after ${uptime:-?}s of uptime"
			return 0
		fi
		if [ "$(date +%s)" -ge "$deadline" ]; then
			warn "$host: still '$last' after ${HEALTH_WAIT}s"
			return 1
		fi
		sleep 5
	done
}

# await_pin <host> — one pinned lookup through beamline. The pin names this
# backend and beamline never substitutes another, so a 200 here means the edge,
# the tunnel and this server are all back, and still agree on a known verdict.
await_pin() {
	host="$1"
	pin_alias=$(pin_for "$host")
	if [ -z "$pin_alias" ] || [ -z "$BEAMLINE" ]; then
		note "$host: no pinned query (no scan-* alias or no BEAMLINE)"
		return 0
	fi
	if [ -z "$BEAMLINE_TOKEN" ]; then
		warn "$host: no beamline token (~/.tok/beamline or BEAMLINE_TOKEN); skipping the pinned query"
		return 0
	fi

	pin="$pin_alias.$PIN_DOMAIN"
	body="$work/$host.pin"
	code=$(curl -s -o "$body" -w '%{http_code}' --max-time 90 \
		-H "Authorization: Bearer $BEAMLINE_TOKEN" \
		-H "X-Beamline-Pin: $pin" \
		"$BEAMLINE/v1/lookup?purl=$PIN_PURL" 2>/dev/null)
	if [ "$code" != "200" ]; then
		warn "$host: pinned lookup on $pin answered HTTP $code — $(head -c 200 "$body")"
		return 1
	fi
	severity=$(grep -o '"severity"[^,}]*' "$body" | head -1 | cut -d'"' -f4)
	if [ "$severity" != "$PIN_SEVERITY" ]; then
		warn "$host: pinned lookup on $pin returned severity '$severity', wanted '$PIN_SEVERITY'"
		return 1
	fi
	note "$host: pinned lookup on $pin returned $severity"
	return 0
}

# --- Phase 0: the queue, alone ----------------------------------------------
#
# Nothing else runs if this fails. Every worker claims its work from hopper and
# files its results back, so a fleet redeployed against a queue that did not
# come back is a fleet of idle processes -- and the failure that stopped it is
# far easier to read here than under seven workers' build output. Clearing the
# remaining phases rather than exiting outright keeps the summary and the log
# tail, which are the whole reason to know which host stopped the rollout.

if [ -n "$hopper" ]; then
	log "Hopper $hopper"
	connect "$hopper" && deploy_host "$hopper"
	if [ "$(field "$hopper" 2)" != "ok" ]; then
		warn "$hopper: the queue did not come back — halting before any worker or server"
		workers=""
		servers=""
	fi
fi

# --- Phase 1: workers, in parallel batches ----------------------------------

run_batch() {
	[ -n "$1" ] || return 0
	log "Opening connections — touch your YubiKey when prompted"
	for host in $1; do connect "$host" || true; done

	log "Deploying $(echo "$1" | wc -w | tr -d ' ') workers in parallel"
	for host in $1; do
		[ -f "$work/$host.status" ] || deploy_host "$host" &
	done
	wait
}

if [ -n "$workers" ]; then
	batch=""
	n=0
	for host in $workers; do
		batch="$batch $host"
		n=$((n + 1))
		if [ "$BATCH" -gt 0 ] && [ "$n" -ge "$BATCH" ]; then
			run_batch "$batch"
			batch=""
			n=0
		fi
	done
	run_batch "$batch"
fi

# --- Phase 2: servers -------------------------------------------------------
#
# One at a time by default, each proving itself healthy before the next is
# touched, because they are what callers reach and taking two down at once is
# an outage.
#
# SERVER_BATCH=0 (`make rollout-servers-fast`) gives that up and rolls the whole
# tier together. It is the right call in exactly two situations -- every server
# is already broken, so there is no working capacity left to protect, or the
# window is one where the tier may go down anyway -- and the wrong call the rest
# of the time. What it does NOT give up is the gate: every server is still
# health-checked and pin-queried afterwards, so a bad build is still caught. It
# is only the sequencing, and therefore the blast radius, that is traded away.

# gate_server <host> — the two questions a redeployed server must answer.
# Exit 0: came back, or had nothing to be asked. Exit 1: did not come back, and
# the chain should stop. Exit 2: something is wrong with this host, but nothing
# that says the NEXT server is at risk -- recorded, and the chain goes on.
# All three have already recorded their own outcome.
gate_server() {
	host="$1"

	# Gate only on what was actually deployed. The roster is a snapshot of the
	# last check-in, so a host that has since been turned back into a worker --
	# nazgul, whose `-idle` row outlived the server that filed it -- lands in
	# this phase with nothing to health-check. That is a demotion, not a
	# failure: the worker deploy already happened, and the gate simply has no
	# question to ask. A host that deployed NOTHING is caught by its status.
	case "$(field "$host" 4)" in
	*server*) ;;
	*worker* | *adhoc* | *hopper*)
		note "$host: roster said server, but this host runs no server — nothing to gate"
		record "$host" "$(field "$host" 2)" "$(field "$host" 3)" "$(field "$host" 4)" worker-only
		return 0
		;;
	*)
		warn "$host: roster says server, but nothing was deployed here"
		record "$host" "$(field "$host" 2)" "$(field "$host" 3)" "$(field "$host" 4)" "no-server"
		return 2
		;;
	esac

	if await_health "$host" && await_pin "$host"; then
		record "$host" ok "$(field "$host" 3)" "$(field "$host" 4)" healthy
		return 0
	fi
	record "$host" ok "$(field "$host" 3)" "$(field "$host" 4)" UNHEALTHY
	return 1
}

if [ "$SERVER_BATCH" -eq 1 ]; then
	for host in $servers; do
		log "Server $host"
		connect "$host" || continue
		deploy_host "$host"
		[ "$(field "$host" 2)" = "ok" ] || {
			warn "$host: deploy failed; not gating, and not moving on"
			break
		}
		gate_server "$host"
		case "$?" in
		0 | 2) ;;
		*)
			warn "stopping: $host did not come back healthy, and the next server is not worth risking"
			break
			;;
		esac
	done
elif [ -n "$servers" ]; then
	log "Opening connections — touch your YubiKey when prompted"
	for host in $servers; do connect "$host" || true; done

	warn "SERVER_BATCH=0: every server goes down together — the tier is offline until they return"
	log "Deploying $(echo "$servers" | wc -w | tr -d ' ') servers in parallel"
	for host in $servers; do
		[ -f "$work/$host.status" ] || deploy_host "$host" &
	done
	wait

	# Gating after the fact rather than between deploys. Nothing is protected
	# by stopping now -- they have all already been restarted -- so every
	# server is asked, and each answer is recorded rather than ending the run.
	log "Checking the tier came back"
	for host in $servers; do
		[ "$(field "$host" 2)" = "ok" ] || continue
		gate_server "$host"
		[ "$?" -ne 1 ] || warn "$host: did not come back healthy"
	done
fi

# --- Summary ----------------------------------------------------------------

printf '\n%-16s %-12s %-10s %-14s %s\n' "HOST" "STATUS" "DURATION" "DEPLOYED" "HEALTH"
printf '%-16s %-12s %-10s %-14s %s\n' "----------------" "------------" "----------" "--------------" "---------"
failed=0
for host in $targets; do
	entry=$(cat "$work/$host.status" 2>/dev/null || echo "$host|not-reached|--|--|--")
	IFS='|' read -r h status duration kind health <<-ENTRY
		$entry
	ENTRY
	printf '%-16s %-12s %-10s %-14s %s\n' "$h" "$status" "$duration" "$kind" "$health"
	case "$status" in
	FAILED | TIMEOUT | UNREACHABLE | not-reached) failed=1 ;;
	esac
	case "$health" in
	UNHEALTHY | no-server) failed=1 ;;
	esac
done

# No re-tail here: each failure already printed its own tail as it happened.

printf '\nLogs: %s\n' "$work"
exit "$failed"
