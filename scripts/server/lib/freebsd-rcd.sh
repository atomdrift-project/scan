#!/bin/sh
# freebsd-rcd.sh - Shared helpers for the FreeBSD Atomdrift Scan server rc.d service.
#
# Sourced by both the native host deploy (server-freebsd.sh) and the jailed
# deploy (rollout-bastille.sh) so a single definition of the service — daemon(8)
# flags, jemalloc tuning, scheduling priority, bounded stop, pidfile/logfile
# layout — drives both. This is the server counterpart of
# scripts/worker/lib/freebsd-rcd.sh, and for the same reason: a fix to the
# service contract lands on the host and in the jail at once.
#
# Pure shell: every function writes to stdout and touches no global state, so
# the caller pipes the output wherever it needs to land — a local file, or
# `bastille cmd <jail> tee`.

# Compose the `serve ...` argument string. Every parameter is optional; an
# empty one omits its flag and leaves atomscan's own default in force. That is
# deliberate: flag defaults are defined once, in atomscan, not repeated here.
#
# --hopper and --idle-worker-slots are deliberately NOT here: both are
# non-secret runtime knobs that live in rc.conf (scan_hopper, scan_idle_slots)
# so they can be changed with sysrc(8) and a restart, without a redeploy.
#
# Usage: scan_server_args [bind] [allow_cidr] [token_file] [workers] [allowed_dirs] [max_rss_gb]
scan_server_args() {
	_ssa_bind="${1:-}"
	_ssa_allow_cidr="${2:-}"
	_ssa_token_file="${3:-}"
	_ssa_workers="${4:-}"
	_ssa_allowed_dirs="${5:-}"
	_ssa_max_rss_gb="${6:-}"
	# -u refreshes models and traits before serving; failures are non-fatal.
	_ssa_args="-u serve"
	[ -n "$_ssa_bind" ] && _ssa_args="$_ssa_args --bind $_ssa_bind"
	[ -n "$_ssa_allow_cidr" ] && _ssa_args="$_ssa_args --allow-cidr $_ssa_allow_cidr"
	# atomscan refuses to start if the file is missing or empty, so a lost token
	# fails loudly instead of silently opening the API.
	[ -n "$_ssa_token_file" ] && _ssa_args="$_ssa_args --token-file $_ssa_token_file"
	[ -n "$_ssa_workers" ] && _ssa_args="$_ssa_args --workers $_ssa_workers"
	[ -n "$_ssa_allowed_dirs" ] && _ssa_args="$_ssa_args --allowed-dirs $_ssa_allowed_dirs"
	[ -n "$_ssa_max_rss_gb" ] && _ssa_args="$_ssa_args --max-rss-gb $_ssa_max_rss_gb"
	printf '%s' "$_ssa_args"
}

# Emit the rc.d service script to stdout.
# Usage: scan_server_rcd_script <binary_path> <server_args> [llm_url] [llm_model] [home]
#
# The LLM second-opinion pass is turned on by the SCAN_LLM environment
# variable rather than a flag, so an operator can switch endpoints — or turn
# the pass off entirely, with scan_llm="" — from rc.conf alone.
#
# The server is expected to run forever; an OOM kill or a panic should bring it
# straight back. daemon(8) is the supervisor: `-r` restarts the child whenever
# it exits, and `-R 5` paces those restarts 5s apart so a hard crash loop (bad
# bind address, missing token) backs off instead of pegging a core.
scan_server_rcd_script() {
	_ssr_bin="$1"
	_ssr_args="$2"
	# No fallback chain here: the deploy script that calls this owns that
	# default, and an empty value must stay empty so LLM= can turn the pass off.
	_ssr_llm="${3-}"
	_ssr_llm_model="${4:-}"
	_ssr_home="${5:-/home/scan}"
	cat <<EOF
#!/bin/sh

# PROVIDE: scan
# REQUIRE: LOGIN DAEMON NETWORKING
# KEYWORD: shutdown

. /etc/rc.subr

name="scan"
rcvar="scan_enable"

load_rc_config \$name

: \${scan_enable:="NO"}
: \${scan_logfile:="/var/log/scan.log"}

# Scheduling priority: the server is the reason this host exists, so it wins
# every CPU contest on it. -20 is the strongest nice(1) level; rc runs as root,
# so it is allowed. Realtime (rtprio) is deliberately not used: a CPU-bound
# analysis at realtime priority can starve sshd and lock the box out. Override
# with scan_nice= in rc.conf.
#
# nice(1) below covers the daemon(8) supervisor only — the priority does NOT
# survive into the server itself, which is why start_postcmd re-applies it.
# See scan_server_prio.
: \${scan_nice:="-20"}

# OOM priority. FreeBSD has no oom_score_adj; protect(1) sets P_PROTECTED,
# which exempts the process from the swap-exhaustion killer. The flag is
# inherited across fork and survives daemon(8)'s setuid to scan. Protection is
# all-or-nothing, so this puts the server *above* an unprotected sshd rather
# than just below it; protect the host's sshd too if you want that ordering:
#   doas protect -p "\$(pgrep -o sshd)"
: \${scan_protect:="YES"}

# OpenAI-compatible endpoint for the LLM second-opinion pass, passed as
# SCAN_LLM below. Empty turns the pass off without touching this script.
: \${scan_llm:="$_ssr_llm"}
# Pinned model (SCAN_LLM_MODEL). Empty leaves atomscan's own default: the
# largest model the endpoint serves, or \`openrouter/auto\` for OpenRouter.
: \${scan_llm_model:="$_ssr_llm_model"}

# --hopper renews every analyzed result on a hopper instance, and defers a
# lookup this server's index cannot answer to the same place. The URL is not a
# secret, so it comes from rc.conf (the deploy sets it with sysrc from HOPPER=);
# empty omits the flag. Its bearer token is a file, $_ssr_home/.tok/hopper,
# which atomscan finds through the scan user's HOME.
: \${scan_hopper:=""}
# --idle-worker-slots caps the embedded idle worker, which fills
# otherwise-idle analysis capacity with hopper queue work and pauses the moment
# a request arrives. Empty omits the flag and leaves the server default (half
# the slots); scan_idle_slots=0 turns background claiming off.
: \${scan_idle_slots:=""}

# Seconds a graceful stop waits for in-flight analyses to finish before the
# whole daemon(8) tree is killed. Bounds how long a redeploy or a reboot
# blocks. See scan_server_stop below.
: \${scan_stop_timeout:="30"}

pidfile="/var/run/\${name}.pid"

# nice(1) execs protect(1), which execs daemon(8), so procname must be set
# explicitly or rc.subr would look for /usr/bin/nice when matching the pidfile
# for status/stop.
command="/usr/bin/nice"
procname="/usr/sbin/daemon"

# MALLOC_CONF tunes FreeBSD's jemalloc to return freed memory to the OS
# promptly instead of holding dirty pages for 10s (the default). Critical
# under bursty analysis workloads where RSS would otherwise drift upward.
# Set via /usr/bin/env so it survives daemon(8)'s user switch and any
# login.conf environment filtering.
#
# Do NOT add background_thread:true here. FreeBSD's in-libc jemalloc is built
# without JEMALLOC_BACKGROUND_THREAD (libc cannot depend on libthr), so
# background_thread_boot0() fails — and it is called from malloc_init_hard()
# *after* malloc_init_state has already been set to malloc_init_recursible.
# Init then returns early and the state never reaches malloc_init_initialized,
# so malloc_initialized() is false forever and EVERY allocation in the process
# re-enters malloc_init_hard() and serializes on the global init_lock. It is
# not a config error, so abort_conf:true does not catch it and nothing is
# logged; the process just runs with a single-threaded malloc.
#
# junk:false turns off jemalloc's fill-on-malloc/fill-on-free debugging.
# FreeBSD builds libc's jemalloc with --enable-fill on -CURRENT (releases
# disable it), so without this every allocation and free memsets its region and
# the cost is charged to whatever called malloc, invisible in any profile.
# Measured on uruk-hai: -20% wall on a warm archive workload, with peak RSS
# 16.0 GiB -> 14.0 GiB over the same pair.
malloc_conf="dirty_decay_ms:1000,muzzy_decay_ms:0,abort_conf:true,junk:false"

_protect=""
case "\$scan_protect" in
[Yy][Ee][Ss] | [Tt][Rr][Uu][Ee] | 1) _protect="/usr/bin/protect" ;;
esac
_hopper=""
[ -n "\$scan_hopper" ] && _hopper="--hopper \${scan_hopper}"
_idle=""
[ -n "\$scan_idle_slots" ] && _idle="--idle-worker-slots \${scan_idle_slots}"
# Omitted rather than passed empty: an empty SCAN_LLM_MODEL is not the same as
# an unset one to the endpoint-probing path.
_llm_model=""
[ -n "\$scan_llm_model" ] && _llm_model="SCAN_LLM_MODEL=\${scan_llm_model}"

# HOME is set explicitly rather than left to daemon(8)'s user switch: atomscan
# resolves the hopper token, the OpenRouter key, and cleave's traits/models
# under it, and a service that silently ran with root's HOME would find none of
# them. RUST_BACKTRACE=1 so a panic inside an analysis names the frame that
# raised it; the cost is paid only on a panic, which is already a lost job.
command_args="-n \${scan_nice} \${_protect} /usr/sbin/daemon -c -f -r -R 5 -P \${pidfile} -o \${scan_logfile} -u scan /usr/bin/env HOME=$_ssr_home MALLOC_CONF=\${malloc_conf} RUST_BACKTRACE=1 SCAN_LLM=\${scan_llm} \${_llm_model} $_ssr_bin $_ssr_args \${_hopper} \${_idle}"

# daemon(8) -u switches user with setusercontext(3), which applies the login
# class's \`priority\` capability — 0 for the default class — *after* rc's
# nice(1) has run. The supervisor keeps \${scan_nice}; the server it forks is
# reset to 0, so the priority this service exists to hold is silently lost.
# Measured on uruk-hai: supervisor ni=-20, atomscan ni=0.
#
# Re-apply it to the child once it appears. rizin and cleave, forked later,
# inherit it. Caveat: a child daemon(8) restarts after a crash starts at the
# class priority again, until the next \`service scan restart\` — the durable
# alternative is a login class for the scan user (\`priority=-20\` in
# login.conf), which this deliberately does not install, since that would also
# reprioritise a worker sharing the account.
start_postcmd="scan_server_prio"
scan_server_prio()
{
	_p=0
	while [ \$_p -lt 10 ]; do
		_sup=\$(cat "\${pidfile}" 2>/dev/null)
		case "\${_sup}" in
		''|*[!0-9]*) _sup="" ;;
		esac
		if [ -n "\${_sup}" ]; then
			_kid=\$(pgrep -P "\${_sup}" 2>/dev/null | head -1)
			if [ -n "\${_kid}" ]; then
				# Absolute priority, not \`renice -n\`, which is an increment
				# on FreeBSD and would compound on every restart.
				renice "\${scan_nice}" -p "\${_kid}" >/dev/null 2>&1
				return 0
			fi
		fi
		sleep 1
		_p=\$((_p + 1))
	done
	echo "scan: could not find the supervised process to set priority \${scan_nice}." >&2
}

# Bounded, orphan-free stop. rc.subr's default sends SIGTERM to the supervisor
# and then waits on it forever (wait_for_pids), so a server draining a slow
# analysis wedges every redeploy until an operator kill -9s it by hand. This
# gives rc.d what the systemd unit gets from TimeoutStopSec: graceful SIGTERM,
# a short grace window, then a forced teardown that cannot orphan the child.
stop_cmd="scan_server_stop"
scan_server_stop()
{
	_sup=\$(cat "\${pidfile}" 2>/dev/null)
	case "\${_sup}" in
	''|*[!0-9]*) _sup="" ;;
	esac
	if [ -z "\${_sup}" ] || ! kill -0 "\${_sup}" 2>/dev/null; then
		echo "scan not running."
		rm -f "\${pidfile}"
		return 0
	fi
	# Collected before the supervisor dies: once it is SIGKILLed the server is
	# reparented to init and pgrep -P can no longer find it. Deliberately not
	# \`pkill -x atomscan\`, which on a host that also runs a worker would take
	# the worker down with it.
	_kids=\$(pgrep -P "\${_sup}" 2>/dev/null)
	echo "Stopping scan (pid \${_sup}); up to \${scan_stop_timeout}s to drain."
	kill -TERM "\${_sup}" 2>/dev/null
	_waited=0
	while kill -0 "\${_sup}" 2>/dev/null; do
		[ "\${_waited}" -ge "\${scan_stop_timeout}" ] && break
		sleep 1
		_waited=\$((_waited + 1))
	done
	if kill -0 "\${_sup}" 2>/dev/null; then
		echo "scan did not stop within \${scan_stop_timeout}s; forcing SIGKILL."
		# The supervisor first, so -r cannot respawn the child we are about to
		# kill; then the child itself, which a SIGKILLed supervisor can no
		# longer reap.
		kill -KILL "\${_sup}" 2>/dev/null
		for _kid in \${_kids}; do
			kill -KILL "\${_kid}" 2>/dev/null
		done
	fi
	rm -f "\${pidfile}"
}

run_rc_command "\$1"
EOF
}
