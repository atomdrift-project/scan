#!/bin/sh
# freebsd-rcd.sh - Shared helpers for the FreeBSD Atomdrift Scan worker rc.d service.
#
# Sourced by both the native host deploy (worker-freebsd.sh) and the jailed
# deploy (worker-bastille.sh) so a single definition of the rc.d service —
# daemon(8) flags, jemalloc tuning, supervised-restart policy, pidfile/logfile
# layout — drives both. Keeping it in one place means a fix to the service
# contract (e.g. restart behaviour) lands in the jail and the host at once.
#
# Pure shell: every function writes to stdout and touches no global state, so
# the caller pipes the output wherever it needs to land — a local file, or
# `bastille cmd <jail> tee`.

# Compose the `worker ...` argument string from a hopper URL and an optional
# worker count. The interpret gate (min ML probability) is left at the binary's
# default. The endpoint itself is supplied via the SCAN_LLM environment variable
# in scan_rcd_script, not here.
# Usage: scan_worker_args <url> [workers] [data_dir] [max_rss_gb]
scan_worker_args() {
	_lwa_url="$1"
	_lwa_workers="$2"
	_lwa_data_dir="${3:-}"
	_lwa_max_rss_gb="${4:-}"
	_lwa_args="worker --url $_lwa_url --interpret"
	[ -n "$_lwa_workers" ] && _lwa_args="$_lwa_args --workers $_lwa_workers"
	[ -n "$_lwa_data_dir" ] && _lwa_args="$_lwa_args --data-dir $_lwa_data_dir"
	[ -n "$_lwa_max_rss_gb" ] && _lwa_args="$_lwa_args --max-rss-gb $_lwa_max_rss_gb"
	printf '%s' "$_lwa_args"
}

# Emit the rc.d service script to stdout.
# Usage: scan_rcd_script <binary_path> <worker_args> [llm_url]
#
# llm_url is baked into the service as the SCAN_LLM environment variable (the
# OpenAI-compatible endpoint for the --interpret LLM second-opinion pass);
# override at runtime via scan_worker_llm in rc.conf.
#
# The worker is expected to run forever; an OOM kill or a panic should bring
# it straight back. daemon(8) is the supervisor: `-r` restarts the child
# whenever it exits (any status), and `-R 5` paces those restarts 5s apart so
# a hard crash loop (bad URL, missing model) backs off instead of pegging a
# core. rc's `scan_worker_enable=YES` brings it back across reboots, so the
# only way the worker stays down is an explicit `service scan-worker stop`.
#
# The generated script overrides rc.subr's default stop with a *bounded* one.
# The default sends SIGTERM to the supervisor and then waits on it forever
# (wait_for_pids), so a busy worker's drain — or a wedged rayon unpack — wedges
# every redeploy until an operator `kill -9`s it by hand. This is FreeBSD-only
# pain: the systemd (TimeoutStopSec) and launchd (worker-macos.sh) deploys
# already SIGTERM-then-SIGKILL. scan_worker_stop below gives rc.d the same:
# graceful SIGTERM, a short grace window, then a forced teardown that can never
# orphan the child.
scan_rcd_script() {
	_lrs_bin="$1"
	_lrs_worker_args="$2"
	_lrs_llm="${3:-http://10.9.8.149:8000/v1}"
	# Basename for the force-kill sweep in scan_worker_stop; matches pkill -x's
	# comm comparison (e.g. /usr/local/bin/atomscan -> atomscan).
	_lrs_binname=$(basename "$_lrs_bin")
	cat <<EOF
#!/bin/sh

# PROVIDE: scan_worker
# REQUIRE: LOGIN DAEMON NETWORKING
# KEYWORD: shutdown

. /etc/rc.subr

name="scan_worker"
rcvar="scan_worker_enable"

load_rc_config \$name

: \${scan_worker_enable:="NO"}
: \${scan_worker_logfile:="/var/log/scan-worker.log"}
# OpenAI-compatible endpoint for the --interpret LLM second-opinion pass.
: \${scan_worker_llm:="$_lrs_llm"}
# The hopper API token is NOT read here. It is a file in the service account's
# home (~/.tok/hopper), installed by worker-freebsd.sh / worker-bastille.sh.
# This service used to source /usr/local/etc/hopper/env for the retired
# HOPPER_UPLOAD_TOKEN; that path is gone, so there is exactly one place the
# token lives and rc.conf never holds a secret.
# Seconds a graceful stop waits for the worker to drain in-flight analyses
# before the whole daemon(8) tree is SIGKILLed. Bounds how long a redeploy or
# reboot blocks; hopper re-leases anything that does not finish. Sized a few
# seconds above the worker's own drain cap so a healthy worker exits cleanly on
# its own and this force-kill is only reached when it is genuinely wedged.
: \${scan_worker_stop_timeout:="20"}

pidfile="/var/run/\${name}.pid"
command="/usr/sbin/daemon"
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
# Measured on uruk-hai 2026-08-03 (128 cores, --workers 96): with the option,
# cpu_cores_busy 14-28/128 and the worker wedged with all 96 slots occupied and
# zero completions for 14h27m; without it, 126/128 and steady completions.
#
# junk:false turns off jemalloc's fill-on-malloc/fill-on-free debugging. FreeBSD
# builds libc's jemalloc with --enable-fill on -CURRENT (releases disable it),
# so without this every allocation and free memsets its region and the cost is
# charged to whatever called malloc, invisible in any profile. Measured on
# uruk-hai 2026-08-27 (128 cores, --workers 96) over four large nested archives
# with warm YARA caches: 284.7s stock, 227.7s with junk:false (-20%), identical
# result hashes. Peak RSS fell 16.0 GiB -> 14.0 GiB over the same pair.
malloc_conf="dirty_decay_ms:1000,muzzy_decay_ms:0,abort_conf:true,junk:false"
# -r -R 5 : supervise the worker and restart it forever (5s back-off) after
#           any exit, so an OOM kill or panic self-heals. -P is the supervisor
#           pidfile; \`service scan-worker stop\` signals it to tear the
#           whole tree down.
command_args="-c -f -r -R 5 -P \${pidfile} -o \${scan_worker_logfile} -u scan /usr/bin/env MALLOC_CONF=\${malloc_conf} SCAN_LLM=\${scan_worker_llm} $_lrs_bin $_lrs_worker_args"

# Bounded, orphan-free stop (see the header comment for why the default won't
# do). SIGTERM the daemon(8) supervisor — it forwards the signal to the worker,
# which drains and exits, after which the supervisor removes \${pidfile} and
# exits too. Wait at most \${scan_worker_stop_timeout}s for that, then force the
# tree down: SIGKILL the supervisor first so -r cannot respawn, then SIGKILL any
# worker child still standing (a SIGKILLed supervisor cannot reap its own
# child). The pkill sweep is by exact name and jail-local, so it also cleans up
# a child orphaned by an older rc.d that predates this stop.
stop_cmd="scan_worker_stop"
scan_worker_stop()
{
	_sup=\$(cat "\${pidfile}" 2>/dev/null)
	case "\${_sup}" in
	''|*[!0-9]*) _sup="" ;;
	esac
	if [ -z "\${_sup}" ] || ! kill -0 "\${_sup}" 2>/dev/null; then
		echo "scan_worker not running."
		rm -f "\${pidfile}"
		return 0
	fi
	echo "Stopping scan_worker (pid \${_sup}); up to \${scan_worker_stop_timeout}s to drain."
	kill -TERM "\${_sup}" 2>/dev/null
	_waited=0
	while kill -0 "\${_sup}" 2>/dev/null; do
		[ "\${_waited}" -ge "\${scan_worker_stop_timeout}" ] && break
		sleep 1
		_waited=\$((_waited + 1))
	done
	if kill -0 "\${_sup}" 2>/dev/null; then
		echo "scan_worker did not stop within \${scan_worker_stop_timeout}s; forcing SIGKILL."
		kill -KILL "\${_sup}" 2>/dev/null
		pkill -9 -x $_lrs_binname 2>/dev/null || true
	fi
	rm -f "\${pidfile}"
}

run_rc_command "\$1"
EOF
}
