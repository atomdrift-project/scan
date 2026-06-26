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
# Usage: scan_worker_args <url> [workers]
scan_worker_args() {
	_lwa_url="$1"
	_lwa_workers="$2"
	_lwa_args="worker --url $_lwa_url --interpret"
	[ -n "$_lwa_workers" ] && _lwa_args="$_lwa_args --workers $_lwa_workers"
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
scan_rcd_script() {
	_lrs_bin="$1"
	_lrs_worker_args="$2"
	_lrs_llm="${3:-http://10.9.8.149:8000/v1}"
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

pidfile="/var/run/\${name}.pid"
command="/usr/sbin/daemon"
# MALLOC_CONF tunes FreeBSD's jemalloc to return freed memory to the OS
# promptly instead of holding dirty pages for 10s (the default). Critical
# under bursty analysis workloads where RSS would otherwise drift upward.
# Set via /usr/bin/env so it survives daemon(8)'s user switch and any
# login.conf environment filtering.
malloc_conf="dirty_decay_ms:1000,muzzy_decay_ms:0,background_thread:true,abort_conf:true"
# -r -R 5 : supervise the worker and restart it forever (5s back-off) after
#           any exit, so an OOM kill or panic self-heals. -P is the supervisor
#           pidfile; \`service scan-worker stop\` signals it to tear the
#           whole tree down.
command_args="-c -f -r -R 5 -P \${pidfile} -o \${scan_worker_logfile} -u scan /usr/bin/env MALLOC_CONF=\${malloc_conf} SCAN_LLM=\${scan_worker_llm} $_lrs_bin $_lrs_worker_args"

run_rc_command "\$1"
EOF
}
