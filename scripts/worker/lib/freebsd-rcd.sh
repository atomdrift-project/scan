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
# worker count. Usage: ascan_worker_args <url> [workers]
ascan_worker_args() {
	_lwa_url="$1"
	_lwa_workers="$2"
	_lwa_args="worker --url $_lwa_url"
	[ -n "$_lwa_workers" ] && _lwa_args="$_lwa_args --workers $_lwa_workers"
	printf '%s' "$_lwa_args"
}

# Emit the rc.d service script to stdout.
# Usage: ascan_rcd_script <binary_path> <worker_args>
#
# The worker is expected to run forever; an OOM kill or a panic should bring
# it straight back. daemon(8) is the supervisor: `-r` restarts the child
# whenever it exits (any status), and `-R 5` paces those restarts 5s apart so
# a hard crash loop (bad URL, missing model) backs off instead of pegging a
# core. rc's `ascan_worker_enable=YES` brings it back across reboots, so the
# only way the worker stays down is an explicit `service ascan-worker stop`.
ascan_rcd_script() {
	_lrs_bin="$1"
	_lrs_worker_args="$2"
	cat <<EOF
#!/bin/sh

# PROVIDE: ascan_worker
# REQUIRE: LOGIN DAEMON NETWORKING
# KEYWORD: shutdown

. /etc/rc.subr

name="ascan_worker"
rcvar="ascan_worker_enable"

load_rc_config \$name

: \${ascan_worker_enable:="NO"}
: \${ascan_worker_logfile:="/var/log/ascan-worker.log"}

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
#           pidfile; \`service ascan-worker stop\` signals it to tear the
#           whole tree down.
command_args="-c -f -r -R 5 -P \${pidfile} -o \${ascan_worker_logfile} -u ascan /usr/bin/env MALLOC_CONF=\${malloc_conf} $_lrs_bin $_lrs_worker_args"

run_rc_command "\$1"
EOF
}
