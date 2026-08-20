#!/bin/sh
# deploy-workers.sh - Redeploy the Atomdrift Scan worker pool, one node at a time.
#
# The hopper node is redeployed FIRST, over SSH:
#     cd hopper && git pull && make deploy
# so the fresh queue server is up before the workers reconnect to it. Then each
# worker node, over SSH:
#     cd scan && git pull && make stop-worker && make deploy-worker URL=<url>
#
# Each worker node installs the hopper API token from its own
# ~/.tok/hopper (that account's, on that host) — nothing is shipped between
# hosts here, so provision that file on each node before the first roll.
#
# Nodes are processed strictly sequentially (never in parallel) because each
# SSH login needs a YubiKey touch — overlapping logins would race the token.
# An interactive TTY is allocated (ssh -t) so the touch/PIN prompt reaches you.
#
# Usage:
#   ./deploy-workers.sh [URL]
#
# Environment overrides:
#   URL            hopper endpoint the workers connect to (default below)
#   WORKER_NODES   space-separated worker hosts (default below)
#   HOPPER_NODE    host to redeploy hopper on   (default below)
set -u

URL="${1:-${URL:-http://10.9.8.10:8081/}}"
WORKER_NODES="${WORKER_NODES:-10.9.8.2 10.9.8.8 10.9.8.9 10.9.8.149 10.9.8.13}"
HOPPER_NODE="${HOPPER_NODE:-10.9.8.10}"

SSH_OPTS="-t -o ConnectTimeout=15"

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*" >&2; }

results=""
failed=0

# run_node <host> <label> <remote-command>
run_node() {
	host="$1"
	label="$2"
	cmd="$3"
	start=$(date +%s)

	log "[$host] $label — touch your YubiKey when prompted"
	printf '    %s\n' "$cmd"

	# SSH_OPTS is intentionally word-split; $cmd is meant to expand here so the
	# URL/targets are baked into the string the remote shell runs.
	# shellcheck disable=SC2086,SC2029
	if ssh $SSH_OPTS "$host" "$cmd"; then
		status="ok"
	else
		status="FAILED"
		failed=1
		warn "[$host] $label FAILED — a worker may be stopped but not redeployed; check this node before moving on."
	fi

	elapsed=$(( $(date +%s) - start ))
	printf '==> [%s] %s in %ds\n' "$host" "$status" "$elapsed"
	results="$results $host:$status:${elapsed}s"
}

# Hopper first: bring up the fresh queue server before the workers reconnect.
run_node "$HOPPER_NODE" "hopper redeploy" \
	"cd hopper && git pull && make deploy"

# Then the workers, in the given order.
for node in $WORKER_NODES; do
	run_node "$node" "worker redeploy" \
		"cd scan && git pull && make stop-worker && make deploy-worker URL=$URL"
done

# Summary.
printf '\n%-16s %-8s %s\n' "NODE" "STATUS" "DURATION"
printf '%-16s %-8s %s\n' "----------------" "--------" "--------"
for entry in $results; do
	host="${entry%%:*}"
	rest="${entry#*:}"
	status="${rest%%:*}"
	duration="${rest#*:}"
	printf '%-16s %-8s %s\n' "$host" "$status" "$duration"
done

exit "$failed"
