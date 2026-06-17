#!/bin/sh
# deploy-workers-tmux.sh - Redeploy the worker pool in parallel tmux windows.
#
# One tmux window per node, all running concurrently — but window launches are
# STAGGERED (default 5s) so the YubiKey only ever faces one SSH touch prompt at
# a time. The touch happens at connect time (~1s), so once a node is past auth
# its multi-minute build runs in parallel with the rest. Net effect: you tap
# the key ~5s apart a handful of times, then everything builds at once.
#
# Hopper is launched first so its (quick) redeploy gets a head start before the
# workers — which spend minutes rebuilding — come back and reconnect. This is a
# head start, not a hard barrier; if you need hopper fully up before any worker
# touches it, use the sequential deploy-workers.sh instead.
#
# Usage:
#   ./deploy-workers-tmux.sh [URL]
#
# Environment overrides:
#   URL            hopper endpoint the workers connect to (default below)
#   WORKER_NODES   space-separated worker hosts (default below)
#   HOPPER_NODE    host to redeploy hopper on   (default below)
#   STAGGER        seconds between window launches / touches (default 5)
#   SESSION        tmux session name (default ascan-deploy)
set -u

# Re-exec entry point: each tmux window runs `$0 _run <host> <label> <cmd>`.
# Splitting this out keeps the per-window command free of nested-quoting hazards
# and holds the pane open afterwards so you can read the result.
if [ "${1:-}" = "_run" ]; then
	host="$2"
	label="$3"
	cmd="$4"
	printf '>>> %s redeploy on %s — touch your YubiKey when it blinks\n\n' "$label" "$host"
	ssh -t -o ConnectTimeout=15 "$host" "$cmd"
	ec=$?
	if [ "$ec" -eq 0 ]; then
		printf '\n=== %s (%s) OK ===\n' "$host" "$label"
	else
		printf '\n=== %s (%s) FAILED (exit %s) ===\n' "$host" "$label" "$ec"
	fi
	printf 'Pane finished; close it with: Ctrl-b then &\n'
	exec "${SHELL:-sh}"
fi

URL="${1:-${URL:-http://10.9.8.5:8081/}}"
WORKER_NODES="${WORKER_NODES:-10.9.8.2 10.9.8.8 10.9.8.9 10.9.8.149 10.9.8.13}"
HOPPER_NODE="${HOPPER_NODE:-10.9.8.5}"
STAGGER="${STAGGER:-5}"
SESSION="${SESSION:-ascan-deploy}"

command -v tmux >/dev/null 2>&1 || { echo "error: tmux not found" >&2; exit 1; }

# Absolute path to this script so tmux windows can re-exec it regardless of cwd.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

worker_cmd="cd scan && git pull && make stop-worker && make deploy-worker URL=$URL"
hopper_cmd="cd hopper && git pull && make deploy"

# Fresh session each run.
tmux kill-session -t "$SESSION" 2>/dev/null || true

session_started=0
launch() {
	host="$1"
	label="$2"
	cmd="$3"
	# "$SELF" is quoted inside the window command in case the repo path has spaces.
	wcmd="\"$SELF\" _run '$host' '$label' '$cmd'"
	if [ "$session_started" = "0" ]; then
		tmux new-session -d -s "$SESSION" -n "$host" "$wcmd"
		session_started=1
	else
		tmux new-window -t "$SESSION" -n "$host" "$wcmd"
	fi
	printf '==> launched %-15s (%s)\n' "$host" "$label"
}

echo "Launching $(echo "$WORKER_NODES" | wc -w | tr -d ' ') workers + hopper into tmux session '$SESSION', ${STAGGER}s apart."
echo "Touch your YubiKey each time the key blinks (one node at a time)."
echo

# Hopper first (head start), then the workers, each spaced by $STAGGER so the
# touch prompts never overlap.
launch "$HOPPER_NODE" "hopper" "$hopper_cmd"
for node in $WORKER_NODES; do
	sleep "$STAGGER"
	launch "$node" "worker" "$worker_cmd"
done

echo
echo "All windows launched. Attaching — switch windows with Ctrl-b then n/p or a number."

# Attach (or switch, if we're already inside tmux).
if [ -n "${TMUX:-}" ]; then
	tmux switch-client -t "$SESSION"
else
	tmux attach-session -t "$SESSION"
fi
