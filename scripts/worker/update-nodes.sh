#!/bin/sh
# update-nodes.sh - Deploy litmus worker to remote nodes via SSH in parallel
# Usage: ./update-nodes.sh <url> <node> [node ...]

if [ $# -lt 2 ]; then
    echo "usage: $0 <url> <node> [node ...]" >&2
    exit 1
fi

URL="$1"
shift

git pull
git push

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

deploy_node() {
    node="$1"
    start=$(date +%s)

    if ! ping -c1 -W2 "$node" >/dev/null 2>&1; then
        printf "==> [%s] UNREACHABLE (ping failed)\n" "$node"
        echo "$node:UNREACHABLE:0s" > "$tmpdir/$node"
        return
    fi

    printf "==> [%s] deploying ...\n" "$node"

    ssh -o ConnectTimeout=10 "$node" "uname -a && uptime && cd litmus && git pull && make deploy-worker URL=$URL" &
    ssh_pid=$!
    ( sleep 300; kill "$ssh_pid" 2>/dev/null ) &
    timer_pid=$!
    wait "$ssh_pid"
    exit_code=$?
    kill "$timer_pid" 2>/dev/null
    wait "$timer_pid" 2>/dev/null

    elapsed=$(( $(date +%s) - start ))
    if [ "$exit_code" -eq 0 ]; then
        status="ok"
    else
        status="FAILED"
    fi

    printf "==> [%s] %s in %ds\n" "$node" "$status" "${elapsed}"
    echo "$node:$status:${elapsed}s" > "$tmpdir/$node"
}

for node in "$@"; do
    deploy_node "$node" &
done
wait

printf "\n%-30s %-8s %s\n" "NODE" "STATUS" "DURATION"
printf "%-30s %-8s %s\n" "------------------------------" "--------" "--------"
failed=0
for node in "$@"; do
    entry=$(cat "$tmpdir/$node" 2>/dev/null || echo "$node:MISSING:0s")
    n="${entry%%:*}"
    rest="${entry#*:}"
    status="${rest%%:*}"
    duration="${rest#*:}"
    printf "%-30s %-8s %s\n" "$n" "$status" "$duration"
    case "$status" in
        FAILED|UNREACHABLE|MISSING) failed=1 ;;
    esac
done

exit $failed
