#!/bin/sh
# update-nodes.sh - Deploy litmus worker to remote nodes via SSH in serial
# Usage: ./update-nodes.sh <url> <node> [node ...]

if [ $# -lt 2 ]; then
    echo "usage: $0 <url> <node> [node ...]" >&2
    exit 1
fi

URL="$1"
shift

git pull
git push

results=""

for node in "$@"; do
    printf "==> [%s] deploying ...\n" "$node"
    start=$(date +%s)

    ssh -t "$node" "uname -a && uptime && cd litmus && make deploy-worker URL=$URL"
    exit_code=$?

    elapsed=$(( $(date +%s) - start ))
    if [ "$exit_code" -eq 0 ]; then
        status="ok"
    else
        status="FAILED"
    fi

    results="$results $node:$status:${elapsed}s"
    printf "==> [%s] %s in %ds\n\n" "$node" "$status" "$elapsed"
done

printf "\n%-30s %-8s %s\n" "NODE" "STATUS" "DURATION"
printf "%-30s %-8s %s\n" "------------------------------" "--------" "--------"
for entry in $results; do
    node="${entry%%:*}"
    rest="${entry#*:}"
    status="${rest%%:*}"
    duration="${rest#*:}"
    printf "%-30s %-8s %s\n" "$node" "$status" "$duration"
done

for entry in $results; do
    case "$entry" in
        *:FAILED:*) exit 1 ;;
    esac
done
