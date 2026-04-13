#!/bin/bash
# update-nodes.sh - Deploy litmus to remote nodes via SSH in serial
# Usage: ./update-nodes.sh node1 [node2 ...]

if [ $# -eq 0 ]; then
    echo "usage: $0 <node> [node ...]" >&2
    exit 1
fi

git pull
git push

# Track results: "node:status:seconds"
results=""

for node in "$@"; do
    printf "==> [%s] deploying ...\n" "$node"
    start=$(date +%s)

    ssh -t "$node" "uname -a && uptime && cd litmus && make deploy"
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

# Summary table
printf "\n%-30s %-8s %s\n" "NODE" "STATUS" "DURATION"
printf "%-30s %-8s %s\n" "------------------------------" "--------" "--------"
for entry in $results; do
    node="${entry%%:*}"
    rest="${entry#*:}"
    status="${rest%%:*}"
    duration="${rest#*:}"
    printf "%-30s %-8s %s\n" "$node" "$status" "$duration"
done

# Exit non-zero if any node failed
for entry in $results; do
    case "$entry" in
        *:FAILED:*) exit 1 ;;
    esac
done
