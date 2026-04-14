#!/bin/sh
# uninstall-nodes.sh - Remove litmus server persistence from remote nodes via SSH
# Usage: ./uninstall-nodes.sh <node> [node ...]

if [ $# -eq 0 ]; then
    echo "usage: $0 <node> [node ...]" >&2
    exit 1
fi

results=""

for node in "$@"; do
    printf "==> [%s] uninstalling ...\n" "$node"
    start=$(date +%s)

    ssh -t "$node" "uname -a && cd litmus && git pull && make uninstall-server"
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
