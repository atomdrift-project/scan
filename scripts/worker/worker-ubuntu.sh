#!/bin/sh
# Deprecated wrapper. The Ubuntu deploy was migrated from cron to systemd and
# now lives in worker-linux.sh, which also supports Arch/CachyOS.
exec "$(dirname "$0")/worker-linux.sh" "$@"
