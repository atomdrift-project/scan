#!/bin/sh
# Deprecated wrapper. Uninstall logic moved to uninstall-linux.sh.
exec "$(dirname "$0")/uninstall-linux.sh" "$@"
