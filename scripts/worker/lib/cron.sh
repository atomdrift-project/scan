#!/bin/sh

# Quote one value for the /bin/sh command stored in a crontab. Newlines cannot
# appear in a crontab entry, so reject them instead of creating a second command.
# Cron consumes unescaped percent signs before invoking the shell, even inside
# quotes, so protect those here too.
scan_shell_quote() {
    case "$1" in
        *'
'*) return 1 ;;
    esac
    printf "'"
    printf '%s' "$1" | sed -e 's/%/\\%/g' -e "s/'/'\\\\''/g"
    printf "'"
}
