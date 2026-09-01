#!/bin/sh
# worker-openbsd.sh - Deploy Atomdrift Scan worker on OpenBSD
# Usage: ./worker-openbsd.sh <url>
# Runs on the local machine. Re-run to update.
# Must be invoked from the repository root.
#
# doas is required only for package installation. Add to /etc/doas.conf:
#   permit nopass <youruser> as root cmd pkg_add

set -ex

URL="$1"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

# Optional: cap concurrent analysis slots (--workers). Unset = worker auto.
# HOPPER_TOKEN_FILE names the hopper API token to install (default: ~/.tok/hopper).
# LLM_TOKEN_FILE names the bearer token for the LLM endpoint, which requires one
# (default: ~/.tok/llm).
WORKERS="${WORKERS:-}"
# LLM second-opinion pass: endpoint (exported as SCAN_LLM) + interpret gate.
LLM="${LLM:-https://llm.isotope13.ai/v1,openrouter}"
BINARY=atomscan
BIN_DIR="$HOME/bin"
LOG="$HOME/.local/share/atomdrift/scan/scan-worker.log"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
# shellcheck source=scripts/worker/lib/cron.sh
. "$SCRIPT_DIR/lib/cron.sh"

log "Installing dependencies"
set --
for p in rust git p7zip rizin innoextract; do
    pkg_info -q | grep -q "^${p}-" || set -- "$@" "$p"
done
if [ "$#" -gt 0 ]; then
    doas pkg_add -I "$@"
fi

log "Building"
# OpenBSD /bin/sh exposes the data-segment hard limit through ulimit -Hd.
# shellcheck disable=SC3045
ulimit -d "$(ulimit -Hd)"
cargo build --release || die "build failed"

mkdir -p "$BIN_DIR" "$(dirname "$LOG")"

# --- Hopper API token --------------------------------------------------------
#
# Hopper requires `Authorization: Bearer <token>` on every API route, so a
# worker without this file cannot claim work. The cron job below runs as this
# same user, so the worker reads ~/.tok/hopper directly; a HOPPER_TOKEN_FILE
# pointing elsewhere is copied there, because cron does not carry the deploy
# environment. A rotated token forces the restart below: the worker reads it
# once, at startup.
restart_needed=0
HOPPER_TOKEN_SRC="${HOPPER_TOKEN_FILE:-$HOME/.tok/hopper}"
HOPPER_TOKEN_DST="$HOME/.tok/hopper"
if [ ! -s "$HOPPER_TOKEN_SRC" ]; then
    # Not fatal: a hopper deployed without --token-file needs no client token.
    log "WARNING: no hopper API token at $HOPPER_TOKEN_SRC; this worker cannot claim work from an authenticated hopper"
elif [ "$HOPPER_TOKEN_SRC" != "$HOPPER_TOKEN_DST" ]; then
    mkdir -p "$HOME/.tok" && chmod 700 "$HOME/.tok"
    cmp -s "$HOPPER_TOKEN_SRC" "$HOPPER_TOKEN_DST" 2>/dev/null || restart_needed=1
    install -m 0600 "$HOPPER_TOKEN_SRC" "$HOPPER_TOKEN_DST"
    log "Installed hopper API token at $HOPPER_TOKEN_DST"
fi

# --- LLM endpoint token ------------------------------------------------------
#
# Our vLLM endpoint requires `Authorization: Bearer <token>`; the worker reads
# it from $HOME/.tok/llm. The cron job below runs as this same user, so
# ~/.tok/llm is read in place; an LLM_TOKEN_FILE pointing elsewhere is copied
# there, because cron does not carry the deploy environment.
#
# Not fatal when absent: every interpret call is refused with 401 and the
# verdict falls back to ML alone. That is silent at runtime, so warn here.
LLM_TOKEN_SRC="${LLM_TOKEN_FILE:-$HOME/.tok/llm}"
LLM_TOKEN_DST="$HOME/.tok/llm"
if [ ! -s "$LLM_TOKEN_SRC" ]; then
    log "WARNING: no LLM token at $LLM_TOKEN_SRC; $LLM will refuse the second-opinion pass with 401"
elif [ "$LLM_TOKEN_SRC" != "$LLM_TOKEN_DST" ]; then
    mkdir -p "$HOME/.tok" && chmod 700 "$HOME/.tok"
    cmp -s "$LLM_TOKEN_SRC" "$LLM_TOKEN_DST" 2>/dev/null || restart_needed=1
    install -m 0600 "$LLM_TOKEN_SRC" "$LLM_TOKEN_DST"
    log "Installed LLM endpoint token at $LLM_TOKEN_DST"
fi

log "Installing binary"
if ! cmp -s "target/release/$BINARY" "$BIN_DIR/$BINARY" 2>/dev/null; then
    install -m 755 "target/release/$BINARY" "$BIN_DIR/$BINARY"
    restart_needed=1
fi

log "Installing cron entry"
cron_url=$(scan_shell_quote "$URL") || die "URL cannot contain a newline"
cron_llm=$(scan_shell_quote "$LLM") || die "LLM URL cannot contain a newline"
cron_binary=$(scan_shell_quote "$BIN_DIR/$BINARY") || die "binary path cannot contain a newline"
cron_log=$(scan_shell_quote "$LOG") || die "log path cannot contain a newline"
cron_args="worker --url $cron_url --interpret"
if [ -n "$WORKERS" ]; then
    cron_workers=$(scan_shell_quote "$WORKERS") || die "worker count cannot contain a newline"
    cron_args="$cron_args --workers $cron_workers"
fi
cron_cmd="* * * * * pgrep -af 'atomscan worker' >/dev/null 2>&1 || { ulimit -d \$(ulimit -Hd); SCAN_LLM=$cron_llm nohup $cron_binary $cron_args < /dev/null >> $cron_log 2>&1 & }"
(crontab -l 2>/dev/null | grep -v "atomscan worker" || true; echo "$cron_cmd") | crontab -

if [ "$restart_needed" -eq 1 ]; then
    log "Restarting atomscan worker"
    pkill -f "atomscan worker" 2>/dev/null || true
    sleep 1
    if [ -n "$WORKERS" ]; then
        SCAN_LLM="$LLM" nohup "$BIN_DIR/$BINARY" worker --url "$URL" --interpret \
            --workers "$WORKERS" < /dev/null >> "$LOG" 2>&1 &
    else
        SCAN_LLM="$LLM" nohup "$BIN_DIR/$BINARY" worker --url "$URL" --interpret \
            < /dev/null >> "$LOG" 2>&1 &
    fi
else
    log "Binary unchanged, skipping restart"
fi

log "Deployment complete"
