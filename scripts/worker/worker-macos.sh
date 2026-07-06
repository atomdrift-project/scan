#!/bin/sh
# worker-macos.sh - Deploy Atomdrift Scan worker on macOS using launchd
# Usage: ./worker-macos.sh <server-url>
# Runs entirely on the local machine. Re-run to update.
# Must be invoked from the repository root.

set -ex

URL="$1"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

# Optional: cap concurrent analysis slots (--workers). Unset = worker auto.
WORKERS="${WORKERS:-}"
# LLM second-opinion pass: endpoint (exported as SCAN_LLM) + interpret gate.
LLM="${LLM:-http://10.9.8.149:8000/v1}"

BINARY=atomscan
INSTALL_DIR=/usr/local/share/atomdrift/scan
MODELS_DIR=/usr/local/share/atomdrift/scan/models
TRAITS_DIR=/usr/local/share/atomdrift/scan/traits
BIN_PATH=/usr/local/bin/atomscan
PLIST=/Library/LaunchDaemons/com.atomdrift.scan-worker.plist
LABEL=com.atomdrift.scan-worker
SERVICE_USER=_scan
LOG=/var/log/scan-worker.log

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

# Privilege escalation: prefer doas, fall back to sudo.
if   command -v doas >/dev/null 2>&1; then SUDO=doas
elif command -v sudo >/dev/null 2>&1; then SUDO=sudo
else die "need doas or sudo"
fi

for brew_candidate in /usr/local/bin/brew /opt/homebrew/bin/brew; do
    [ -x "$brew_candidate" ] && eval "$("$brew_candidate" shellenv)" && break
done
command -v brew >/dev/null 2>&1 || die "brew not found"

BREW_PREFIX=$(brew --prefix)

log "Installing dependencies"
brew install rust sccache p7zip upx rizin innoextract

log "Building release binary"
RUSTC_WRAPPER=sccache make release || die "build failed"

log "Ensuring service user '$SERVICE_USER' exists"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    uid=300
    while dscl . -list /Users UniqueID | awk '{print $2}' | grep -qx "$uid"; do
        uid=$((uid + 1))
    done
    $SUDO dscl . -create "/Users/$SERVICE_USER"
    $SUDO dscl . -create "/Users/$SERVICE_USER" UserShell /usr/bin/false
    $SUDO dscl . -create "/Users/$SERVICE_USER" RealName "Atomdrift Scan Worker"
    $SUDO dscl . -create "/Users/$SERVICE_USER" UniqueID "$uid"
    $SUDO dscl . -create "/Users/$SERVICE_USER" PrimaryGroupID 1
    $SUDO dscl . -create "/Users/$SERVICE_USER" NFSHomeDirectory /var/empty
fi

if [ ! -d "$INSTALL_DIR" ] || [ ! -w "$INSTALL_DIR" ]; then
    $SUDO mkdir -p "$INSTALL_DIR"
    $SUDO chown "$(id -un)" "$INSTALL_DIR"
fi

# Models and traits directories are owned by the service user so it can auto-clone/update.
if [ ! -d "$MODELS_DIR" ]; then
    $SUDO mkdir -p "$MODELS_DIR"
    $SUDO chown "$SERVICE_USER" "$MODELS_DIR"
fi
if [ ! -d "$TRAITS_DIR" ]; then
    $SUDO mkdir -p "$TRAITS_DIR"
    $SUDO chown "$SERVICE_USER" "$TRAITS_DIR"
fi

log "Installing binary"
restart_needed=0
if ! cmp -s "out/$BINARY" "$BIN_PATH" 2>/dev/null; then
    $SUDO install -m 755 "out/$BINARY" "$BIN_PATH"
    $SUDO codesign --force --sign - "$BIN_PATH"
    restart_needed=1
fi

if [ ! -f "$LOG" ]; then
    $SUDO touch "$LOG"
    $SUDO chown "$SERVICE_USER" "$LOG"
fi

log "Installing launchd plist"
workers_args=""
[ -n "$WORKERS" ] && workers_args="        <string>--workers</string>
        <string>$WORKERS</string>
"
new_plist=$(mktemp)
cat > "$new_plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$BIN_PATH</string>
        <string>worker</string>
        <string>--url</string>
        <string>$URL</string>
        <string>--traits-dir</string>
        <string>$TRAITS_DIR</string>
        <string>--interpret</string>
${workers_args}    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$BREW_PREFIX/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>SCAN_MODELS_DIR</key>
        <string>$MODELS_DIR</string>
        <key>SCAN_LLM</key>
        <string>$LLM</string>
    </dict>
    <key>UserName</key>
    <string>$SERVICE_USER</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG</string>
    <key>StandardErrorPath</key>
    <string>$LOG</string>
</dict>
</plist>
EOF
if [ ! -f "$PLIST" ] || ! cmp -s "$new_plist" "$PLIST"; then
    $SUDO cp "$new_plist" "$PLIST"
    $SUDO chown root:wheel "$PLIST"
    $SUDO chmod 644 "$PLIST"
    restart_needed=1
fi
rm -f "$new_plist"

# Migrate hosts that ran under the legacy unsuffixed label (com.atomdrift.litmus,
# pre-rename in commit 65440b1). The old label collided with the macOS server
# service name; if a stale plist is still on disk, bootout and delete it so
# launchd doesn't resurrect a mismatched daemon on next boot.
LEGACY_LABEL=com.atomdrift.litmus
LEGACY_PLIST=/Library/LaunchDaemons/com.atomdrift.litmus.plist
if [ -f "$LEGACY_PLIST" ]; then
    log "Removing legacy unsuffixed worker service ($LEGACY_LABEL)"
    $SUDO launchctl bootout "system/$LEGACY_LABEL" 2>/dev/null || true
    $SUDO rm -f "$LEGACY_PLIST"
fi

service_loaded() {
    $SUDO launchctl print "system/$LABEL" >/dev/null 2>&1
}

if [ "$restart_needed" -eq 1 ] || ! service_loaded; then
    log "Restarting launchd service"

    if service_loaded; then
        $SUDO launchctl bootout "system/$LABEL"
        i=0
        while service_loaded; do
            sleep 1
            i=$((i + 1))
            [ "$i" -lt 10 ] || { log "Timed out waiting for service to unregister"; break; }
        done
    fi

    i=0
    while $SUDO pgrep -x "$BINARY" >/dev/null 2>&1; do
        sleep 1
        i=$((i + 1))
        if [ "$i" -ge 10 ]; then
            log "Process did not exit cleanly; sending SIGKILL"
            $SUDO pkill -9 -x "$BINARY" 2>/dev/null || true
            sleep 1
            break
        fi
    done

    $SUDO launchctl bootstrap system "$PLIST" || die "launchctl bootstrap failed"
else
    log "Binary and plist unchanged, service already running, skipping restart"
fi

log "Service status:"
$SUDO launchctl print "system/$LABEL" || die "service failed to start"

log "Deployment complete"
