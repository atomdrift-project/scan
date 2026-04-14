#!/bin/sh
# worker-macos.sh - Deploy litmus worker on macOS using launchd
# Usage: ./worker-macos.sh <server-url>
# Runs entirely on the local machine. Re-run to update.
# Must be invoked from the repository root.

set -ex

URL="$1"
[ -n "$URL" ] || { echo "error: URL required" >&2; exit 1; }

BINARY=litmus
INSTALL_DIR=/usr/local/share/litmus
MODELS_DIR=/usr/local/share/litmus/models
TRAITS_DIR=/usr/local/share/litmus/traits
BIN_LINK=/usr/local/bin/litmus
PLIST=/Library/LaunchDaemons/com.atomdrift.litmus-worker.plist
LABEL=com.atomdrift.litmus-worker
SERVICE_USER=_litmus
LOG=/var/log/litmus-worker.log

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

for brew_candidate in /usr/local/bin/brew /opt/homebrew/bin/brew; do
    [ -x "$brew_candidate" ] && eval "$("$brew_candidate" shellenv)" && break
done
command -v brew >/dev/null 2>&1 || die "brew not found"

BREW_PREFIX=$(brew --prefix)

log "Installing build dependencies"
brew install rust sccache

log "Building release binary"
RUSTC_WRAPPER=sccache make release || die "build failed"

log "Ensuring service user '$SERVICE_USER' exists"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    uid=300
    while dscl . -list /Users UniqueID | awk '{print $2}' | grep -qx "$uid"; do
        uid=$((uid + 1))
    done
    sudo dscl . -create "/Users/$SERVICE_USER"
    sudo dscl . -create "/Users/$SERVICE_USER" UserShell /usr/bin/false
    sudo dscl . -create "/Users/$SERVICE_USER" RealName "Litmus Worker"
    sudo dscl . -create "/Users/$SERVICE_USER" UniqueID "$uid"
    sudo dscl . -create "/Users/$SERVICE_USER" PrimaryGroupID 1
    sudo dscl . -create "/Users/$SERVICE_USER" NFSHomeDirectory /var/empty
fi

if [ ! -d "$INSTALL_DIR" ] || [ ! -w "$INSTALL_DIR" ]; then
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown "$(id -un)" "$INSTALL_DIR"
fi

# Models and traits directories are owned by the service user so it can auto-clone/update.
if [ ! -d "$MODELS_DIR" ]; then
    sudo mkdir -p "$MODELS_DIR"
    sudo chown "$SERVICE_USER" "$MODELS_DIR"
fi
if [ ! -d "$TRAITS_DIR" ]; then
    sudo mkdir -p "$TRAITS_DIR"
    sudo chown "$SERVICE_USER" "$TRAITS_DIR"
fi

log "Installing binary"
restart_needed=0
if ! cmp -s "out/litmus" "$INSTALL_DIR/$BINARY" 2>/dev/null; then
    install -m 755 out/litmus "$INSTALL_DIR/$BINARY"
    restart_needed=1
fi

if [ ! -L "$BIN_LINK" ] || [ "$(readlink "$BIN_LINK")" != "$INSTALL_DIR/$BINARY" ]; then
    sudo ln -sf "$INSTALL_DIR/$BINARY" "$BIN_LINK"
fi

if [ ! -f "$LOG" ]; then
    sudo touch "$LOG"
    sudo chown "$SERVICE_USER" "$LOG"
fi

log "Installing launchd plist"
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
        <string>$INSTALL_DIR/$BINARY</string>
        <string>worker</string>
        <string>--url</string>
        <string>$URL</string>
        <string>--traits-dir</string>
        <string>$TRAITS_DIR</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$BREW_PREFIX/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>LITMUS_MODELS_DIR</key>
        <string>$MODELS_DIR</string>
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
    sudo cp "$new_plist" "$PLIST"
    sudo chown root:wheel "$PLIST"
    sudo chmod 644 "$PLIST"
    restart_needed=1
fi
rm -f "$new_plist"

service_loaded() {
    sudo launchctl print "system/$LABEL" >/dev/null 2>&1
}

if [ "$restart_needed" -eq 1 ] || ! service_loaded; then
    log "Restarting launchd service"

    if service_loaded; then
        sudo launchctl bootout "system/$LABEL"
        i=0
        while service_loaded; do
            sleep 1
            i=$((i + 1))
            [ "$i" -lt 10 ] || { log "Timed out waiting for service to unregister"; break; }
        done
    fi

    i=0
    while sudo pgrep -x "$BINARY" >/dev/null 2>&1; do
        sleep 1
        i=$((i + 1))
        if [ "$i" -ge 10 ]; then
            log "Process did not exit cleanly; sending SIGKILL"
            sudo pkill -9 -x "$BINARY" 2>/dev/null || true
            sleep 1
            break
        fi
    done

    sudo launchctl bootstrap system "$PLIST" || die "launchctl bootstrap failed"
else
    log "Binary and plist unchanged, service already running, skipping restart"
fi

log "Service status:"
sudo launchctl print "system/$LABEL" || die "service failed to start"

log "Deployment complete"
