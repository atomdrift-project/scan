#!/bin/sh
# rollout-macos.sh - Deploy litmus on macOS using launchd
# Runs entirely on the local machine. Re-run to update.
# Must be invoked from the repository root.
#
# sudo is required on every run only for launchctl (system LaunchDaemon management).
# All other sudo calls are first-run-only, guarded by existence/ownership checks.
# Models and traits are cloned once; the service handles git pull at runtime.

set -ex

BINARY=litmus
INSTALL_DIR=/usr/local/share/litmus
BIN_LINK=/usr/local/bin/litmus
PLIST=/Library/LaunchDaemons/com.atomdrift.litmus.plist
LABEL=com.atomdrift.litmus
SERVICE_USER=_litmus
LOG=/var/log/litmus.log
BIND="${BIND:-0.0.0.0:49999}"
ALLOW_CIDR="${ALLOW_CIDR:-10.0.0.0/8}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

# Non-login SSH shells don't source profile, so brew may not be in PATH.
# Try the standard install locations for Intel and Apple Silicon.
for brew_candidate in /usr/local/bin/brew /opt/homebrew/bin/brew; do
    [ -x "$brew_candidate" ] && eval "$("$brew_candidate" shellenv)" && break
done
command -v brew >/dev/null 2>&1 || die "brew not found"

BREW_PREFIX=$(brew --prefix)
TRAITS_SRC="$HOME/Library/Application Support/cleave/traits"

log "Installing build dependencies"
brew install rust sccache p7zip upx rizin innoextract

log "Building release binary"
RUSTC_WRAPPER=sccache make release || die "build failed"

# First-run only: create service user
log "Ensuring service user '$SERVICE_USER' exists"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    uid=300
    while dscl . -list /Users UniqueID | awk '{print $2}' | grep -qx "$uid"; do
        uid=$((uid + 1))
    done
    sudo dscl . -create "/Users/$SERVICE_USER"
    sudo dscl . -create "/Users/$SERVICE_USER" UserShell /usr/bin/false
    sudo dscl . -create "/Users/$SERVICE_USER" RealName "Litmus Service"
    sudo dscl . -create "/Users/$SERVICE_USER" UniqueID "$uid"
    sudo dscl . -create "/Users/$SERVICE_USER" PrimaryGroupID 1
    sudo dscl . -create "/Users/$SERVICE_USER" NFSHomeDirectory /var/empty
fi

# First-run only: create install dir owned by current user
if [ ! -d "$INSTALL_DIR" ] || [ ! -w "$INSTALL_DIR" ]; then
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown "$(id -un)" "$INSTALL_DIR"
fi

log "Installing binary"
restart_needed=0
if ! cmp -s "out/litmus" "$INSTALL_DIR/$BINARY" 2>/dev/null; then
    install -m 755 out/litmus "$INSTALL_DIR/$BINARY"
    restart_needed=1
fi

# First-run only: create symlink
if [ ! -L "$BIN_LINK" ] || [ "$(readlink "$BIN_LINK")" != "$INSTALL_DIR/$BINARY" ]; then
    sudo ln -sf "$INSTALL_DIR/$BINARY" "$BIN_LINK"
fi

# Clone models and traits on first run; the service handles git pull at runtime
log "Ensuring models repo"
if [ ! -d "$INSTALL_DIR/models/.git" ]; then
    sudo git clone https://codeberg.org/atomdrift/litmus-models.git "$INSTALL_DIR/models"
    sudo chown -R "$SERVICE_USER" "$INSTALL_DIR/models"
fi

log "Ensuring traits repo"
if [ ! -d "$INSTALL_DIR/traits/.git" ]; then
    [ -d "$TRAITS_SRC" ] || die "traits not found at '$TRAITS_SRC'; run 'litmus update-rules' first"
    traits_remote=$(git -C "$TRAITS_SRC" remote get-url origin 2>/dev/null) || die "'$TRAITS_SRC' is not a git repo"
    sudo git clone "$traits_remote" "$INSTALL_DIR/traits"
    sudo chown -R "$SERVICE_USER" "$INSTALL_DIR/traits"
fi

# First-run only: prepare log file owned by service user
if [ ! -f "$LOG" ]; then
    sudo touch "$LOG"
    sudo chown "$SERVICE_USER" "$LOG"
fi

# Write plist only if content has changed
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
        <string>-u</string>
        <string>serve</string>
        <string>--bind</string>
        <string>$BIND</string>
        <string>--allow-cidr</string>
        <string>$ALLOW_CIDR</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>LITMUS_MODELS_DIR</key>
        <string>$INSTALL_DIR/models</string>
        <key>CLEAVE_TRAITS_DIR</key>
        <string>$INSTALL_DIR/traits</string>
        <key>PATH</key>
        <string>$BREW_PREFIX/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
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

# Only restart if binary or plist changed - avoid disrupting a healthy service
if [ "$restart_needed" -eq 1 ]; then
    log "Restarting launchd service"
    sudo launchctl bootout "system/$LABEL" 2>/dev/null || true
    sudo pkill -x "$BINARY" 2>/dev/null || true
    i=0
    while sudo pgrep -x "$BINARY" >/dev/null 2>&1; do
        sleep 1
        i=$((i + 1))
        if [ "$i" -ge 10 ]; then
            sudo pkill -9 -x "$BINARY" 2>/dev/null || true
            sleep 1
            break
        fi
    done
    sudo launchctl bootstrap system "$PLIST"
else
    log "Binary and plist unchanged, skipping service restart"
fi

log "Service status:"
sudo launchctl print "system/$LABEL" || true

log "Deployment complete"
