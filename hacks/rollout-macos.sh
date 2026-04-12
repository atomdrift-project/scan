#!/bin/sh
# rollout-macos.sh - Deploy litmus on macOS using launchd
# Runs entirely on the local machine. Re-run to update.
# Must be invoked from the repository root.

set -e

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

[ "$(id -u)" -eq 0 ] || exec sudo "$0" "$@"

REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || id -un)}"

log "Installing build dependencies"
sudo -u "$REAL_USER" brew install rust sccache p7zip upx rizin innoextract

log "Building release binary"
sudo -u "$REAL_USER" env RUSTC_WRAPPER=sccache make release || die "build failed"

log "Upgrading rules"
sudo -u "$REAL_USER" out/litmus update-rules || die "update-rules failed"

REAL_HOME=$(dscl . -read "/Users/$REAL_USER" NFSHomeDirectory | sed 's/NFSHomeDirectory: //')
BREW_PREFIX=$(sudo -u "$REAL_USER" brew --prefix)
MODELS_SRC="$REAL_HOME/Library/Application Support/litmus/models"
TRAITS_SRC="$REAL_HOME/Library/Application Support/cleave/traits"
[ -d "$MODELS_SRC" ] || die "models not found at '$MODELS_SRC' after update-rules"
[ -d "$TRAITS_SRC" ] || die "traits not found at '$TRAITS_SRC' after update-rules"

log "Ensuring service user '$SERVICE_USER' exists"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    uid=300
    while dscl . -list /Users UniqueID | awk '{print $2}' | grep -qx "$uid"; do
        uid=$((uid + 1))
    done
    dscl . -create "/Users/$SERVICE_USER"
    dscl . -create "/Users/$SERVICE_USER" UserShell /usr/bin/false
    dscl . -create "/Users/$SERVICE_USER" RealName "Litmus Service"
    dscl . -create "/Users/$SERVICE_USER" UniqueID "$uid"
    dscl . -create "/Users/$SERVICE_USER" PrimaryGroupID 1
    dscl . -create "/Users/$SERVICE_USER" NFSHomeDirectory /var/empty
fi

log "Installing binary"
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR" /usr/local/bin
install -m 755 out/litmus "$INSTALL_DIR/$BINARY"
chown root:wheel "$INSTALL_DIR/$BINARY"
chmod 755 "$INSTALL_DIR/$BINARY"
ln -sf "$INSTALL_DIR/$BINARY" "$BIN_LINK"

log "Installing models and traits"
cp -R "$MODELS_SRC" "$INSTALL_DIR/models"
cp -R "$TRAITS_SRC" "$INSTALL_DIR/traits"
chown -R "$SERVICE_USER" "$INSTALL_DIR/models" "$INSTALL_DIR/traits"

log "Preparing log file"
touch "$LOG"
chown "$SERVICE_USER" "$LOG"

log "Installing launchd plist"
cat > "$PLIST" <<EOF
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
        <string>--verbose</string>
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
chown root:wheel "$PLIST"
chmod 644 "$PLIST"

log "Loading launchd service"
launchctl bootout "system/$LABEL" 2>/dev/null || true
launchctl bootout system "$PLIST" 2>/dev/null || true
pkill -9 -x "$BINARY" 2>/dev/null || true
launchctl bootstrap system "$PLIST"

log "Service status:"
launchctl print "system/$LABEL" || true

log "Deployment complete"
