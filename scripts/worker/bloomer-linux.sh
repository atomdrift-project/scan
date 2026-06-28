#!/bin/sh
# bloomer-linux.sh - Install the hourly Atomdrift bloom build+publish timer.
#
# The bloom filters (known-good/known-bad) are rebuilt from hopper's labelled
# samples once an hour, committed+pushed to the source-of-truth repo, and
# uploaded to R2. This installs that cycle as a systemd timer (scan-bloomer.timer
# -> scan-bloomer.service, Type=oneshot) running `make bloom-cron`.
#
# It runs as a DEDICATED `bloom` system user — separate from the worker's `scan`
# user — so the publishing credentials (codeberg push key, R2 token, DB password)
# are isolated from the worker and vice versa. Everything lives under one state
# tree, /var/lib/bloom:
#
#   /var/lib/bloom            HOME + StateDirectory. Holds the bloom user's creds
#                             and caches: .ssh, .pgpass, .config/rclone, .cargo
#   /var/lib/bloom/scan-src   scan source checkout (WorkingDirectory): the
#                             Makefile + scripts/bloom_pool.sql the cycle runs
#   /var/lib/bloom/repo       bloom repo checkout (BLOOM_REPO), committed and
#                             pushed to codeberg each cycle
#
# Re-runnable: idempotent. Re-running refreshes the source checkout to the
# committed HEAD of this repo, re-asserts the units, and reloads the timer.
#
# Secrets it CANNOT create for you (it checks and reports what is missing, but
# still installs the timer so you can drop them in afterwards):
#   ~bloom/.ssh/<key>                 codeberg key allowed to push atomdrift/bloom
#   ~bloom/.config/rclone/rclone.conf  rclone remote backing $(R2_REMOTE) (R2)
#   ~bloom/.pgpass                    password for BLOOM_DB, chmod 600
#
# Usage: ./scripts/worker/bloomer-linux.sh        (run from the repository root)
#
# Environment overrides:
#   BLOOM_DB     hopper DSN the cycle exports from   (default: postgres://hopper@localhost:5432/hopper)
#   BLOOM_REMOTE git URL of the bloom repo to clone  (default: ssh://git@codeberg.org/atomdrift/bloom.git)
#   ON_CALENDAR  systemd OnCalendar= cadence          (default: hourly)
#   RUN_NOW=1    kick one cycle immediately after install (otherwise wait for the timer)

set -eu

SERVICE_USER=bloom
SERVICE_NAME=scan-bloomer
STATE_HOME=/var/lib/bloom
SCAN_SRC=${STATE_HOME}/scan-src
BLOOM_DIR=${STATE_HOME}/repo
CARGO_HOME_DIR=${STATE_HOME}/.cargo
RUSTUP_HOME_DIR=${STATE_HOME}/.rustup

SERVICE_FILE=/etc/systemd/system/${SERVICE_NAME}.service
TIMER_FILE=/etc/systemd/system/${SERVICE_NAME}.timer

BLOOM_DB="${BLOOM_DB:-postgres://hopper@localhost:5432/hopper}"
BLOOM_REMOTE="${BLOOM_REMOTE:-ssh://git@codeberg.org/atomdrift/bloom.git}"
# scan-src is read-only (fetched, never pushed) → HTTP, no deploy key needed.
# The bloom repo above stays SSH because the cycle pushes to it.
SCAN_REMOTE="${SCAN_REMOTE:-https://codeberg.org/atomdrift/scan.git}"
ON_CALENDAR="${ON_CALENDAR:-hourly}"
RUN_NOW="${RUN_NOW:-}"

die() { echo "error: $*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

# Run a command as the bloom user with its HOME/toolchain environment, so
# cargo/git/rclone/psql resolve their config and caches under $STATE_HOME.
RUN_PATH="${CARGO_HOME_DIR}/bin:/usr/local/bin:/usr/bin:/bin"
as_bloom() {
    sudo -u "$SERVICE_USER" -H env -i \
        HOME="$STATE_HOME" \
        CARGO_HOME="$CARGO_HOME_DIR" RUSTUP_HOME="$RUSTUP_HOME_DIR" \
        PATH="$RUN_PATH" \
        "$@"
}

# --- Preconditions -----------------------------------------------------------

[ -f Makefile ]                      || die "run from the repository root"
[ "$(uname -s)" = "Linux" ]          || die "this script is for Linux"
command -v systemctl >/dev/null 2>&1 || die "systemctl not found (systemd required)"
command -v sudo      >/dev/null 2>&1 || die "sudo required"
command -v git       >/dev/null 2>&1 || die "git required"
MAKE_BIN=$(command -v make) || die "make not found"

# --- Service user + dir layout ----------------------------------------------

if ! getent passwd "${SERVICE_USER}" >/dev/null; then
    log "Creating service user '${SERVICE_USER}'"
    sudo useradd --system --home-dir "${STATE_HOME}" --no-create-home \
                 --shell /usr/sbin/nologin \
                 --comment "Atomdrift bloom publisher" "${SERVICE_USER}"
fi

# Pre-create the state tree. systemd re-asserts /var/lib/bloom via
# StateDirectory=bloom on each start, but creating it now lets us seed the
# checkouts and run a warm build before the first timer tick.
sudo install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${STATE_HOME}"
sudo install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${SCAN_SRC}"

# --- Source checkout --------------------------------------------------------

# scan-src is a shallow read-only checkout of the canonical scan repo. The timer
# fetches origin/main fresh before each build (ExecStartPre below), so deployed
# code tracks the repo without a redeploy. Seed it once here (init+fetch over the
# existing dir keeps any target/ build cache), then refresh to the current tip.
log "Setting up scan source checkout at ${SCAN_SRC} (tracking ${SCAN_REMOTE})"
if ! as_bloom git -C "${SCAN_SRC}" rev-parse --git-dir >/dev/null 2>&1; then
    as_bloom git -C "${SCAN_SRC}" init -q -b main
    as_bloom git -C "${SCAN_SRC}" remote add origin "${SCAN_REMOTE}"
fi
as_bloom git -C "${SCAN_SRC}" remote set-url origin "${SCAN_REMOTE}"
as_bloom git -C "${SCAN_SRC}" fetch -q --depth=1 origin main \
    || die "cannot fetch ${SCAN_REMOTE} as ${SERVICE_USER}"
as_bloom git -C "${SCAN_SRC}" reset --hard -q FETCH_HEAD

# --- Rust toolchain (bloom-owned) -------------------------------------------

if as_bloom sh -c 'command -v cargo >/dev/null 2>&1'; then
    log "Rust toolchain already present for ${SERVICE_USER}"
else
    command -v curl >/dev/null 2>&1 || die "curl required to install the Rust toolchain"
    log "Installing Rust toolchain for ${SERVICE_USER} (into ${CARGO_HOME_DIR})"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | as_bloom sh -s -- -y --no-modify-path --default-toolchain stable \
        || die "rustup install failed"
fi

# --- SSH known_hosts for the push -------------------------------------------

# Pin codeberg's host keys so the non-interactive `git push` never blocks on a
# host-key prompt. (The deploy key itself is a secret you provide; see below.)
codeberg_host=$(printf '%s\n' "$BLOOM_REMOTE" | sed -n 's#^[a-z+]*://\(git@\)\?\([^/:]*\).*#\2#p')
codeberg_host="${codeberg_host:-codeberg.org}"
if command -v ssh-keyscan >/dev/null 2>&1; then
    as_bloom install -d -m 0700 "${STATE_HOME}/.ssh"
    if ! as_bloom sh -c "grep -q '${codeberg_host}' ~/.ssh/known_hosts 2>/dev/null"; then
        log "Pinning ${codeberg_host} host keys in ~bloom/.ssh/known_hosts"
        ssh-keyscan -t rsa,ecdsa,ed25519 "${codeberg_host}" 2>/dev/null \
            | as_bloom sh -c "cat >> ~/.ssh/known_hosts" || true
    fi
fi

# --- Bloom repo checkout ----------------------------------------------------

bloom_ready=0
if [ -d "${BLOOM_DIR}/.git" ]; then
    log "Bloom checkout present at ${BLOOM_DIR}; normalizing to HEAD"
    # A prior clone may have fetched objects but failed to materialize the working
    # tree (e.g. tripping on stray files a pre-clone build left behind). Reset to
    # HEAD so the tree is whole and clean before the cycle commits into it.
    as_bloom git -C "${BLOOM_DIR}" reset --hard HEAD >/dev/null 2>&1 || true
    as_bloom git -C "${BLOOM_DIR}" clean -fd >/dev/null 2>&1 || true
    bloom_ready=1
else
    # A non-git directory here is stale build output (filters are rebuilt every
    # cycle) and would make `git clone` trip on untracked files — clear it first.
    if [ -e "${BLOOM_DIR}" ]; then
        log "Clearing stale non-git ${BLOOM_DIR} before clone"
        as_bloom rm -rf "${BLOOM_DIR}"
    fi
    log "Cloning bloom repo into ${BLOOM_DIR} (${BLOOM_REMOTE})"
    if as_bloom git clone "${BLOOM_REMOTE}" "${BLOOM_DIR}"; then
        bloom_ready=1
    else
        log "WARNING: could not clone ${BLOOM_REMOTE} as ${SERVICE_USER}."
        log "  Add a codeberg deploy key with push access in ${STATE_HOME}/.ssh,"
        log "  then re-run this script (or clone ${BLOOM_DIR} by hand)."
    fi
fi

if [ "$bloom_ready" = 1 ]; then
    # Commit identity for the unattended commits — without it the cycle's git
    # commit fails ("Please tell me who you are") — and make sure origin points at
    # the push remote even if the repo was seeded some other way.
    as_bloom git -C "${BLOOM_DIR}" config user.name  "Atomdrift Bloomer"
    as_bloom git -C "${BLOOM_DIR}" config user.email "bloomer@atomdrift"
    as_bloom git -C "${BLOOM_DIR}" remote set-url origin "${BLOOM_REMOTE}" 2>/dev/null \
        || as_bloom git -C "${BLOOM_DIR}" remote add origin "${BLOOM_REMOTE}"
fi

# --- Warm build (surface toolchain errors before the first timer tick) ------

log "Warm-building scan-bloom-build as ${SERVICE_USER}"
if ! as_bloom sh -c "cd '${SCAN_SRC}' && cargo build --release --bin scan-bloom-build"; then
    log "WARNING: warm build failed; the timer will retry, but check the toolchain."
fi

# --- Units ------------------------------------------------------------------

TMP_SERVICE=$(mktemp -t scan-bloomer.service.XXXXXX)
TMP_TIMER=$(mktemp -t scan-bloomer.timer.XXXXXX)
trap 'rm -f "$TMP_SERVICE" "$TMP_TIMER"' EXIT

cat >"$TMP_SERVICE" <<EOF
[Unit]
Description=Atomdrift bloom filter build + publish (rebuild -> commit+push -> R2)
Documentation=https://codeberg.org/atomdrift/scan
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=${SERVICE_USER}
Group=${SERVICE_USER}

# Dedicated state tree for the bloom publisher (separate from the worker).
StateDirectory=bloom
StateDirectoryMode=0750

WorkingDirectory=${SCAN_SRC}
Environment=HOME=${STATE_HOME}
Environment=CARGO_HOME=${CARGO_HOME_DIR}
Environment=RUSTUP_HOME=${RUSTUP_HOME_DIR}
Environment=PATH=${RUN_PATH}
# The bloom repo the cycle commits/pushes (BLOOM_REPO overrides the Makefile's
# ../bloom default).
Environment=BLOOM_REPO=${BLOOM_DIR}
# hopper replica the labelled pool is exported from (auth via ~bloom/.pgpass).
Environment=BLOOM_DB=${BLOOM_DB}
# 365-day window for the published filters (see 'make bloom-cron').
Environment=BLOOM_CRON_MAX_AGE_DAYS=365
# Pull the latest scan source before each build so deployed code tracks
# origin/main without a redeploy — a clone-once, fetch-each-cycle checkout, never
# a re-clone. Best-effort ('-'): on fetch failure, build the code already on disk
# rather than skip the whole cycle.
ExecStartPre=-/bin/sh -c '/usr/bin/git -C ${SCAN_SRC} fetch -q --depth=1 origin main && /usr/bin/git -C ${SCAN_SRC} reset --hard -q FETCH_HEAD'
ExecStart=${MAKE_BIN} bloom-cron

# Yield to the worker: the rebuild + multi-million-row pool export is heavy.
Nice=10
CPUWeight=20
IOSchedulingClass=idle
MemoryMax=50%
TasksMax=4096
# A cycle that overruns the hour is killed; the next tick retries cleanly.
TimeoutStartSec=45min

# Filesystem isolation. Everything the cycle writes (checkouts, caches, creds)
# lives under StateDirectory, so strict confinement still leaves it room to work.
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
PrivateDevices=true
PrivateMounts=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectKernelLogs=true
ProtectControlGroups=true
ProtectClock=true
ProtectHostname=true
ProtectProc=invisible
ProcSubset=pid
UMask=0077

NoNewPrivileges=true
RestrictSUIDSGID=true
RestrictRealtime=true
RestrictNamespaces=true
LockPersonality=true
SystemCallArchitectures=native
SystemCallFilter=@system-service
CapabilityBoundingSet=
AmbientCapabilities=
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

StandardOutput=journal
StandardError=journal
EOF

cat >"$TMP_TIMER" <<EOF
[Unit]
Description=Hourly Atomdrift bloom build + publish
Documentation=https://codeberg.org/atomdrift/scan

[Timer]
OnCalendar=${ON_CALENDAR}
# Spread off the top of the hour so we don't collide with other on-the-hour jobs.
RandomizedDelaySec=5min
AccuracySec=1min
# Run a cycle missed during downtime once on boot, instead of waiting an hour.
Persistent=true

[Install]
WantedBy=timers.target
EOF

units_changed=0
for pair in "$TMP_SERVICE:$SERVICE_FILE" "$TMP_TIMER:$TIMER_FILE"; do
    src=${pair%%:*}; dst=${pair#*:}
    if sudo cmp -s "$src" "$dst" 2>/dev/null; then
        log "$(basename "$dst") unchanged"
    else
        log "Writing $dst"
        sudo install -m 0644 -o root -g root "$src" "$dst"
        units_changed=1
    fi
done

# --- Activate ---------------------------------------------------------------

[ "$units_changed" -eq 1 ] && sudo systemctl daemon-reload

sudo systemctl enable --now "${SERVICE_NAME}.timer" >/dev/null
log "Timer enabled:"
sudo systemctl --no-pager list-timers "${SERVICE_NAME}.timer" || true

if [ -n "$RUN_NOW" ]; then
    log "Running one cycle now (RUN_NOW=1)"
    sudo systemctl start "${SERVICE_NAME}.service" || true
    sudo systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
fi

# --- Credential readiness summary -------------------------------------------

echo
log "Credential check (the timer fires regardless; fix any MISSING before it runs):"
missing=0

if [ "$bloom_ready" = 1 ]; then
    printf '    [ ok ] bloom checkout + push remote: %s\n' "${BLOOM_DIR}"
else
    printf '    [MISS] bloom checkout: clone failed; add a codeberg push key in %s/.ssh\n' "${STATE_HOME}"
    missing=1
fi

if as_bloom sh -c 'command -v rclone >/dev/null 2>&1'; then
    if as_bloom rclone listremotes 2>/dev/null | grep -q .; then
        printf '    [ ok ] rclone remote(s) configured for %s\n' "${SERVICE_USER}"
    else
        printf '    [MISS] rclone present but no remote: configure the R2 remote in ~%s/.config/rclone\n' "${SERVICE_USER}"
        missing=1
    fi
else
    printf '    [MISS] rclone not installed for %s (needed for the R2 upload)\n' "${SERVICE_USER}"
    missing=1
fi

if as_bloom sh -c 'test -f ~/.pgpass'; then
    printf '    [ ok ] ~%s/.pgpass present\n' "${SERVICE_USER}"
else
    printf '    [MISS] ~%s/.pgpass absent (needed for BLOOM_DB=%s)\n' "${SERVICE_USER}" "${BLOOM_DB}"
    missing=1
fi

echo
if [ "$missing" -eq 0 ]; then
    log "Install complete. Watch a cycle:  journalctl -u ${SERVICE_NAME}.service -f"
else
    log "Install complete, but fix the [MISS] items above; test with:  sudo systemctl start ${SERVICE_NAME}.service"
    log "then watch:  journalctl -u ${SERVICE_NAME}.service -e"
fi
