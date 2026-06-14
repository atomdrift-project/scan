#!/bin/sh
# worker-omnios.sh - Deploy Atomdrift Scan worker inside an OmniOS zone (as root).
# Usage: ./worker-omnios.sh <hopper-url>
# Must be invoked from the scan repository root inside the zone.
# Idempotent: re-run to update.
#
# rust + 7zip come from pkgsrc; innoextract, upx, and rizin (HEAD) are
# built from upstream source only when they are not already available. The
# worker runs as the unprivileged `ascan` user under SMF, with
# ignore_error=core,signal so a crashing
# backend triggers a restart instead of dropping into maintenance.

set -eu

URL="${1:-}"
[ -n "$URL" ] || { echo "error: hopper URL required as first argument" >&2; exit 1; }

# Optional: cap concurrent analysis slots (--workers). Unset = worker auto.
WORKERS="${WORKERS:-}"
# LLM second-opinion pass: endpoint (exported as SCAN_LLM) + interpret gate.
LLM="${LLM:-http://10.9.8.149:8000/v1}"
INTERPRET_MIN_PROB="${INTERPRET_MIN_PROB:-0.15}"
worker_args="worker --url $URL --interpret --interpret-min-prob $INTERPRET_MIN_PROB"
[ -n "$WORKERS" ] && worker_args="$worker_args --workers $WORKERS"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

PKG_PREFIX=/opt/local
# Source builds and the ascan binary install under /opt/atomdrift/scan rather than
# /usr/local: in sparse zones /usr is a read-only lofs mount from the global,
# so /usr/local is unwritable. /opt is writable per-zone.
SRC_PREFIX=/opt/atomdrift/scan
SRC_DIR=/var/atomdrift/scan/src
SCAN_USER=ascan
SCAN_GROUP=ascan
SCAN_HOME=/var/atomdrift/scan
SCAN_LOG_DIR=/var/log/atomdrift/scan
# Disk-backed scratch dir. /tmp on illumos is tmpfs (swap-backed); large
# unpacks (innoextract, 7z, rizin) can exhaust swap and wedge the whole
# zone — including /etc/svc/volatile, which then blocks svc.startd.
SCAN_TMP_DIR=/var/atomdrift/scan/tmp
SCAN_BIN=$SRC_PREFIX/bin/ascan
SMF_MANIFEST=/lib/svc/manifest/site/ascan-worker.xml
SMF_FMRI=svc:/site/ascan-worker:default

PATH=$PKG_PREFIX/sbin:$PKG_PREFIX/bin:$SRC_PREFIX/sbin:$SRC_PREFIX/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

[ "$(id -u)" -eq 0 ] || die "must be run as root inside the zone"
[ -f Cargo.toml ] || die "must be invoked from the scan repository root"
command -v pkgin >/dev/null 2>&1 || die "pkgin not found; install pkgsrc bootstrap first"
command -v pkg >/dev/null 2>&1   || die "IPS pkg(1) not found; not running on illumos?"

tool_available() {
    command -v "$1" >/dev/null 2>&1
}

JOBS=$(psrinfo | wc -l | tr -d ' ')
[ "$JOBS" -gt 0 ] 2>/dev/null || JOBS=2

# rust + 7zip are explicitly requested via pkgsrc. git is needed for
# rules/models updates. The remainder are build-time prerequisites for source
# building missing runtime tools.
log "Installing pkgsrc binary dependencies"
to_install=""
add_pkg() {
    pkg_info -qe "$1" 2>/dev/null || to_install="$to_install $1"
}

need_source_builds=0
for tool in innoextract upx rizin; do
    tool_available "$tool" || need_source_builds=1
done

add_pkg rust
add_pkg p7zip
add_pkg git
if [ "$need_source_builds" -eq 1 ]; then
    add_pkg cmake
    add_pkg meson
    add_pkg ninja-build
    add_pkg gmake
    add_pkg pkgconf
    add_pkg boost-headers
    add_pkg boost-libs
    add_pkg xz
    add_pkg zlib
    add_pkg libiconv
else
    log "All source-built runtime tools already available; skipping pkgsrc build dependencies"
fi
if [ -n "$to_install" ]; then
    log "Refreshing pkgin catalog"
    pkgin -y update >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    pkgin -y install $to_install
else
    log "All pkgsrc dependencies already installed"
fi

current_ascan_rev() {
    if ! command -v git >/dev/null 2>&1; then
        echo unknown
        return
    fi
    rev=$(git rev-parse HEAD 2>/dev/null || echo unknown)
    git update-index -q --refresh 2>/dev/null || true
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        echo "$rev-dirty"
    else
        echo "$rev"
    fi
}

SCAN_REV=$(current_ascan_rev)
SCAN_REV_MARKER=$SRC_PREFIX/.ascan-installed-rev
ascan_build_needed=1
case "$SCAN_REV" in
    unknown|*-dirty) ascan_build_needed=1 ;;
    *)
        if [ -x "$SCAN_BIN" ] && [ -f "$SCAN_REV_MARKER" ] \
                && [ "$(cat "$SCAN_REV_MARKER")" = "$SCAN_REV" ]; then
            ascan_build_needed=0
        fi
        ;;
esac

###############################################################################
# IPS system prerequisites: crt1.o, headers, ld, libc — needed by pkgsrc gcc
# to link anything. Without build-essential the linker fails with
# "ld: fatal: file crt1.o: open failed".
###############################################################################

ips_installed() {
    pkg list -Hv "$1" >/dev/null 2>&1
}

# IPS pkg(1) returns 4 when nothing needs doing — treat that as success.
ensure_ips() {
    ips_installed "$1" && return 0
    pkg install -q "$1" && return 0
    rc=$?
    [ "$rc" -eq 4 ] && return 0
    return "$rc"
}

if [ "$need_source_builds" -eq 1 ] || [ "$ascan_build_needed" -eq 1 ]; then
    log "Ensuring IPS build prerequisites (headers, ld, crt files)"
    # build-essential pulls headers + ld; c-runtime supplies
    # /usr/lib/{,amd64/}crt1.o which build-essential does NOT pull in on
    # r151058+, despite the name.
    ensure_ips build-essential          || die "failed to install IPS build-essential"
    ensure_ips system/library/c-runtime || die "failed to install IPS c-runtime"
else
    log "No source builds needed; skipping IPS build prerequisites"
fi

###############################################################################
# Source builds: innoextract, upx, rizin (HEAD)
###############################################################################

mkdir -p "$SRC_DIR" "$SRC_PREFIX/bin" "$SRC_PREFIX/lib" "$SRC_PREFIX/share"

# ensure_source_tool <name> <repo> <ref> <builder-fn>
# Runtime tools may come from pkgsrc, IPS, or a prior manual install. Skip the
# source-build path entirely when the command is already on PATH.
ensure_source_tool() {
    name=$1; repo=$2; ref=$3; builder=$4

    if tool_available "$name"; then
        log "$name already available at $(command -v "$name"); skipping source build"
        return 0
    fi

    build_from_git "$name" "$repo" "$ref" "$builder"
}

# build_from_git <name> <repo> <ref> <builder-fn>
# Idempotent: skips the build when the resolved SHA matches the marker.
build_from_git() {
    name=$1; repo=$2; ref=$3; builder=$4
    dir=$SRC_DIR/$name

    if [ ! -d "$dir/.git" ]; then
        log "Cloning $name @ $ref (shallow)"
        case "$ref" in
            HEAD) git clone --depth 1 "$repo" "$dir" ;;
            *)    git clone --depth 1 --branch "$ref" "$repo" "$dir" ;;
        esac
    fi

    # Shallow-fetch just the requested ref so re-runs don't pull history.
    git -C "$dir" fetch --depth 1 origin "$ref"
    git -C "$dir" reset --hard FETCH_HEAD
    git -C "$dir" submodule update --init --recursive --depth 1

    sha=$(git -C "$dir" rev-parse HEAD)
    marker=$dir/.installed-sha
    if [ -f "$marker" ] && [ "$(cat "$marker")" = "$sha" ] \
            && command -v "$name" >/dev/null 2>&1; then
        log "$name @ $sha already installed"
        return 0
    fi

    log "Building $name @ $sha"
    ( cd "$dir" && "$builder" )
    echo "$sha" > "$marker"
}

build_innoextract() {
    # Boost 1.90 finally removed boost_system as a library (header-only since
    # 1.69). innoextract still lists `system` in find_package COMPONENTS, so
    # we point cmake at a stub config that declares Boost::system as a
    # header-only INTERFACE target. Other components (filesystem, iostreams,
    # date_time, program_options) come from pkgsrc as normal.
    stub=$PWD/.boost-system-stub
    mkdir -p "$stub"
    cat > "$stub/boost_system-config.cmake" <<'STUB'
add_library(Boost::system INTERFACE IMPORTED)
set(boost_system_FOUND TRUE)
set(boost_system_VERSION 1.90.0)
STUB
    cat > "$stub/boost_system-config-version.cmake" <<'STUB'
set(PACKAGE_VERSION "1.90.0")
set(PACKAGE_VERSION_COMPATIBLE TRUE)
set(PACKAGE_VERSION_EXACT TRUE)
STUB

    rm -rf build
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX="$SRC_PREFIX" \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH="$PKG_PREFIX" \
          -Dboost_system_DIR="$stub" \
          ..
    gmake -j"$JOBS"
    gmake install
}

build_upx() {
    rm -rf build
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX="$SRC_PREFIX" \
          -DCMAKE_BUILD_TYPE=Release \
          ..
    gmake -j"$JOBS"
    gmake install
}

build_rizin() {
    rm -rf build
    meson setup --prefix="$SRC_PREFIX" --buildtype=release build

    # tree-sitter's portable/endian.h has no illumos branch and falls into a
    # "platform not supported" #error. illumos provides the standard
    # le16toh/be16toh names via <endian.h>, so wrap the #error in a guard
    # that includes <endian.h> on __sun. Idempotent — skipped if already
    # patched. Path globs the version since rizin can bump tree-sitter.
    ts_endian=$(find subprojects -path '*tree-sitter*/portable/endian.h' 2>/dev/null | head -1)
    if [ -n "$ts_endian" ] && ! grep -q '__sun' "$ts_endian"; then
        log "Patching tree-sitter portable/endian.h for illumos"
        awk '
            /^#    error platform not supported$/ {
                print "#  if defined(__sun) || defined(__illumos__)"
                print "#    include <endian.h>"
                print "#  else"
                print "#    error platform not supported"
                print "#  endif"
                next
            }
            { print }
        ' "$ts_endian" > "$ts_endian.tmp" && mv "$ts_endian.tmp" "$ts_endian"
    fi

    meson compile -C build
    meson install -C build
}

ensure_source_tool innoextract https://github.com/dscharrer/innoextract.git HEAD       build_innoextract
ensure_source_tool upx          https://github.com/upx/upx.git                v4.2.4   build_upx
ensure_source_tool rizin        https://github.com/rizinorg/rizin.git         HEAD     build_rizin

###############################################################################
# ascan user, build, install
###############################################################################

log "Ensuring $SCAN_USER user/group exist"
getent group "$SCAN_GROUP" >/dev/null 2>&1 || groupadd "$SCAN_GROUP"
id "$SCAN_USER" >/dev/null 2>&1 || \
    useradd -m -d "$SCAN_HOME" -g "$SCAN_GROUP" -s /bin/sh \
            -c "Atomdrift Scan Worker" "$SCAN_USER"
mkdir -p "$SCAN_HOME" "$SCAN_LOG_DIR" "$SCAN_TMP_DIR"
chown "$SCAN_USER:$SCAN_GROUP" "$SCAN_HOME" "$SCAN_LOG_DIR" "$SCAN_TMP_DIR"
chmod 700 "$SCAN_TMP_DIR"
# Sweep leftovers from any prior wedged run so the dir doesn't grow unbounded
# across restarts. Safe: only the worker writes here.
find "$SCAN_TMP_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

ascan_changed=0
if [ "$ascan_build_needed" -eq 1 ]; then
    log "Building ascan from source tree ($SCAN_REV)"
    # unrar-sys builds the unrar library as C++ but doesn't emit a link-lib for
    # libstdc++, so the final link (with -nodefaultlibs) leaves operator new,
    # std::__cxx11 string symbols, and __gxx_personality_v0 unresolved on illumos.
    # Force rustc to append -lstdc++ to the link line.
    RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-lstdc++" cargo build --release \
        || die "ascan build failed"

    log "Installing ascan binary to $SCAN_BIN"
    # illumos install(1) is the old SunOS variant with incompatible flags;
    # use cp+chmod for portability. Stage to .new then rename for atomicity
    # (replacing a running binary while the SMF service is up is otherwise racy).
    cp -f target/release/ascan "$SCAN_BIN.new"
    chmod 755 "$SCAN_BIN.new"
    mv -f "$SCAN_BIN.new" "$SCAN_BIN"
    case "$SCAN_REV" in
        unknown|*-dirty) rm -f "$SCAN_REV_MARKER" ;;
        *) echo "$SCAN_REV" > "$SCAN_REV_MARKER" ;;
    esac
    ascan_changed=1
else
    log "ascan $SCAN_REV already installed; skipping cargo build"
fi

log "Refreshing rules/models as $SCAN_USER"
rules_changed=0
update_log=/tmp/ascan-update-rules.$$.log
if su - "$SCAN_USER" -c "PATH=$PATH $SCAN_BIN update-rules" > "$update_log" 2>&1; then
    cat "$update_log"
else
    cat "$update_log" >&2
    rm -f "$update_log"
    die "update-rules failed"
fi
if grep -q '^Updated:' "$update_log"; then
    rules_changed=1
fi
rm -f "$update_log"

###############################################################################
# SMF manifest
###############################################################################

manifest_tmp=/tmp/ascan-worker.$$.xml
manifest_changed=0

# duration=child: ascan runs in the foreground (doesn't fork), so SMF must
# treat the start method's process itself as the service. With contract mode
# SMF would expect start to fork+exit and would kill the worker as a
# "method exit timeout" after timeout_seconds. With child mode SMF tracks
# the start process directly and restarts it when it exits.
# ignore_error=core,signal: restart on crash/signal instead of going to
# maintenance. timeout_seconds=0 on start: no timeout (run forever).
cat > "$manifest_tmp" <<EOF
<?xml version="1.0"?>
<!DOCTYPE service_bundle SYSTEM "/usr/share/lib/xml/dtd/service_bundle.dtd.1">
<service_bundle type="manifest" name="ascan-worker">
  <service name="site/ascan-worker" type="service" version="1">
    <create_default_instance enabled="true"/>
    <single_instance/>
    <dependency name="network" grouping="require_all" restart_on="error" type="service">
      <service_fmri value="svc:/milestone/network:default"/>
    </dependency>
    <dependency name="filesystem-local" grouping="require_all" restart_on="error" type="service">
      <service_fmri value="svc:/system/filesystem/local:default"/>
    </dependency>
    <method_context>
      <method_credential user="$SCAN_USER" group="$SCAN_GROUP"/>
      <method_environment>
        <envvar name="PATH" value="$PATH"/>
        <envvar name="HOME" value="$SCAN_HOME"/>
        <!-- illumos's ld.so.1 doesn't search /opt/atomdrift/scan/lib by default;
             set LD_LIBRARY_PATH so rizin can find librz_util.so.0.x and
             friends, plus pkgsrc libs at /opt/local/lib for transitive deps. -->
        <envvar name="LD_LIBRARY_PATH" value="$SRC_PREFIX/lib:$PKG_PREFIX/lib"/>
        <envvar name="TMPDIR" value="$SCAN_TMP_DIR"/>
        <envvar name="TMP"    value="$SCAN_TMP_DIR"/>
        <envvar name="TEMP"   value="$SCAN_TMP_DIR"/>
        <!-- OpenAI-compatible endpoint for the --interpret LLM second-opinion pass. -->
        <envvar name="SCAN_LLM" value="$LLM"/>
      </method_environment>
    </method_context>
    <exec_method type="method" name="start"
                 exec="$SCAN_BIN $worker_args"
                 timeout_seconds="0"/>
    <exec_method type="method" name="stop" exec=":kill" timeout_seconds="60"/>
    <property_group name="startd" type="framework">
      <propval name="duration" type="astring" value="child"/>
      <propval name="ignore_error" type="astring" value="core,signal"/>
    </property_group>
    <stability value="Unstable"/>
    <template>
      <common_name><loctext xml:lang="C">Atomdrift Scan Worker</loctext></common_name>
    </template>
  </service>
</service_bundle>
EOF

service_configured() {
    svcs -H -o state "$SMF_FMRI" >/dev/null 2>&1
}

service_was_configured=0
if service_configured; then
    service_was_configured=1
fi

if [ ! -f "$SMF_MANIFEST" ] || ! cmp -s "$manifest_tmp" "$SMF_MANIFEST"; then
    log "Updating SMF manifest at $SMF_MANIFEST"
    mkdir -p "$(dirname "$SMF_MANIFEST")"
    cp -f "$manifest_tmp" "$SMF_MANIFEST"
    manifest_changed=1
else
    log "SMF manifest unchanged"
fi
rm -f "$manifest_tmp"

if [ "$manifest_changed" -eq 1 ] || [ "$service_was_configured" -eq 0 ]; then
    log "Importing SMF manifest"
    svccfg import "$SMF_MANIFEST"
fi

if [ "$ascan_changed" -eq 1 ] || [ "$rules_changed" -eq 1 ] \
        || [ "$manifest_changed" -eq 1 ] || [ "$service_was_configured" -eq 0 ]; then
    log "Clearing maintenance state and restarting $SMF_FMRI"
    svcadm clear "$SMF_FMRI" 2>/dev/null || true
    svcadm restart "$SMF_FMRI"
else
    log "$SMF_FMRI already current; skipping restart"
fi

log "Deployment complete"
