#!/bin/sh
# worker-omnios.sh - Deploy litmus worker inside an OmniOS zone (as root).
# Usage: ./worker-omnios.sh <hopper-url>
# Must be invoked from the litmus repository root inside the zone.
# Idempotent: re-run to update.
#
# rust + 7zip come from pkgsrc; innoextract, upx, and rizin (HEAD) are
# built from upstream source. The worker runs as the unprivileged
# `litmus` user under SMF, with ignore_error=core,signal so a crashing
# backend triggers a restart instead of dropping into maintenance.

set -eu

URL="${1:-}"
[ -n "$URL" ] || { echo "error: hopper URL required as first argument" >&2; exit 1; }

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

PKG_PREFIX=/opt/local
# Source builds and the litmus binary install under /opt/litmus rather than
# /usr/local: in sparse zones /usr is a read-only lofs mount from the global,
# so /usr/local is unwritable. /opt is writable per-zone.
SRC_PREFIX=/opt/litmus
SRC_DIR=/var/litmus/src
LITMUS_USER=litmus
LITMUS_GROUP=litmus
LITMUS_HOME=/var/litmus
LITMUS_LOG_DIR=/var/log/litmus
LITMUS_BIN=$SRC_PREFIX/bin/litmus
SMF_MANIFEST=/lib/svc/manifest/site/litmus-worker.xml
SMF_FMRI=svc:/site/litmus-worker:default

PATH=$PKG_PREFIX/sbin:$PKG_PREFIX/bin:$SRC_PREFIX/sbin:$SRC_PREFIX/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

[ "$(id -u)" -eq 0 ] || die "must be run as root inside the zone"
[ -f Cargo.toml ] || die "must be invoked from the litmus repository root"
command -v pkgin >/dev/null 2>&1 || die "pkgin not found; install pkgsrc bootstrap first"
command -v pkg >/dev/null 2>&1   || die "IPS pkg(1) not found; not running on illumos?"

###############################################################################
# IPS system prerequisites: crt1.o, headers, ld, libc — needed by pkgsrc gcc
# to link anything. Without build-essential the linker fails with
# "ld: fatal: file crt1.o: open failed".
###############################################################################

# IPS pkg(1) returns 4 when nothing needs doing — treat that as success.
ensure_ips() {
    pkg install -q "$1" && return 0
    rc=$?
    [ "$rc" -eq 4 ] && return 0
    return "$rc"
}

log "Ensuring IPS build prerequisites (headers, ld, crt files)"
# build-essential pulls headers + ld; c-runtime supplies /usr/lib/{,amd64/}crt1.o
# which build-essential does NOT pull in on r151058+, despite the name.
ensure_ips build-essential        || die "failed to install IPS build-essential"
ensure_ips system/library/c-runtime || die "failed to install IPS c-runtime"

JOBS=$(psrinfo | wc -l | tr -d ' ')
[ "$JOBS" -gt 0 ] 2>/dev/null || JOBS=2

###############################################################################
# pkgsrc binary dependencies
###############################################################################

log "Refreshing pkgin catalog"
pkgin -y update >/dev/null 2>&1 || true

# rust + 7zip are explicitly requested via pkgsrc. The remainder are
# build-time prerequisites for the source builds below.
log "Installing pkgsrc binary dependencies"
to_install=""
add_pkg() {
    pkg_info -qe "$1" 2>/dev/null || to_install="$to_install $1"
}
add_pkg rust
add_pkg p7zip
add_pkg git
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
if [ -n "$to_install" ]; then
    # shellcheck disable=SC2086
    pkgin -y install $to_install
else
    log "All pkgsrc dependencies already installed"
fi

###############################################################################
# Source builds: innoextract, upx, rizin (HEAD)
###############################################################################

mkdir -p "$SRC_DIR" "$SRC_PREFIX/bin" "$SRC_PREFIX/lib" "$SRC_PREFIX/share"

# build_from_git <name> <repo> <ref> <builder-fn>
# Idempotent: skips the build when the resolved SHA matches the marker.
build_from_git() {
    name=$1; repo=$2; ref=$3; builder=$4
    dir=$SRC_DIR/$name

    if [ ! -d "$dir/.git" ]; then
        log "Cloning $name"
        git clone "$repo" "$dir"
    fi

    git -C "$dir" fetch --tags --prune origin
    case "$ref" in
        HEAD) target=origin/HEAD ;;
        *)    target=$ref ;;
    esac
    git -C "$dir" reset --hard "$target"
    git -C "$dir" submodule update --init --recursive

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
    meson compile -C build
    meson install -C build
}

build_from_git innoextract https://github.com/dscharrer/innoextract.git HEAD       build_innoextract
build_from_git upx          https://github.com/upx/upx.git                v4.2.4   build_upx
build_from_git rizin        https://github.com/rizinorg/rizin.git         HEAD     build_rizin

###############################################################################
# litmus user, build, install
###############################################################################

log "Ensuring $LITMUS_USER user/group exist"
getent group "$LITMUS_GROUP" >/dev/null 2>&1 || groupadd "$LITMUS_GROUP"
id "$LITMUS_USER" >/dev/null 2>&1 || \
    useradd -m -d "$LITMUS_HOME" -g "$LITMUS_GROUP" -s /bin/sh \
            -c "Litmus Worker" "$LITMUS_USER"
mkdir -p "$LITMUS_HOME" "$LITMUS_LOG_DIR"
chown "$LITMUS_USER:$LITMUS_GROUP" "$LITMUS_HOME" "$LITMUS_LOG_DIR"

log "Building litmus from source tree"
cargo build --release || die "litmus build failed"

log "Installing litmus binary to $LITMUS_BIN"
install -m 755 target/release/litmus "$LITMUS_BIN"

log "Refreshing rules/models as $LITMUS_USER"
su - "$LITMUS_USER" -c "PATH=$PATH $LITMUS_BIN update-rules" \
    || die "update-rules failed"

###############################################################################
# SMF manifest
###############################################################################

log "Writing SMF manifest at $SMF_MANIFEST"
mkdir -p "$(dirname "$SMF_MANIFEST")"

# duration=contract + ignore_error=core,signal: SMF restarts the worker
# whether it exits cleanly, dies on signal, or dumps core, instead of
# parking the service in maintenance. exec is run via /bin/sh -c so the
# pkgsrc PATH is in effect for child processes (rizin, 7z, upx, innoextract).
cat > "$SMF_MANIFEST" <<EOF
<?xml version="1.0"?>
<!DOCTYPE service_bundle SYSTEM "/usr/share/lib/xml/dtd/service_bundle.dtd.1">
<service_bundle type="manifest" name="litmus-worker">
  <service name="site/litmus-worker" type="service" version="1">
    <create_default_instance enabled="true"/>
    <single_instance/>
    <dependency name="network" grouping="require_all" restart_on="error" type="service">
      <service_fmri value="svc:/milestone/network:default"/>
    </dependency>
    <dependency name="filesystem-local" grouping="require_all" restart_on="error" type="service">
      <service_fmri value="svc:/system/filesystem/local:default"/>
    </dependency>
    <method_context>
      <method_environment>
        <envvar name="PATH" value="$PATH"/>
        <envvar name="HOME" value="$LITMUS_HOME"/>
      </method_environment>
      <method_credential user="$LITMUS_USER" group="$LITMUS_GROUP"/>
    </method_context>
    <exec_method type="method" name="start"
                 exec="$LITMUS_BIN worker --url $URL"
                 timeout_seconds="120"/>
    <exec_method type="method" name="stop" exec=":kill" timeout_seconds="60"/>
    <property_group name="startd" type="framework">
      <propval name="duration" type="astring" value="contract"/>
      <propval name="ignore_error" type="astring" value="core,signal"/>
    </property_group>
    <stability value="Unstable"/>
    <template>
      <common_name><loctext xml:lang="C">Litmus Worker</loctext></common_name>
    </template>
  </service>
</service_bundle>
EOF

log "Importing SMF manifest"
svccfg import "$SMF_MANIFEST"

log "Clearing maintenance state and restarting $SMF_FMRI"
svcadm clear "$SMF_FMRI" 2>/dev/null || true
svcadm restart "$SMF_FMRI"

log "Deployment complete"
