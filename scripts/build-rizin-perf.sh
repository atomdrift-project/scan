#!/bin/sh
# Build an isolated Rizin 0.9.1 with the offset-index fix and optional compact
# filefacts output. Nothing is installed outside OUTPUT_DIR.
# Usage: sh scripts/build-rizin-perf.sh [SOURCE_ARCHIVE] [OUTPUT_DIR]
# RIZIN_NATIVE=1 enables native CPU tuning and LTO (for this host only).
# RIZIN_PGO_TRAIN_DIR optionally holds extracted native binaries to train on.
# RIZIN_STATIC=1 links Rizin libraries together (the system runtime stays dynamic).
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
archive=${1:-}
output_dir=${2:-out/rizin-perf-0.9.1}
mkdir -p "$output_dir"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd)
if [ -z "$archive" ]; then
    archive="$output_dir/rizin-src-v0.9.1.tar.xz"
    if [ ! -f "$archive" ]; then
        curl -fL --retry 2 \
            https://github.com/rizinorg/rizin/releases/download/v0.9.1/rizin-src-v0.9.1.tar.xz \
            -o "$archive"
    fi
fi
python3 - "$archive" <<'PY'
import hashlib, sys
with open(sys.argv[1], 'rb') as source:
    digest = hashlib.sha256(source.read()).hexdigest()
expected = '7ac1cd7daca7afdda742e15478b1f747fc1f813e496fee71839d1e109e543dca'
if digest != expected:
    raise SystemExit(f'Unexpected Rizin source SHA-256: {digest}')
PY

mkdir -p "$output_dir/src" "$output_dir/bin"
tar -xJf "$archive" -C "$output_dir/src" --strip-components=1
patch -d "$output_dir/src" -p1 < "$repo_dir/patches/rizin-0.9.1-offset-index.patch"
patch -d "$output_dir/src" -p1 < "$repo_dir/patches/rizin-0.9.1-filefacts-json.patch"
if ! command -v meson >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
    python3 -m venv "$output_dir/build-tools"
    "$output_dir/build-tools/bin/pip" install meson==1.12.1 ninja==1.13.2
    PATH="$output_dir/build-tools/bin:$PATH"
    export PATH
fi
set --
if [ "${RIZIN_NATIVE:-0}" = 1 ]; then
    set -- -Db_lto=true '-Dc_args=-march=native -mtune=native'
fi
if [ "${RIZIN_STATIC:-0}" = 1 ]; then
    set -- "$@" -Ddefault_library=static
fi
if [ -n "${RIZIN_PGO_TRAIN_DIR:-}" ]; then
    [ -d "$RIZIN_PGO_TRAIN_DIR" ] || { echo "Training directory not found: $RIZIN_PGO_TRAIN_DIR" >&2; exit 1; }
    set -- "$@" -Db_pgo=generate
fi
meson setup "$output_dir/build" "$output_dir/src" \
    --prefix="$output_dir/install" --libdir=lib --buildtype=release \
    -Dlocal=enabled -Dportable=true -Denable_tests=false "$@"
meson compile -C "$output_dir/build" -j "${RIZIN_BUILD_JOBS:-8}"
meson install -C "$output_dir/build"

if [ -n "${RIZIN_PGO_TRAIN_DIR:-}" ]; then
    python3 - "$output_dir" "$RIZIN_PGO_TRAIN_DIR" <<'TRAIN'
import os
from pathlib import Path
import subprocess
import sys

output, training = map(Path, sys.argv[1:])
samples = sorted(path for path in training.iterdir() if path.is_file())
if not samples:
    raise SystemExit("PGO training directory must contain native binary samples")
command = [str(output / 'install/bin/rizin'), '-NN', '-q', '-T', '-z',
           '-e', 'scr.color=0', '-e', 'log.level=3', '-e', 'analysis.vars=false',
           '-c', 'iij; echo ===SEP===; iEj; echo ===SEP===; aalg; aa; aac; '
                 'echo ===SEP===; aflj; echo ===SEP===; iSj']
with (output / 'pgo-training.log').open('wb') as log:
    for sample in samples:
        subprocess.run([*command, str(sample.resolve())], check=True,
                       stdout=subprocess.DEVNULL, stderr=log,
                       env=dict(os.environ, RZ_FACTS_JSON='1'))
        print(f'Trained: {sample}', flush=True)
TRAIN
    meson configure "$output_dir/build" -Db_pgo=use
    meson compile -C "$output_dir/build" -j "${RIZIN_BUILD_JOBS:-8}"
    meson install -C "$output_dir/build"
fi

# The library regression deliberately links against this build's headers and
# library, not the system Rizin installation.
python3 - "$repo_dir" "$output_dir" <<'LINK'
import os
from pathlib import Path
import shlex
import subprocess
import sys
repo, output = map(Path, sys.argv[1:])
env = dict(os.environ)
env['PKG_CONFIG_PATH'] = str(output / 'install/lib/pkgconfig') + os.pathsep + env.get('PKG_CONFIG_PATH', '')
flags = shlex.split(subprocess.check_output(
    ['pkg-config', '--static', '--cflags', '--libs', 'rz_flag'], env=env, text=True))
subprocess.run([*shlex.split(env.get('CC', 'cc')), '-O2',
                str(repo / 'scripts/rizin/offset-index.c'), *flags,
                '-o', str(output / 'test-offset-index')], check=True)
LINK
LD_LIBRARY_PATH="$output_dir/install/lib" "$output_dir/test-offset-index"
python3 "$repo_dir/scripts/rizin/function-projection.py" \
    "$output_dir/install/bin/rizin" "$output_dir/test-offset-index"
cat > "$output_dir/bin/rizin" <<'EOF'
#!/bin/sh
# Only the scanner-facing wrapper projects the function table. The installed
# tool at install/bin/rizin retains its complete default JSON output.
export RZ_FACTS_JSON=1
wrapper_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "$wrapper_dir/install/bin/rizin" "$@"
EOF
chmod +x "$output_dir/bin/rizin"
echo "Built and tested: $output_dir/bin/rizin"
echo "Prepend $output_dir/bin to PATH for the benchmark."
