#!/bin/sh
# Build only the ABI-compatible flag library from a Rizin v0.8.2 checkout.
# Usage: sh scripts/build-rizin-flag-fix.sh /path/to/rizin-v0.8.2 [output-dir]
set -eu
source_dir=${1:?usage: build-rizin-flag-fix.sh RIZIN_SOURCE [OUTPUT_DIR]}
output_dir=${2:-out/rizin-perf}
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test "$(pkg-config --modversion rz_util)" = 0.8.2 || {
    echo 'This patch requires Rizin 0.8.2 development headers and libraries.' >&2
    exit 1
}
test "$(git -C "$source_dir" describe --tags --exact-match HEAD)" = v0.8.2
mkdir -p "$output_dir/src" "$output_dir/lib" "$output_dir/bin"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd)
# Export the committed sources, so the source checkout is never modified.
git -C "$source_dir" archive HEAD librz/flag | tar -x -C "$output_dir/src"
patch -d "$output_dir/src" -p1 < "$repo_dir/patches/rizin-0.8.2-offset-index.patch"
# pkg-config intentionally produces separate compiler/linker arguments.
${CC:-cc} -O2 -g -fPIC -shared -Wl,-soname,librz_flag.so.0.8 \
    $(pkg-config --cflags rz_util) \
    "$output_dir"/src/librz/flag/*.c \
    $(pkg-config --libs rz_util) -o "$output_dir/lib/librz_flag.so.0.8.2"
ln -sf librz_flag.so.0.8.2 "$output_dir/lib/librz_flag.so.0.8"
ln -sf librz_flag.so.0.8.2 "$output_dir/lib/librz_flag.so"
system_rizin=$(pkg-config --variable=prefix rz_util)/bin/rizin
test -x "$system_rizin"
cat > "$output_dir/bin/rizin" <<EOF
#!/bin/sh
export LD_LIBRARY_PATH='$output_dir/lib'
exec '$system_rizin' "\$@"
EOF
chmod +x "$output_dir/bin/rizin"
${CC:-cc} -O2 $(pkg-config --cflags rz_flag) \
    "$repo_dir/scripts/rizin/offset-index.c" $(pkg-config --libs rz_flag) \
    -o "$output_dir/test-offset-index"
LD_LIBRARY_PATH="$output_dir/lib" "$output_dir/test-offset-index"
echo "Built and tested: $output_dir/bin/rizin"
echo "Prepend $output_dir/bin to PATH for the benchmark."
