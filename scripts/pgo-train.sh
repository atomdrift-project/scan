#!/bin/sh
# Train a PGO profile for atomscan: build it instrumented, scan the training
# paths with that binary, merge the counters into one .profdata.
#
# This is the one definition of training. `make pgo-train` runs it for local
# optimized builds and .github/workflows/build.yml runs it for the natively
# built release targets, so the two cannot drift.
#
# usage: scripts/pgo-train.sh -p PROFILE -o PROFDATA [-t TARGET] PATH...
#
#   -p PROFILE   cargo profile to instrument (`release` or `release-lto`). The
#                optimized build the profile is applied to must use the same
#                one: a thin-LTO profile applied to the fat-LTO build measured
#                slower than no profile at all.
#   -o PROFDATA  where the merged profile goes; its directory is created, and
#                the raw counters live beside it in raw-PROFILE/ while
#                training runs.
#   -t TARGET    pass `--target TARGET` to cargo; the instrumented binary is
#                then under target/TARGET/PROFILE/.
#   PATH...      files or directories to scan. Paths that do not exist are
#                skipped; at least one must exist. Any representative corpus
#                works -- PGO only needs the hot paths exercised.
#
# Environment: LLVM_PROFDATA names the merge tool when it is not in the
# toolchain's sysroot (`rustup component add llvm-tools`); RUSTFLAGS is kept
# and `-C profile-generate` appended; BINARY overrides the binary name.
#
# The scans run analysis only: fetching, the analysis cache and the update
# check are off, so the counters reflect the scan itself. The instrumented
# binary is about twice as slow as an optimized one. On Windows (MSYS/Git
# Bash) rustc and the binary get native paths through `cygpath -m`.
set -eu

profile=""
profdata=""
target=""
while getopts "p:o:t:" opt; do
	case "$opt" in
	p) profile=$OPTARG ;;
	o) profdata=$OPTARG ;;
	t) target=$OPTARG ;;
	*) echo "usage: $0 -p PROFILE -o PROFDATA [-t TARGET] PATH..." >&2; exit 2 ;;
	esac
done
shift $((OPTIND - 1))
[ -n "$profile" ] && [ -n "$profdata" ] && [ $# -gt 0 ] || {
	echo "usage: $0 -p PROFILE -o PROFDATA [-t TARGET] PATH..." >&2
	exit 2
}
bin=${BINARY:-atomscan}

found=0
for path in "$@"; do
	[ -e "$path" ] && found=1
done
[ "$found" = 1 ] || {
	echo "error: no training corpus found among: $*" >&2
	exit 1
}

# llvm-profdata: LLVM_PROFDATA, else the active toolchain's llvm-tools, else
# any rustup toolchain that has it.
lp=${LLVM_PROFDATA:-}
if [ -z "$lp" ]; then
	sysroot=$(rustc --print sysroot)
	host=$(rustc -vV | sed -n 's/^host: //p')
	for cand in "$sysroot/lib/rustlib/$host/bin/llvm-profdata" "$sysroot/lib/rustlib/$host/bin/llvm-profdata.exe"; do
		if [ -x "$cand" ]; then
			lp=$cand
			break
		fi
	done
fi
if [ -z "$lp" ]; then
	lp=$(find "${HOME:-/nonexistent}/.rustup" -type f -name 'llvm-profdata*' 2>/dev/null | head -1)
fi
[ -n "$lp" ] || {
	echo "error: llvm-profdata not found -- run: rustup component add llvm-tools (or set LLVM_PROFDATA)" >&2
	exit 1
}

native() {
	if command -v cygpath >/dev/null 2>&1; then
		cygpath -m "$1"
	else
		printf '%s\n' "$1"
	fi
}

profdir=$(dirname "$profdata")
mkdir -p "$profdir"
profdir=$(cd "$profdir" && pwd)
prof="$profdir/$(basename "$profdata")"
raw="$profdir/raw-$profile"
rm -rf "$raw"
mkdir -p "$raw"

echo "pgo-train: instrumented '$profile' build${target:+ for $target}"
RUSTFLAGS="${RUSTFLAGS:-} -C profile-generate=$(native "$raw")" \
	cargo build --locked --profile "$profile" ${target:+--target "$target"} --bin "$bin"

exe="target/${target:+$target/}$profile/$bin"
[ -x "$exe" ] || exe="$exe.exe"
[ -x "$exe" ] || {
	echo "error: instrumented binary not found at $exe" >&2
	exit 1
}

for path in "$@"; do
	[ -e "$path" ] || continue
	echo "pgo-train: scanning $path with the instrumented binary"
	SCAN_FOLLOW=none SCAN_NO_ANALYSIS_CACHE=1 CLEAVE_SKIP_CACHE=1 SCAN_NO_UPDATE_CHECK=1 \
		"$exe" -f json --mode balanced --no-update --follow=none "$path" >/dev/null || true
done

set -- "$raw"/*.profraw
[ -e "$1" ] || {
	echo "error: the training scans wrote no counters under $raw" >&2
	exit 1
}
"$lp" merge -o "$(native "$prof")" "$@"
rm -rf "$raw"
echo "pgo-train: profile written to $prof"
