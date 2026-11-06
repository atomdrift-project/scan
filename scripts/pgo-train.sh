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
#                then under $CARGO_TARGET_DIR/TARGET/PROFILE/.
#   PATH...      files or directories to scan. Paths that do not exist are
#                skipped; at least one must exist. Any representative corpus
#                works -- PGO only needs the hot paths exercised.
#
# Environment: LLVM_PROFDATA names the merge tool; otherwise it must be the
# active toolchain's own (`rustup component add llvm-tools`), since a merge
# tool from another LLVM can write a profile rustc rejects or misreads.
# RUSTFLAGS is kept and `-C profile-generate` appended; CARGO_TARGET_DIR is
# honored; BINARY overrides the binary name.
#
# The scans run analysis only: fetching, the analysis cache and the update
# check are off, so the counters reflect the scan itself. Every scan must end
# in one of atomscan's verdict exit codes (0 clean, 1 hostile, 2 suspicious,
# 3 file errors, 4 degraded YARA engine) and write a report; a panic, a usage
# error or a fatal error fails the training rather than yielding a profile of
# startup alone. The instrumented binary is about twice as slow as an
# optimized one. On Windows (MSYS/Git Bash) rustc and the binary get native
# paths through `cygpath -m`.
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

# llvm-profdata: LLVM_PROFDATA, else the active toolchain's llvm-tools.
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
[ -n "$lp" ] || {
	echo "error: llvm-profdata not found in $(rustc --print sysroot) -- run: rustup component add llvm-tools (or set LLVM_PROFDATA)" >&2
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

exe="${CARGO_TARGET_DIR:-target}/${target:+$target/}$profile/$bin"
[ -x "$exe" ] || exe="$exe.exe"
[ -x "$exe" ] || {
	echo "error: instrumented binary not found at $exe" >&2
	exit 1
}

# A scan's report and diagnostics go beside the counters, so they leave with
# them. stderr is replayed after each scan, so the log still shows it.
out="$raw/scan.out"
err="$raw/scan.err"
scans=0
for path in "$@"; do
	[ -e "$path" ] || continue
	echo "pgo-train: scanning $path with the instrumented binary"
	status=0
	SCAN_NO_ANALYSIS_CACHE=1 CLEAVE_SKIP_CACHE=1 SCAN_NO_UPDATE_CHECK=1 \
		"$exe" -f json --mode balanced --no-update --follow=none "$path" >"$out" 2>"$err" || status=$?
	cat "$err" >&2
	case "$status" in
	0 | 1 | 2 | 3 | 4) ;;
	*)
		echo "error: scanning $path exited $status, which is not a verdict" >&2
		exit 1
		;;
	esac
	# 1 and 2 are also what a fatal error and a usage error exit with; a
	# finished scan prints no `Error: ` line and always writes a report.
	if grep -q '^Error: ' "$err" || [ ! -s "$out" ]; then
		echo "error: scanning $path failed (exit $status) -- see above" >&2
		exit 1
	fi
	if [ "$status" = 4 ]; then
		echo "warning: the YARA engine degraded while scanning $path; the profile under-represents rule matching" >&2
	fi
	scans=$((scans + 1))
done
rm -f "$out" "$err"
[ "$scans" -gt 0 ] || {
	echo "error: no training scan completed" >&2
	exit 1
}

set -- "$raw"/*.profraw
[ -e "$1" ] || {
	echo "error: the training scans wrote no counters under $raw" >&2
	exit 1
}
"$lp" merge -o "$(native "$prof")" "$@"
rm -rf "$raw"
echo "pgo-train: profile written to $prof"
