#!/bin/sh
# install.sh — install Atomdrift Scan (the `atomscan` CLI).
#
#   curl -fsSL https://install.atomdrift.org | sh
#
# Passing options through a pipe needs `sh -s --`:
#
#   curl -fsSL https://install.atomdrift.org | sh -s -- --dir ~/bin --method binary
#
# What it does, in order: work out the platform, pick an install method, fetch a
# release binary (checksum- and provenance-verified), fall back to a source
# build when no binary exists for the platform, then report on the optional
# analysis tools that make scans deeper.
#
# Re-running is safe and cheap: an install that is already current is left
# alone, and everything written to the install directory is written atomically.
#
# POSIX sh only — no bashisms, no `local`, no arrays. This has to run on macOS,
# Linux (glibc and musl), FreeBSD, OpenBSD, NetBSD, DragonFly, and illumos,
# where /bin/sh may be dash, ash, busybox, or ksh93. Function-scoped variables
# do not exist here, so each function prefixes its own.

set -eu

REPO="atomdrift-project/scan"
BIN="atomscan"
TAP="atomdrift/tap"
TAP_URL="https://github.com/atomdrift-project/homebrew-tap.git"

# Targets published by .github/workflows/release.yml. Anything else takes the
# source path.
TARGETS="
x86_64-unknown-linux-gnu
aarch64-unknown-linux-gnu
x86_64-unknown-linux-musl
aarch64-unknown-linux-musl
s390x-unknown-linux-gnu
riscv64gc-unknown-linux-gnu
powerpc64le-unknown-linux-gnu
aarch64-apple-darwin
x86_64-apple-darwin
x86_64-unknown-freebsd
x86_64-unknown-openbsd
x86_64-unknown-netbsd
x86_64-unknown-dragonfly
x86_64-unknown-illumos
"

# Settings, overridable by flag or environment.
OPT_VERSION="${ATOMSCAN_VERSION:-}"
OPT_DIR="${ATOMSCAN_INSTALL_DIR:-}"
OPT_METHOD="${ATOMSCAN_METHOD:-auto}"
OPT_TOOLS=1
OPT_FORCE=0
OPT_QUIET=0
[ -n "${ATOMSCAN_NO_TOOLS:-}" ] && OPT_TOOLS=0

# Filled in as we go; declared here so the shape of the script is visible.
SELF=""         # this machine's target triple, published or not
TARGET=""       # same, but empty unless a release actually carries it
PLATFORM=""     # human-readable platform description
METHOD=""       # resolved: brew | binary | source
VERSION=""      # version being installed, without the leading v
INSTALL_DIR=""  # directory the binary lands in
INSTALLED=""    # full path of the binary we installed
CHANGED=0       # whether this run actually replaced anything
BREW_PREFIX=""  # Homebrew root, empty when there is no Homebrew
DOWNLOADER=""   # curl | wget | fetch | ftp
WGET_MODERN=0   # GNU wget (spider, header dumps) rather than busybox wget
NAP=1           # progress-bar poll interval, seconds
TMP=""          # scratch directory, removed on exit
DL_PID=""       # background downloader, killed on interrupt

usage() {
	cat <<EOF
Install Atomdrift Scan — malware and supply-chain analysis for files,
directories, archives, packages, URLs, and processes.

Usage: install.sh [options]
       curl -fsSL https://install.atomdrift.org | sh -s -- [options]

Options:
  --version VERSION   Install a specific version (default: the latest release).
  --dir DIR           Install into DIR (default: a writable bin dir on PATH).
  --method METHOD     auto, binary, brew, or source. auto prefers Homebrew on
                      macOS, then a release binary, then a source build.
  --no-tools          Skip the optional analysis tool check (rizin, upx, ...).
  --force             Reinstall even when the target version is already there.
  --quiet             Only report problems.
  --help              Show this message.

Environment:
  ATOMSCAN_VERSION, ATOMSCAN_INSTALL_DIR, ATOMSCAN_METHOD, ATOMSCAN_NO_TOOLS
  NO_COLOR            Disable colour (any value).

To uninstall: delete the binary whose path this prints, or run
\`brew uninstall $TAP/scan\` if it was installed with Homebrew.
EOF
}

parse_args() {
	while [ $# -gt 0 ]; do
		case $1 in
		--version) [ $# -ge 2 ] || die "--version needs a value"; OPT_VERSION=$2; shift 2 ;;
		--version=*) OPT_VERSION=${1#*=}; shift ;;
		--dir) [ $# -ge 2 ] || die "--dir needs a value"; OPT_DIR=$2; shift 2 ;;
		--dir=*) OPT_DIR=${1#*=}; shift ;;
		--method) [ $# -ge 2 ] || die "--method needs a value"; OPT_METHOD=$2; shift 2 ;;
		--method=*) OPT_METHOD=${1#*=}; shift ;;
		--no-tools) OPT_TOOLS=0; shift ;;
		--force) OPT_FORCE=1; shift ;;
		--quiet | -q) OPT_QUIET=1; shift ;;
		--help | -h) usage; exit 0 ;;
		*) usage >&2; printf '\nunknown option: %s\n' "$1" >&2; exit 2 ;;
		esac
	done

	case $OPT_METHOD in
	auto | binary | brew | source) ;;
	*) die "--method must be auto, binary, brew, or source (got '$OPT_METHOD')" ;;
	esac
	OPT_VERSION=${OPT_VERSION#v}
}

# ---------------------------------------------------------------------------
# Style
#
# Three independent decisions — colour depth, character set, and whether we are
# drawing on a terminal at all — each degrading on its own, so a log file gets
# clean ASCII and a 24-bit terminal gets the brand palette.
# ---------------------------------------------------------------------------

setup_style() {
	ESC=$(printf '\033')
	TTY=0
	[ -t 1 ] && TTY=1

	# The palette is the six-bar spectrum from media/logo.svg.
	if [ -n "${NO_COLOR:-}" ] || [ "$TTY" = 0 ] || [ "${TERM:-dumb}" = dumb ]; then
		C_RED='' C_ORANGE='' C_AMBER='' C_GREEN='' C_TEAL='' C_BLUE=''
		C_DIM='' C_BOLD='' C_RESET=''
	else
		case "${COLORTERM:-}" in
		truecolor | 24bit)
			C_RED="${ESC}[38;2;226;75;74m" C_ORANGE="${ESC}[38;2;216;90;48m"
			C_AMBER="${ESC}[38;2;239;159;39m" C_GREEN="${ESC}[38;2;99;153;34m"
			C_TEAL="${ESC}[38;2;29;158;117m" C_BLUE="${ESC}[38;2;55;138;221m"
			;;
		*)
			case "${TERM:-}" in
			*256color* | *-direct*)
				C_RED="${ESC}[38;5;167m" C_ORANGE="${ESC}[38;5;166m" C_AMBER="${ESC}[38;5;214m"
				C_GREEN="${ESC}[38;5;106m" C_TEAL="${ESC}[38;5;36m" C_BLUE="${ESC}[38;5;68m"
				;;
			*)
				C_RED="${ESC}[31m" C_ORANGE="${ESC}[31m" C_AMBER="${ESC}[33m"
				C_GREEN="${ESC}[32m" C_TEAL="${ESC}[36m" C_BLUE="${ESC}[34m"
				;;
			esac
			;;
		esac
		C_DIM="${ESC}[2m" C_BOLD="${ESC}[1m" C_RESET="${ESC}[0m"
	fi

	# Box-drawing and block elements only. Unlike the geometric shapes (● ◆ ·)
	# they are unambiguously one column wide, so the art cannot smear in a CJK
	# locale — and every one of them has an ASCII twin below.
	case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
	*[Uu][Tt][Ff]-8* | *[Uu][Tt][Ff]8*) UTF8=1 ;;
	*) UTF8=0 ;;
	esac
	if [ "$UTF8" = 1 ]; then
		G_STEP="▸" G_OK="✓" G_WARN="!" G_ERR="✗" G_BAR="━━" G_FULL="█" G_EMPTY="░"
	else
		G_STEP=">" G_OK="+" G_WARN="!" G_ERR="x" G_BAR="==" G_FULL="#" G_EMPTY="."
	fi

	BLANKS="                                                                              "
	STEP_N=0 # the bullet colour walks the spectrum as the install progresses
}

step_color() {
	STEP_N=$((STEP_N + 1))
	case $((STEP_N % 6)) in
	1) printf '%s' "$C_RED" ;;
	2) printf '%s' "$C_ORANGE" ;;
	3) printf '%s' "$C_AMBER" ;;
	4) printf '%s' "$C_GREEN" ;;
	5) printf '%s' "$C_TEAL" ;;
	*) printf '%s' "$C_BLUE" ;;
	esac
}

# step LABEL VALUE — one aligned line of progress.
step() {
	if [ "$OPT_QUIET" = 0 ]; then
		printf ' %s%s%s %s%-11s%s%s\n' "$(step_color)" "$G_STEP" "$C_RESET" "$C_DIM" "$1" "$C_RESET" "$2"
	fi
}

ok() {
	if [ "$OPT_QUIET" = 0 ]; then
		printf ' %s%s%s %s%-11s%s%s\n' "$C_GREEN" "$G_OK" "$C_RESET" "$C_DIM" "$1" "$C_RESET" "$2"
	fi
}

note() {
	if [ "$OPT_QUIET" = 0 ]; then
		printf '   %s%s%s\n' "$C_DIM" "$1" "$C_RESET"
	fi
}

warn() {
	printf ' %s%s%s %s\n' "$C_AMBER" "$G_WARN" "$C_RESET" "$1" >&2
}

die() {
	printf '\n %s%s%s %s%s%s\n\n' "$C_RED" "$G_ERR" "$C_RESET" "$C_BOLD" "$1" "$C_RESET" >&2
	exit 1
}

# The logo's six-bar spectrum, in text.
spectrum_rule() {
	printf '%s%s %s%s %s%s %s%s %s%s %s%s%s' \
		"$C_RED" "$G_BAR" "$C_ORANGE" "$G_BAR" "$C_AMBER" "$G_BAR" \
		"$C_GREEN" "$G_BAR" "$C_TEAL" "$G_BAR" "$C_BLUE" "$G_BAR" "$C_RESET"
}

# An atom: a nucleus inside an orbit drawn in the six brand colours, running
# clockwise from red at ten o'clock round to blue at nine.
banner() {
	if [ "$OPT_QUIET" = 1 ]; then
		return 0
	fi
	ba_tag="static analysis + local ML malware detection"
	printf '\n'
	if [ "$UTF8" = 1 ]; then
		printf '   %s╭────%s────╮%s\n' "$C_RED" "$C_ORANGE" "$C_RESET"
		printf ' %s╭─╯%s        %s╰─╮%s    %satomdrift scan%s\n' \
			"$C_RED" "$C_RESET" "$C_AMBER" "$C_RESET" "$C_BOLD" "$C_RESET"
		printf ' %s│%s     %s%s██%s     %s│%s    %s\n' \
			"$C_BLUE" "$C_RESET" "$C_BOLD" "$C_AMBER" "$C_RESET" "$C_AMBER" "$C_RESET" "$(spectrum_rule)"
		printf ' %s╰─╮%s        %s╭─╯%s    %s%s%s\n' \
			"$C_TEAL" "$C_RESET" "$C_GREEN" "$C_RESET" "$C_DIM" "$ba_tag" "$C_RESET"
		printf '   %s╰────%s────╯%s\n' "$C_TEAL" "$C_GREEN" "$C_RESET"
	else
		printf '   %s.----%s----.%s\n' "$C_RED" "$C_ORANGE" "$C_RESET"
		printf " %s.-'%s        %s'-.%s    %satomdrift scan%s\n" \
			"$C_RED" "$C_RESET" "$C_AMBER" "$C_RESET" "$C_BOLD" "$C_RESET"
		printf ' %s|%s     %s%s**%s     %s|%s    %s\n' \
			"$C_BLUE" "$C_RESET" "$C_BOLD" "$C_AMBER" "$C_RESET" "$C_AMBER" "$C_RESET" "$(spectrum_rule)"
		printf " %s'-.%s        %s.-'%s    %s%s%s\n" \
			"$C_TEAL" "$C_RESET" "$C_GREEN" "$C_RESET" "$C_DIM" "$ba_tag" "$C_RESET"
		printf "   %s'----%s----'%s\n" "$C_TEAL" "$C_GREEN" "$C_RESET"
	fi
	printf '\n'
}

# Every target a release carries, two to a row under the `targets` label, with
# this machine's own picked out. A platform with no published binary is listed
# too, as the source build it is about to get.
#
# Padding is applied to the bare text and colour wrapped around the result:
# escape sequences are characters as far as printf's %-30s is concerned, so
# colouring first would throw the columns out.
list_targets() {
	if [ "$OPT_QUIET" = 1 ]; then
		return 0
	fi
	lt_label=targets lt_line="" lt_here=0 lt_n=0
	for lt_t in $TARGETS; do
		# Pad the left column only, so rows carry no trailing whitespace.
		lt_pad=%-30s
		[ $((lt_n % 2)) = 1 ] && lt_pad=%s
		if [ "$lt_t" = "$SELF" ]; then
			lt_here=1
			# shellcheck disable=SC2059 # lt_pad is a format we chose, not input
			lt_line="$lt_line${C_BOLD}${C_TEAL}$(printf "%s $lt_pad" "$G_STEP" "$lt_t")${C_RESET}"
		else
			# shellcheck disable=SC2059
			lt_line="$lt_line${C_DIM}$(printf "  $lt_pad" "$lt_t")${C_RESET}"
		fi
		lt_n=$((lt_n + 1))
		if [ $((lt_n % 2)) = 0 ]; then
			lt_row "$lt_label" "$lt_line" "$lt_here"
			lt_label="" lt_line="" lt_here=0
		fi
	done
	[ -n "$lt_line" ] && lt_row "$lt_label" "$lt_line" "$lt_here"

	if [ -z "$TARGET" ]; then
		lt_row "" "${C_BOLD}${C_AMBER}$(printf '%s %s' "$G_STEP" "$SELF")${C_RESET}${C_AMBER} (source-based install)${C_RESET}" 0
	fi
}

# lt_row LABEL CELLS MINE — one row of the target table, aligned to the step
# lines above it: three leading spaces plus an eleven-column label.
lt_row() {
	if [ "$3" = 1 ]; then
		printf '   %s%-11s%s%s%s\n' "$C_DIM" "$1" "$C_RESET" "$2" "${C_DIM}  this machine${C_RESET}"
	else
		printf '   %s%-11s%s%s\n' "$C_DIM" "$1" "$C_RESET" "$2"
	fi
}

# ---------------------------------------------------------------------------
# Scratch space
# ---------------------------------------------------------------------------

cleanup() {
	# A download in flight owns a child process; an interrupt has to take it
	# with us, or it keeps writing into a directory we are about to delete.
	if [ -n "$DL_PID" ]; then
		kill "$DL_PID" 2>/dev/null || :
	fi
	[ -n "$TMP" ] && rm -rf "$TMP"
	return 0
}

make_tmpdir() {
	TMP=$(mktemp -d 2>/dev/null) || TMP=""
	if [ -z "$TMP" ]; then
		TMP="${TMPDIR:-/tmp}/atomscan-install.$$"
		(umask 077 && mkdir "$TMP") || die "cannot create a temporary directory"
	fi
}

# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------

detect_platform() {
	dp_os=$(uname -s 2>/dev/null || echo unknown)
	dp_arch=$(uname -m 2>/dev/null || echo unknown)
	dp_desc=""

	case $dp_arch in
	x86_64 | amd64) dp_arch=x86_64 ;;
	aarch64 | arm64) dp_arch=aarch64 ;;
	riscv64 | riscv64gc) dp_arch=riscv64gc ;;
	ppc64le | powerpc64le) dp_arch=powerpc64le ;;
	esac

	case $dp_os in
	Linux)
		# musl announces itself only in ldd's usage message, and that ldd exits
		# non-zero while printing it.
		if ls /lib/ld-musl-* >/dev/null 2>&1 || (ldd --version 2>&1 || :) | grep -qi musl; then
			SELF="$dp_arch-unknown-linux-musl"
		else
			SELF="$dp_arch-unknown-linux-gnu"
		fi
		dp_desc="Linux"
		if [ -r /etc/os-release ]; then
			# shellcheck disable=SC1091
			dp_pretty=$(. /etc/os-release 2>/dev/null && printf '%s' "${PRETTY_NAME:-}")
			[ -n "$dp_pretty" ] && dp_desc=$dp_pretty
		fi
		;;
	Darwin)
		# Under Rosetta `uname -m` says x86_64. Install the native binary.
		if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = 1 ]; then
			dp_arch=aarch64
		fi
		SELF="$dp_arch-apple-darwin"
		dp_desc="macOS $(sw_vers -productVersion 2>/dev/null || :)"
		;;
	FreeBSD) SELF="$dp_arch-unknown-freebsd" dp_desc="FreeBSD $(uname -r 2>/dev/null || :)" ;;
	OpenBSD) SELF="$dp_arch-unknown-openbsd" dp_desc="OpenBSD $(uname -r 2>/dev/null || :)" ;;
	NetBSD) SELF="$dp_arch-unknown-netbsd" dp_desc="NetBSD $(uname -r 2>/dev/null || :)" ;;
	DragonFly) SELF="$dp_arch-unknown-dragonfly" dp_desc="DragonFly $(uname -r 2>/dev/null || :)" ;;
	SunOS) SELF="$dp_arch-unknown-illumos" dp_desc=$(uname -v 2>/dev/null || echo SunOS) ;;
	CYGWIN* | MINGW* | MSYS* | Windows_NT)
		die "on Windows, use install.ps1 instead:
   irm https://install.atomdrift.org/ps1 | iex"
		;;
	*)
		SELF="$dp_arch-unknown-$(printf '%s' "$dp_os" | tr '[:upper:]' '[:lower:]')"
		dp_desc="$dp_os $(uname -r 2>/dev/null || :)"
		;;
	esac

	# TARGET is SELF only when a release actually carries it; otherwise the
	# source path takes over.
	TARGET=""
	for dp_t in $TARGETS; do
		[ "$dp_t" = "$SELF" ] && TARGET=$SELF
	done

	PLATFORM="$SELF"
	if [ -n "$dp_desc" ]; then
		PLATFORM="$PLATFORM  ${C_DIM}${dp_desc}${C_RESET}"
	fi
	return 0
}

# ---------------------------------------------------------------------------
# HTTP
#
# curl, wget, fetch, and ftp in that order: OpenBSD and NetBSD ship the last two
# in the base system and nothing else, and both are targets we publish.
# ---------------------------------------------------------------------------

find_downloader() {
	for fd_c in curl wget fetch ftp; do
		if command -v "$fd_c" >/dev/null 2>&1; then
			DOWNLOADER=$fd_c
			break
		fi
	done
	[ -n "$DOWNLOADER" ] || die "no HTTP client found — install curl or wget and re-run"

	# busybox wget accepts neither --spider nor -S. Ask once instead of
	# discovering it mid-install.
	if [ "$DOWNLOADER" = wget ] && wget --help 2>&1 | grep -q -- --spider; then
		WGET_MODERN=1
	fi
	return 0
}

# http_get URL OUTFILE — writes the body; non-zero on any HTTP or transport error.
http_get() {
	case $DOWNLOADER in
	curl) curl -fsSL --proto '=https' --tlsv1.2 --retry 3 --retry-delay 1 -o "$2" "$1" ;;
	wget) wget -q -O "$2" "$1" ;;
	fetch) fetch -q -o "$2" "$1" ;;
	ftp) ftp -o "$2" "$1" ;;
	esac
}

# http_ok URL — false only when we can cheaply prove the URL is not there.
#
# Deliberately narrow: only a 404 counts as absent. A HEAD refused for any other
# reason must not be read as "this platform has no binary", or a working release
# would quietly become a twenty-minute source build.
http_ok() {
	case $DOWNLOADER in
	curl)
		ho_code=$(curl -sSLI --proto '=https' --tlsv1.2 -o /dev/null \
			-w '%{http_code}' "$1" 2>/dev/null || printf 000)
		[ "$ho_code" != 404 ]
		;;
	wget)
		[ "$WGET_MODERN" = 1 ] || return 0
		! wget --spider -S "$1" 2>&1 | grep -q ' 404 '
		;;
	*) : ;;
	esac
}

# content_length URL — size in bytes, or empty when it cannot be known cheaply.
content_length() {
	{
		case $DOWNLOADER in
		curl) curl -fsSLI --proto '=https' --tlsv1.2 "$1" 2>/dev/null ;;
		wget) [ "$WGET_MODERN" = 0 ] || wget -q --spider -S "$1" 2>&1 ;;
		*) : ;;
		esac
	} | tr -d '\r' |
		awk 'tolower($1) == "content-length:" { n = $2 } END { if (n ~ /^[0-9]+$/) print n }'
}

# resolve_latest — the newest release tag.
#
# Read from the redirect that /releases/latest performs, because the REST API is
# rate-limited per IP and CI runners share addresses. The API is the fallback
# for clients that cannot report a redirect target.
resolve_latest() {
	rl_url="https://github.com/$REPO/releases/latest"
	rl_final=""
	case $DOWNLOADER in
	curl) rl_final=$(curl -fsSLI --proto '=https' -o /dev/null -w '%{url_effective}' "$rl_url" 2>/dev/null || :) ;;
	wget)
		if [ "$WGET_MODERN" = 1 ]; then
			rl_final=$(wget -q --spider -S "$rl_url" 2>&1 | awk '/^ *Location:/ { print $2 }' | tail -n 1 || :)
		fi
		;;
	esac
	case $rl_final in
	*/releases/tag/*)
		printf '%s' "${rl_final##*/tag/}"
		return 0
		;;
	esac

	http_get "https://api.github.com/repos/$REPO/releases/latest" "$TMP/latest.json" 2>/dev/null || return 1
	rl_tag=$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$TMP/latest.json" | head -n 1)
	[ -n "$rl_tag" ] || return 1
	printf '%s' "$rl_tag"
}

# ---------------------------------------------------------------------------
# Download with a progress bar
#
# The downloader runs in the background and the bar is drawn from the size of
# the file it is filling, so every client gets the same display — including the
# BSD ones with no progress output of their own.
# ---------------------------------------------------------------------------

# Fractional sleep is universal on Linux, macOS, and the BSDs but not on
# illumos. Ask once rather than once per frame.
probe_sleep() {
	if sleep 0.08 2>/dev/null; then NAP=0.08; else NAP=1; fi
}

file_size() {
	# `ls -ln` is the portable stat: GNU wants -c%s, BSD wants -f%z.
	# shellcheck disable=SC2012 # the path is ours, in a mode 700 temp dir
	fs_n=$(ls -ln "$1" 2>/dev/null | awk 'NR == 1 { print $5 }')
	case $fs_n in
	'' | *[!0-9]*) fs_n=0 ;;
	esac
	printf '%s' "$fs_n"
}

human_size() {
	awk -v b="$1" 'BEGIN {
		if (b >= 1048576) printf "%.1f MB", b / 1048576
		else if (b >= 1024) printf "%.0f KB", b / 1024
		else printf "%d B", b
	}'
}

# draw_bar DONE TOTAL — redraws the download line in place.
draw_bar() {
	db_done=$1 db_total=$2 db_width=18 db_pct="" db_fill=0
	if [ -n "$db_total" ] && [ "$db_total" -gt 0 ]; then
		db_pct=$((db_done * 100 / db_total))
		[ "$db_pct" -gt 100 ] && db_pct=100
		db_fill=$((db_pct * db_width / 100))
	fi

	db_bar="" db_i=0
	while [ "$db_i" -lt "$db_width" ]; do
		if [ "$db_i" -lt "$db_fill" ]; then db_bar="$db_bar$G_FULL"; else db_bar="$db_bar$G_EMPTY"; fi
		db_i=$((db_i + 1))
	done

	if [ -n "$db_pct" ]; then
		printf '\r %s%s%s %s%-11s%s%s%s%s %3s%%  %s' \
			"$C_TEAL" "$G_STEP" "$C_RESET" "$C_DIM" download "$C_RESET" \
			"$C_TEAL" "$db_bar" "$C_RESET" "$db_pct" "$(human_size "$db_done")"
	else
		printf '\r %s%s%s %s%-11s%s%s' \
			"$C_TEAL" "$G_STEP" "$C_RESET" "$C_DIM" download "$C_RESET" "$(human_size "$db_done")"
	fi
}

# download URL OUTFILE LABEL
download() {
	dl_url=$1 dl_out=$2 dl_label=$3

	if [ "$TTY" = 0 ] || [ "$OPT_QUIET" = 1 ]; then
		step download "$dl_label"
		http_get "$dl_url" "$dl_out" || return 1
		return 0
	fi

	dl_total=$(content_length "$dl_url")
	: >"$TMP/dl.err"
	http_get "$dl_url" "$dl_out" 2>"$TMP/dl.err" &
	DL_PID=$!
	while kill -0 "$DL_PID" 2>/dev/null; do
		draw_bar "$(file_size "$dl_out")" "$dl_total"
		sleep "$NAP" 2>/dev/null || sleep 1
	done
	if wait "$DL_PID"; then
		DL_PID=""
	else
		DL_PID=""
		printf '\r%s\r' "$BLANKS"
		[ -s "$TMP/dl.err" ] && cat "$TMP/dl.err" >&2
		return 1
	fi

	dl_got=$(file_size "$dl_out")
	printf '\r%s\r' "$BLANKS"
	ok download "$dl_label  ${C_DIM}$(human_size "$dl_got")${C_RESET}"
}

# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

# sha256_of FILE — lowercase hex digest, empty when nothing here can produce one.
sha256_of() {
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum "$1" | awk '{ print $1 }'
	elif command -v shasum >/dev/null 2>&1; then
		shasum -a 256 "$1" | awk '{ print $1 }'
	elif command -v sha256 >/dev/null 2>&1; then
		sha256 -q "$1"
	elif command -v openssl >/dev/null 2>&1; then
		openssl dgst -sha256 "$1" | awk '{ print $NF }'
	elif command -v digest >/dev/null 2>&1; then
		digest -a sha256 "$1"
	elif cksum -a sha256 "$1" >/dev/null 2>&1; then
		cksum -a sha256 "$1" | awk '{ print $NF }'
	fi
}

# verify_checksum TARBALL SUMSFILE NAME — prints a short digest, non-zero on
# any doubt whatsoever. The caller treats failure as fatal.
#
# The digest travels over the same TLS connection as the archive, so this is a
# corruption and truncation check; verify_provenance below is the trust anchor.
verify_checksum() {
	vc_want=$(awk -v want="$3" '
		{ n = $2; sub(/^\.\//, "", n); sub(/^\*/, "", n)
		  if (n == want) { print tolower($1); exit } }' "$2")
	if [ -z "$vc_want" ]; then
		printf '%s is not listed in SHA256SUMS\n' "$3" >&2
		return 1
	fi

	vc_got=$(sha256_of "$1" | tr '[:upper:]' '[:lower:]')
	if [ -z "$vc_got" ]; then
		printf 'no sha256 tool found (sha256sum, shasum, openssl)\n' >&2
		return 1
	fi
	if [ "$vc_got" != "$vc_want" ]; then
		printf 'expected %s\n     got %s\n' "$vc_want" "$vc_got" >&2
		return 1
	fi

	printf '%s' "$vc_got" | cut -c1-12
}

# Signed build provenance, when the GitHub CLI is here to check it. Its absence
# is not an error: this is a stronger check than we can otherwise make, not a
# required one.
verify_provenance() {
	command -v gh >/dev/null 2>&1 || return 1
	gh attestation verify "$1" --repo "$REPO" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Install locations
# ---------------------------------------------------------------------------

on_path() {
	case ":$PATH:" in
	*":$1:"*) return 0 ;;
	*) return 1 ;;
	esac
}

writable_dir() {
	[ -d "$1" ] && [ -w "$1" ]
}

# Path of an atomscan already on PATH, if any.
current_install() {
	command -v "$BIN" 2>/dev/null
}

# installed_version PATH — version string of a binary already on disk.
installed_version() {
	[ -x "$1" ] || return 1
	iv_out=$("$1" --version 2>/dev/null) || return 1
	printf '%s' "$iv_out" | awk '{ print $2; exit }'
}

resolve_install_dir() {
	if [ -n "$OPT_DIR" ]; then
		INSTALL_DIR=$OPT_DIR
		mkdir -p "$INSTALL_DIR" 2>/dev/null || die "cannot create $INSTALL_DIR"
		writable_dir "$INSTALL_DIR" || die "$INSTALL_DIR is not writable
   Re-run with --dir \$HOME/.local/bin, or as root if you meant a system path."
		return 0
	fi

	# Upgrading in place beats installing a second copy that shadows the first.
	# Anything Homebrew owns is left to Homebrew.
	rid_cur=$(current_install || :)
	if [ -n "$rid_cur" ]; then
		rid_dir=$(dirname "$rid_cur")
		rid_brewed=0
		if [ -n "$BREW_PREFIX" ] && [ "${rid_dir#"$BREW_PREFIX"}" != "$rid_dir" ]; then
			rid_brewed=1
		fi
		if [ "$rid_brewed" = 0 ] && writable_dir "$rid_dir"; then
			INSTALL_DIR=$rid_dir
			return 0
		fi
	fi

	# root installs system-wide, which is what /usr/local/bin is for on every
	# Unix here — and what a container or CI image expects. Everyone else gets a
	# user-owned directory, so nothing ever needs sudo.
	if [ "$(id -u)" = 0 ] && writable_dir /usr/local/bin && on_path /usr/local/bin; then
		INSTALL_DIR=/usr/local/bin
		return 0
	fi

	for rid_d in "$HOME/.local/bin" "$HOME/bin" "$HOME/.cargo/bin" /usr/local/bin; do
		if writable_dir "$rid_d" && on_path "$rid_d"; then
			INSTALL_DIR=$rid_d
			return 0
		fi
	done

	INSTALL_DIR="$HOME/.local/bin"
	mkdir -p "$INSTALL_DIR" || die "cannot create $INSTALL_DIR"
}

# install_binary_file SRC — move SRC into place atomically.
#
# Writing beside the destination and renaming means a reader sees either the old
# binary or the new one and never a half-written file, which matters most when
# the thing being replaced is a binary that is currently running.
install_binary_file() {
	ibf_dest="$INSTALL_DIR/$BIN"
	ibf_tmp="$INSTALL_DIR/.$BIN.new.$$"
	cp "$1" "$ibf_tmp" || die "cannot write to $INSTALL_DIR"
	chmod 755 "$ibf_tmp"
	mv -f "$ibf_tmp" "$ibf_dest" || {
		rm -f "$ibf_tmp"
		die "cannot replace $ibf_dest"
	}
	INSTALLED=$ibf_dest
	CHANGED=1
}

# ---------------------------------------------------------------------------
# Method: Homebrew
#
# The native package manager on macOS: it owns upgrades, PATH, and — through the
# cleave formula — rizin and upx. The formula builds from source, which is slow,
# so say so and leave `--method binary` one flag away.
# ---------------------------------------------------------------------------

brew_works() {
	command -v brew >/dev/null 2>&1 && brew --version >/dev/null 2>&1
}

install_brew() {
	step method "Homebrew  ${C_DIM}$TAP/scan${C_RESET}"

	if ! brew tap 2>/dev/null | grep -qx "$TAP"; then
		# The tap's Homebrew handle and its GitHub org differ, so it can only be
		# added with an explicit URL.
		brew tap "$TAP" "$TAP_URL" >/dev/null 2>&1 || return 1
	fi

	br_prefix=$(brew --prefix 2>/dev/null) || return 1
	if brew list --formula "$TAP/scan" >/dev/null 2>&1; then
		if [ "$OPT_FORCE" = 1 ]; then
			note "reinstalling through Homebrew — it builds from source, so this takes a while"
			brew reinstall --formula "$TAP/scan" || return 1
			CHANGED=1
			INSTALLED="$br_prefix/bin/$BIN"
			VERSION=$(installed_version "$INSTALLED" || :)
			return 0
		fi
		if [ -z "$(brew outdated --formula --quiet "$TAP/scan" 2>/dev/null)" ]; then
			INSTALLED="$br_prefix/bin/$BIN"
			VERSION=$(installed_version "$INSTALLED" || :)
			ok "up to date" "$INSTALLED  ${C_DIM}${VERSION}${C_RESET}"
			return 0
		fi
		note "upgrading through Homebrew — it builds from source, so this takes a while"
		brew upgrade --formula "$TAP/scan" || return 1
	else
		note "installing through Homebrew — it builds from source, so this takes a while"
		note "for a prebuilt binary instead, re-run with --method binary"
		brew install --formula "$TAP/scan" || return 1
	fi

	CHANGED=1
	INSTALLED="$br_prefix/bin/$BIN"
	[ -x "$INSTALLED" ] || return 1
	VERSION=$(installed_version "$INSTALLED" || :)
	return 0
}

# ---------------------------------------------------------------------------
# Method: release binary
#
# Returns non-zero when this platform has no published binary — the signal for
# main() to fall back to a source build.
# ---------------------------------------------------------------------------

install_binary() {
	if [ -z "$TARGET" ]; then
		warn "no published binary for this platform"
		return 1
	fi

	if [ -n "$OPT_VERSION" ]; then
		VERSION=$OPT_VERSION
	else
		VERSION=$(resolve_latest || :)
		VERSION=${VERSION#v}
		if [ -z "$VERSION" ]; then
			warn "could not work out the latest release"
			return 1
		fi
	fi
	step version "$VERSION${OPT_VERSION:+  ${C_DIM}(pinned)${C_RESET}}"

	# Idempotence: an install that is already what we would install is done.
	ib_have=$(installed_version "$INSTALL_DIR/$BIN" || :)
	if [ "$OPT_FORCE" = 0 ] && [ "$ib_have" = "$VERSION" ]; then
		INSTALLED="$INSTALL_DIR/$BIN"
		ok "up to date" "$INSTALLED  ${C_DIM}$VERSION${C_RESET}"
		return 0
	fi

	ib_name="$BIN-$VERSION-$TARGET.tar.gz"
	ib_base="https://github.com/$REPO/releases/download/v$VERSION"

	if ! http_ok "$ib_base/$ib_name"; then
		warn "release v$VERSION publishes no binary for $TARGET"
		return 1
	fi

	step method "release binary  ${C_DIM}$TARGET${C_RESET}"
	download "$ib_base/$ib_name" "$TMP/$ib_name" "$ib_name" || {
		warn "download failed"
		return 1
	}

	http_get "$ib_base/SHA256SUMS" "$TMP/SHA256SUMS" ||
		die "release v$VERSION publishes no SHA256SUMS — refusing to install unverified"
	ib_digest=$(verify_checksum "$TMP/$ib_name" "$TMP/SHA256SUMS" "$ib_name") ||
		die "$ib_name failed verification — refusing to install"

	if verify_provenance "$TMP/$ib_name"; then
		ok verified "sha256 $ib_digest  ${C_DIM}·  provenance attested${C_RESET}"
	else
		ok verified "sha256 $ib_digest"
	fi

	# Solaris and illumos tar have no -z; piping gzip works everywhere.
	mkdir -p "$TMP/x"
	gzip -dc "$TMP/$ib_name" | (cd "$TMP/x" && tar -xf -) || die "cannot unpack $ib_name"
	[ -f "$TMP/x/$BIN" ] || die "$ib_name does not contain $BIN"

	install_binary_file "$TMP/x/$BIN"
}

# ---------------------------------------------------------------------------
# Method: source
#
# The fallback for platforms with no published binary, and for anyone who asks
# for it. The checkout stays in the cache directory so a later re-run is an
# incremental rebuild rather than a cold one.
# ---------------------------------------------------------------------------

rust_hint() {
	if [ "$(uname -s)" = Darwin ] && brew_works; then
		printf 'brew install rust'
	else
		printf "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
	fi
}

install_source() {
	step method "source build  ${C_DIM}git + cargo${C_RESET}"

	command -v git >/dev/null 2>&1 || die "a source build needs git"
	command -v cargo >/dev/null 2>&1 || die "a source build needs Rust 1.94 or newer:
   $(rust_hint)"
	if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 &&
		! command -v clang >/dev/null 2>&1; then
		die "a source build needs a C/C++ toolchain (cc, gcc, or clang)"
	fi

	is_ref=main
	if [ -n "$OPT_VERSION" ]; then
		is_ref="v$OPT_VERSION"
	else
		is_latest=$(resolve_latest || :)
		[ -n "$is_latest" ] && is_ref=$is_latest
	fi
	VERSION=${is_ref#v}

	is_have=$(installed_version "$INSTALL_DIR/$BIN" || :)
	if [ "$OPT_FORCE" = 0 ] && [ -n "$is_have" ] && [ "$is_have" = "$VERSION" ]; then
		INSTALLED="$INSTALL_DIR/$BIN"
		ok "up to date" "$INSTALLED  ${C_DIM}$VERSION${C_RESET}"
		return 0
	fi

	is_src="${XDG_CACHE_HOME:-$HOME/.cache}/atomdrift/scan"
	mkdir -p "$(dirname "$is_src")" || die "cannot create $(dirname "$is_src")"

	# The analysis stack is large; the target directory wants about 10 GB.
	is_free=$(df -Pk "$(dirname "$is_src")" 2>/dev/null | awk 'NR == 2 { print int($4 / 1048576) }')
	case $is_free in
	'' | *[!0-9]*) : ;;
	*) [ "$is_free" -lt 10 ] && warn "only ${is_free} GB free at $is_src — a source build wants about 10 GB" ;;
	esac

	if [ -d "$is_src/.git" ]; then
		step source "updating $is_src"
		git -C "$is_src" fetch --quiet --depth 1 origin "$is_ref" || die "cannot fetch $is_ref"
		git -C "$is_src" checkout --quiet --force FETCH_HEAD || die "cannot check out $is_ref"
	else
		step source "cloning $is_ref into $is_src"
		git clone --quiet --depth 1 --branch "$is_ref" "https://github.com/$REPO.git" "$is_src" ||
			die "cannot clone $REPO at $is_ref"
	fi

	step build "cargo build --release  ${C_DIM}(the analysis stack is large — expect minutes)${C_RESET}"
	if command -v make >/dev/null 2>&1 && [ -f "$is_src/Makefile" ]; then
		# `make release` also ad-hoc signs the binary on macOS, which cargo does
		# not, and leaves it somewhere known.
		(cd "$is_src" && make release) || die "build failed in $is_src"
		is_built="$is_src/out/$BIN"
	else
		(cd "$is_src" && cargo build --release --locked --bin "$BIN") || die "build failed in $is_src"
		is_built="$is_src/target/release/$BIN"
	fi
	[ -f "$is_built" ] || die "the build produced no $BIN"

	install_binary_file "$is_built"
	note "source checkout kept at $is_src — delete it to reclaim the space"
}

# ---------------------------------------------------------------------------
# Optional analysis tools
#
# None of these are required: scans work without them, with less depth on some
# file types. Install them when that can be done without asking anyone for a
# password, and print the exact command when it cannot.
# ---------------------------------------------------------------------------

have_tool() {
	for ht_n in $1; do
		command -v "$ht_n" >/dev/null 2>&1 && return 0
	done
	return 1
}

detect_pkg_manager() {
	PM="" PM_INSTALL="" PM_SUDO=""
	if brew_works; then
		PM=brew PM_INSTALL="brew install"
		return 0
	fi
	for pm_c in apt-get dnf pacman zypper apk pkg pkgin pkg_add; do
		command -v "$pm_c" >/dev/null 2>&1 || continue
		case $pm_c in
		apt-get) PM=apt PM_INSTALL="apt-get install -y" ;;
		dnf) PM=dnf PM_INSTALL="dnf install -y" ;;
		pacman) PM=pacman PM_INSTALL="pacman -S --noconfirm --needed" ;;
		zypper) PM=zypper PM_INSTALL="zypper --non-interactive install" ;;
		apk) PM=apk PM_INSTALL="apk add" ;;
		pkg) PM=pkg PM_INSTALL="pkg install -y" ;;
		pkgin) PM=pkgin PM_INSTALL="pkgin -y install" ;;
		pkg_add) PM=pkg_add PM_INSTALL="pkg_add" ;;
		esac
		break
	done
	[ -n "$PM" ] || return 1

	# Escalation we can perform silently, or none at all: prompting for a
	# password from inside a `curl | sh` pipeline is how terminals get wedged.
	if [ "$(id -u)" = 0 ]; then
		PM_SUDO=""
	elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
		PM_SUDO="sudo -n"
	else
		PM_SUDO=none
	fi
	return 0
}

# pkg_name TOOL — the package providing TOOL under the detected manager, empty
# when this manager has no name we trust for it.
pkg_name() {
	case "$PM:$1" in
	brew:rizin | apt:rizin | dnf:rizin | pacman:rizin | pkg:rizin) printf rizin ;;
	brew:upx | dnf:upx | pacman:upx | zypper:upx | apk:upx | pkg:upx | pkgin:upx | pkg_add:upx) printf upx ;;
	apt:upx) printf upx-ucl ;;
	brew:7z) printf sevenzip ;;
	apt:7z) printf p7zip-full ;;
	dnf:7z | pacman:7z | zypper:7z | apk:7z | pkgin:7z | pkg_add:7z) printf p7zip ;;
	pkg:7z) printf 7-zip ;;
	brew:innoextract | apt:innoextract | dnf:innoextract | pacman:innoextract | zypper:innoextract | pkg:innoextract) printf innoextract ;;
	*) : ;;
	esac
}

check_tools() {
	if [ "$OPT_TOOLS" = 0 ]; then
		return 0
	fi
	detect_pkg_manager || :
	ct_report="" ct_missing="" ct_cmds=""

	# tool : binaries that provide it
	for ct_spec in "rizin:rizin radare2 r2" "upx:upx" "7z:7zz 7z" "innoextract:innoextract"; do
		ct_tool=${ct_spec%%:*}
		ct_bins=${ct_spec#*:}

		if have_tool "$ct_bins"; then
			ct_report="$ct_report $C_GREEN$G_OK$C_RESET$ct_tool"
			continue
		fi

		ct_pkg=$(pkg_name "$ct_tool")
		if [ -n "$ct_pkg" ] && [ -n "$PM" ] && [ "$PM_SUDO" != none ]; then
			# One package at a time: a name this distribution happens not to
			# carry must not take the others down with it.
			# shellcheck disable=SC2086
			if $PM_SUDO $PM_INSTALL "$ct_pkg" >"$TMP/pm.log" 2>&1 && have_tool "$ct_bins"; then
				ct_report="$ct_report $C_GREEN$G_OK$C_RESET$ct_tool"
				continue
			fi
		fi

		ct_missing="$ct_missing $ct_tool"
		ct_report="$ct_report $C_DIM-$ct_tool$C_RESET"
		[ -n "$ct_pkg" ] && ct_cmds="$ct_cmds $ct_pkg"
	done

	step tools "${ct_report# }  ${C_DIM}optional${C_RESET}"
	if [ -n "$ct_cmds" ]; then
		ct_sudo=""
		if [ "$PM" != brew ] && [ "$(id -u)" != 0 ]; then
			ct_sudo="sudo "
		fi
		note "for deeper analysis:  $ct_sudo$PM_INSTALL$ct_cmds"
	elif [ -n "$ct_missing" ]; then
		note "optional, not packaged here:$ct_missing"
	fi
}

# ---------------------------------------------------------------------------
# Wrap-up
# ---------------------------------------------------------------------------

path_advice() {
	pa_dir=$(dirname "$INSTALLED")
	on_path "$pa_dir" && return 0

	warn "$pa_dir is not on your PATH"
	case "$(basename "${SHELL:-sh}")" in
	fish) note "fish_add_path $pa_dir" ;;
	zsh) note "echo 'export PATH=\"$pa_dir:\$PATH\"' >> ~/.zshrc" ;;
	bash)
		if [ "$(uname -s)" = Darwin ]; then
			note "echo 'export PATH=\"$pa_dir:\$PATH\"' >> ~/.bash_profile"
		else
			note "echo 'export PATH=\"$pa_dir:\$PATH\"' >> ~/.bashrc"
		fi
		;;
	*) note "export PATH=\"$pa_dir:\$PATH\"" ;;
	esac
}

shadow_check() {
	sc_found=$(current_install || :)
	[ -n "$sc_found" ] || return 0
	[ "$sc_found" != "$INSTALLED" ] || return 0
	on_path "$(dirname "$INSTALLED")" || return 0
	warn "an earlier $BIN on your PATH will still win: $sc_found"
}

summary() {
	if [ "$OPT_QUIET" = 1 ]; then
		return 0
	fi
	sm_v=$(installed_version "$INSTALLED" || printf '%s' "${VERSION:-}")
	printf '\n %s%s%s %s%s %s%s  %s%s%s\n\n' \
		"$C_GREEN" "$G_OK" "$C_RESET" "$C_BOLD" "$BIN" "$sm_v" "$C_RESET" \
		"$C_DIM" "$INSTALLED" "$C_RESET"
	printf '   %sscan a project%s     %s ./project\n' "$C_DIM" "$C_RESET" "$BIN"
	printf '   %sscan a package%s     %s purl npm/left-pad@1.3.0\n' "$C_DIM" "$C_RESET" "$BIN"
	printf '   %severything else%s    %s --help\n' "$C_DIM" "$C_RESET" "$BIN"
	printf '\n   %sThe first scan downloads the model, rule, and bloom-filter bundles.%s\n\n' \
		"$C_DIM" "$C_RESET"
}

# ---------------------------------------------------------------------------

main() {
	# Style first: parse_args can die, and die prints in colour.
	setup_style
	parse_args "$@"
	trap cleanup EXIT
	trap 'cleanup; exit 130' INT
	trap 'cleanup; exit 143' TERM HUP
	make_tmpdir
	probe_sleep
	find_downloader

	banner
	detect_platform
	step platform "$PLATFORM"
	list_targets

	BREW_PREFIX=$(brew --prefix 2>/dev/null || :)

	# Homebrew is the right owner of a macOS install when it is there: it holds
	# upgrades, PATH, and the rizin and upx dependencies. It cannot honour a
	# chosen directory or version, though, so asking for either says plainly
	# that this is not a Homebrew install.
	METHOD=$OPT_METHOD
	if [ "$METHOD" = auto ]; then
		if [ "$(uname -s)" = Darwin ] && brew_works && [ -z "$OPT_DIR" ] && [ -z "$OPT_VERSION" ]; then
			METHOD=brew
		else
			METHOD=binary
		fi
	fi
	if [ "$METHOD" = brew ]; then
		if ! brew_works; then
			warn "Homebrew is not usable here — falling back to a release binary"
			METHOD=binary
		elif [ -n "$OPT_DIR" ] || [ -n "$OPT_VERSION" ]; then
			warn "Homebrew chooses its own directory and version — ignoring --dir and --version"
		fi
	fi

	if [ "$METHOD" = brew ]; then
		install_brew || {
			warn "the Homebrew install did not finish — falling back to a release binary"
			METHOD=binary
		}
	fi

	if [ "$METHOD" = binary ]; then
		resolve_install_dir
		install_binary || {
			note "no binary available — building from source instead"
			METHOD=source
		}
	fi

	if [ "$METHOD" = source ]; then
		[ -n "$INSTALL_DIR" ] || resolve_install_dir
		install_source
	fi

	[ -n "$INSTALLED" ] || die "the install produced no binary"
	"$INSTALLED" --version >/dev/null 2>&1 || die "$INSTALLED does not run on this machine"
	[ "$CHANGED" = 1 ] && ok installed "$INSTALLED"

	check_tools
	path_advice
	shadow_check
	summary
}

main "$@"
