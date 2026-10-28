#!/bin/sh
# Refuse a dependency graph that only builds on this machine.
#
# A local override has three places to hide: a `path =` dependency in
# Cargo.toml, a `paths` or `[patch]` entry in .cargo/config.toml, and — after a
# build with `--config patch.toml` — a Cargo.lock package that lost its
# `source` line. Any of them produces a release nobody else can reproduce.
# Run by CI's guard job and by scripts/pre-commit.

set -eu

root=$(git rev-parse --show-toplevel)
cd "$root"
status=0

# An uncommented `path = "../x"`, `"./x"` or `"/x"` in Cargo.toml, inline or in
# a table. Target paths (`path = "src/lib.rs"`) are relative to the crate and
# pass; commented-out patch recipes are skipped.
if grep -nE '^[^#]*\bpath[[:space:]]*=[[:space:]]*"[./~]' Cargo.toml; then
	echo "check-lock: Cargo.toml has a local path dependency (shown above)" >&2
	status=1
fi

if [ -f .cargo/config.toml ] &&
	grep -nE '^[[:space:]]*(paths[[:space:]]*=|\[patch)' .cargo/config.toml; then
	echo "check-lock: .cargo/config.toml overrides a dependency (shown above)" >&2
	status=1
fi

# Every locked package except this crate must name where it came from.
unsourced=$(awk '
	/^\[\[package\]\]/ { if (name != "" && !src) print name; name = ""; src = 0 }
	/^name = /         { name = $3 }
	/^source = /       { src = 1 }
	END                { if (name != "" && !src) print name }
' Cargo.lock | grep -vx '"atomdrift-scan"' || true)
if [ -n "$unsourced" ]; then
	echo "check-lock: Cargo.lock resolves these packages from local paths:" >&2
	echo "$unsourced" >&2
	status=1
fi

exit "$status"
