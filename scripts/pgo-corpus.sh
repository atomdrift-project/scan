#!/bin/sh
# Fetch the PGO training corpus: the testdata of atomdrift-project/traits at
# the commit pinned in scripts/pgo-traits.rev. That testdata is real malware;
# it is only ever read by the scanner, never executed.
#
# This is the one definition of the corpus. `make pgo-corpus` runs it for
# local builds and .github/workflows/build.yml runs it in the PGO training
# job, so both train on the same bytes. Bump the pin by writing a new commit
# SHA into scripts/pgo-traits.rev.
#
# usage: scripts/pgo-corpus.sh DEST
#
#   DEST   directory to check the corpus out into. Anything already there is
#          deleted first, and a failed fetch leaves nothing behind, so a
#          partial checkout is never mistaken for a corpus.
#
# Environment: PGO_TRAITS_URL overrides the repository URL (a mirror).
set -eu

[ $# -eq 1 ] || {
	echo "usage: $0 DEST" >&2
	exit 2
}
dest=$1
url=${PGO_TRAITS_URL:-https://github.com/atomdrift-project/traits}

rev=$(tr -d '[:space:]' <"$(dirname "$0")/pgo-traits.rev")
case "$rev" in
*[!0-9a-f]*) rev="" ;;
esac
[ "${#rev}" -eq 40 ] || {
	echo "error: scripts/pgo-traits.rev must hold one full lowercase commit SHA" >&2
	exit 1
}

rm -rf "$dest"
trap 'rm -rf "$dest"' EXIT
echo "pgo-corpus: fetching $url at $rev into $dest"
git init --quiet "$dest"
# Samples are byte-exact evidence: no line-ending rewrites on checkout.
git -C "$dest" config core.autocrlf false
git -C "$dest" fetch --quiet --depth 1 "$url" "$rev"
git -C "$dest" -c advice.detachedHead=false checkout --quiet FETCH_HEAD
got=$(git -C "$dest" rev-parse HEAD)
[ "$got" = "$rev" ] || {
	echo "error: fetched $got, expected $rev" >&2
	exit 1
}
[ -d "$dest/testdata" ] || {
	echo "error: $url at $rev has no testdata/" >&2
	exit 1
}
trap - EXIT
echo "pgo-corpus: training corpus at $dest/testdata"
