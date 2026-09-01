#!/usr/bin/env python3
"""Diff two atomscan JSON outputs by detection outcome.

Used by `make check-typed-detection` to prove a cleave performance change did
not alter WHAT is detected, only how fast. Compares, per output record:

  - the record verdict (ml.lvl, ml.prob to 4 places), keyed by the record's
    outer file sha (raw.files[0].sha) so file paths/temp dirs don't matter
  - every inner file's risk, keyed by (record sha, inner sha, inner path) —
    path included because one sha can legitimately appear at several members

Works on both shapes: a loose-tree scan (one record per file) and an archive
scan (one record for the whole archive). Exit 0 = identical detection,
1 = any difference (printed), 2 = usage/parse error.
"""

import json
import sys


def load(path):
    verdicts = {}  # record sha -> (lvl, prob4)
    risks = {}  # (record sha, member sha, member path) -> risk
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            files = (rec.get("raw") or {}).get("files") or []
            if not files:
                continue
            rsha = files[0].get("sha", "")
            ml = rec.get("ml") or {}
            verdicts[rsha] = (ml.get("lvl"), round(ml.get("prob", 0.0), 4))
            for m in files:
                risks[(rsha, m.get("sha", ""), m.get("path", ""))] = m.get("risk")
    return verdicts, risks


def diff_maps(name, a, b, limit=20):
    bad = 0
    for k in a.keys() - b.keys():
        bad += 1
        if bad <= limit:
            print(f"  {name} removed: {k}")
    for k in b.keys() - a.keys():
        bad += 1
        if bad <= limit:
            print(f"  {name} added:   {k}")
    for k in a.keys() & b.keys():
        if a[k] != b[k]:
            bad += 1
            if bad <= limit:
                print(f"  {name} changed: {k}: {a[k]} -> {b[k]}")
    if bad > limit:
        print(f"  ... and {bad - limit} more {name} differences")
    return bad


def is_fetch_derived(path):
    """Nodes pulled over the network by --fetch, not shipped in the corpus.

    Registry provenance sidecars and fetched package payloads. Their presence
    varies run to run (transient registry failures, rate limits — measured
    ~400-node drift between two otherwise identical runs), so a fetch-enabled
    comparison must not hard-fail on them.
    """
    return ".registry.json" in path or path.startswith("pkg:") or "!!pkg:" in path or "∴ pkg:" in path


def main():
    args = [a for a in sys.argv[1:] if a != "--allow-fetch-drift"]
    allow_fetch_drift = len(args) != len(sys.argv) - 1
    if len(args) != 2:
        print(
            f"usage: {sys.argv[0]} [--allow-fetch-drift] baseline.json current.json",
            file=sys.stderr,
        )
        return 2
    try:
        av, ar = load(args[0])
        bv, br = load(args[1])
    except (OSError, json.JSONDecodeError) as e:
        print(f"detect-diff: {e}", file=sys.stderr)
        return 2
    if allow_fetch_drift:
        # Fetch-derived member risks are compared only for drift accounting;
        # corpus members and record verdicts remain the hard gate. (A record
        # verdict CAN legitimately move if fetched content changes it — that is
        # exactly what --fetch is for — so verdict changes are still reported
        # and still fail: a tuning change should alter cost, not conclusions,
        # between two runs made minutes apart.)
        af = {k: v for k, v in ar.items() if is_fetch_derived(k[2])}
        bf = {k: v for k, v in br.items() if is_fetch_derived(k[2])}
        ar = {k: v for k, v in ar.items() if k not in af}
        br = {k: v for k, v in br.items() if k not in bf}
        drift_gone = len(af.keys() - bf.keys())
        drift_new = len(bf.keys() - af.keys())
        drift_changed = sum(1 for k in af.keys() & bf.keys() if af[k] != bf[k])
        print(
            f"fetch drift (non-fatal): {drift_gone} gone, {drift_new} new, "
            f"{drift_changed} changed of {len(af)}/{len(bf)} fetch nodes"
        )
    bad = diff_maps("verdict", av, bv) + diff_maps("risk", ar, br)
    if bad:
        print(f"DETECTION CHANGED: {bad} difference(s)")
        return 1
    print(f"detection identical: {len(av)} verdicts, {len(ar)} member risks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
