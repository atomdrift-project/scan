#!/usr/bin/env python3
"""Compare filefacts-consumed fields from full and compact Rizin output.

Usage: python3 scripts/rizin/function-projection.py RIZIN_BINARY INPUT_BINARY
The input is statically analyzed, never executed.
"""

import json
import os
import subprocess
import sys


def functions(compact):
    env = dict(os.environ)
    env.pop("RZ_FACTS_JSON", None)
    if compact:
        env["RZ_FACTS_JSON"] = "1"
    result = subprocess.run(
        [sys.argv[1], "-NN", "-q", "-T", "-z", "-e", "scr.color=0",
         "-e", "log.level=4", "-e", "analysis.vars=false",
         "-c", "aa; aac; aflj", sys.argv[2]],
        env=env, check=True, capture_output=True, timeout=60,
    )
    return json.loads(result.stdout), len(result.stdout)


def project(function):
    # Mirrors the wire contract of filefacts::rizin::{RawFunction, RawCallref},
    # including serde defaults. Reference order has no semantic significance;
    # retain multiplicity while comparing independently of traversal order.
    fields = {key: function.get(key, default) for key, default in
              [("name", None), ("offset", 0), ("size", 0), ("cc", None), ("nbbs", None)]}
    fields["callrefs"] = sorted(
        json.dumps({key: ref.get(key) for key in ("name", "from", "to", "type")},
                   sort_keys=True)
        for ref in function.get("callrefs", [])
    )
    return fields


full, full_bytes = functions(False)
compact, compact_bytes = functions(True)
assert full, "The fixture must produce a nonempty function table"
assert any(f.get("callrefs") for f in full), "The fixture must exercise call references"
assert {f["offset"]: project(f) for f in full} == {
    f["offset"]: project(f) for f in compact
}, "Compact output changed consumed function facts"
assert compact_bytes < full_bytes, "The compact mode did not reduce output"
print(f"Rizin function projection passed: {len(full)} functions, "
      f"{full_bytes} -> {compact_bytes} bytes")
