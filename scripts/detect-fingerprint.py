#!/usr/bin/env python3
"""Emit deterministic level decisions for server-response parity checks."""

from __future__ import annotations

import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RESPONSE_DIR", file=sys.stderr)
        return 2

    response_dir = pathlib.Path(sys.argv[1])
    fingerprint = []
    for path in sorted(response_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as source:
            response = json.load(source)
        ml = response.get("ml")
        if not isinstance(ml, dict):
            # Worker result dumps carry an `error` instead of `ml` when the
            # analysis failed; that outcome is part of the fingerprint.
            error = response.get("error")
            if error is None:
                raise ValueError(f"{path}: response has no ml object")
            fingerprint.append(
                {"file": path.name.removesuffix(".json"), "error": error}
            )
            continue
        fingerprint.append(
            {
                "file": path.name.removesuffix(".json"),
                "lvl": ml.get("lvl"),
                "files": [
                    {
                        "id": member.get("id"),
                        "type": member.get("type"),
                        # Missing means the member was not evaluated. Preserve
                        # that distinction from a manual-threshold null.
                        **({"lvl": member["lvl"]} if "lvl" in member else {}),
                    }
                    for member in ml.get("files", [])
                ],
            }
        )

    json.dump(fingerprint, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
