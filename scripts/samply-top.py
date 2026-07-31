#!/usr/bin/env python3
"""Print a text summary of Samply's Firefox-profile JSON.

Samply deliberately leaves native PCs unresolved in saved profiles.  This
helper resolves addresses against the profiled binary with addr2line, then
aggregates sampled leaf and inclusive function counts.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import subprocess
from pathlib import Path


def load_profile(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as profile:
        return json.load(profile)


def thread_chains(thread: dict) -> list[list[str]]:
    strings = thread["stringArray"]
    funcs = thread["funcTable"]
    frames = thread["frameTable"]
    stacks = thread["stackTable"]
    names = [
        strings[funcs["name"][frames["func"][frame]]]
        for frame in stacks["frame"]
    ]

    chains: list[list[str]] = []
    for stack in thread["samples"]["stack"]:
        chain: list[str] = []
        while stack is not None:
            chain.append(names[stack])
            stack = stacks["prefix"][stack]
        chains.append(chain)
    return chains


def resolve(binary: Path, addresses: set[str]) -> dict[str, str]:
    ordered = sorted(addresses, key=lambda address: int(address, 16))
    if not ordered:
        return {}
    process = subprocess.run(
        ["addr2line", "-Cfp", "-e", str(binary)],
        input="\n".join(ordered) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    lines = process.stdout.splitlines()
    if len(lines) != len(ordered):
        raise RuntimeError(
            f"addr2line returned {len(lines)} lines for {len(ordered)} addresses"
        )
    return dict(zip(ordered, lines, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--process", default="atomscan.bench")
    parser.add_argument("--minimum-samples", type=int, default=1)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    chains: list[list[str]] = []
    thread_counts = collections.Counter()
    for thread in profile["threads"]:
        count = thread.get("samples", {}).get("length", 0)
        if thread.get("processName") != args.process or count < args.minimum_samples:
            continue
        thread_counts[thread["name"]] += count
        chains.extend(thread_chains(thread))

    addresses = {
        name
        for chain in chains
        for name in chain
        if name.startswith("0x")
    }
    symbols = resolve(args.binary, addresses)

    leaf = collections.Counter()
    inclusive = collections.Counter()
    for chain in chains:
        resolved = [symbols.get(name, name) for name in chain]
        if resolved:
            leaf[resolved[0]] += 1
        inclusive.update(set(resolved))

    print(f"samples: {len(chains):,}; resolved addresses: {len(symbols):,}")
    print("threads:")
    for name, count in thread_counts.most_common():
        print(f"{count:9,d}  {name}")

    for heading, counts in (("leaf", leaf), ("inclusive", inclusive)):
        print(f"\n{heading}:")
        for name, count in counts.most_common(args.top):
            percentage = count * 100 / len(chains)
            print(f"{count:9,d} {percentage:6.2f}%  {name}")


if __name__ == "__main__":
    main()
