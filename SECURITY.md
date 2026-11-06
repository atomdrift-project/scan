# Security policy

Atomdrift Scan reads hostile input for a living, so we treat a bug that lets
that input misbehave as a security issue.

## Reporting a vulnerability

Open a [public issue](https://github.com/atomdrift-project/scan/issues/new)
with "security" in the title. Include what you can of:

- the affected version (`atomscan version`) and platform;
- the input, command line or request that triggers it. Name a sample by its
  SHA-256 and where it can be found; never attach live malware to an issue;
- what happens, and what you expected instead.

## Scope

In scope:

- **The scanner** — anything a scanned file, archive, URL, package or
  process can make `atomscan` do beyond being judged: code execution, memory
  corruption, path traversal or writes outside temporary directories, hangs or
  resource exhaustion out of proportion to the input, or a crafted input that
  hides itself by crashing or disabling analysis.
- **`atomscan serve`** — authentication and access-control bypasses
  (`--token-file`, `--allow-cidr`, `--allow-local-paths`), reading files
  outside the allowed directories, and denial of service past the documented
  limits.
- **The update and bundle path** — anything that lets a party other than the
  project get rules, models or filters installed by `atomscan update-rules`
  or the automatic refresh, or makes a tampered or downgraded bundle load.
- **Release artifacts and CI** — a way to alter what the release workflows
  build, sign or publish.

Out of scope: detection quality. A missed sample or a false positive is a
bug, not a vulnerability; report it as a regular issue.

## Supported versions

Only the [latest release](https://github.com/atomdrift-project/scan/releases/latest)
receives security fixes. Please check that a problem reproduces there before
reporting it.

## What to expect

We will reply on the issue, fix it in the open, and name the release that
contains the fix there. Say in the issue how you would like to be credited.
