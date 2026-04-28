<p align="center">
  <img src="media/logo.svg" alt="litmus" width="240">
</p>

# litmus

> **Status: beta.** Litmus is usable today and we run it ourselves, but the CLI surface, JSON schema, and default thresholds may still change before 1.0.

Point litmus at a package, directory, or running process and it tells you whether it's `hostile`, `suspicious`, or `benign` — and exactly which behaviors drove that call. Litmus is the reference scanner for [azoth](https://codeberg.org/atomdrift/azoth), the first open-source AI model for general malware detection across binaries, source, and packaged formats. No cloud calls, no opaque signatures, no telemetry.

<p align="center">
  <img src="media/screenshot.png" alt="litmus terminal output" width="760">
</p>

## What It Does

Litmus runs each sample through [cleave](https://codeberg.org/atomdrift/cleave), which catalogs 50,000+ known malicious behaviors against [MBC](https://github.com/MBCProject/mbc-markdown) and [MITRE ATT&CK](https://attack.mitre.org/). It feeds those capabilities to the [azoth](https://codeberg.org/atomdrift/azoth) model and returns a score. Every flagged file comes with a ranked list of the capabilities that pushed it over the line — so when litmus flags something, you know *why*, not just *what*.

Azoth runs on CPU. No GPU, no accelerator, no per-call charge. Same weights on a laptop, a CI runner, or a fleet of endpoints.

One binary, four modes: an interactive CLI, a JSON emitter for CI, an HTTP service, and a pull worker for distributed fleets. Designed up front to be useful standalone *and* as a component embedded in other open-source or commercial software — Apache-2.0, no telemetry, stable JSON contract.

## Quick Start

Install with Homebrew (macOS/Linux):

```bash
brew tap atomdrift/tap https://codeberg.org/atomdrift/homebrew-tap.git
brew install litmus
```

Or build from source:

```bash
git clone https://codeberg.org/atomdrift/litmus.git
cd litmus && make install
```

Then run:

```bash
litmus suspect.tgz                            # single sample
litmus /srv/npm-mirror                        # recursive; archives unpacked
litmus --format json --show all pkg/          # pipeline-friendly output
litmus ps                                     # classify every running process
litmus -1 release.tgz                         # loose: zero-FP target
litmus -9 release.tgz                         # paranoid: catch the long tail
```

**Sensitivity is gzip-style.** `-1` through `-9` shift the verdict thresholds; `-5` is the default. `-1` (`--loose`) targets zero false positives, `-9` (`--paranoid`) catches the long tail at the cost of noise. For exact cutoffs use `--threshold-hostile` / `--threshold-suspicious`.

Exit codes: `0` clean, `1` hostile, `2` suspicious, `3` error. Wire them straight into CI.

Optional: [rizin](https://github.com/rizinorg/rizin) for disassembly, [upx](https://github.com/upx/upx) for runtime unpacking.

## Design

- **Tunable sensitivity.** `-1` through `-9` (gzip-style) shift the verdict thresholds; `-5` is the default. `--threshold-hostile` / `--threshold-suspicious` set exact cutoffs. `--model-dir` swaps in a custom model.
- **Explained verdicts.** Every flagged file ships with the capabilities that drove the score, computed via TreeSHAP on the live model — not a post-hoc story.
- **Offline.** Models and rules live in plain git repos. `--update` pulls new versions on demand. No license servers, no sample upload, no telemetry.
- **Streaming output.** JSONL per-file verdicts with the full cleave report attached. Progress writes to stderr.
- **Fleet-ready.** `litmus serve` exposes an HTTP API with loopback-default binding, CIDR allowlists, bounded concurrency, and an RSS ceiling that rejects requests before the box starts swapping. `litmus worker` pulls jobs from a [hopper](https://codeberg.org/atomdrift/hopper) queue with SHA256-verified local paths.

## Related

- [azoth](https://codeberg.org/atomdrift/azoth) — the model litmus runs (Apache-2.0 weights, schema, and training pipeline)
- [cleave](https://codeberg.org/atomdrift/cleave) — the capability analyzer litmus is built on
- [azoth-trainer](https://codeberg.org/atomdrift/azoth-trainer) — XGBoost training pipeline that produces the azoth model
- [hopper](https://codeberg.org/atomdrift/hopper) — work queue for distributed scanning fleets
- [xgboost-ars](https://codeberg.org/atomdrift/xgboost-ars) — pure-Rust inference with exact TreeSHAP
- [Atomdrift Lab](https://lab.atomdrift.org/) — submit samples for free analysis

## License

Apache-2.0
