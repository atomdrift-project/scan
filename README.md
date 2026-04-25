<p align="center">
  <img src="media/logo.svg" alt="litmus" width="240">
</p>

# litmus

Point litmus at a package, directory, or running process and it tells you whether it's `hostile`, `suspicious`, or `benign` — and exactly which behaviors drove that call. A local XGBoost model trained on millions of real packages runs the classification. No cloud calls, no opaque signatures, no telemetry.

## What It Does

Litmus looks for 47,000+ known malicious behaviors, cataloged against [MBC](https://github.com/MBCProject/mbc-markdown) and [MITRE ATT&CK](https://attack.mitre.org/), by running each sample through [cleave](https://codeberg.org/atomdrift/cleave). The model weighs what it finds and returns a score. Every flagged file comes with a ranked list of the capabilities that pushed it over the line — so when litmus flags something, you know *why*, not just *what*.

One binary, four modes: an interactive CLI, a JSON emitter for CI, an HTTP service, and a pull worker for distributed fleets. The same model runs on a laptop or a jail classifying a million packages a day.

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
```

Exit codes: `0` clean, `1` hostile, `2` suspicious, `3` error. Wire them straight into CI.

Optional: [rizin](https://github.com/rizinorg/rizin) for disassembly, [upx](https://github.com/upx/upx) for runtime unpacking.

## Why Security Engineers Use It

- **Tune the sensitivity** — `--threshold-hostile` and `--threshold-suspicious` shift the verdict cutoffs at runtime. `--model-dir` swaps in your own model when the stock one doesn't match your threat profile.
- **Every verdict is explained** — flagged files come with the exact capabilities that drove the score, computed via TreeSHAP on the live model. Not a post-hoc story.
- **Works offline** — models and rules live in plain git repos: clone once, run forever. `--update` pulls new versions when you want them. No license servers, no sample upload.
- **Built for pipelines** — JSONL output streams per-file verdicts with the full cleave report attached. Progress writes to stderr, so it stays out of your pipe.
- **Scales to fleets** — `litmus serve` exposes an HTTP API with loopback-default binding, CIDR allowlists, bounded concurrency, and an RSS ceiling that rejects requests before the box starts swapping. `litmus worker` pulls jobs from a [hopper](https://codeberg.org/atomdrift/hopper) queue with SHA256-verified local paths.

## Related

- [cleave](https://codeberg.org/atomdrift/cleave) — the capability analyzer litmus is built on
- [hopper](https://codeberg.org/atomdrift/hopper) — work queue for distributed scanning fleets
- [xgboost-ars](https://codeberg.org/atomdrift/xgboost-ars) — pure-Rust inference with exact TreeSHAP
- [Atomdrift Lab](https://lab.atomdrift.org/) — submit samples for free analysis

## License

Apache-2.0
