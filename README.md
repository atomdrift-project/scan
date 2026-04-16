<p align="center">
  <img src="media/logo.svg" alt="litmus" width="240">
</p>

# litmus

Malware classifier for the modern software supply chain. litmus turns [cleave](https://codeberg.org/atomdrift/cleave)'s capability analysis into a verdict — `hostile`, `suspicious`, or `benign` — using a local XGBoost model trained on millions of real packages. No cloud calls, no opaque signatures, no telemetry.

## What It Does

Every sample passes through cleave's AST-aware decomposition, which emits a vector of behaviors drawn from 47,000+ rules aligned to [MBC](https://github.com/MBCProject/mbc-markdown) and [ATT&CK](https://attack.mitre.org/). That vector feeds the classifier, which returns a probability and an exact TreeSHAP ranking of the capabilities that moved the score. You see not just *what* litmus decided, but *why*.

The same binary runs four ways: a CLI at the desk, a JSON emitter in CI, an HTTP service under load, and a pull worker for fleets — handing the same model to a laptop, a build agent, or a jail analyzing a million packages a day.

## Quick Start

```bash
brew install atomdrift/tap/litmus            # macOS/Linux — first run: brew tap atomdrift/tap https://codeberg.org/atomdrift/homebrew-tap.git
make install                                  # from source

litmus suspect.tgz                            # single sample
litmus /srv/npm-mirror                        # recursive; archives unpacked
litmus --format json --show all pkg/          # pipeline-friendly output
litmus ps                                     # classify every running process
```

Exit codes: `0` clean, `1` hostile, `2` suspicious, `3` error. Wire them straight into CI.

Optional: [rizin](https://github.com/rizinorg/rizin) for disassembly, [upx](https://github.com/upx/upx) for runtime unpacking.

## Why Security Engineers Use It

- **Tunable paranoia** — `--threshold-hostile` and `--threshold-suspicious` move the goalposts at runtime; `--model-dir` swaps in your own weights when the stock model doesn't match your threat model.
- **Explanations by default** — every flagged file ships with the capabilities that drove its score, computed via exact TreeSHAP against the production model. No post-hoc justification.
- **Air-gap ready** — the model and trait repositories are plain git: fetch once, run forever. `--update` pulls new versions when you want them. No license servers, no sample upload.
- **Pipeline-native** — JSONL output streams per-file verdicts with the full cleave report attached; progress lives on stderr so it never contaminates a pipe.
- **Fleet-scale** — `litmus serve` exposes an HTTP API with loopback-default binding, CIDR allowlists, bounded concurrency, and an RSS ceiling that rejects requests before the box swaps. `litmus worker` pulls jobs from a [hopper](https://codeberg.org/atomdrift/hopper) queue with SHA256-verified local paths.

## Related

- [cleave](https://codeberg.org/atomdrift/cleave) — the capability analyzer litmus is built on
- [hopper](https://codeberg.org/atomdrift/hopper) — work queue for distributed scanning fleets
- [xgboost-native](https://codeberg.org/atomdrift/xgboost-native) — pure-Rust inference with exact TreeSHAP
- [Atomdrift Lab](https://lab.atomdrift.org/) — submit samples for free analysis

## License

Apache-2.0
