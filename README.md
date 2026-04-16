<p align="center">
  <img src="media/logo.svg" alt="litmus" width="240">
</p>

# litmus

Malware classifier for the modern software supply chain. litmus turns [cleave](https://codeberg.org/atomdrift/cleave)'s capability analysis into a verdict — `hostile`, `suspicious`, or `benign` — using a local gradient-boosted model trained on millions of real packages. No cloud calls, no opaque signatures, no vendor lock-in.

## What It Does

litmus scans files, directories, archives, and running processes. Every sample passes through cleave's AST-aware decomposition, which emits a vector of behaviors drawn from 47,000+ rules aligned to [MBC](https://github.com/MBCProject/mbc-markdown) and [ATT&CK](https://attack.mitre.org/). That vector feeds the classifier, which returns a probability and a SHAP-ranked list of the capabilities that moved the score. You see not just *what* litmus decided, but *why*.

The same binary runs four ways. At the desk it's a CLI. In a pipeline it emits JSON and exits with a code your CI can branch on. Under load it runs as an HTTP service. Across a fleet it runs as a pull worker against a [hopper](https://codeberg.org/atomdrift/hopper) queue — handing the same model to a laptop, a build agent, or a jail full of FreeBSD boxes analyzing a million packages a day.

## Quick Start

```bash
brew install atomdrift/tap/litmus            # macOS/Linux — first run: brew tap atomdrift/tap https://codeberg.org/atomdrift/homebrew-tap.git
make install                                  # from source

litmus suspect.tgz                            # single sample
litmus /srv/npm-mirror                        # recursive; archives unpacked
litmus --format json --show all pkg/          # pipeline-friendly output
litmus ps                                     # classify every running process
```

Exit codes are deliberate: `0` clean, `1` hostile present, `2` suspicious present, `3` analysis error. Wire them into CI directly.

Optional: [rizin](https://github.com/rizinorg/rizin) for binary disassembly, [upx](https://github.com/upx/upx) for runtime unpacking. cleave drives both when present.

## Why Security Engineers Use It

**Dial in your paranoia.** The model ships with thresholds chosen from a held-out evaluation set, but a SOC triaging npm publishes and an IR team sweeping for post-compromise artifacts have different tolerances for noise. `--threshold-hostile` and `--threshold-suspicious` move the goalposts at runtime. Bring your own model with `--model-dir` when the stock weights don't fit your threat model.

**Explanations by default.** Every flagged file comes with the top capabilities that drove its score — credential theft, anti-debug, network exfiltration, whatever the sample actually did. This is SHAP against the production model, not a post-hoc justification. When a verdict is wrong you know immediately which feature to investigate.

**Built for air-gapped defenders.** The model and trait repository are both plain git: fetch once, run forever. `litmus --update` pulls new versions when you want them. No telemetry, no license servers, no sample upload.

**Pipeline-native.** JSONL output streams per-file verdicts with the full cleave report attached. Terminal output is concise by default — benign files are silent, hostile and suspicious render with context — and configurable with `--show`. Progress bars live on stderr so they never contaminate a pipe.

**Scales past the laptop.** `litmus serve` exposes a classification HTTP API with loopback-by-default binding, CIDR allowlists for remote access, bounded concurrency, and an RSS ceiling that rejects requests before the machine starts swapping. `litmus worker` connects to a hopper queue and pulls jobs; SHA256-verified local paths avoid re-downloading when the worker shares storage with the orchestrator.

**Honest about cost.** cleave does real work — disassembly, unpacking, YARA-X, AST walks — and litmus inherits that budget. In return you get a verdict backed by observed behavior, not a hash lookup that goes stale the moment an attacker recompiles.

## How It Works

1. **Extract** — cleave analyzes the sample against its rule corpus, emitting a report of matched traits and structural signals.
2. **Featurize** — litmus compresses that report into a fixed-length numeric vector defined by [feature_spec.json](https://codeberg.org/atomdrift/litmus-models/). The spec is versioned; model and features move together.
3. **Classify** — the vector is scored by an [XGBoost model](https://codeberg.org/atomdrift/litmus-models/) via [xgboost-native](https://codeberg.org/atomdrift/xgboost-native), with exact TreeSHAP producing per-feature attributions.
4. **Arbitrate** — an optional heuristic layer upgrades verdicts when cleave reports capabilities that obviously disagree with a benign score (`--upgrade-heuristic=false` disables it for raw model output).

## Related

- [cleave](https://codeberg.org/atomdrift/cleave) — the capability analyzer litmus is built on
- [hopper](https://codeberg.org/atomdrift/hopper) — work queue for distributed scanning fleets
- [xgboost-native](https://codeberg.org/atomdrift/xgboost-native) — pure-Rust inference with exact TreeSHAP
- [Atomdrift Lab](https://lab.atomdrift.org/) — submit samples for free analysis

## License

Apache-2.0
