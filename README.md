# litmus

**Experimental ML-powered malware classifier using [cleave](https://github.com/chainguard-dev/cleave) static analysis.**

> **Warning:** This is alpha software. It will produce false positives and false negatives. Do not use for production security decisions. May kill kittens.

## Features

- Train and consume custom ML models tailored to your threat model
- Scan files, directories, and archives for malicious indicators
- Compare package versions to detect supply chain attacks

## Usage

```bash
# Build (requires cleave at ../cleave)
cargo build --release

# Scan a file
./target/release/litmus scan /path/to/file

# Compare package versions
./target/release/litmus diff old.tar.gz new.tar.gz
```

## Training

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) to train your own model.

## License

Apache 2.0
