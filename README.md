<p align="center">
  <img src="media/logo.svg" alt="litmus" width="240">
</p>

# litmus

ML-powered malware classifier for supply-chain security. Uses [cleave](https://codeberg.org/atomdrift/cleave) static analysis to extract capabilities, then classifies threat level — built for security engineers and automated pipelines alike.

> **Note:** Alpha software. Expect false positives and false negatives.

## What It Does

- **Scan** files, directories, and archives — evaluates them against a local [ML model](https://codeberg.org/atomdrift/litmus-models/)

## Usage

```bash
# macOS (Homebrew)
brew tap atomdrift/tap https://codeberg.org/atomdrift/homebrew-tap.git
brew install atomdrift/tap/litmus

# From source
make install
```

```bash
litmus scan /path/to/file
```

## License

Apache-2.0
