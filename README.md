# Atomdrift Scan

Atomdrift Scan is an embeddable open-source malware scanner, tuned for open-source ecosystems, designed to catch supply-chain attacks no matter the file format: ELF, Ruby, Python, Shell, PHP, C, Go, PE, whatever (we process 95+ different filetypes!)

AS is designed to replace proprietary scanners such as socket.dev, ReversingLabs, and Aikido, as well as legacy open-source
scanners such as ClamAV and malcontent. To fight fire with fire, AS is based on efficient and deterministic local AI models - designed to run on any hardware or operating system. Unlike most scanners that rely on pattern mattching, it dissects each file - through active reverse engineering using tools like rizin and tree-sitter. Atomdrift decomposes each file into a tree of atoms, and from there looks for "malecule" shapes to derive probability.

Unlike most scanners, AS allows you to set your acceptable false-positive level, based on predicted occurrence over 100 million samples based on the specific filetype involved. Paranoid about your CI pipeline? use `atomscan -l5000 <files>`; don't want to bombard the SOC with alerts? use `atomscan -l0` (the default is L50, or 50-per-100 million).

We're Apache 2.0 licensed, and in active development on the following architectures - chances are if you are on something else, it should still work (if not, PR's welcome).

* Linux [x86-64]
* macOS [arm64]
* FreeBSD [arm64, x86-64]
* OpenBSD [x86-64]
* OmniOS/illumos [x86-64]

Atomdrift's core values are: privacy-first, local-first, efficiency, and transparency.

<p align="center">
  <img src="media/screenshot.png" alt="Atomdrift Scan terminal output" width="760">
</p>

## How it works

Atomdrift Scan is a multi-stage analyzer bringing together the best that open-source has to offer for reverse-engineering
binaries and source code. 

AS is able to cover as much ground as it does by expressing the AI model in terms of a YAML-based expert system with over 75,000 rules, analyzing using a large ensemble of LightGBM ML models. AS also supports the use of local GPU-based analysis via OpenAI-compatible endpoints [vLLM, for example] for additional accuracy and interpretation, but that's entirely optional.

```mermaid
flowchart LR
    IN([file · dir · process]) --> CLEAVE

    subgraph CLEAVE[cleave — capability extraction]
        direction TB
        UPX[upx<br/>unpack]
        TS[tree-sitter<br/>parse scripts]
        YARA[YARA<br/>pattern match]
        RIZIN[rizin<br/>disassemble]
    end

    CLEAVE -->|AnalysisReport<br/>75k rules → MBC + ATT&CK| FF[filefacts<br/>feature extraction]
    FF -->|standardized<br/>feature vector| SCAN[scan<br/>ONNX inference]
    AZOTH[(azoth<br/>GBT ensemble)] -.loads.-> SCAN
    SCAN -->|Decision + SHAP reasons| OUT{{verdict<br/>hostile · suspicious · benign}}
    SCAN -.->|prob ≥ gate| INTERPRET[--interpret<br/>local LLM blend]
    INTERPRET -.-> OUT

    click CLEAVE "https://atomdrift.org/cleave" _blank
    click FF "https://atomdrift.org/filefacts" _blank
    click AZOTH "https://atomdrift.org/azoth" _blank
    click SCAN "https://atomdrift.org/scan" _blank
    click UPX "https://upx.github.io/" _blank
    click TS "https://tree-sitter.github.io/tree-sitter/" _blank
    click YARA "https://virustotal.github.io/yara-x/" _blank
    click RIZIN "https://rizin.re/" _blank
```

## Dependencies

* Rust 1.96 or higher
* [upx](https://upx.github.io/) [optional, recommended for binary analysis]
* [rizin](https://rizin.re/) [optional, recommended for binary reverse-analysis]
* [innoextract](https://github.com/dscharrer/innoextract) [optional, recommended for PE archive analysis]

## Install

For Linux and macOS users using Homebrew:

```bash
brew tap atomdrift/tap https://codeberg.org/atomdrift/homebrew-tap.git
brew install atomdrift-scan
```

For everyone else, source compiles are trivial:

```bash
git clone https://codeberg.org/atomdrift/scan.git
cd scan
make install
```

`make install` builds and installs the CLI as **`atomscan`** — a unique,
collision-free command name (avast ships its own `/usr/bin/scan`, so `scan`
alone isn't safe as a global command). All examples below use `atomscan`. Run
`make uninstall` to remove it; that also cleans up any stale `scan`/`ascan`
symlinks left by older installs.

## Usage

```bash
atomscan path /bin/                     # recursive; archives unpacked
atomscan ps                             # classify running processes
```

By default, atomscan is tuned for 50 false-positives per 100-million files, tune it for your use case using -l <X-per-100M>. To be ultra conservative and avoid any likelihood of false-positive, use:

```bash
atomscan -l 0 /sbin/sulogin            # 0-fp scan against a file
```

If you want a second opinion for added accuracy, AS recently added support for efficiently sending evidence to a local LLM for analysis using `--interpret`. It defaults to the local OpenAI endpoint at https://127.0.0.1:8000/v1 - we currently recommend vLLM with Qwen 3.6-27B as a model. Models down to 9B are likely sufficient as well. LLM scores are blended against the ML scores for a final adjusted outcome.

If you only care about rules for a particular platform, say macos or JunOS; use `--platform` to mask everything else out. 
Nothing is more annoying than seeing a Windows-specific alert on your ArchLinux CI pipeline.

## What AS groks.

**95 file types** across binaries, source, packages, and documents — **25 platforms** from desktop OSes to network appliances. 

### File types

| Category | Formats |
|----------|---------|
| **Binaries** | Mach-O, ELF, PE, Java `.class`, Python `.pyc`, BEAM |
| **Source** | Python, JavaScript, TypeScript, Go, Rust, Java, C, C#, Ruby, PHP, Perl, Lua, Swift, Objective-C, Kotlin, Scala, Groovy, Zig, Elixir, Clojure, Shell, PowerShell, Batch, VBScript, AppleScript |
| **Build & config** | package.json, Cargo.toml, pyproject.toml, composer.json, binding.gyp, GitHub Actions, systemd units, `.desktop`, Makefile, Dockerfile, JSON, XML |
| **Archives** | ZIP, TAR (gz/bz2/xz/zst), 7-Zip, RAR, CAB, ASAR, gzip/bzip2/xz/zstd |
| **Packages** | deb, rpm, APK, npm, wheel, egg, gem, crate, conda, NuGet, `.ipa`, `.crx`, `.xpi`, `.vsix`, macOS/FreeBS/Arch `.pkg` |
| **Documents** | OLE2, OOXML, OpenDocument, PDF, RTF, Markdown, HTML, plist |
| **Images & other** | JPEG, PNG, `.lnk`, CHM, Python pickle |

Binaries are automatically reverse-engineered using Rizin, Source automatically reverse-engineered using Treesitter.

### Platforms

We currently have specific rules for detecting malware for the following operating systems:

`Linux` · `macOS` · `Windows` · `Android` · `iOS` · `FreeBSD` · `OpenBSD` · `NetBSD` · `DragonFly BSD` · `AIX` · `Solaris` · `QNX` · `z/OS` · `ESXi` · `OpenWrt` · `VxWorks` · `RouterOS` · `FortiOS` · `PAN-OS` · `IOS-XE` · `Junos` · `NetScaler` · `Ivanti`

Yes, we got creative with generating synthetic malware just to prove coverage on the more exotic language/platform combinations, just in case.

## Documentation

- [Integration guide](docs/INTEGRATION.md) — pick CLI, server, or worker for your volume
- [JSON report schema](docs/JSON.md) · [Server API](docs/SERVER_API.md) · [Workers](docs/WORKERS.md)

## Related

- [cleave](https://codeberg.org/atomdrift/cleave) — the capability analyzer underneath
- [azoth](https://codeberg.org/atomdrift/azoth) — model weights, thresholds, and feature spec
- [hopper](https://codeberg.org/atomdrift/hopper) — distributed work queue
- [Atomdrift Lab](https://lab.atomdrift.org/) — submit samples for free analysis

## License

Apache-2.0 - because why the fuck not?
