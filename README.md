# Atomdrift Scan

[![CI](https://github.com/atomdrift-project/scan/actions/workflows/ci.yml/badge.svg)](https://github.com/atomdrift-project/scan/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/atomdrift-project/scan)](https://github.com/atomdrift-project/scan/releases/latest)
[![License](https://img.shields.io/github/license/atomdrift-project/scan)](LICENSE)

Atomdrift Scan is a modern ML malware scanner designed to detect 0-day attacks against the software supply-chain. 

It's designed to be deterministic, fast, and flexible, and embeddable in any workflow or security tool you have in mind, and can operate against files, archives, URLs, PURLs, or processes. 

As of August 2026, Atomdrift Scan has a [82% 0-day detection rate](https://atomdrift.org/compare/), +18% ahead of any other scanner: commercial or open.

<p align="center">
  <img src="media/screenshot.png" alt="Atomdrift Scan terminal output" width="760">
</p>

How does Atomdrift get such great results? First, Atomdrift covers more ground than any other single malware scanner:

- 100+ supported file formats: from C source to ELF to PDF
- 100,000+ detection rules covering malware on every platform from AIX to iOS to Windows
- 4,000,000+ hashes for known good/badware
- Integrated AST analysis using [tree-sitter](https://tree-sitter.github.io/)
- Automated binary reverse engineering via [rizin](https://rizin.re/)

Most importantly, rules are constantly refreshed using reinforcement learning against new samples, blogs, and technical articles, resulting in ~1000 updated rules daily.

## Install

If you are an a UNIX-flavored host (macOS, Linux, BSD, Solaris, illumos, Android):

```bash
curl -fsSL https://install.atomdrift.org/scan.sh | sh
```

If you are on Windows:

```powershell
irm https://install.atomdrift.org/scan.ps1 | iex
```

Or, if you have [Rust](https://rust-lang.org/) installed and just want to build it from source:

```shell
make install
```

If deeper binary analysis is required, you should also install
[rizin](https://rizin.re/), [upx](https://upx.github.io/), and [innoextract](https://github.com/dscharrer/innoextract).

## Usage

```bash
# Scan a file, directory, or archive recusively:
atomscan ./project
atomscan release.tgz

# Fetch a package from its registry and scan it.
atomscan purl npm/left-pad@1.3.0

# Fetch and scan a URL.
atomscan url https://example.com/download

# EXPERIMENTAL: Scan running process executables, or triage the wider host
atomscan ps
atomscan sys

# Emit one JSON object per scanned file.
atomscan -f json ./project
```

Exit codes make CI integration trivial:

- `0`: all samples are benign
- `1`: hostile sample detected
- `2`: suspicious sample detected
- `3` or more: analysis error

## How does it work?

### Networking and privacy

atomscan will never send telemetry data. It will however reach out to the Internet for 2 reasons:

- **rule updates**: every 24h, can be disabled using `--no-update` or `SCAN_NO_UPDATE_CHECK=1`
- **dependency fetching**: to detect if a benign package depends on downloading a compromised package or  payload. Set `--fetch=none` to prevent this.

## Tune the false-positive level

Unlike other malware scanners, Atomdrift allows you to adjust sensitivity in terms of an acceptable false-positive level using the `-l` flag:

* `-l0`: tight, sets the confidence cutoff to a point where no false-positives have been observed.
* `-l25`: the default shipping point: 25 false-positives per 100 million files.
* `-l1000`: loose, roughly 1 false positive per 100,000 files.

NOTE: For file formats where we don't have 100 million samples, the observed false-positive rate may be up to 5-6X the requested level. YMMV.


Raw `--threshold-hostile` and `--threshold-suspicious` overrides are available
for users who want to micromanage probability thresholds directly, but these numbers are not guaranteed to be stable, and these flags cannot be combined with `-l`.

## How it works

Atomdrift Scan uses several local analysis stages:

1. [cleave](https://github.com/atomdrift-project/cleave) unpacks containers and
   extracts capabilities from binaries and source with tools including Rizin,
   tree-sitter, YARA, and UPX.
2. The report is converted into a standardized feature vector.
3. [azoth](https://github.com/atomdrift-project/azoth) ONNX model ensembles
   select a route for the file type and produce a probability and explanation.
4. The configured false-positive level turns that result into a benign,
   suspicious, or hostile verdict.

```mermaid
flowchart LR
    IN([file · directory · archive<br/>package · URL · process]) --> BLOOM{known-good /<br/>known-bad filters}
    BLOOM -->|no decisive match| CLEAVE[cleave<br/>unpack, parse, disassemble, match]
    CLEAVE --> FEATURES[feature vector]
    FEATURES --> MODEL[azoth<br/>ONNX ensemble]
    MODEL --> OUT{{benign · suspicious · hostile}}
    BLOOM -->|decisive match| OUT
    MODEL -. optional .-> LLM[local LLM second opinion]
    LLM -. blended verdict .-> OUT
```

### Optional local LLM

For additional interpretation, users can provide access to an LLM via the `--llm` flag. This flag serves two purposes:

* Provides a large-language model text interpretation of the results
* Steer edge cases based on agreement/disagreement with the ML model.

By default, it sends the interpreted evidence (not the original file) to `http://localhost:8000/v1` - meant to be used with a local service like Ollama or vLLM; but it can also be setup to use a remote service like Claude, ChatGPT, or DeepSeek.

```bash
atomscan --llm ./project
atomscan --llm http://model-host:8000/v1 --llm-model my-model ./project
```

This feature is optional. The default scanner needs neither an LLM nor a GPU.

## Coverage

The scanner recognizes more than 100 file and container types. Representative
coverage includes:

| Category | Formats |
| --- | --- |
| **Binaries and bytecode** | Mach-O, ELF, PE, WebAssembly, Android DEX, Java `.class`, Python `.pyc`, BEAM |
| **Source** | Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, C#, Ruby, PHP, Perl, Lua, Swift, Objective-C, Kotlin, Scala, Groovy, Zig, Elixir, Clojure, Shell, PowerShell, Batch, VBScript, AppleScript, JCL |
| **Build, manifest, and lock files** | package.json, package-lock.json, Cargo.toml, Cargo.lock, pyproject.toml, requirements.txt, Poetry, Pipenv, Composer, Yarn, pnpm, Go modules, binding.gyp, GitHub Actions, systemd units, Makefile, Dockerfile |
| **Archives and disk images** | ZIP, TAR, gzip, bzip2, XZ, zstd, 7-Zip, RAR, CAB, ASAR, DMG, ISO |
| **Packages and containers** | deb, rpm, APK, npm, wheel, egg, sdist, gem, crate, conda, NuGet, IPA, CRX, XPI, VSIX, OCI/Docker images, FreeBSD, Arch, Void, and Gentoo packages |
| **Documents and data** | OLE2, OOXML, OpenDocument, PDF, RTF, Markdown, HTML, XML, SVG, plist, JPEG, PNG, LNK, CHM, Python pickle |

The rule set includes platform-specific behaviors for Linux, macOS, Windows,
Android, iOS, the BSDs, AIX, Solaris, QNX, z/OS, ESXi, OpenWrt, VxWorks,
RouterOS, FortiOS, PAN-OS, IOS-XE, Junos, NetScaler, and Ivanti appliances.

The default build matrix covers Linux, macOS, FreeBSD, OpenBSD, NetBSD, illumos, Solaris, and Windows across several CPU architectures. Test depth
varies by target; see the [build workflow](.github/workflows/build.yml) for the
current matrix.

## Documentation

- [Integration guide](docs/INTEGRATION.md) — CLI, server, worker, exit codes,
  and deployment choices
- [JSON report schema](docs/JSON.md) — machine-readable output fields
- [Server API](docs/SERVER_API.md) — long-running HTTP service
- [Workers](docs/WORKERS.md) — distributed scanning with hopper
- [Dependency behavior](docs/DEPENDENCIES.md) — fetched dependency graph and
  provenance

## Related projects

- [cleave](https://github.com/atomdrift-project/cleave) — capability extraction
  and static analysis
- [azoth](https://github.com/atomdrift-project/azoth) — model weights,
  thresholds, and feature specification
- [hopper](https://github.com/atomdrift-project/hopper) — distributed work queue
- [Atomdrift Lab](https://lab.atomdrift.org/) — free sample analysis

Issues and pull requests are welcome in the
[GitHub repository](https://github.com/atomdrift-project/scan).

## License

Atomdrift Scan is available under the [Apache License 2.0](LICENSE).
