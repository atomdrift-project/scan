//! Command smoke tests: `cargo test --locked --test cli_smoke`.
//! The update test (ignored by default; `--include-ignored` runs it) downloads
//! published models and traits into a tempdir.
//! Other tests use a tiny local model and never need production artifacts.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::fs::{self, File};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

mod bundle;

struct Fixture {
    root: tempfile::TempDir,
    models: PathBuf,
    traits: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("temporary installation");
        let models = root.path().join("models");
        let traits = root.path().join("traits");
        bundle::write(&models).unwrap();
        fs::create_dir_all(traits.join("metadata")).unwrap();
        Self {
            root,
            models,
            traits,
        }
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_atomscan"));
        cmd.env_clear();
        // Retain only what the OS, runtime, and HTTPS client need. Developer
        // SCAN_*/CLEAVE_* settings and credentials must not change these tests.
        for name in [
            "PATH",
            "SystemRoot",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "USERPROFILE",
            "TMPDIR",
            "TMP",
            "TEMP",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "no_proxy",
            "all_proxy",
        ] {
            #[expect(
                clippy::disallowed_methods,
                reason = "passing the OS's own variables through"
            )]
            if let Some(value) = std::env::var_os(name) {
                cmd.env(name, value);
            }
        }
        cmd.current_dir(self.root.path())
            .env("HOME", self.root.path())
            .env("XDG_DATA_HOME", self.root.path().join("data"))
            .env("XDG_CACHE_HOME", self.root.path().join("cache"))
            .env("SCAN_MODELS_DIR", &self.models)
            .env("CLEAVE_TRAITS_DIR", &self.traits)
            .env("CLEAVE_CACHE_DIR", self.root.path().join("cleave-cache"))
            .env("SCAN_NO_UPDATE_CHECK", "1")
            .env("NO_COLOR", "1")
            .stdin(Stdio::null())
            // Never refresh rules in the background; `update-rules` still does.
            .arg("--no-update")
            .args(args);
        cmd
    }

    fn run(&self, cmd: &mut Command, expected: i32, timeout: Duration) -> Output {
        let args = cmd
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        // Files avoid undrained pipes blocking a verbose command before timeout.
        let stdout = tempfile::tempfile().unwrap();
        let stderr = tempfile::tempfile().unwrap();
        cmd.stdout(stdout.try_clone().unwrap())
            .stderr(stderr.try_clone().unwrap());
        let mut child = cmd
            .spawn()
            .unwrap_or_else(|e| panic!("spawn {args:?}: {e}"));
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().expect("poll command") {
                break status;
            }
            if started.elapsed() > timeout {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "command exceeded {timeout:?}: {args:?}\n{}",
                    String::from_utf8_lossy(&read_bytes(stderr))
                );
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        let output = Output {
            status,
            stdout: read_bytes(stdout),
            stderr: read_bytes(stderr),
        };
        assert_eq!(
            output.status.code(),
            Some(expected),
            "{args:?}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        output
    }

    fn smoke(&self, args: &[&str], expected: i32) -> Output {
        self.run(&mut self.command(args), expected, Duration::from_secs(60))
    }

    fn report(&self) -> PathBuf {
        let mut report = cleave::AnalysisReport::new(cleave::TargetInfo::default());
        report.files.push(cleave::FileAnalysis {
            id: 0,
            path: "payload.js".into(),
            file_type: "javascript".into(),
            sha256: "a".repeat(64),
            size: 12,
            findings: vec![cleave::Finding::new(
                "micro-behaviors/process/interpreter::eval",
                cleave::FindingKind::Indicator,
                "Executes a variable",
                0.95,
            )],
            ..Default::default()
        });
        report.files[0].findings[0].crit = cleave::Criticality::Notable;
        // The match the renderer annotates: `eval`, the first four bytes.
        report.files[0].findings[0].evidence = vec![cleave::Evidence {
            method: "ast".into(),
            value: "eval".into(),
            offsets: vec![0],
            ..Default::default()
        }];
        // A window as schema 8 wrote it (Z85 `b`), which readers still accept.
        report.files[0].context = serde_json::from_value(serde_json::json!([{
            "ln": 0, "line": 1, "col": 1,
            "b": "wP?G#c$D+dvjsnB",
        }]))
        .unwrap();
        let path = self.root.path().join("report.json");
        fs::write(&path, serde_json::to_vec(&report).unwrap()).unwrap();
        path
    }
}

fn read_bytes(mut file: File) -> Vec<u8> {
    use std::io::{Read as _, Seek as _};
    file.rewind().unwrap();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    bytes
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn browser_delivery_follows_references_by_default() {
    use fletch::fetch::{BlobCache, Fixtures, Outcome, fetch_ref};
    use fletch::{RefKind, RefLocator, Reference};

    const CHILD: &str = "https://assets.atomdrift-replay.test/assets/child";
    const MODULE: &str = "https://assets.atomdrift-replay.test/assets/entry.js";
    const ENTRY: &str = "https://assets.atomdrift-replay.test/entry.js";
    const CLEAN: &str = "https://assets.atomdrift-replay.test/clean.js";
    const MISSING: &str = "https://assets.atomdrift-replay.test/assets/unavailable";
    let f = Fixture::new();
    // A synthetic string carries the verdict. The gate exercises the shipped
    // reference extractor, default fetch policy, MIME hint and child verdict.
    let rule_dir = f
        .traits
        .join("objectives/impact/crypto-manipulation/hijack-transfer/address");
    fs::create_dir_all(&rule_dir).unwrap();
    fs::write(
        rule_dir.join("release-fixture.yaml"),
        r#"
defaults:
  platforms: [unix, windows]
  for: [javascript]
traits:
  - id: browser-selected-wallet-copy-override
    desc: Synthetic browser delivery test marker
    crit: hostile
    conf: 0.99
    if: {type: string_literal, exact: atomscan-browser-release-test}
"#,
    )
    .unwrap();
    fs::write(
        f.traits.join("metadata/reference-outcome.yaml"),
        r#"
defaults:
  platforms: [unix, windows]
  for: [javascript]
traits:
  - id: browser-script-fetch-failed
    desc: Browser script fetch failed
    crit: notable
    conf: 1.0
    if: {type: metrics, field: references.unavailable_browser_script_count, min: 1}
"#,
    )
    .unwrap();

    let cache_root = f.root.path().join(if cfg!(target_os = "macos") {
        "Library/Caches"
    } else {
        "cache"
    });
    let cache = BlobCache::with_dir(cache_root.join("fletch/refs"));
    let net = Fixtures::default()
        .with_headers(
            CHILD,
            b"const fixture = 'atomscan-browser-release-test';",
            &[("content-type", "application/javascript")],
        )
        .with(MODULE, format!("import('{CHILD}');").as_bytes())
        .with(ENTRY, format!("import('{MODULE}');").as_bytes())
        .with(CLEAN, format!("// import('{CHILD}');").as_bytes());
    for url in [ENTRY, CLEAN, MODULE, CHILD] {
        let reference = Reference::new(RefLocator::Url(url.into()), RefKind::UrlFetch, "test", url);
        assert_eq!(fetch_ref(&reference, &net, &cache).outcome, Outcome::Ok);
    }
    let loader = f.root.path().join("loader.js");
    fs::write(
        &loader,
        format!(
            "function fixture(document) {{ const s = document.createElement('script'); \
             s.src = '{CHILD}'; document.head.appendChild(s); }}"
        ),
    )
    .unwrap();
    let image = f.root.path().join("image.js");
    fs::write(
        &image,
        format!("function fixture(document) {{ const i = document.createElement('img'); i.src = '{CHILD}'; document.head.appendChild(i); }}"),
    ).unwrap();
    let missing = f.root.path().join("missing.js");
    fs::write(
        &missing,
        format!("function fixture(document) {{ const s = document.createElement('script'); s.src = '{MISSING}'; document.head.appendChild(s); }}"),
    ).unwrap();
    let offline_command = |args: &[&str]| {
        let mut cmd = f.command(args);
        for key in [
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "https_proxy",
            "http_proxy",
        ] {
            cmd.env(key, "http://127.0.0.1:1");
        }
        cmd.env_remove("NO_PROXY")
            .env_remove("no_proxy")
            .env("ATOMDRIFT_UPLINK", "0")
            .env("SCAN_NO_ANALYSIS_CACHE", "1");
        cmd
    };

    for (target, remote, blocked) in [
        (loader.to_str().unwrap(), false, true),
        (image.to_str().unwrap(), false, false),
        (ENTRY, true, true),
        (CLEAN, true, false),
    ] {
        let mut args = vec!["--analyze", "local", "-f", "json"];
        if remote {
            args.push("url");
        }
        args.push(target);
        let mut cmd = offline_command(&args);
        let output = f.run(&mut cmd, i32::from(blocked), Duration::from_secs(60));
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let child_verdict = report["findings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|finding| {
                finding["id"] == "fetch/dependency-verdict"
                    && finding["desc"]
                        .as_str()
                        .is_some_and(|desc| desc.contains(CHILD))
            });
        assert_eq!(child_verdict, blocked, "{target}: {report}");
    }

    for (target, follow, expected) in [
        (&missing, None, true),
        (&missing, Some("--follow=none"), false),
        (&image, None, false),
    ] {
        let mut args = vec!["--analyze", "local", "-f", "json-full"];
        args.extend(follow);
        args.push(target.to_str().unwrap());
        let mut cmd = offline_command(&args);
        let output = f.run(&mut cmd, 0, Duration::from_secs(60));
        let report = String::from_utf8(output.stdout).unwrap();
        let mut findings = report
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter_map(|line| line["traits"].as_array().cloned())
            .flatten();
        let failure =
            findings.find(|finding| finding["id"] == "metadata::browser-script-fetch-failed");
        assert_eq!(failure.is_some(), expected, "{target:?}: {report}");
        if let Some(finding) = failure {
            assert!(
                finding["spans"]
                    .as_array()
                    .is_some_and(|spans| !spans.is_empty()),
                "{finding}"
            );
        }
    }
}

#[test]
#[ignore = "downloads the published rules over the network; CI's release builds run it"]
fn update_rules_installs_models_and_traits() {
    let fixture = Fixture::new();
    // `SCAN_MODELS_DIR` pins a bundle the updater never touches, so unpinned
    // the models install into the platform data directory. Elsewhere that
    // follows the fixture's `HOME`/`XDG_DATA_HOME`; Windows resolves it as a
    // Known Folder that ignores both, so this process finds the same one.
    let mut cmd = fixture.command(&["update-rules"]);
    cmd.env_remove("SCAN_MODELS_DIR");
    fixture.run(&mut cmd, 0, Duration::from_secs(600));
    let data = if cfg!(windows) {
        scan::data_dir().expect("data directory")
    } else {
        fixture.root.path().to_path_buf()
    };

    // Fresh destinations and sidecars prove this fetched/installed bundles,
    // rather than merely succeeding after skipping a pinned directory.
    let model_meta = walkdir::WalkDir::new(&data)
        .into_iter()
        .filter_map(Result::ok)
        .map(walkdir::DirEntry::into_path)
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name == ".scan-models.toml")
        })
        .expect("model installation sidecar");
    let models = model_meta
        .parent()
        .expect("sidecar has a parent")
        .to_path_buf();
    let trait_meta = fixture.traits.join(".cleave-rules.toml");
    let model_before = fs::read(&model_meta).expect("model installation sidecar");
    let traits_before = fs::read(&trait_meta).expect("traits installation sidecar");
    assert!(
        models.join("feature_spec.json").is_file()
            || models.join("general/feature_spec.json").is_file()
    );
    assert!(
        walkdir::WalkDir::new(&models)
            .into_iter()
            .filter_map(Result::ok)
            .any(|entry| entry.path().extension().is_some_and(|ext| ext == "onnx"))
    );
    assert!(
        walkdir::WalkDir::new(&fixture.traits)
            .into_iter()
            .filter_map(Result::ok)
            .any(|entry| matches!(
                entry.path().extension().and_then(|e| e.to_str()),
                Some("yaml" | "yml")
            ))
    );

    // A second run finds everything current and reinstalls nothing.
    let mut cmd = fixture.command(&["update-rules"]);
    cmd.env_remove("SCAN_MODELS_DIR");
    fixture.run(&mut cmd, 0, Duration::from_secs(180));
    assert_eq!(fs::read(&model_meta).unwrap(), model_before);
    assert_eq!(fs::read(&trait_meta).unwrap(), traits_before);
}

#[test]
fn render_saved_reports_and_stdin_succeed() {
    let fixture = Fixture::new();
    let path = fixture.report();
    for format in ["terminal", "tiny", "interpret"] {
        let mut cmd = fixture.command(&["render", "--format", format]);
        cmd.arg(&path);
        let output = fixture.run(&mut cmd, 0, Duration::from_secs(60));
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("eval(data)"),
            "{format}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
    let mut cmd = fixture.command(&["render", "-", "--format", "tiny"]);
    cmd.stdin(File::open(path).unwrap());
    let output = fixture.run(&mut cmd, 0, Duration::from_secs(60));
    assert!(String::from_utf8_lossy(&output.stdout).contains("eval(data)"));
}

#[test]
fn render_invalid_reports_fail() {
    let fixture = Fixture::new();
    let path = fixture.root.path().join("invalid.json");
    fs::write(&path, br#"{"v":"unsupported"}"#).unwrap();
    let mut cmd = fixture.command(&["render", "--format", "tiny"]);
    cmd.arg(path);
    let output = fixture.run(&mut cmd, 1, Duration::from_secs(60));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unsupported report version"));
    assert!(output.stdout.is_empty());
}

#[test]
fn url_rejects_an_invalid_locator() {
    let fixture = Fixture::new();
    let output = fixture.smoke(&["url", "not-a-url", "--mode", "slow"], 1);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not-a-url"), "{stderr}");
    assert!(!stderr.contains("model bundle is incomplete"), "{stderr}");
}

#[test]
fn purl_rejects_an_invalid_coordinate() {
    let fixture = Fixture::new();
    let output = fixture.smoke(&["purl", "not-a-purl", "--mode", "slow"], 1);
    assert!(String::from_utf8_lossy(&output.stderr).contains("not a package URL"));
}

#[test]
fn ps_reports_a_missing_model_bundle() {
    let fixture = Fixture::new();
    let empty = fixture.root.path().join("empty-models");
    fs::create_dir(&empty).unwrap();
    let mut cmd = fixture.command(&["ps", "--mode", "slow"]);
    cmd.env("SCAN_MODELS_DIR", &empty);
    let output = fixture.run(&mut cmd, 1, Duration::from_secs(60));
    // A bundle is a routed ensemble: an empty directory lacks its `general/`
    // route, and the error says so and how to fix it.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("incomplete"), "{stderr}");
    assert!(stderr.contains("general/"), "{stderr}");
    assert!(stderr.contains("update-rules"), "{stderr}");
}

#[test]
fn version_json_reports_the_build_and_rule_sources() {
    let fixture = Fixture::new();
    let output = fixture.smoke(&["version", "--format", "json"], 0);
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["version"], env!("CARGO_PKG_VERSION"));
    for key in ["models", "traits", "bloom"] {
        assert!(value.get(key).is_some(), "missing {key}: {value}");
    }
}

#[test]
fn required_operands_are_usage_errors() {
    let fixture = Fixture::new();
    for name in ["url", "purl", "render"] {
        let output = fixture.smoke(&[name], 2);
        assert!(String::from_utf8_lossy(&output.stderr).contains("required"));
    }
}
