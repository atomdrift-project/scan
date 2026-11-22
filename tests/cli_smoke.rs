//! Command smoke tests: `cargo test --locked --test cli_smoke`.
//! The update test (ignored by default; `--include-ignored` runs it) downloads
//! published models and traits into a tempdir.
//! Other tests use a tiny local model and never need production artifacts.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::fs::{self, File};
use std::path::{Path, PathBuf};
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
            // No test asks Atomdrift Uplink about anything.
            .env("ATOMDRIFT_UPLINK", "false")
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

    // `entry.js` imports the module that imports the child: the child is two
    // levels down, past the default depth of one and within an explicit two.
    for (target, remote, depth, blocked) in [
        (loader.to_str().unwrap(), false, None, true),
        (image.to_str().unwrap(), false, None, false),
        (ENTRY, true, None, false),
        (ENTRY, true, Some("--follow-depth=2"), true),
        (CLEAN, true, Some("--follow-depth=2"), false),
    ] {
        let mut args = vec!["--uplink", "false", "-f", "json"];
        args.extend(depth);
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
        assert_eq!(child_verdict, blocked, "{target} {depth:?}: {report}");
    }

    for (target, follow, expected) in [
        (&missing, None, true),
        (&missing, Some("--follow=none"), false),
        (&image, None, false),
    ] {
        let mut args = vec!["--uplink", "false", "-f", "json-full"];
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

/// Scanning one named file again, unchanged and the same way, prints what
/// the first scan printed without analyzing it; a changed file or a changed
/// option is scanned. A copy under another name is the same file: its scan
/// is the first one's, shown under its own path.
#[test]
fn a_repeat_scan_of_an_unchanged_file_is_replayed() {
    const REPLAYED: &str = "scan replayed";
    let f = Fixture::new();
    let rule_dir = f.traits.join("objectives/test");
    fs::create_dir_all(&rule_dir).unwrap();
    fs::write(
        rule_dir.join("replay-fixture.yaml"),
        r"
defaults:
  platforms: [unix, windows]
  for: [javascript]
traits:
  - id: replay-fixture-marker
    desc: Synthetic replay test marker
    crit: hostile
    conf: 0.99
    if: {type: string_literal, exact: atomscan-replay-test}
",
    )
    .unwrap();
    let sample = f.root.path().join("sample.js");
    fs::write(&sample, "const marker = 'atomscan-replay-test';\n").unwrap();
    let scan_as = |path: &Path, format: &str, cached: bool| {
        let mut cmd = f.command(&[
            "--uplink",
            "false",
            "-f",
            format,
            "path",
            path.to_str().unwrap(),
        ]);
        cmd.env("ATOMDRIFT_UPLINK", "0")
            .env("RUST_LOG", "scan::engine::run=info");
        if !cached {
            cmd.env("SCAN_NO_ANALYSIS_CACHE", "1");
        }
        let output = f.run(&mut cmd, 0, Duration::from_secs(60));
        let replayed = String::from_utf8_lossy(&output.stderr).contains(REPLAYED);
        (String::from_utf8(output.stdout).unwrap(), replayed)
    };
    let scan = |format: &str| scan_as(&sample, format, true);

    let (first, replayed) = scan("json");
    assert!(!replayed, "nothing to replay yet");
    assert!(first.contains("replay-fixture-marker"), "{first}");
    assert_eq!(scan("json"), (first.clone(), true));

    // Another format is another scan, and is then replayed in its own right.
    let (tiny, replayed) = scan("tiny");
    assert!(!replayed, "a format never scanned");
    assert_ne!(tiny, first);
    assert_eq!(scan("tiny"), (tiny, true));

    // A copy is not analyzed: it prints the first scan with its own path,
    // which is what an analysis of the copy alone prints. The time of the
    // analysis is the only thing that tells the two apart.
    let copy = f.root.path().join("renamed copy.js");
    fs::copy(&sample, &copy).unwrap();
    let path_of = |path: &Path| serde_json::to_string(path.to_str().unwrap()).unwrap();
    let undated = |decision: &str| {
        let (decision, at) = decision.split_once("\"analyzed_at\"").expect(decision);
        assert!(!at.contains("renamed") && !at.contains("sample"), "{at}");
        decision.to_string()
    };
    let (copied, replayed) = scan_as(&copy, "json", true);
    assert!(replayed, "the same bytes under another name");
    assert_eq!(copied, first.replace(&path_of(&sample), &path_of(&copy)));
    // In its `path` and with the finding that fired on it.
    assert_eq!(copied.matches(&path_of(&copy)).count(), 2, "{copied}");
    let (alone, replayed) = scan_as(&copy, "json", false);
    assert!(!replayed, "nothing is kept without the cache");
    assert_eq!(undated(&copied), undated(&alone));
    // The text format is right for a copy as well, whether its output could
    // be shown under another name or the copy was scanned for itself.
    let (copied, _) = scan_as(&copy, "tiny", true);
    assert_eq!(copied, scan_as(&copy, "tiny", false).0);
    assert_eq!(scan_as(&copy, "tiny", true), (copied, true));

    // The path is the same and the bytes are not: nothing is replayed, and
    // the answer is the new file's.
    fs::write(&sample, "const marker = 'nothing of note';\n").unwrap();
    let (changed, replayed) = scan("json");
    assert!(!replayed, "changed bytes");
    assert!(!changed.contains("replay-fixture-marker"), "{changed}");
}

/// Scanning a directory again replays every file that has not changed,
/// each under its own path, and analyzes only bytes it has not scanned: a
/// copy of a scanned file is that file's scan, shown under its own path. A
/// file found in a walk and the same file named alone are different scans.
#[test]
fn a_repeat_scan_of_a_directory_replays_each_unchanged_file() {
    const REPLAYED: &str = "scan replayed";
    const MARKED: &str = "const marker = 'atomscan-replay-test';\n";
    let f = Fixture::new();
    let rule_dir = f.traits.join("objectives/test");
    fs::create_dir_all(&rule_dir).unwrap();
    fs::write(
        rule_dir.join("replay-fixture.yaml"),
        r"
defaults:
  platforms: [unix, windows]
  for: [javascript]
traits:
  - id: replay-fixture-marker
    desc: Synthetic replay test marker
    crit: hostile
    conf: 0.99
    if: {type: string_literal, exact: atomscan-replay-test}
",
    )
    .unwrap();
    let dir = f.root.path().join("batch");
    fs::create_dir_all(dir.join("nested")).unwrap();
    // Two files of the same bytes are two results, one per path.
    fs::write(dir.join("first.js"), MARKED).unwrap();
    fs::write(dir.join("nested/copy.js"), MARKED).unwrap();
    fs::write(dir.join("plain.js"), "const marker = 'nothing of note';\n").unwrap();
    let scan = |target: &Path| {
        let mut cmd = f.command(&[
            "--uplink",
            "false",
            "-f",
            "json",
            "path",
            target.to_str().unwrap(),
        ]);
        cmd.env("ATOMDRIFT_UPLINK", "0")
            .env("RUST_LOG", "scan::engine::run=info,scan::bloom_repo=info");
        let output = f.run(&mut cmd, 0, Duration::from_secs(60));
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Every scan says once what its artifacts met on the way to analysis.
        assert_eq!(
            stderr.matches("what each artifact met").count(),
            1,
            "{stderr}"
        );
        assert!(stderr.contains("payloads_known_good="), "{stderr}");
        let replayed = stderr.matches(REPLAYED).count();
        // Results arrive as they finish, so compare them in a fixed order.
        let mut results: Vec<String> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        results.sort_unstable();
        (results, replayed)
    };
    let marked = |results: &[String]| {
        results
            .iter()
            .filter(|line| line.contains("replay-fixture-marker"))
            .count()
    };

    // Each result names its own file, whichever scan it came from.
    let named = |results: &[String], name: &str| {
        let path = serde_json::to_string(dir.join(name).to_str().unwrap()).unwrap();
        results
            .iter()
            .filter(|line| line.contains(&format!("\"path\":{path}")))
            .count()
    };
    let each_named_once = |results: &[String]| {
        ["first.js", "nested/copy.js", "plain.js"]
            .iter()
            .all(|name| named(results, name) == 1)
    };

    // The copy is not analyzed: it is the first file's scan under its path.
    let (first, replayed) = scan(&dir);
    assert_eq!(replayed, 1, "the second of the two files of the same bytes");
    assert_eq!((first.len(), marked(&first)), (3, 2), "{first:?}");
    assert!(each_named_once(&first), "{first:?}");
    assert_eq!(scan(&dir), (first.clone(), 3));

    // One file changes into bytes already scanned: nothing is analyzed, and
    // its answer is the one those bytes have, under its own path.
    fs::write(dir.join("plain.js"), MARKED).unwrap();
    let (changed, replayed) = scan(&dir);
    assert_eq!(replayed, 3, "bytes scanned before");
    assert_eq!((changed.len(), marked(&changed)), (3, 3), "{changed:?}");
    assert!(each_named_once(&changed), "{changed:?}");
    assert_eq!(scan(&dir), (changed, 3));

    // And into bytes never scanned: it alone is analyzed.
    fs::write(dir.join("plain.js"), "const marker = 'something new';\n").unwrap();
    let (fresh, replayed) = scan(&dir);
    assert_eq!(replayed, 2, "the two files left alone");
    assert_eq!((fresh.len(), marked(&fresh)), (3, 2), "{fresh:?}");
    assert!(each_named_once(&fresh), "{fresh:?}");

    // Named alone, a file the walk scanned is scanned again: nothing a walk
    // kept answers for it.
    let (alone, replayed) = scan(&dir.join("first.js"));
    assert_eq!((alone.len(), replayed), (1, 0));
    assert_eq!(scan(&dir.join("first.js")), (alone, 1));
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

/// `--offline` refuses every command that exists to reach the network, before
/// it reaches it, and cannot be combined with a forced update.
#[test]
fn offline_refuses_what_needs_the_network() {
    let fixture = Fixture::new();
    for args in [
        &["--offline", "url", "https://example.com/x.js"][..],
        &["--offline", "purl", "pkg:npm/left-pad@1.3.0"],
        &["--offline", "update-rules"],
        &["--offline", "feed", "supporter"],
    ] {
        let output = fixture.smoke(args, 1);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("--offline forbids"), "{args:?}: {stderr}");
    }
    let output = fixture.smoke(&["--offline", "-u", "version"], 2);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot be used with"), "{stderr}");
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
