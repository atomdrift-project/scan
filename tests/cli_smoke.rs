//! Command smoke tests: `cargo test --locked --test cli_smoke`.
//! The update test downloads published models and traits from R2 into a tempdir.
//! Other tests use a tiny local model and never need production artifacts.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::fs::{self, File};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

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
        fs::create_dir_all(&models).unwrap();
        fs::create_dir_all(traits.join("metadata")).unwrap();
        fs::write(
            models.join("model.onnx"),
            include_bytes!("fixtures/cli-model.onnx"),
        )
        .unwrap();
        fs::write(
            models.join("feature_spec.json"),
            serde_json::to_vec(&serde_json::json!({
                "version": scan::features::EXPECTED_SPEC_VERSION,
                "abi_version": scan::features::EXPECTED_MODEL_ABI_VERSION,
                "presence_vocab": ["metadata"],
                "feature_names": ["present:metadata"],
                "total_features": 1,
                "standardized": false
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            models.join("config.json"),
            br#"{"suspicious":0.9,"hostile":0.99}"#,
        )
        .unwrap();
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
            if let Some(value) = std::env::var_os(name) {
                cmd.env(name, value);
            }
        }
        cmd.current_dir(self.root.path())
            .env("HOME", self.root.path())
            .env("XDG_DATA_HOME", self.root.path().join("data"))
            .env("XDG_CACHE_HOME", self.root.path().join("cache"))
            .env("SCAN_MODELS_DIR", &self.models)
            .env("SCAN_BLOOM_DIR", self.root.path().join("bloom"))
            .env("CLEAVE_TRAITS_DIR", &self.traits)
            .env("CLEAVE_CACHE_DIR", self.root.path().join("cleave-cache"))
            .env("SCAN_NO_UPDATE", "1")
            .env("SCAN_NO_UPDATE_CHECK", "1")
            .env("NO_COLOR", "1")
            .stdin(Stdio::null())
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
                    read_capture(stderr)
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
        report.files[0].context = serde_json::from_value(serde_json::json!([{
            "ln": 0, "line": 1, "col": 1,
            "b": "wP?G#c$D+dvjsnB",
            "n": [{"c":3,"i":"micro-behaviors/process/interpreter::eval",
                "d":"Executes a variable","o":0,"z":4,"cf":0.95}]
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

fn read_capture(file: File) -> String {
    String::from_utf8_lossy(&read_bytes(file)).into_owned()
}

#[test]
fn update_rules_installs_models_and_traits_from_r2() {
    let fixture = Fixture::new();
    let models = fixture.root.path().join("r2-models");
    let mut cmd = fixture.command(&["update-rules"]);
    cmd.arg("--model-dir").arg(&models);
    fixture.run(&mut cmd, 0, Duration::from_secs(600));

    // Fresh destinations and sidecars prove this fetched/installed bundles,
    // rather than merely succeeding after skipping a pinned directory.
    let model_meta = models.join(".litmus-models.toml");
    let trait_meta = fixture.traits.join(".cleave-rules.toml");
    let model_before = fs::read(&model_meta).expect("R2 model installation sidecar");
    let traits_before = fs::read(&trait_meta).expect("R2 traits installation sidecar");
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

    for extra in [
        &["--check"][..],
        &["--models-only", "--check"][..],
        &["--models-only"][..],
    ] {
        let mut cmd = fixture.command(&["update-rules"]);
        cmd.arg("--model-dir").arg(&models).args(extra);
        fixture.run(&mut cmd, 0, Duration::from_secs(180));
        assert_eq!(fs::read(&model_meta).unwrap(), model_before);
        assert_eq!(fs::read(&trait_meta).unwrap(), traits_before);
    }
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
    let mut cmd = fixture.command(&["ps", "--mode", "slow", "--model-dir"]);
    cmd.arg(empty);
    let output = fixture.run(&mut cmd, 1, Duration::from_secs(60));
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("feature_spec.json"));
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
