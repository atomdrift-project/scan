//! End-to-end LLM grading of `testdata/llm/`: at least 20 hostile and 20
//! benign samples (plus a few dual-use ones expected to grade suspicious), each
//! rendered exactly as `--llm` sends it and graded by the live endpoint, which
//! must return the grade an expert security engineer would give the file if
//! they found it on an endpoint.
//!
//! This is the regression gate for the interpret render: a render change that
//! hides the evidence a grade rests on, or buries it in noise, shows up here
//! as a wrong grade, and the summary reports the prompt bytes each sample
//! cost, so a smaller render can be shown to grade as well.
//!
//! It sends live traffic to our LLM endpoint, so it is `#[ignore]`d and never
//! runs under `make test`; `make test-llm` runs it. It needs the network,
//! installed rules and models, and `~/.tok/llm`, and skips (and says so) when
//! the token file is absent. The endpoint defaults to
//! `https://llm.isotope13.ai/v1`; `SCAN_LLM` overrides it, as it does for scan.
//! `SCAN_LLM_FIXTURES_DIR` grades another directory laid out the same way (a
//! `MANIFEST.tsv` beside xz-compressed samples) — a held-out set, or goldens
//! built from a larger corpus — and `SCAN_LLM_SYSTEM_PROMPT_FILE` swaps in a
//! candidate prompt, so a prompt or render change can be measured on both.
//! Grading is not bit-reproducible even at temperature 0, so each sample is
//! decided by the majority of up to three grades.
//!
//! The render is taken from the real binary (`atomscan path -f interpret`) and
//! graded in-process with [`scan::interpret::grade_render`], the function
//! behind `atomscan render --llm`. That grades every sample, where `path
//! --llm` would let the admission gate skip the quiet benign ones.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use clap::Parser as _;

const DEFAULT_ENDPOINT: &str = "https://llm.isotope13.ai/v1";
/// Grades asked per sample at most; two agreeing grades settle it early.
const MAX_VOTES: usize = 3;
/// Samples rendered and graded at once.
const WORKERS: usize = 4;

#[derive(clap::Parser)]
struct Cli {
    #[command(flatten)]
    global: scan::cli::GlobalArgs,
}

struct Fixture {
    path: PathBuf,
    expected: String,
}

struct Outcome {
    name: String,
    expected: String,
    grades: Vec<String>,
    prompt_bytes: usize,
    reason: String,
}

impl Outcome {
    fn passed(&self) -> bool {
        self.grades.iter().filter(|g| **g == self.expected).count() * 2 > self.grades.len()
    }

    fn line(&self) -> String {
        format!(
            "{} {:10} {} prompt={}B grades={:?} {}",
            if self.passed() { "ok  " } else { "FAIL" },
            self.expected,
            self.name,
            self.prompt_bytes,
            self.grades,
            if self.passed() { "" } else { &self.reason },
        )
    }
}

fn fixtures(root: &Path) -> Vec<Fixture> {
    let manifest = std::fs::read_to_string(root.join("MANIFEST.tsv")).expect("MANIFEST.tsv");
    manifest
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut cols = line.split('\t');
            let (Some(file), Some(expected)) = (cols.next(), cols.next()) else {
                panic!("malformed MANIFEST.tsv row: {line}");
            };
            assert!(
                matches!(expected, "hostile" | "suspicious" | "benign"),
                "unexpected grade {expected:?} in: {line}"
            );
            Fixture {
                path: root.join(file),
                expected: expected.to_string(),
            }
        })
        .collect()
}

/// A held-out set to grade instead of the built-in fixtures.
#[expect(
    clippy::disallowed_methods,
    reason = "a switch for this test alone; scan never reads it, so it is not in crate::env"
)]
fn fixtures_dir() -> Option<PathBuf> {
    std::env::var_os("SCAN_LLM_FIXTURES_DIR").map(PathBuf::from)
}

/// Restore a fixture under `dir` with its original name (the render shows it).
fn decompress(fixture: &Path, dir: &Path) -> PathBuf {
    let name = fixture
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix(".xz"))
        .expect("fixture names end in .xz");
    let out = dir.join(name);
    let bytes = Command::new("xz")
        .args(["-dc"])
        .arg(fixture)
        .output()
        .expect("xz is required to decompress the LLM fixtures");
    assert!(bytes.status.success(), "xz -dc {}", fixture.display());
    std::fs::write(&out, bytes.stdout).unwrap();
    out
}

/// The exact user message a live `--llm` query would send for `sample`.
fn render(sample: &Path) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_atomscan"))
        .args([
            "path",
            "--no-update",
            "--uplink",
            "false",
            "--follow-depth",
            "0",
            "-f",
            "interpret",
        ])
        .arg(sample)
        .output()
        .unwrap();
    // 0 clean, 1 hostile, 2 suspicious; 3 (errors) and 4 (degraded YARA) mean
    // the render was made without the full analysis and is not worth grading.
    assert!(
        matches!(out.status.code(), Some(0..=2)),
        "atomscan exited {:?} on {}: {}",
        out.status.code(),
        sample.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("interpret render is UTF-8")
}

fn grade(cfg: &scan::interpret::InterpretConfig, render: &str) -> (String, String) {
    let reply = scan::interpret::grade_render(cfg, render).expect("LLM endpoint answered");
    let field = |k: &str| reply[k].as_str().unwrap_or_default().to_string();
    (field("grade"), field("reason"))
}

fn judge(cfg: &scan::interpret::InterpretConfig, fixture: &Fixture, scratch: &Path) -> Outcome {
    let dir = tempfile::tempdir_in(scratch).unwrap();
    let sample = decompress(&fixture.path, dir.path());
    let context = render(&sample);
    let mut outcome = Outcome {
        name: sample.file_name().unwrap().to_string_lossy().into_owned(),
        expected: fixture.expected.clone(),
        grades: Vec::new(),
        // `-f interpret` prints the user message byte for byte.
        prompt_bytes: context.len(),
        reason: String::new(),
    };
    while outcome.grades.len() < MAX_VOTES {
        let (grade, reason) = grade(cfg, &context);
        if grade != fixture.expected {
            outcome.reason = reason;
        }
        outcome.grades.push(grade);
        let agree = outcome
            .grades
            .iter()
            .filter(|g| **g == fixture.expected)
            .count();
        let disagree = outcome.grades.len() - agree;
        if agree * 2 > MAX_VOTES || disagree * 2 > MAX_VOTES {
            break;
        }
    }
    outcome
}

#[test]
#[ignore = "sends live traffic to the LLM endpoint; run with `make test-llm`"]
fn llm_grades_every_fixture_as_labelled() {
    let Some(token) = dirs::home_dir().map(|h| h.join(".tok/llm")) else {
        return;
    };
    if !token.is_file() {
        eprintln!("skipping: {} is absent", token.display());
        return;
    }
    let endpoint = scan::env::text(scan::env::SCAN_LLM).unwrap_or_else(|| DEFAULT_ENDPOINT.into());
    let cfg = Cli::try_parse_from(["atomscan", "--llm", &endpoint])
        .unwrap()
        .global
        .interpret_config()
        .expect("LLM endpoint configuration")
        .expect("--llm enables interpretation");

    let builtin = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/llm");
    let root = fixtures_dir().unwrap_or_else(|| builtin.clone());
    let fixtures = fixtures(&root);
    if root == builtin {
        for class in ["hostile", "benign"] {
            let n = fixtures.iter().filter(|f| f.expected == class).count();
            assert!(n >= 20, "{n} {class} fixtures; the set needs at least 20");
        }
    }

    let scratch = tempfile::tempdir().unwrap();
    let queue = Mutex::new(fixtures.iter());
    let outcomes = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..WORKERS {
            s.spawn(|| {
                loop {
                    let Some(fixture) = queue.lock().unwrap().next() else {
                        break;
                    };
                    let outcome = judge(&cfg, fixture, scratch.path());
                    eprintln!("{}", outcome.line());
                    outcomes.lock().unwrap().push(outcome);
                }
            });
        }
    });
    let mut outcomes = outcomes.into_inner().unwrap();
    outcomes.sort_by(|a, b| (&a.expected, &a.name).cmp(&(&b.expected, &b.name)));

    let mut report = String::new();
    for o in &outcomes {
        report.push_str(&o.line());
        report.push('\n');
    }
    let total: usize = outcomes.iter().map(|o| o.prompt_bytes).sum();
    let failed = outcomes.iter().filter(|o| !o.passed()).count();
    report.push_str(&format!(
        "{} of {} graded as labelled; total prompt {} bytes\n",
        outcomes.len() - failed,
        outcomes.len(),
        total
    ));
    eprint!("{report}");
    assert_eq!(failed, 0, "\n{report}");
}
