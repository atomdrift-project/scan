//! A finished scan of a named file, kept so that scanning the same bytes
//! again the same way prints the same answer without doing the work.
//!
//! The analysis caches already make a repeat scan cheap to *compute*: cleave
//! replays the file's report, and every dependency's report is kept under its
//! content hash. What they do not keep is the result. A wheel with forty
//! fresh dependencies still spent ten seconds asking registries which
//! releases those are, merging their reports and rendering the card, to print
//! what it printed a minute earlier.
//!
//! An entry is named by the file's SHA-256 and a digest of everything else
//! what was printed depends on: the configuration, the path it is shown
//! under, both terminals, and the environment variables the stack reads. It lives in the
//! analysis cache's namespace, so a rules, engine, bloom or model change
//! starts from nothing.
//!
//! One input is outside that key: which releases the file's references
//! resolve to. A registry can answer differently a minute later, so a scan
//! that followed references is replayed only for [`RESOLUTION_TTL`] after it
//! was resolved. The same limit holds when a remote is asked about a file
//! before it is analyzed here: its answer can change too. After that the references are resolved again, and each
//! dependency that resolves to bytes already seen is served from the same
//! content-keyed cache as before.
//!
//! Only a settled scan is kept (see `ClassifiedReport::settled`): never one
//! cut short by a deadline or built on a fetch that failed by accident.

use std::io::IsTerminal as _;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{ScanConfig, ScanResult};
use crate::analysis_cache::AnalysisCache;
use crate::model::Classification;

/// How long a scan that followed references, or deferred to a remote first,
/// answers from its replay: the time a resolved release set is trusted before
/// the registries are asked again.
const RESOLUTION_TTL: Duration = Duration::from_secs(60 * 60);

/// Environment variables whose names start with one of these can change an
/// analysis or its output, so all of them are part of the key.
const ENV_PREFIXES: [&str; 5] = ["SCAN_", "CLEAVE_", "ATOMDRIFT_", "FLETCH_", "STNG_"];

/// What a finished scan contributed: its line in the tally, and what it
/// printed.
#[derive(Serialize, Deserialize)]
pub(super) struct Replay {
    /// When the scan finished, in seconds since the Unix epoch (UTC).
    at: u64,
    /// Whether something outside the file and this machine could have fed
    /// the result, so that it expires: references it followed, or a remote
    /// that was asked about the file first.
    expires: bool,
    /// SHA-256 of the file.
    #[serde(skip)]
    pub(super) sha256: String,
    pub(super) class: Classification,
    /// Whether it carried suspicious or hostile static findings.
    pub(super) severe: bool,
    pub(super) dependency_gaps: u32,
    pub(super) flagged_candidates: u32,
    /// What its fetch phase left on stderr: fetch rows or the dependency tree.
    pub(super) notes: String,
    /// What it wrote to stdout; `None` when the display filter hid it.
    pub(super) output: Option<String>,
}

impl Replay {
    /// The replay of `result`, which left `notes` on stderr and wrote
    /// `output`.
    pub(super) fn new(
        result: &ScanResult,
        notes: String,
        output: Option<String>,
        config: &ScanConfig,
    ) -> Self {
        Self {
            at: now(),
            expires: config.fetch_policy().enabled()
                || crate::remote::mode() != crate::remote::Analyze::Local,
            sha256: result.sha256.clone(),
            class: result.classification,
            severe: result.has_severe_findings(),
            dependency_gaps: result.dependency_attention.gaps,
            flagged_candidates: result.dependency_attention.flagged_candidates,
            notes,
            output,
        }
    }

    /// The replay of an earlier scan of the file at `path` under `config`,
    /// if one is kept and has not expired. Hashes the file.
    pub(super) fn load(path: &Path, config: &ScanConfig) -> Option<Self> {
        let cache = AnalysisCache::shared()?;
        let sha256 = crate::digest::sha256_file(path).ok()?;
        let label = path.display().to_string();
        let replay: Self = cache.read(&key(&sha256, &label, config))?;
        let fresh = replay.fresh(now());
        tracing::debug!(path = %label, sha256, at = replay.at, fresh, "scan replay found");
        fresh.then_some(Self { sha256, ..replay })
    }

    /// Whether this replay still answers at `now` (seconds since the epoch):
    /// always when the scan read nothing but the file, otherwise while its
    /// resolution is trusted. A stamp from the future is a clock that moved,
    /// and is not.
    fn fresh(&self, now: u64) -> bool {
        now.checked_sub(self.at)
            .is_some_and(|age| !self.expires || age <= RESOLUTION_TTL.as_secs())
    }

    /// Keep this replay for the next scan of the same bytes shown as `label`.
    pub(super) fn store(&self, label: &str, config: &ScanConfig) {
        if let Some(cache) = AnalysisCache::shared() {
            cache.write(&key(&self.sha256, label, config), self);
        }
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// The cache key for a scan of the bytes `sha256` shown as `label` under
/// `config`: the hash, and a digest of every other input to the output.
fn key(sha256: &str, label: &str, config: &ScanConfig) -> String {
    let mut digest = Sha256::new();
    for part in [
        config.fingerprint(),
        label.to_string(),
        format!(
            "{:?}|{:?}|{}|{}|{}|{}|{}",
            crate::remote::mode(),
            crate::settings::get(),
            cleave::output::terminal_width(),
            crate::output::color_enabled(),
            std::io::stdout().is_terminal(),
            std::io::stderr().is_terminal(),
            // Whether the fetch phase draws a tree or streams rows.
            crate::deptree::room().is_some(),
        ),
    ] {
        digest.update(part.as_bytes());
        digest.update([0]);
    }
    let mut env: Vec<_> = std::env::vars_os()
        .filter(|(name, _)| {
            let name = name.to_string_lossy();
            ENV_PREFIXES.iter().any(|prefix| name.starts_with(prefix))
        })
        .collect();
    env.sort_unstable();
    for (name, value) in env {
        digest.update(name.as_encoded_bytes());
        digest.update(b"=");
        digest.update(value.as_encoded_bytes());
        digest.update([0]);
    }
    let digest = digest.finalize();
    let rest: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
    format!("{sha256}-{rest}.replay")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OutputFormat;

    fn replay(expires: bool, at: u64) -> Replay {
        Replay {
            at,
            expires,
            sha256: "ab".repeat(32),
            class: Classification::Hostile,
            severe: true,
            dependency_gaps: 3,
            flagged_candidates: 1,
            notes: "    \u{26a0} gap\n".to_string(),
            output: Some("card\n".to_string()),
        }
    }

    /// A scan that followed references answers for an hour and no longer; one
    /// that read only the file answers for as long as its namespace lives. A
    /// stamp ahead of the clock answers for neither.
    #[test]
    fn a_replay_expires_only_when_its_scan_followed_references() {
        let hour = RESOLUTION_TTL.as_secs();
        assert!(replay(true, 1_000).fresh(1_000 + hour));
        assert!(!replay(true, 1_000).fresh(1_001 + hour));
        assert!(replay(false, 1_000).fresh(1_000 + 1_000 * hour));
        assert!(!replay(true, 1_000).fresh(999));
        assert!(!replay(false, 1_000).fresh(999));
    }

    /// What is kept comes back as it was stored, under the hash it was
    /// stored by; a different path, format or file is a different entry.
    #[test]
    fn a_replay_is_found_only_for_the_same_bytes_shown_the_same_way() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sample.bin");
        std::fs::write(&path, b"the same bytes").expect("write");
        let sha256 = crate::digest::sha256_file(&path).expect("hash");
        let config = |format| ScanConfig::new(dir.path(), format, None).expect("config");
        let terminal = config(OutputFormat::Terminal);
        let label = path.display().to_string();
        let kept = Replay {
            sha256: sha256.clone(),
            ..replay(true, now())
        };
        kept.store(&label, &terminal);

        let found = Replay::load(&path, &terminal).expect("kept a moment ago");
        assert_eq!(found.sha256, sha256);
        assert_eq!(found.class, Classification::Hostile);
        assert!(found.severe);
        assert_eq!((found.dependency_gaps, found.flagged_candidates), (3, 1));
        assert_eq!(found.notes, kept.notes);
        assert_eq!(found.output, kept.output);

        let json = config(OutputFormat::Json);
        assert!(Replay::load(&path, &json).is_none(), "another format");
        let copy = dir.path().join("copy.bin");
        std::fs::copy(&path, &copy).expect("copy");
        assert!(Replay::load(&copy, &terminal).is_none(), "another path");
        std::fs::write(&path, b"other bytes").expect("rewrite");
        assert!(Replay::load(&path, &terminal).is_none(), "other bytes");
    }

    /// The key names the file by its hash and nothing else in the clear: the
    /// rest, archive passwords included, is behind a digest.
    #[test]
    fn a_key_shows_the_hash_and_hides_the_configuration() {
        let plain = || ScanConfig::new("models", OutputFormat::Terminal, None).expect("config");
        let config = plain().with_zip_passwords(vec!["hunter2-not-in-the-key".to_string()]);
        let sha256 = "cd".repeat(32);
        let name = key(&sha256, "a/label", &config);
        let (hash, rest) = name.split_once('-').expect("hash, then digest");
        assert_eq!(hash, sha256);
        let digest = rest.strip_suffix(".replay").expect("suffix");
        assert_eq!(digest.len(), 16);
        assert!(digest.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(name, key(&sha256, "a/label", &plain()));
        assert_ne!(name, key(&sha256, "b/label", &config));
    }
}
