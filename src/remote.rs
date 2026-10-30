//! Where an artifact is judged under `--enriched`: here, or by isotope13.
//!
//! `--analyze=auto` asks isotope13 for verdicts it already holds and analyzes
//! locally whatever it does not know. The asking goes through
//! [`crate::corpus_precheck`] — the same batched `/v1/lookup` negotiation, the
//! same circuit breaker, the same fail-open rule hopper's fleet lookups use —
//! armed against isotope13 with the subscription token. `local` never asks,
//! and is the reproducible mode. `remote` sends every artifact to
//! `/v1/analyze`, uploading local files' contents.
//!
//! What may leave the machine is decided here and nowhere else:
//!
//! - the SHA-256 of a local file of at least
//!   [`crate::analysis::SMALL_JOB_MAX_BYTES`] (1 MiB) — below that, analyzing
//!   is cheaper than asking, and such files are mostly source nobody else has;
//! - the SHA-256 of a fetched artifact;
//! - a PURL only when [`public_purl`] says it names a package on a mainstream
//!   public registry.
//!
//! Never a URL, and never a file's contents except under `remote`. A remote
//! verdict replaces local analysis and is reported as a `"source": "remote"`
//! record, the way a bloom decision is reported as `"source": "bloom"`.

use std::io::Read as _;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use crate::corpus_precheck::{Standing, Verdict};
use crate::model::{Classification, Level};

/// isotope13's API. Fixed: an override would let any step of a CI job send the
/// subscription token, and every file it is shown, somewhere else.
pub const API_URL: &str = "https://api.isotope13.ai";

/// Beamline's upload cap.
const MAX_UPLOAD_BYTES: u64 = 16 * 1024 * 1024;
/// Connect budget: an unreachable API costs one of these per run, then the
/// breaker opens.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(4);
/// Whole-request ceiling for a lookup.
pub(crate) const LOOKUP_TIMEOUT: Duration = Duration::from_secs(15);
/// Whole-request ceiling for a remote analysis, which streams progress.
const ANALYZE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
/// How long a remote analysis keeps retrying transient failures before the
/// artifact is reported as failed. Under `remote` there is no local analysis
/// to fall back to, so it is the longest any call waits.
const ANALYZE_RETRY_BUDGET: Duration = Duration::from_secs(120);

/// `--analyze`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Analyze {
    /// Ask isotope13 first under `--enriched`; otherwise the same as `local`.
    #[default]
    Auto,
    /// Analyze everything here.
    Local,
    /// Send everything to isotope13 for analysis.
    Remote,
}

/// The mode this run resolved to, and the token `remote` uploads with.
static RESOLVED: OnceLock<(Analyze, Option<cleave::enriched::Token>)> = OnceLock::new();

/// Resolve `--analyze` against whether this run is Enriched (a held token,
/// or `--enriched`), once, before any scan starts.
///
/// `auto` on an open-source run is `local`: no new traffic for anyone who did
/// not ask for the service. On an Enriched run, `auto` arms the isotope13
/// lookups and `remote` holds the token for uploads. A missing token is asked
/// for here, on a terminal, and checked against the rules channel before it
/// is used.
///
/// # Errors
/// `remote` on an open-source run, or an Enriched run with no usable token.
pub fn resolve(analyze: Analyze, enriched: bool) -> Result<Analyze> {
    let resolved = match (analyze, enriched) {
        (Analyze::Remote, false) => {
            anyhow::bail!("--analyze=remote requires isotope13 Enriched: a token, or --enriched")
        }
        (Analyze::Local, _) | (Analyze::Auto, false) => (Analyze::Local, None),
        (mode, true) => {
            let token = token()?;
            if mode == Analyze::Auto {
                crate::corpus_precheck::configure_isotope13(token.clone());
            }
            (mode, Some(token))
        }
    };
    tracing::info!(event = "remote.mode", analyze = ?resolved.0, enriched);
    let mode = resolved.0;
    // A second resolve in one process (tests) keeps the first answer.
    let _ = RESOLVED.set(resolved);
    Ok(mode)
}

/// The resolved mode; `local` until [`resolve`] runs.
pub fn mode() -> Analyze {
    RESOLVED.get().map_or(Analyze::Local, |(mode, _)| *mode)
}

/// The subscription token, asking for one (and saving it, once the rules
/// channel has honoured it) when there is none.
fn token() -> Result<cleave::enriched::Token> {
    if let Some((token, _)) = cleave::enriched::token()? {
        return Ok(token);
    }
    cleave::rule_update::enrich()?;
    cleave::enriched::token()?
        .map(|(token, _)| token)
        .context("isotope13 token was accepted but could not be read back")
}

/// The process's client for isotope13: TLS only, and no redirects, so the
/// token can reach no host but [`API_URL`]'s.
pub(crate) fn http() -> Option<&'static reqwest::blocking::Client> {
    static HTTP: OnceLock<Option<reqwest::blocking::Client>> = OnceLock::new();
    HTTP.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .https_only(true)
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_TIMEOUT)
            .user_agent(concat!("atomdrift-scan/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| tracing::error!(error = %e, "cannot build the isotope13 HTTP client; remote analysis disabled"))
            .ok()
    })
    .as_ref()
}

/// Whether `purl` may be sent: a package on one of the mainstream public
/// registries.
///
/// The type names the registry, and these are the ones whose default is a
/// single public registry anyone can read: npm, PyPI, crates.io, RubyGems,
/// NuGet, the Go module proxy, Maven Central, Packagist, Hex, pub.dev,
/// CocoaPods, conda, CRAN, and Hackage. A `repository_url` qualifier names a
/// registry other than that default — a private mirror — and is never sent.
/// Anything else (`github`, `generic`, a type we do not know) is judged by
/// digest or here.
#[must_use]
pub fn public_purl(purl: &str) -> bool {
    const MAINSTREAM: &[&str] = &[
        "npm",
        "pypi",
        "cargo",
        "gem",
        "nuget",
        "golang",
        "maven",
        "composer",
        "hex",
        "pub",
        "cocoapods",
        "conda",
        "cran",
        "hackage",
    ];
    let Some(rest) = purl.strip_prefix("pkg:") else {
        return false;
    };
    let Some((kind, path)) = rest.split_once('/') else {
        return false;
    };
    let qualifiers = path.split_once('?').map_or("", |(_, q)| q);
    // Each names bytes other than the public registry's under this coordinate.
    if qualifiers.split('&').any(|q| {
        let q = q.to_ascii_lowercase();
        ["repository_url=", "download_url=", "vcs_url="]
            .iter()
            .any(|key| q.starts_with(key))
    }) {
        return false;
    }
    MAINSTREAM.contains(&kind.to_ascii_lowercase().as_str())
}

/// How a remote level grades at this run's operating level — the same rule a
/// local verdict is graded by. `None` for a verdict with no level.
#[must_use]
pub fn classification(fires_at: Level, level: Option<u16>) -> Option<Classification> {
    match fires_at {
        Level::Manual => None,
        Level::Clean => Some(Classification::Benign),
        Level::At(n) => Some(crate::model::verdict_for_level(
            n,
            level.unwrap_or(crate::model::DEFAULT_SEVERITY_LEVEL),
            u16::MAX,
        )),
    }
}

/// The local files isotope13 already judges, by path, and the files left to
/// analyze here, in their original order.
///
/// Only files of at least [`crate::analysis::SMALL_JOB_MAX_BYTES`] are asked
/// about. Each is hashed by streaming it — bounded memory, never a second
/// resident copy — and the digests go out in batches. A file whose digest
/// comes back with a standing verdict is answered; everything else, including
/// every file when lookups are off or the API is unreachable, is analyzed.
pub(crate) fn answer_known(paths: Vec<PathBuf>) -> (Vec<(PathBuf, String, Verdict)>, Vec<PathBuf>) {
    let Some(precheck) = crate::corpus_precheck::armed().filter(|p| p.is_isotope13()) else {
        return (Vec::new(), paths);
    };
    let floor = crate::analysis::SMALL_JOB_MAX_BYTES;
    let digests: Vec<Option<String>> = paths
        .par_iter()
        .map(|path| {
            let big = std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() >= floor);
            big.then(|| crate::digest::sha256_file(path).ok()).flatten()
        })
        .collect();
    let mut asked: Vec<String> = digests.iter().flatten().cloned().collect();
    asked.sort_unstable();
    asked.dedup();
    if asked.is_empty() {
        return (Vec::new(), paths);
    }
    let known = precheck.shas(&asked);
    let mut answered = Vec::new();
    let mut rest = Vec::with_capacity(paths.len());
    for (path, digest) in paths.into_iter().zip(digests) {
        // Cloned rather than removed: two paths may hold the same bytes.
        if let Some(sha) = digest
            && let Some(Standing::Adopt(verdict)) = known.get(&sha)
        {
            answered.push((path, sha, verdict.clone()));
        } else {
            rest.push(path);
        }
    }
    (answered, rest)
}

/// The verdict isotope13 holds for a fetched artifact's bytes, under `auto`.
pub(crate) fn known_bytes(bytes: &[u8]) -> Option<(String, Verdict)> {
    let precheck = crate::corpus_precheck::armed().filter(|p| p.is_isotope13())?;
    let sha = format!("{:x}", Sha256::digest(bytes));
    match precheck.standing(&sha) {
        Standing::Adopt(verdict) => Some((sha, verdict)),
        _ => None,
    }
}

/// The verdict isotope13 holds for a package coordinate, under `auto`, without
/// fetching it.
pub(crate) fn known_purl(purl: &str) -> Option<(String, Verdict)> {
    let precheck = crate::corpus_precheck::armed().filter(|p| p.is_isotope13())?;
    if !public_purl(purl) {
        tracing::debug!(
            event = "remote.withheld",
            kind = "purl",
            reason = "not_public",
            purl
        );
        return None;
    }
    // A verdict stored against a coordinate stands in for a fetch only when
    // the coordinate can name no other bytes: not a version-less or ranged
    // PURL (whatever is latest now), not a gem or a young PyPI release. The
    // rest are fetched, and asked about again by digest.
    if !crate::fetch::immutable_coordinate(purl, None) {
        tracing::debug!(
            event = "remote.withheld",
            kind = "purl",
            reason = "mutable",
            purl
        );
        return None;
    }
    let mut hits = precheck.purls(&[purl.to_string()]);
    let hit = hits.remove(purl)?;
    match hit.standing {
        Standing::Adopt(verdict) => Some((hit.sha, verdict)),
        _ => None,
    }
}

/// What to send `/v1/analyze` under `remote`.
pub(crate) enum Upload<'a> {
    /// A package isotope13 fetches itself.
    Purl(&'a str),
    /// A file's contents.
    File(&'a Path),
    /// Bytes already in hand (a fetched artifact).
    Bytes(Vec<u8>),
}

/// Have isotope13 analyze one artifact, and return its verdict and digest.
///
/// The answer streams: progress frames, then one line carrying `status`, which
/// is the answer. Anything but an `analyzed` answer is an error for this
/// artifact — under `remote` there is no local analysis to fall back to.
///
/// # Errors
/// No `remote` mode, a file over the upload cap, a transport failure, or an
/// answer that is not an analyzed verdict.
pub(crate) fn analyze(upload: Upload<'_>) -> Result<(String, Verdict)> {
    let Some((Analyze::Remote, Some(token))) = RESOLVED.get() else {
        anyhow::bail!("remote analysis was not enabled");
    };
    let http = http().context("no isotope13 HTTP client")?;
    let url = format!("{API_URL}/v1/analyze");
    // Held as `Bytes` so a retry resends the body without copying it.
    let (purl, body): (Option<&str>, Option<bytes::Bytes>) = match upload {
        Upload::Purl(purl) => (Some(purl), None),
        Upload::File(path) => {
            // The cap bounds the read itself: a file that grows after a size
            // check, or a device or FIFO reporting no length, cannot be
            // read past it.
            let mut bytes = Vec::new();
            std::fs::File::open(path)
                .and_then(|file| file.take(MAX_UPLOAD_BYTES + 1).read_to_end(&mut bytes))
                .with_context(|| format!("reading {}", path.display()))?;
            if bytes.len() as u64 > MAX_UPLOAD_BYTES {
                anyhow::bail!(
                    "{} is over isotope13's {MAX_UPLOAD_BYTES}-byte upload cap; scan it with --analyze=auto or --analyze=local",
                    path.display()
                );
            }
            (None, Some(bytes.into()))
        }
        Upload::Bytes(bytes) => {
            if bytes.len() as u64 > MAX_UPLOAD_BYTES {
                anyhow::bail!(
                    "artifact is {} bytes, over isotope13's {MAX_UPLOAD_BYTES}-byte upload cap",
                    bytes.len()
                );
            }
            (None, Some(bytes.into()))
        }
    };
    let started = std::time::Instant::now();
    // A permanent 4xx, or an answer that has begun streaming, is final.
    let resp = crate::retry::send("isotope13 analyze", ANALYZE_RETRY_BUDGET, |_| {
        let mut request = http
            .post(&url)
            .timeout(ANALYZE_TIMEOUT)
            .bearer_auth(token.secret());
        if let Some(purl) = purl {
            request = request.query(&[("purl", purl)]);
        }
        if let Some(body) = &body {
            request = request
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .body(body.clone());
        }
        request
    })
    .with_context(|| format!("POST {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        // Bounded and escaped: the body is the remote's, and this error is
        // printed to the operator's terminal.
        let mut body = String::new();
        let _ = std::io::Read::read_to_string(&mut resp.take(4096), &mut body);
        anyhow::bail!(
            "isotope13 refused the analysis ({status}): {:?}",
            body.trim()
        );
    }
    let answer = BufReader::new(resp)
        .lines()
        .map_while(std::result::Result::ok)
        .find(|line| {
            serde_json::from_str::<serde_json::Value>(line).is_ok_and(|v| v.get("status").is_some())
        })
        .context("isotope13's analysis stream ended without an answer")?;
    let (sha, verdict) =
        crate::corpus_precheck::analyzed_verdict(&answer).map_err(anyhow::Error::msg)?;
    tracing::info!(
        event = "remote.analyze",
        status = status.as_u16(),
        ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        %sha,
    );
    Ok((sha, verdict))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_mainstream_registry_coordinates_may_be_sent() {
        for purl in [
            "pkg:npm/left-pad@1.3.0",
            "pkg:npm/%40babel/core@7.24.0",
            "pkg:pypi/requests@2.31.0",
            "pkg:cargo/serde@1.0.0",
            "pkg:gem/rails@7.1.0",
            "pkg:nuget/Newtonsoft.Json@13.0.3",
            "pkg:golang/github.com/spf13/cobra@v1.8.0",
            "pkg:maven/org.apache.commons/commons-lang3@3.14.0",
            "pkg:composer/laravel/framework@11.0.0",
            "pkg:NPM/left-pad@1.3.0",
        ] {
            assert!(public_purl(purl), "{purl}");
        }
        for purl in [
            // A registry that is not the public default.
            "pkg:pypi/requests@2.31.0?repository_url=https://pypi.acme.internal/simple",
            "pkg:npm/left-pad@1.3.0?Repository_URL=https://npm.acme.internal",
            "pkg:maven/com.acme/core@1.0?repository_url=https://nexus.acme.internal/repo",
            // Not a registry, or one we do not know.
            "pkg:github/acme/private@v1",
            "pkg:generic/thing@1",
            "pkg:bitbucket/acme/thing@1",
            // Not a coordinate at all.
            "npm/left-pad@1.3.0",
            "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz",
        ] {
            assert!(!public_purl(purl), "{purl}");
        }
    }

    #[test]
    fn a_remote_level_grades_like_a_local_one() {
        assert_eq!(classification(Level::Manual, Some(25)), None);
        assert_eq!(
            classification(Level::Clean, Some(25)),
            Some(Classification::Benign)
        );
        assert_eq!(
            classification(Level::At(3), Some(25)),
            Some(Classification::Hostile)
        );
        assert_eq!(
            classification(Level::At(25), Some(25)),
            Some(Classification::Hostile)
        );
        assert_eq!(
            classification(Level::At(500), Some(25)),
            Some(Classification::Suspicious)
        );
        assert_eq!(
            classification(Level::At(5000), Some(25)),
            Some(Classification::Benign)
        );
        // The caller's operating level moves the grade, as it does locally.
        assert_eq!(
            classification(Level::At(500), Some(1000)),
            Some(Classification::Hostile)
        );
    }

    #[test]
    fn hashing_streams_the_file() {
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        std::fs::write(tmp.path(), b"abc").expect("write");
        assert_eq!(
            crate::digest::sha256_file(tmp.path()).expect("hash"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn nothing_is_asked_when_lookups_are_off() {
        let tmp = tempfile::tempdir().expect("tmp");
        let path = tmp.path().join("big.bin");
        std::fs::write(&path, vec![0u8; 2 << 20]).expect("write");
        let (answered, rest) = answer_known(vec![path.clone()]);
        assert!(answered.is_empty());
        assert_eq!(rest, vec![path]);
    }
}
