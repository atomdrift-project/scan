//! Atomdrift Uplink: the service a scan may ask before analyzing.
//!
//! `--uplink=true` asks Uplink for verdicts it already holds and analyzes
//! locally whatever it does not know. It is the default, except for a run
//! whose rules are a git checkout: a rule author's edits are not in the
//! service's verdicts, so that run asks only when told to. The asking goes through
//! [`crate::corpus_precheck`] — the same batched `/v1/lookup` negotiation, the
//! same circuit breaker, the same fail-open rule hopper lookups use — armed
//! against Uplink. `false` never asks, and is the reproducible mode. `always`
//! sends every artifact to `/v1/analyze`, uploading local files' contents,
//! and needs a subscription.
//!
//! A scan with no subscription asks `api.atomdrift.org` and sends no
//! credential. A subscriber's scan asks `api.atomdrift.com`, the one host its
//! token is sent to.
//!
//! What may leave the machine is decided here and nowhere else:
//!
//! - the SHA-256 of a local file of at least
//!   [`crate::analysis::SMALL_JOB_MAX_BYTES`] (1 MiB) — below that, analyzing
//!   is cheaper than asking, and such files are mostly source nobody else has;
//! - the SHA-256 of a fetched artifact;
//! - a PURL only when `public_purl` says it names a package on a mainstream
//!   public registry.
//!
//! Never a URL, a path or a file name, and never a file's contents except
//! under `always`. A remote verdict replaces local analysis and is reported
//! as a `"source": "remote"` record, the way a bloom decision is reported as
//! `"source": "bloom"`.

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

/// Uplink's API for a subscriber. Fixed: an override would let any step of a
/// CI job send the subscription token, and every file it is shown, somewhere
/// else.
pub(crate) const API_URL: &str = "https://api.atomdrift.com";

/// Uplink's API for everyone else: lookups, with no credential.
pub(crate) const OPEN_API_URL: &str = "https://api.atomdrift.org";

/// The largest body Uplink's `/v1/analyze` accepts.
const MAX_UPLOAD_BYTES: u64 = 16 * 1024 * 1024;
/// Connect budget: an unreachable API costs one of these per run, then the
/// breaker opens.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(4);
/// Whole-request ceiling for a lookup.
pub(crate) const LOOKUP_TIMEOUT: Duration = Duration::from_secs(15);
/// Whole-request ceiling for a remote analysis, which streams progress.
const ANALYZE_TIMEOUT: Duration = Duration::from_mins(30);
/// How long a remote analysis keeps retrying transient failures before the
/// artifact is reported as failed. Under `always` there is no local analysis
/// to fall back to, so it is the longest any call waits.
const ANALYZE_RETRY_BUDGET: Duration = Duration::from_secs(120);

/// `--uplink`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Uplink {
    /// Ask Atomdrift Uplink about large files and dependencies it may already
    /// have judged; analyze the rest here.
    #[default]
    #[value(name = "true", alias = "1")]
    Ask,
    /// Never ask: analyze everything here.
    #[value(name = "false", alias = "0")]
    Off,
    /// Send everything to Atomdrift Uplink for analysis.
    Always,
}

/// The mode this run resolved to, and the subscriber's token when it has one.
///
/// Process-wide: [`resolve`] sets it once, before any scan starts, and the
/// scan paths read it through [`mode`] and [`analyze`] rather than being
/// handed it.
static RESOLVED: OnceLock<(Uplink, Option<cleave::feed::Token>)> = OnceLock::new();

/// What a run that did not say means by `--uplink`, given the rules
/// directory it analyzes with: ask, unless the rules are a git checkout (or a
/// link to one). The service judged under published rules, so its verdicts
/// would hide whatever the checkout changes.
fn unsaid(rules: Option<&Path>) -> Uplink {
    if rules.is_some_and(|dir| dir.join(".git").exists()) {
        Uplink::Off
    } else {
        Uplink::Ask
    }
}

/// Resolve `--uplink` against the rules in use and whether this run is on a
/// paid plan — a paid rule feed in use, which a refused token has already
/// fallen back from — once, before any scan starts.
///
/// `true` arms the Uplink lookups: with the subscription token on a paid run,
/// with no credential otherwise. `always` holds the token for uploads. A
/// missing token is asked for here, on a terminal, and checked against the
/// rules service before it is used. A run that did not say asks unless
/// `rules` is a git checkout.
///
/// # Errors
/// `always` on an open-source run, or a paid run with no usable token.
pub fn resolve(said: Option<Uplink>, paid: bool, rules: Option<&Path>) -> Result<Uplink> {
    let uplink = said.unwrap_or_else(|| unsaid(rules));
    let token = match (uplink, paid) {
        (Uplink::Always, false) => anyhow::bail!(
            "--uplink=always uploads to Atomdrift Uplink, which needs a paid plan: switch to a paid rule feed (`atomscan feed supporter`), which connects this computer"
        ),
        (Uplink::Off, _) | (Uplink::Ask, false) => None,
        (Uplink::Ask | Uplink::Always, true) => Some(token()?),
    };
    if uplink == Uplink::Ask {
        crate::corpus_precheck::configure_uplink(token.clone());
    }
    tracing::info!(
        event = "remote.mode",
        ?uplink,
        said = said.is_some(),
        paid,
        rules = %rules.map(Path::display).map(|d| d.to_string()).unwrap_or_default(),
    );
    // Once per process. A second resolve keeps the first answer, which every
    // scan already started has acted on.
    if RESOLVED.set((uplink, token)).is_err() {
        tracing::warn!(
            kept = ?mode(),
            ignored = ?uplink,
            "--uplink resolved twice in one process; keeping the first"
        );
    }
    Ok(uplink)
}

/// The resolved mode; [`Uplink::Off`] until [`resolve`] runs.
pub(crate) fn mode() -> Uplink {
    RESOLVED.get().map_or(Uplink::Off, |(mode, _)| *mode)
}

/// The subscription token, asking for one (and saving it, once the rules
/// channel has honoured it) when there is none.
fn token() -> Result<cleave::feed::Token> {
    if let Some((token, _)) = cleave::feed::token()? {
        return Ok(token);
    }
    cleave::rule_update::connect()?;
    cleave::feed::token()?
        .map(|(token, _)| token)
        .context("Uplink token was accepted but could not be read back")
}

/// The process's client for Uplink: TLS only, and no redirects, so a
/// request, and the token it may carry, reaches no host but the one named.
pub(crate) fn http() -> Option<&'static reqwest::blocking::Client> {
    static HTTP: OnceLock<Option<reqwest::blocking::Client>> = OnceLock::new();
    HTTP.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .https_only(true)
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_TIMEOUT)
            .user_agent(concat!("atomdrift-scan/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| tracing::error!(error = %e, "cannot build the Uplink HTTP client; remote analysis disabled"))
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
pub(crate) fn public_purl(purl: &str) -> bool {
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
        ["repository_url=", "download_url=", "vcs_url="]
            .iter()
            .any(|key| {
                q.get(..key.len())
                    .is_some_and(|k| k.eq_ignore_ascii_case(key))
            })
    }) {
        return false;
    }
    MAINSTREAM.iter().any(|m| m.eq_ignore_ascii_case(kind))
}

/// How a remote level grades at this run's operating level — the same rule a
/// local verdict is graded by. `None` for a verdict with no level.
#[must_use]
pub(crate) fn classification(fires_at: Level, level: Option<u16>) -> Option<Classification> {
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

/// The local files Uplink already judges, by path, and the files left to
/// analyze here, in their original order.
///
/// Only files of at least [`crate::analysis::SMALL_JOB_MAX_BYTES`] are asked
/// about, and, when `limit` is not zero, of at most `limit` bytes: a file the
/// scan will not analyze is not worth a question, and neither is one
/// `passed_over` says this machine already vouches for (by path and digest).
/// A digest in `known` is used as it is; any other is hashed by streaming the
/// file — bounded memory, never a second resident copy — and the digests go
/// out in batches. A file
/// whose digest comes back with a standing verdict is answered; everything
/// else, including every file when lookups are off or the API is
/// unreachable, is analyzed.
pub(crate) fn answer_known(
    paths: Vec<PathBuf>,
    known: &std::collections::HashMap<PathBuf, String>,
    limit: u64,
    passed_over: impl Fn(&Path, &str) -> bool + Sync,
) -> (Vec<(PathBuf, String, Verdict)>, Vec<PathBuf>) {
    let Some(precheck) = crate::corpus_precheck::armed().filter(|p| p.is_uplink()) else {
        return (Vec::new(), paths);
    };
    let floor = crate::analysis::SMALL_JOB_MAX_BYTES;
    let asked = |len: u64| len >= floor && (limit == 0 || len <= limit);
    let digests: Vec<Option<String>> = paths
        .par_iter()
        .map(|path| {
            let size = std::fs::metadata(path)
                .ok()
                .filter(std::fs::Metadata::is_file)?
                .len();
            if !asked(size) {
                return None;
            }
            known
                .get(path)
                .cloned()
                .or_else(|| crate::digest::sha256_file(path).ok())
                .filter(|sha| !passed_over(path, sha))
        })
        .collect();
    let mut asked: Vec<String> = digests.iter().flatten().cloned().collect();
    asked.sort_unstable();
    asked.dedup();
    if asked.is_empty() {
        return (Vec::new(), paths);
    }
    // What Uplink did not know a little while ago it is not asked again: a
    // scan whose every file is saved, judged or unknown makes no request.
    let cache = crate::analysis_cache::AnalysisCache::shared();
    let recently_unknown = |sha: &str| {
        cache
            .and_then(|cache| cache.read::<Unknown>(&unknown_key(sha)))
            .is_some_and(|unknown| unknown.fresh(crate::civil::now_secs()))
    };
    asked.retain(|sha| !recently_unknown(sha));
    let (known, unknown) = if asked.is_empty() {
        Default::default()
    } else {
        precheck.shas(&asked)
    };
    if let Some(cache) = cache {
        let now = crate::civil::now_secs();
        for sha in &unknown {
            cache.write(&unknown_key(sha), &Unknown { at: now });
        }
    }
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

/// How long Uplink not knowing a digest is believed: as long as a saved result
/// that rests on Uplink, so a repeat scan asks no more than it replays.
const UNKNOWN_TTL_SECS: u64 = 2 * 3600;

/// That Uplink, asked about some bytes at `at` (Unix seconds), knew nothing
/// of them.
#[derive(serde::Serialize, serde::Deserialize)]
struct Unknown {
    at: u64,
}

impl Unknown {
    /// Whether this is still believed at `now`. A stamp from the future is a
    /// clock that moved, and is not.
    fn fresh(&self, now: u64) -> bool {
        now.checked_sub(self.at)
            .is_some_and(|age| age <= UNKNOWN_TTL_SECS)
    }
}

/// The cache entry recording that Uplink did not know `sha256`.
fn unknown_key(sha256: &str) -> String {
    format!("{sha256}-uplink-unknown")
}

/// The verdict Uplink holds for a fetched artifact's bytes, when asking.
pub(crate) fn known_bytes(bytes: &[u8]) -> Option<(String, Verdict)> {
    let precheck = crate::corpus_precheck::armed().filter(|p| p.is_uplink())?;
    let sha = crate::digest::hex(&Sha256::digest(bytes));
    match precheck.standing(&sha) {
        Standing::Adopt(verdict) => Some((sha, verdict)),
        _ => None,
    }
}

/// The verdict Uplink holds for a package coordinate, when asking, without
/// fetching it.
pub(crate) fn known_purl(purl: &str) -> Option<(String, Verdict)> {
    let precheck = crate::corpus_precheck::armed().filter(|p| p.is_uplink())?;
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

/// What to send `/v1/analyze` under `always`.
pub(crate) enum Upload<'a> {
    /// A package Uplink fetches itself.
    Purl(&'a str),
    /// A file's contents.
    File(&'a Path),
    /// Bytes already in hand (a fetched artifact).
    Bytes(Vec<u8>),
}

/// Have Uplink analyze one artifact, and return its verdict and digest.
///
/// The answer streams: progress frames, then one line carrying `status`, which
/// is the answer. Anything but an `analyzed` answer is an error for this
/// artifact — under `always` there is no local analysis to fall back to — and
/// so is an answer about bytes other than the ones uploaded.
///
/// # Errors
/// Not under `always`, a file over the upload cap, a transport failure, or an
/// answer that is not an analyzed verdict.
pub(crate) fn analyze(upload: Upload<'_>) -> Result<(String, Verdict)> {
    let Some((Uplink::Always, Some(token))) = RESOLVED.get() else {
        anyhow::bail!("remote analysis was not enabled");
    };
    let http = http().context("no Uplink HTTP client")?;
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
                    "{} is over Uplink's {MAX_UPLOAD_BYTES}-byte upload cap; scan it with --uplink=true or --uplink=false",
                    path.display()
                );
            }
            (None, Some(bytes.into()))
        }
        Upload::Bytes(bytes) => {
            if bytes.len() as u64 > MAX_UPLOAD_BYTES {
                anyhow::bail!(
                    "artifact is {} bytes, over Uplink's {MAX_UPLOAD_BYTES}-byte upload cap",
                    bytes.len()
                );
            }
            (None, Some(bytes.into()))
        }
    };
    let started = std::time::Instant::now();
    // A permanent 4xx, or an answer that has begun streaming, is final.
    let resp = crate::retry::send("Uplink analyze", ANALYZE_RETRY_BUDGET, |_| {
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
        let _ = resp.take(4096).read_to_string(&mut body);
        anyhow::bail!("Uplink refused the analysis ({status}): {:?}", body.trim());
    }
    let answer = BufReader::new(resp)
        .lines()
        .map_while(std::result::Result::ok)
        .find(|line| {
            serde_json::from_str::<serde_json::Value>(line).is_ok_and(|v| v.get("status").is_some())
        })
        .context("Uplink's analysis stream ended without an answer")?;
    let (sha, verdict) = crate::corpus_precheck::analyzed_verdict(&answer)?;
    if let Some(body) = &body {
        let sent = crate::digest::hex(&Sha256::digest(body));
        if sent != sha {
            anyhow::bail!("Uplink answered for {sha}, but the upload was {sent}");
        }
    }
    tracing::info!(
        event = "remote.analyze",
        status = status.as_u16(),
        ms = crate::duration_ms(started.elapsed()),
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

    /// `--uplink` and `ATOMDRIFT_UPLINK` read the same words, asking is the
    /// default, and the modes the flag replaced are not accepted.
    #[test]
    fn the_flag_reads_true_false_and_always() {
        use clap::ValueEnum as _;
        assert_eq!(Uplink::default(), Uplink::Ask);
        for (text, mode) in [
            ("true", Uplink::Ask),
            ("1", Uplink::Ask),
            ("false", Uplink::Off),
            ("0", Uplink::Off),
            ("always", Uplink::Always),
        ] {
            assert_eq!(Uplink::from_str(text, false).ok(), Some(mode), "{text}");
        }
        for text in ["auto", "local", "remote", "ask", "off", ""] {
            assert!(Uplink::from_str(text, false).is_err(), "{text}");
        }
    }

    /// A run that did not say asks Uplink, unless its rules are a git
    /// checkout, a worktree, or a link to one: the service's verdicts know
    /// nothing of local rule edits.
    #[test]
    fn a_run_on_checked_out_rules_does_not_ask_unless_told() {
        let tmp = tempfile::tempdir().expect("tmp");
        let installed = tmp.path().join("installed");
        let checkout = tmp.path().join("checkout");
        let worktree = tmp.path().join("worktree");
        for dir in [&installed, &worktree] {
            std::fs::create_dir_all(dir).expect("mkdir");
        }
        std::fs::create_dir_all(checkout.join(".git")).expect("mkdir");
        std::fs::write(worktree.join(".git"), "gitdir: elsewhere\n").expect("write");

        assert_eq!(unsaid(None), Uplink::Ask);
        assert_eq!(unsaid(Some(&installed)), Uplink::Ask);
        assert_eq!(unsaid(Some(&checkout)), Uplink::Off);
        assert_eq!(unsaid(Some(&worktree)), Uplink::Off);
        #[cfg(unix)]
        {
            let link = tmp.path().join("link");
            std::os::unix::fs::symlink(&checkout, &link).expect("symlink");
            assert_eq!(unsaid(Some(&link)), Uplink::Off);
        }
    }

    /// Uplink not knowing some bytes is believed for as long as a saved
    /// result resting on it, and not from a clock that moved back.
    #[test]
    fn an_unknown_answer_is_believed_for_two_hours() {
        let at = 1_000_000;
        let unknown = Unknown { at };
        assert!(unknown.fresh(at));
        assert!(unknown.fresh(at + 2 * 3600));
        assert!(!unknown.fresh(at + 2 * 3600 + 1));
        assert!(!unknown.fresh(at - 1), "from the future");
        let sha = "f".repeat(64);
        assert!(unknown_key(&sha).starts_with(&sha), "{}", unknown_key(&sha));
        assert_ne!(unknown_key(&sha), sha);
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
        let (answered, rest) =
            answer_known(vec![path.clone()], &Default::default(), 0, |_, _| false);
        assert!(answered.is_empty());
        assert_eq!(rest, vec![path]);
    }
}
