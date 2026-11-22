//! A finished scan of a file, kept so that scanning the same bytes again the
//! same way prints the same answer without doing the work.
//!
//! The analysis caches already make a repeat scan cheap to *compute*: cleave
//! replays the file's report, and every dependency's report is kept under its
//! content hash. What they do not keep is the result. A wheel with forty
//! fresh dependencies still spent ten seconds asking registries which
//! releases those are, merging their reports and rendering the card, to print
//! what it printed a minute earlier.
//!
//! An entry is named by the file's SHA-256 and a digest of everything else
//! what was printed depends on: the configuration, how it came to be scanned
//! (see [`Role`]), both terminals, and the environment variables the stack
//! reads. It lives in the analysis cache's namespace, so a rules, engine,
//! bloom or model change starts from nothing.
//!
//! The path the file is shown under is no part of that name. A file is its
//! bytes: a copy kept under another name is the same scan, printed under its
//! own name (see [`Replay::shown_as`]). Only a scan whose output spells the
//! name outside the fields that hold the path, as the text formats do before
//! every member, is named by its path as well, and answers for that path
//! alone.
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
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{ScanConfig, ScanResult};
use crate::OutputFormat;
use crate::analysis_cache::AnalysisCache;
use crate::model::Classification;

/// How long a scan that followed references, or deferred to a remote first,
/// answers from its replay: the time a resolved release set is trusted before
/// the registries are asked again. The same two hours fletch keeps a
/// versionless registry document, so a replay never outlives what a fresh
/// scan would read, nor expires before it could read anything newer.
const RESOLUTION_TTL: Duration = Duration::from_hours(2);

/// Environment variables whose names start with one of these can change an
/// analysis or its output, so all of them are part of the key.
const ENV_PREFIXES: [&str; 5] = ["SCAN_", "CLEAVE_", "ATOMDRIFT_", "FLETCH_", "STNG_"];

/// How a file came to be scanned, which decides what its scan prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Role {
    /// The one file named: a spinner, then its dependency tree and card.
    Alone,
    /// Named among other files: analyzed whatever the bloom filters say.
    Named,
    /// Found under a directory that was named, where a known-good file is
    /// passed over.
    Found,
}

/// A file hashed to look for its replay.
pub(super) struct Hashed {
    /// SHA-256 of the file.
    pub(super) sha256: String,
    /// Its replay, when one is kept and has not expired.
    pub(super) replay: Option<Replay>,
}

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
    /// The path the scan was shown under.
    label: String,
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
    /// The replay of `result`, shown as `label`, which left `notes` on
    /// stderr and wrote `output`.
    pub(super) fn new(
        result: &ScanResult,
        label: &str,
        notes: String,
        output: Option<String>,
        config: &ScanConfig,
    ) -> Self {
        Self {
            at: now(),
            expires: config.fetch_policy().enabled()
                || crate::remote::mode() != crate::remote::Uplink::Off,
            sha256: result.sha256.clone(),
            label: label.to_string(),
            class: result.classification,
            severe: result.has_severe_findings(),
            dependency_gaps: result.dependency_attention.gaps,
            flagged_candidates: result.dependency_attention.flagged_candidates,
            notes,
            output,
        }
    }

    /// The replay of Uplink's verdict on the bytes `sha256`, shown as `label`
    /// as `class`, which wrote `output`. The service may change its mind, so
    /// it expires as a followed scan does.
    pub(super) fn remote(
        sha256: &str,
        label: &str,
        class: Classification,
        output: Option<String>,
    ) -> Self {
        Self {
            at: now(),
            expires: true,
            sha256: sha256.to_string(),
            label: label.to_string(),
            class,
            severe: false,
            dependency_gaps: 0,
            flagged_candidates: 0,
            notes: String::new(),
            output,
        }
    }

    /// The file at `path`, hashed, with the replay of an earlier scan of
    /// its bytes as `role` under `config` if one is kept and has not expired.
    /// `None` when nothing is kept at all, or the file cannot be read.
    pub(super) fn load(path: &Path, role: Role, config: &ScanConfig) -> Option<Hashed> {
        AnalysisCache::shared()?;
        let sha256 = crate::digest::sha256_file(path).ok()?;
        let replay = Self::find(&sha256, &path.display().to_string(), role, config);
        Some(Hashed { sha256, replay })
    }

    /// The replay of an earlier scan of the bytes `sha256` as `role` under
    /// `config`, as it is shown for `label`, if one is kept and has not
    /// expired: a scan of the same bytes under any name when its output can
    /// be shown under another, else the scan under this very name.
    pub(super) fn find(sha256: &str, label: &str, role: Role, config: &ScanConfig) -> Option<Self> {
        let cache = AnalysisCache::shared()?;
        let kept = |named: Option<&str>| {
            cache
                .read::<Self>(&key(sha256, named, role, config))
                .filter(|replay| {
                    let fresh = replay.fresh(now());
                    tracing::debug!(path = label, shown_as = %replay.label, sha256, ?role, at = replay.at, fresh, "scan replay found");
                    fresh
                })
        };
        kept(None)
            .and_then(|replay| replay.shown_as(label, config.format()))
            .or_else(|| kept(Some(label)))
            .map(|replay| Self {
                sha256: sha256.to_string(),
                ..replay
            })
    }

    /// This replay as it is shown for the same bytes at `label`: itself for
    /// the path it was shown under, and for another path every mention of
    /// the first replaced. `None` when its output, written as `format`,
    /// cannot be shown under another name (see [`mentions`]), or `label` is
    /// not [`plain`].
    fn shown_as(mut self, label: &str, format: OutputFormat) -> Option<Self> {
        if self.label == label {
            return Some(self);
        }
        let mentions = self.mentions(format).filter(|_| plain(label))?;
        if let Some(output) = &mut self.output {
            // Last first, so the offsets before it still hold.
            for at in mentions.into_iter().rev() {
                output.replace_range(at..at + self.label.len(), label);
            }
        }
        self.label = label.to_string();
        Some(self)
    }

    /// Where this scan's output, written as `format`, names the path it was
    /// shown under. `None` when the output cannot be shown under another
    /// name: the file's name is spelled anywhere else in it or in the notes,
    /// or the path is not [`plain`].
    fn mentions(&self, format: OutputFormat) -> Option<Vec<usize>> {
        let name = Path::new(&self.label)
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| plain(&self.label) && !self.notes.contains(name))?;
        mentions(
            self.output.as_deref().unwrap_or_default(),
            &self.label,
            name,
            format,
        )
    }

    /// Each of `files` as [`Self::load`] finds it scanned as `role`, in their
    /// order. Hashes several files at a time. A file
    /// longer than `limit` bytes is not read when `limit` is not zero: cleave
    /// leaves such a file unread in a walk, and hashing gigabytes to find
    /// nothing kept for them would cost more than every replay saves.
    pub(super) fn load_each(
        files: &[PathBuf],
        role: Role,
        limit: u64,
        config: &ScanConfig,
    ) -> Vec<Option<Hashed>> {
        use rayon::prelude::*;
        files
            .par_iter()
            .map(|path| {
                let unread = limit > 0 && path.metadata().is_ok_and(|m| m.len() > limit);
                (!unread).then(|| Self::load(path, role, config)).flatten()
            })
            .collect()
    }

    /// Whether this replay still answers at `now` (seconds since the epoch):
    /// always when the scan read nothing but the file, otherwise while its
    /// resolution is trusted. A stamp from the future is a clock that moved,
    /// and is not.
    fn fresh(&self, now: u64) -> bool {
        now.checked_sub(self.at)
            .is_some_and(|age| !self.expires || age <= RESOLUTION_TTL.as_secs())
    }

    /// Keep this replay for the next scan of the same bytes as `role`:
    /// under any name when its output can be shown under another, else
    /// under the name it was shown as.
    pub(super) fn store(&self, role: Role, config: &ScanConfig) {
        if let Some(cache) = AnalysisCache::shared() {
            let named = self
                .mentions(config.format())
                .is_none()
                .then_some(self.label.as_str());
            cache.write(&key(&self.sha256, named, role, config), self);
        }
    }
}

/// The offset of each mention of the path `label`, whose file name is `name`,
/// in `output` written as `format`. `None` when `name` is spelled anywhere
/// else: output that cannot be shown under another name.
///
/// A renderer names the file by its path, whole. A decision carries it as
/// its `path` and at the head of each finding's `file`, and a JSON string
/// cannot hold either key's quotes unescaped, so those mentions are the
/// renderer's. Every other format names it once at most, in a card's head.
/// A name spelled outside those mentions is not told apart from a member, a
/// string or a finding that happens to spell it — the text formats spell it
/// before every member — and such output stays with its path.
fn mentions(output: &str, label: &str, name: &str, format: OutputFormat) -> Option<Vec<usize>> {
    let mut found: Vec<usize> = if format == OutputFormat::Json {
        ["\"path\":\"", "\"file\":\""]
            .iter()
            .flat_map(|key| {
                output
                    .match_indices(&format!("{key}{label}"))
                    .map(|(at, _)| at + key.len())
                    .collect::<Vec<_>>()
            })
            .collect()
    } else {
        output.match_indices(label).map(|(at, _)| at).collect()
    };
    found.sort_unstable();
    let whole = format == OutputFormat::Json || found.len() <= 1;
    (whole && output.matches(name).count() == found.len()).then_some(found)
}

/// Whether `label` is written the same in every output: a terminal shows it
/// unescaped, and so does a JSON string.
fn plain(label: &str) -> bool {
    !label.is_empty()
        && !label
            .chars()
            .any(|c| crate::output::tty_hostile(c) || matches!(c, '"' | '\\'))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// The cache key for a scan of the bytes `sha256` as `role` under `config`:
/// the hash, and a digest of every other input to the output. `named` is the
/// path of a scan that answers for that path alone.
fn key(sha256: &str, named: Option<&str>, role: Role, config: &ScanConfig) -> String {
    let mut digest = Sha256::new();
    for part in [
        config.fingerprint(),
        // A path cannot hold a NUL, so no path reads as "no path".
        named.map_or_else(|| "\0".to_string(), str::to_string),
        format!(
            "{role:?}|{:?}|{:?}|{}|{}|{}|{}|{}",
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
    let rest = crate::digest::hex(&digest.finalize()[..8]);
    format!("{sha256}-{rest}.replay")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay(expires: bool, at: u64) -> Replay {
        Replay {
            at,
            expires,
            sha256: "ab".repeat(32),
            label: String::new(),
            class: Classification::Hostile,
            severe: true,
            dependency_gaps: 3,
            flagged_candidates: 1,
            notes: "    \u{26a0} gap\n".to_string(),
            output: Some("card\n".to_string()),
        }
    }

    /// A scan that followed references answers for two hours and no longer; one
    /// that read only the file answers for as long as its namespace lives. A
    /// stamp ahead of the clock answers for neither.
    #[test]
    fn a_replay_expires_only_when_its_scan_followed_references() {
        let ttl = RESOLUTION_TTL.as_secs();
        assert!(replay(true, 1_000).fresh(1_000 + ttl));
        assert!(!replay(true, 1_000).fresh(1_001 + ttl));
        assert!(replay(false, 1_000).fresh(1_000 + 1_000 * ttl));
        assert!(!replay(true, 1_000).fresh(999));
        assert!(!replay(false, 1_000).fresh(999));
    }

    /// What is kept comes back as it was stored, under the hash it was
    /// stored by; a different format, role or file is a different entry. A
    /// copy under another name is the same entry.
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
            label,
            ..replay(true, now())
        };
        kept.store(Role::Alone, &terminal);

        let found = Replay::load(&path, Role::Alone, &terminal)
            .and_then(|hashed| hashed.replay)
            .expect("kept a moment ago");
        assert_eq!(found.sha256, sha256);
        assert_eq!(found.class, Classification::Hostile);
        assert!(found.severe);
        assert_eq!((found.dependency_gaps, found.flagged_candidates), (3, 1));
        assert_eq!(found.notes, kept.notes);
        assert_eq!(found.output, kept.output);

        let json = config(OutputFormat::Json);
        let load = |path, role, config| {
            let hashed = Replay::load(path, role, config).expect("hashed");
            assert_eq!(
                hashed.sha256,
                crate::digest::sha256_file(path).expect("hash")
            );
            hashed.replay
        };
        assert!(load(&path, Role::Alone, &json).is_none(), "another format");
        // A known-good file is passed over in a walk and analyzed when named,
        // and only the one file named draws a tree: three different outputs.
        assert!(
            load(&path, Role::Named, &terminal).is_none(),
            "among others"
        );
        assert!(load(&path, Role::Found, &terminal).is_none(), "in a walk");
        // The output does not name the file, so a copy prints the same.
        let copy = dir.path().join("copy.bin");
        std::fs::copy(&path, &copy).expect("copy");
        let copied = load(&copy, Role::Alone, &terminal).expect("the same bytes");
        assert_eq!(copied.output, kept.output);
        assert_eq!(copied.sha256, sha256);
        assert!(load(&copy, Role::Alone, &json).is_none(), "another format");
        std::fs::write(&path, b"other bytes").expect("rewrite");
        assert!(load(&path, Role::Alone, &terminal).is_none(), "other bytes");
    }

    /// A replay shown as `label` that left `notes` and wrote `output`.
    fn shown(label: &str, notes: &str, output: Option<&str>) -> Replay {
        Replay {
            label: label.to_string(),
            notes: notes.to_string(),
            output: output.map(str::to_string),
            ..replay(false, now())
        }
    }

    /// A copy under another name prints what the first file printed, with
    /// each mention of the first file's path replaced by its own: a card's
    /// head; a decision's `path` and the `file` of each of its findings, a
    /// member's included. Everything else is left as it was.
    #[test]
    fn a_copy_is_shown_the_first_scan_under_its_own_name() {
        use OutputFormat::{Json, Terminal};
        let (first, copy) = ("/dl/installer.dmg", "/dl/installer(1).dmg");
        let card = |label: &str| {
            format!(
                "\nHOSTILE[99%] {label} \u{b7} DMG \u{b7} 306.8MB\n abc123\n  drops installer(1).dmg.sig\n"
            )
        };
        let decision = |label: &str| {
            format!(
                "{{\"decision\":\"block\",\"purl\":null,\"path\":\"{label}\",\"findings\":[\
                 {{\"id\":\"a\",\"file\":\"{label}\",\"desc\":\"reads \\\"file\\\":\\\"x\"}},\
                 {{\"id\":\"b\",\"file\":\"{label}!!App.app/main.js\"}}]}}\n"
            )
        };
        for (format, render) in [
            (Terminal, &card as &dyn Fn(&str) -> String),
            (Json, &decision),
        ] {
            let kept = shown(first, "    \u{26a0} gap\n", Some(&render(first)));
            let copied = kept.shown_as(copy, format).expect("the same bytes");
            assert_eq!(copied.output.as_deref(), Some(render(copy).as_str()));
            assert_eq!(copied.notes, "    \u{26a0} gap\n");
            assert_eq!(copied.label, copy);
            assert_eq!(copied.class, Classification::Hostile);
        }
        // A copy's own replay answers for the first file in turn.
        let back = shown(copy, "", Some(&decision(copy))).shown_as(first, Json);
        assert_eq!(back.and_then(|back| back.output), Some(decision(first)));
        // A relative path is its file name, and is replaced whole.
        let kept = shown("a.sh", "", Some("SUSPICIOUS a.sh \u{b7} 1KB\n"));
        assert_eq!(
            kept.shown_as("copy of a.sh", Terminal)
                .and_then(|copied| copied.output),
            Some("SUSPICIOUS copy of a.sh \u{b7} 1KB\n".to_string())
        );
    }

    /// Output that does not name the file is the same for every copy: a
    /// benign file a terminal scan shows nothing of, the full document whose
    /// paths start at the archive's root.
    #[test]
    fn output_that_names_no_file_is_shown_as_it_is() {
        let document = "{\"v\":8}\n{\"path\":\"App.app/Info.plist\"}\n";
        for (format, output) in [
            (OutputFormat::Terminal, None),
            (OutputFormat::Tiny, Some("")),
            (OutputFormat::JsonFull, Some(document)),
        ] {
            let kept = shown("/dl/installer.dmg", "", output);
            let copied = kept
                .shown_as("/elsewhere/renamed.bin", format)
                .expect("the same bytes");
            assert_eq!(copied.output.as_deref(), output);
        }
    }

    /// Output that spells the file's name anywhere but where the path is
    /// held answers for its own path alone: a mention that might be a
    /// member's or a finding's is never rewritten, and never left naming
    /// another file.
    #[test]
    fn output_that_spells_the_name_elsewhere_stays_with_its_path() {
        use OutputFormat::{Json, Terminal, Tiny};
        let (first, copy) = ("/dl/installer.dmg", "/dl/installer(1).dmg");
        let bound = [
            // The text formats spell it before every member.
            (
                Tiny,
                shown(
                    first,
                    "",
                    Some("installer.dmg\tdmg\n  installer.dmg/App.app/main.js\n"),
                ),
            ),
            // The path twice: one of them is not the renderer's.
            (
                Terminal,
                shown(
                    first,
                    "",
                    Some("HOSTILE /dl/installer.dmg\n  runs /dl/installer.dmg\n"),
                ),
            ),
            // The name in a finding beside the head.
            (
                Terminal,
                shown(
                    first,
                    "",
                    Some("HOSTILE /dl/installer.dmg\n  renames installer.dmg\n"),
                ),
            ),
            // The name without the path: not the path's mention.
            (Terminal, shown(first, "", Some("HOSTILE installer.dmg\n"))),
            // The path in a decision's text, not in a field that holds it.
            (
                Json,
                shown(
                    first,
                    "",
                    Some(
                        "{\"path\":\"/dl/installer.dmg\",\"reason\":\"runs /dl/installer.dmg\"}\n",
                    ),
                ),
            ),
            (
                Json,
                shown(
                    first,
                    "",
                    Some(
                        "{\"path\":\"/dl/installer.dmg\",\"desc\":\"\\\"file\\\":\\\"/dl/installer.dmg\"}\n",
                    ),
                ),
            ),
            // The name in what the fetch phase left on stderr.
            (
                Terminal,
                shown(first, "  fetched for installer.dmg\n", None),
            ),
            // A path a terminal or a JSON string writes otherwise.
            (
                Terminal,
                shown(
                    "/dl/a\u{1b}[2K.dmg",
                    "",
                    Some("HOSTILE /dl/a\\u{1b}[2K.dmg\n"),
                ),
            ),
            (Json, shown("/dl/a\"b.dmg", "", None)),
            (Terminal, shown("", "", None)),
        ];
        for (format, kept) in bound {
            assert!(kept.mentions(format).is_none(), "{:?}", kept.output);
            let (label, output) = (kept.label.clone(), kept.output.clone());
            assert!(kept.shown_as(copy, format).is_none());
            let own = shown(&label, "", output.as_deref());
            let own = own.shown_as(&label, format).expect("its own path");
            assert_eq!(own.output, output);
        }
        // A copy whose own name is written otherwise is scanned for itself.
        let kept = shown(first, "", Some("HOSTILE /dl/installer.dmg\n"));
        assert!(kept.mentions(Terminal).is_some());
        assert!(kept.shown_as("/dl/a\u{202e}gmd.exe", Terminal).is_none());
    }

    /// A scan bound to its path is kept beside the scan of the same bytes
    /// that any name can use, and each path finds its own.
    #[test]
    fn a_scan_bound_to_its_path_is_kept_for_that_path_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ScanConfig::new(dir.path(), OutputFormat::Tiny, None).expect("config");
        let paths: Vec<PathBuf> = ["one.sh", "two.sh"]
            .iter()
            .map(|name| dir.path().join(name))
            .collect();
        for path in &paths {
            std::fs::write(path, b"echo the same bytes").expect("write");
        }
        let sha256 = crate::digest::sha256_file(&paths[0]).expect("hash");
        let tiny = |path: &Path| {
            let name = path.file_name().and_then(|n| n.to_str()).expect("name");
            format!("{name}\tshell\n  {name}##base64\n")
        };
        let find = |path: &Path| {
            Replay::find(&sha256, &path.display().to_string(), Role::Found, &config)
                .and_then(|replay| replay.output)
        };
        let keep = |path: &Path| {
            Replay {
                sha256: sha256.clone(),
                ..shown(&path.display().to_string(), "", Some(&tiny(path)))
            }
            .store(Role::Found, &config);
        };
        keep(&paths[0]);
        assert_eq!(find(&paths[0]), Some(tiny(&paths[0])));
        assert_eq!(find(&paths[1]), None, "spelled before every member");
        keep(&paths[1]);
        assert_eq!(find(&paths[0]), Some(tiny(&paths[0])), "still its own");
        assert_eq!(find(&paths[1]), Some(tiny(&paths[1])));
    }

    /// Uplink's answer for a file is kept like a local scan: the next scan of
    /// the same bytes replays it, under its own name, asking nobody, until it
    /// expires as a followed scan does.
    #[test]
    fn an_uplink_answer_is_replayed_by_the_next_scan() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ScanConfig::new(dir.path(), OutputFormat::Json, None).expect("config");
        let sha256 = "e".repeat(64);
        let shown = |label: &str| format!("{{\"path\":\"{label}\",\"source\":\"uplink\"}}\n");
        let first = dir.path().join("big.dmg").display().to_string();
        let copy = dir.path().join("big (1).dmg").display().to_string();
        Replay::remote(
            &sha256,
            &first,
            Classification::Hostile,
            Some(shown(&first)),
        )
        .store(Role::Found, &config);
        for label in [&first, &copy] {
            let replay = Replay::find(&sha256, label, Role::Found, &config).expect("kept");
            assert_eq!(replay.class, Classification::Hostile);
            assert_eq!(replay.output, Some(shown(label)), "{label}");
            assert!(replay.notes.is_empty());
        }
        let kept = Replay::remote(&sha256, &first, Classification::Benign, None);
        assert!(kept.expires, "the service may change its mind");
        assert!(kept.fresh(kept.at + RESOLUTION_TTL.as_secs()));
        assert!(!kept.fresh(kept.at + RESOLUTION_TTL.as_secs() + 1));
    }

    /// A batch finds each file's replay in the files' own order, and reads no
    /// file over the size limit a walk is held to.
    #[test]
    fn a_batch_finds_each_replay_in_order_and_skips_oversized_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ScanConfig::new(dir.path(), OutputFormat::Terminal, None).expect("config");
        let files: Vec<PathBuf> = ["kept.bin", "never-scanned.bin", "kept-too.bin"]
            .iter()
            .map(|name| dir.path().join(name))
            .collect();
        for (path, bytes) in files
            .iter()
            .zip([&b"first bytes"[..], b"second", b"third bytes"])
        {
            std::fs::write(path, bytes).expect("write");
        }
        for path in [&files[0], &files[2]] {
            let kept = Replay {
                sha256: crate::digest::sha256_file(path).expect("hash"),
                label: path.display().to_string(),
                notes: path.display().to_string(),
                ..replay(false, now())
            };
            kept.store(Role::Found, &config);
        }

        let notes = |limit| -> Vec<Option<String>> {
            Replay::load_each(&files, Role::Found, limit, &config)
                .into_iter()
                .map(|found| found?.replay.map(|replay| replay.notes))
                .collect()
        };
        let shown = |path: &PathBuf| Some(path.display().to_string());
        assert_eq!(notes(0), [shown(&files[0]), None, shown(&files[2])]);
        assert_eq!(notes(1 << 20), notes(0), "a limit none of them reaches");
        // "first bytes" and "third bytes" are 11 bytes long.
        assert_eq!(notes(10), [None, None, None]);
        assert_eq!(notes(11), notes(0), "a file at the limit is read");
        let named = Replay::load_each(&files, Role::Named, 0, &config);
        assert!(
            named.iter().flatten().all(|hashed| hashed.replay.is_none()),
            "kept as found in a walk"
        );
        // Hashed all the same, so files of the same bytes can be told.
        assert_eq!(named.iter().flatten().count(), 3);
    }

    /// The key names the file by its hash and nothing else in the clear: the
    /// rest, archive passwords included, is behind a digest.
    #[test]
    fn a_key_shows_the_hash_and_hides_the_configuration() {
        let plain = || ScanConfig::new("models", OutputFormat::Terminal, None).expect("config");
        let config = plain().with_zip_passwords(vec!["hunter2-not-in-the-key".to_string()]);
        let sha256 = "cd".repeat(32);
        let name = key(&sha256, Some("a/label"), Role::Alone, &config);
        let (hash, rest) = name.split_once('-').expect("hash, then digest");
        assert_eq!(hash, sha256);
        let digest = rest.strip_suffix(".replay").expect("suffix");
        assert_eq!(digest.len(), 16);
        assert!(digest.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(name, key(&sha256, Some("a/label"), Role::Alone, &plain()));
        assert_ne!(name, key(&sha256, Some("b/label"), Role::Alone, &config));
        assert_ne!(name, key(&sha256, None, Role::Alone, &config));
        assert_ne!(name, key(&sha256, Some("a/label"), Role::Found, &config));
    }
}
