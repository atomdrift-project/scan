//! Handing a scan result and the artifact behind it to hopper.

use std::path::Path;
use std::sync::OnceLock;

use fletch::fetch::FetchRecord;

use super::{DepResult, HopperRoute, ScanResult, ScanResultEnvelope, now_rfc3339};
use crate::provenance::{RegistryProvenance, Upload};
use crate::upload::{ArtifactBytes, UploadArtifact, Uploader};

/// Where a scanned artifact came from: its path (for a fetched root, its
/// display label) and the provenance that travels with it to hopper.
#[derive(Clone, Copy)]
pub(crate) struct Origin<'a> {
    pub(crate) path: &'a Path,
    /// Registry metadata for the artifact itself, when a registry or a
    /// collector supplied it.
    pub(crate) registry: Option<&'a RegistryProvenance>,
    /// How a fetched root (`scan url|purl`) was acquired.
    pub(crate) fetch: Option<&'a FetchRecord>,
}

impl<'a> Origin<'a> {
    /// A file on disk, with no provenance beyond what may sit beside it.
    pub(crate) const fn local(path: &'a Path) -> Self {
        Self {
            path,
            registry: None,
            fetch: None,
        }
    }
}

/// The collector name recorded on every artifact this process files with
/// hopper, so its provenance reads the same whichever path ingested a sample.
pub(crate) fn upload_collector() -> &'static str {
    static COLLECTOR: OnceLock<String> = OnceLock::new();
    COLLECTOR.get_or_init(|| format!("scan+{}", crate::upload::default_worker_name()))
}

/// Renew one scan result on hopper: ensure hopper has the scanned file and any
/// fetched dependency archives (with provenance), then renew the verdict. Used
/// by both the CLI `--hopper` path and the serve-mode `--hopper` upload. The
/// artifacts are queued before the result so a never-seen top-level file's row
/// exists before its verdict lands. Blocking (reads sidecars from disk); callers
/// on an async runtime must run it off the executor.
pub(crate) fn upload_scan_result(
    uploader: &Uploader,
    origin: Origin<'_>,
    sha256: String,
    size_bytes: u64,
    dependency_results: Vec<DepResult>,
    envelope: ScanResultEnvelope,
) {
    uploader.submit_artifacts(collect_upload_artifacts(
        origin.path,
        &sha256,
        size_bytes,
        upload_collector(),
        origin.registry,
        origin.fetch,
    ));
    // Each fetched dependency, mirrored into hopper as its own sample: bytes (only
    // if missing) + provenance + the verdict scan computed for it. Queued before
    // the root verdict so a dependency's row exists before its own verdict lands.
    if !dependency_results.is_empty() {
        uploader.submit_dependencies(
            dependency_results,
            envelope.ml.version.clone(),
            envelope.ml.analyzed_at.clone(),
        );
    }
    // `scan purl` names a package outright and a registry URL still identifies
    // one; a local path or an arbitrary URL identifies nothing, and the
    // uploader names those by digest alone.
    uploader.submit(sha256, artifact_purl(origin.fetch), envelope);
}

/// Renew a CLI result on hopper, routed by [`HopperRoute`] exactly as the
/// server routes its own. `fallback_purl` marks a registry-metadata document
/// standing in for a package whose artifact could not be fetched.
pub(super) fn renew(
    uploader: &Uploader,
    hopper_url: &str,
    origin: Origin<'_>,
    fallback_purl: Option<&str>,
    mut result: ScanResult,
) {
    let route = fallback_purl.map_or(HopperRoute::Normal, |purl| {
        registry_fallback_route(uploader, hopper_url, purl, origin)
    });
    match route {
        HopperRoute::Normal => {
            let sha256 = std::mem::take(&mut result.sha256);
            let size = result.size_bytes;
            let dependencies = std::mem::take(&mut result.dependency_results);
            upload_scan_result(
                uploader,
                origin,
                sha256,
                size,
                dependencies,
                result.into_envelope(),
            );
        }
        HopperRoute::Redirect(sha256) => {
            uploader.submit(
                sha256,
                fallback_purl.map(str::to_string),
                result.into_envelope(),
            );
        }
        HopperRoute::Suppress => {}
    }
}

/// Where the verdict on a registry-metadata stand-in for `purl` belongs.
///
/// The stand-in is the registry's JSON record, not an artifact, and it hashes
/// differently on every fetch: posted under its own sha256 it would mint hopper
/// a fresh, never-deduplicating row each time. So it backs onto the real
/// content hopper already holds for `purl` — refreshing that row's provenance —
/// or, when hopper holds none, goes nowhere.
fn registry_fallback_route(
    uploader: &Uploader,
    hopper_url: &str,
    purl: &str,
    origin: Origin<'_>,
) -> HopperRoute {
    let Some(real_sha) = crate::upload::hopper_http()
        .and_then(|http| crate::upload::known_sha_for_purl(http, hopper_url, purl))
    else {
        return HopperRoute::Suppress;
    };
    let name = origin
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
    if let Some(artifact) = registry_fallback_artifact(name, &real_sha, purl, origin.registry) {
        uploader.submit_artifacts(vec![artifact]);
    }
    HopperRoute::Redirect(real_sha)
}

/// The provenance-only artifact a registry-metadata stand-in backfills onto
/// the real content hopper already holds for `purl` under `real_sha`; `None`
/// (logged) when its sidecar cannot be built.
pub(crate) fn registry_fallback_artifact(
    name: &str,
    real_sha: &str,
    purl: &str,
    provenance: Option<&RegistryProvenance>,
) -> Option<UploadArtifact> {
    let now = now_rfc3339();
    let upload = Upload {
        filename: name,
        sha256: real_sha,
        size_bytes: 0,
        collector: upload_collector(),
        at: &now,
        url: "",
        purl: (!purl.is_empty()).then_some(purl),
    };
    let sidecar = match provenance {
        Some(provenance) => upload.sidecar_from_provenance(provenance),
        None => upload.sidecar(None, &[]),
    };
    let sidecar = sidecar
        .inspect_err(|error| {
            tracing::error!(sha256 = real_sha, %error, "registry fallback: sidecar not built; not uploading");
        })
        .ok()?;
    Some(UploadArtifact {
        sha256: real_sha.to_string(),
        size: 0,
        filename: name.to_string(),
        bytes: ArtifactBytes::None,
        sidecar,
        backfill: true,
    })
}

/// The package locator an artifact was fetched as, when it was fetched by one.
///
/// A fetch record's locator is either a PURL or a plain URL, and only the
/// former is a package identity. Shared by the sidecar's package slot — which
/// hopper projects into its queryable `purl_base` column — and by the verdict
/// handed to the uploader, so the bytes and the verdict can never disagree
/// about what this artifact is.
fn fetched_purl(root_fetch: Option<&FetchRecord>) -> Option<&str> {
    root_fetch
        .map(|rec| rec.locator.as_str())
        .filter(|locator| locator.starts_with("pkg:"))
}

/// The package an artifact *is*, as far as anything here can tell.
///
/// Broader than [`fetched_purl`], which asks only whether the *request* named a
/// package. This also recovers the coordinate from a registry URL, because
/// `scan url https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz` is a
/// request about a package whether or not it was spelled as one — and hopper
/// should store it under that package either way, or the same artifact lands
/// twice depending on which spelling the operator reached for.
///
/// Feeds both the sidecar's package slot — which hopper projects into its
/// queryable `purl_base` column — and the identity the uploader logs.
fn artifact_purl(root_fetch: Option<&FetchRecord>) -> Option<String> {
    if let Some(purl) = fetched_purl(root_fetch) {
        return Some(purl.to_string());
    }
    fletch::purl::url_to_purl(&root_fetch?.locator)
}

/// The scanned file itself, offered to hopper so a `--upload` run can store a
/// locally-analyzed file hopper has never seen. Just the one artifact — fetched
/// dependencies are mirrored separately (bytes, provenance, *and* verdict) by
/// [`crate::upload::Uploader::submit_dependencies`], so they never ride here.
pub(crate) fn collect_upload_artifacts(
    file_path: &Path,
    sha256: &str,
    size_bytes: u64,
    collector: &str,
    root_provenance: Option<&RegistryProvenance>,
    root_fetch: Option<&FetchRecord>,
) -> Vec<UploadArtifact> {
    let now = now_rfc3339();

    // For a fetched root (`scan url|purl`), `file_path` is the display locator,
    // not a readable file: it takes its name from the fetch URL, its bytes from
    // fletch's blob cache (where the fetch stored them), that URL for the
    // sidecar's fetch slot, and — when the locator is a PURL — the package slot
    // hopper projects into its queryable purl_base column. A local `scan path`
    // root reads from disk and claims none of the rest.
    let purl = artifact_purl(root_fetch).unwrap_or_default();
    let purl = purl.as_str();
    let (root_name, bytes, url) = match root_fetch {
        Some(rec) => {
            let url = crate::fetch::fetched_url(rec).unwrap_or_default();
            (
                artifact_filename(url, &rec.locator),
                ArtifactBytes::Cached {
                    locator: rec.locator.clone(),
                },
                url,
            )
        }
        None => (
            file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("file")
                .to_string(),
            ArtifactBytes::File(file_path.to_path_buf()),
            "",
        ),
    };
    // A collector that fetched this file left its capture record beside it.
    // That record holds the source URL and package identity the bytes came from
    // — which a local scan cannot reconstruct, and which for a since-deleted
    // package exists nowhere else — so it is uploaded as this artifact's
    // provenance instead of the thin one built below. A fetched root has no such
    // neighbour on disk, and an explicit `--registry-map` entry still wins.
    let collected = root_fetch
        .is_none()
        .then(|| crate::provenance::collector_sidecar(file_path, &root_name, sha256, size_bytes))
        .flatten();
    let from_collector = collected.is_some();
    let upload = Upload {
        filename: &root_name,
        sha256,
        size_bytes,
        collector,
        at: &now,
        url,
        purl: (!purl.is_empty()).then_some(purl),
    };
    let sidecar = match (root_provenance, collected) {
        (Some(provenance), collected) => {
            upload
                .sidecar_from_provenance(provenance)
                .map(|supplied| match collected {
                    Some(collected) => crate::provenance::with_registry_from(&collected, &supplied),
                    None => supplied,
                })
        }
        (None, Some(collected)) => Ok(collected),
        (None, None) => upload.sidecar(None, &[]),
    };
    let sidecar = match sidecar {
        Ok(sidecar) => sidecar,
        Err(error) => {
            tracing::error!(%sha256, %error, "upload: sidecar not built; not uploading the artifact");
            return Vec::new();
        }
    };
    vec![UploadArtifact {
        sha256: sha256.to_string(),
        size: size_bytes,
        filename: root_name,
        bytes,
        sidecar,
        // Registry data or a PURL identity is worth backfilling onto a sample
        // hopper already has; a plain local file's thin sidecar is not.
        backfill: root_provenance.is_some() || from_collector || !purl.is_empty(),
    }]
}

/// Longest stored filename we will emit. Well under the 255-byte cap every
/// mainstream filesystem imposes, leaving a consumer room for its own prefix.
const MAX_ARTIFACT_FILENAME: usize = 128;

/// A filename for an uploaded artifact: the last path segment of the fetch URL
/// (query/fragment stripped), falling back to the locator's tail with PURL
/// punctuation flattened. hopper uses it for the stored filename and type sniff.
///
/// Both inputs are hostile. A redirect chooses `url`'s final segment, and a
/// locator can come from references discovered inside the sample, so the result
/// is [sanitized](sanitize_artifact_filename) before it leaves this process.
pub(crate) fn artifact_filename(url: &str, locator: &str) -> String {
    // Split on the Windows separator too: a consumer that resolves `a\..\..\b`
    // as a path must not be handed one.
    let from_url = url
        .rsplit(['/', '\\'])
        .next()
        .map(|seg| seg.split(['?', '#']).next().unwrap_or(seg))
        .filter(|seg| !seg.is_empty());
    let raw = match from_url {
        Some(name) => std::borrow::Cow::Borrowed(name),
        None => std::borrow::Cow::Owned(
            locator
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(locator)
                .replace(['@', ':'], "-"),
        ),
    };
    sanitize_artifact_filename(&raw)
}

/// Reduce an untrusted path segment to a name that is inert everywhere it
/// lands: hopper's stored filename, a `Content-Disposition` field, a log line,
/// and an analyst's screen.
///
/// An allowlist rather than a denylist, because the interesting attacks are the
/// ones we would forget to enumerate: `%2f` that a consumer later decodes into
/// a separator, a `U+202E` right-to-left override that makes a `.exe` render as
/// a `.png` to the analyst reading the verdict, a control character that forges
/// a log line, a 64 KiB segment. Everything outside `[A-Za-z0-9.-_+~]` → `_`.
///
/// Leading dots and dashes go too: they produce hidden files, the `.`/`..` path
/// components, and flag-like arguments. Registry filenames are ASCII in
/// practice, so this is lossless for real packages.
fn sanitize_artifact_filename(raw: &str) -> String {
    let safe: String = raw
        .chars()
        .take(MAX_ARTIFACT_FILENAME)
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '-' | '_' | '+' | '~' => c,
            _ => '_',
        })
        .collect();
    match safe.trim_start_matches(['.', '-']) {
        // Nothing survived: a segment of only dots/dashes, or empty to begin
        // with. Name it rather than emit "" for a consumer to interpret.
        "" => "artifact".to_string(),
        trimmed => trimmed.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upload::ArtifactBytes;

    /// A fetched package carries its locator to the uploader; a fetched URL and
    /// a local path carry none, because neither names a package. The verdict's
    /// identity and the sidecar's package slot read the same rule.
    #[test]
    fn fetched_purl_is_only_a_package_locator() {
        // Built from JSON rather than a struct literal: FetchRecord has a
        // dozen fields this rule does not read, and listing them here would
        // make the test break on every unrelated field added to it.
        let record = |locator: &str| {
            serde_json::from_value::<fletch::fetch::FetchRecord>(serde_json::json!({
                "locator": locator,
                "outcome": "ok",
            }))
            .expect("FetchRecord from locator alone")
        };
        assert_eq!(
            fetched_purl(Some(&record("pkg:npm/left-pad@1.3.0"))),
            Some("pkg:npm/left-pad@1.3.0"),
        );
        assert_eq!(
            fetched_purl(Some(&record("https://example.com/a.tgz"))),
            None
        );
        assert_eq!(fetched_purl(None), None);

        // The logged identity recovers a package from a registry URL too, so a
        // `scan url` of a tarball reports what a reader recognises. An
        // arbitrary URL still identifies nothing.
        assert_eq!(
            artifact_purl(Some(&record(
                "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz"
            ))),
            Some("pkg:npm/left-pad@1.3.0".to_string()),
        );
        assert_eq!(
            artifact_purl(Some(&record("pkg:npm/left-pad@1.3.0"))),
            Some("pkg:npm/left-pad@1.3.0".to_string()),
        );
        assert_eq!(
            artifact_purl(Some(&record("https://example.com/a.tgz"))),
            None
        );
        assert_eq!(artifact_purl(None), None);
    }

    /// The sidecar hopper stores binds the package a URL-fetched artifact
    /// belongs to, so `scan url <registry tarball>` and the equivalent `scan
    /// purl` deposit the same identity rather than one bound row and one
    /// anonymous one.
    #[test]
    fn collect_upload_artifacts_binds_a_recovered_package() {
        let fetch = |locator: &str, url: &str| {
            serde_json::from_value::<fletch::fetch::FetchRecord>(serde_json::json!({
                "locator": locator,
                "resolved_url": url,
                "outcome": "ok",
            }))
            .expect("FetchRecord")
        };
        let package_of = |rec: &fletch::fetch::FetchRecord| {
            let arts = collect_upload_artifacts(
                Path::new("ignored"),
                &"a".repeat(64),
                10,
                "t",
                None,
                Some(rec),
            );
            let sidecar: serde_json::Value =
                serde_json::from_slice(&arts[0].sidecar).expect("sidecar json");
            sidecar["package"].clone()
        };

        let url = "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz";
        assert_eq!(
            package_of(&fetch(url, url))["purl"],
            serde_json::json!("pkg:npm/left-pad@1.3.0"),
            "a registry URL is stored under the package it names",
        );
        assert_eq!(
            package_of(&fetch("pkg:npm/left-pad@1.3.0", url))["purl"],
            serde_json::json!("pkg:npm/left-pad@1.3.0"),
            "and matches what the PURL spelling stores",
        );
        // A URL that names no package must not acquire an invented identity.
        assert!(
            package_of(&fetch(
                "https://example.com/a.tgz",
                "https://example.com/a.tgz"
            ))
            .is_null(),
            "no package slot without a package",
        );
    }

    #[test]
    fn collect_upload_artifacts_offers_root_only() {
        // Fetched dependencies are mirrored separately (bytes + provenance + their
        // own verdict) via the uploader's dependency path, so this offers only the
        // scanned file itself — a local artifact hopper may never have seen.
        let arts = collect_upload_artifacts(
            Path::new("/tmp/proj.tgz"),
            &"a".repeat(64),
            10,
            "scan+test",
            None,
            None,
        );

        assert_eq!(arts.len(), 1, "only the scanned file is offered here");
        assert_eq!(arts[0].sha256, "a".repeat(64));
        assert!(matches!(
            arts[0].bytes,
            crate::upload::ArtifactBytes::File(_)
        ));
        assert!(
            !arts[0].backfill,
            "root file's thin sidecar is not backfilled"
        );
        assert_eq!(arts[0].filename, "proj.tgz", "filename is the file's name");
    }

    #[test]
    fn collect_upload_artifacts_backfills_preserved_root_provenance() {
        let provenance = crate::provenance::registry_provenance(
            br#"{"record":{"ecosystem":"npm","name":"proj","version":"1.0.0"},"sources":[{"url":"https://registry.example/proj","status":200,"body":{"provider_only":42}}]}"#,
        )
        .unwrap();
        let arts = collect_upload_artifacts(
            Path::new("/tmp/proj.tgz"),
            &"a".repeat(64),
            10,
            "scan+test",
            Some(&provenance),
            None,
        );
        let sidecar: serde_json::Value = serde_json::from_slice(&arts[0].sidecar).unwrap();
        assert!(arts[0].backfill);
        assert_eq!(sidecar["registry"]["raw"][0]["body"]["provider_only"], 42);
    }

    fn fetch_record(locator: &str, resolved: &str) -> fletch::fetch::FetchRecord {
        serde_json::from_value(serde_json::json!({
            "locator": locator,
            "resolved_url": resolved,
            "outcome": "ok",
        }))
        .expect("minimal FetchRecord")
    }

    #[test]
    fn fetched_purl_root_carries_package_identity() {
        let rec = fetch_record(
            "pkg:npm/lodash@4.17.21",
            "https://registry.npmjs.org/lodash/-/lodash-4.17.21.tgz",
        );
        let arts = collect_upload_artifacts(
            Path::new("pkg:npm/lodash@4.17.21"),
            "aa11",
            42,
            "scan+test",
            None,
            Some(&rec),
        );
        let art = &arts[0];
        assert!(art.backfill, "a PURL identity is worth backfilling");
        assert_eq!(
            art.filename, "lodash-4.17.21.tgz",
            "named from the fetch URL"
        );
        assert!(
            matches!(&art.bytes, ArtifactBytes::Cached { locator } if locator == "pkg:npm/lodash@4.17.21"),
            "fetched root loads from the blob cache, not the display label"
        );
        let sidecar: serde_json::Value = serde_json::from_slice(&art.sidecar).unwrap();
        assert_eq!(sidecar["package"]["purl"], "pkg:npm/lodash@4.17.21");
        assert_eq!(
            sidecar["fetch"]["url"],
            "https://registry.npmjs.org/lodash/-/lodash-4.17.21.tgz"
        );
        assert_eq!(sidecar["artifact"]["filename"], "lodash-4.17.21.tgz");
    }

    /// A collector's capture record is the only surviving account of where a
    /// deleted package's bytes came from, so an upload must carry it rather than
    /// the thin record a local scan can rebuild.
    #[test]
    fn local_root_uploads_the_collector_sidecar_beside_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = dir.path().join("evil-1.0.0.tgz");
        std::fs::write(&artifact, b"bytes").expect("write artifact");
        std::fs::write(
            dir.path().join("evil-1.0.0.tgz.forage.json"),
            serde_json::json!({
                "schema_version": "1.0",
                "artifact": {"filename": "evil-1.0.0.tgz", "sha256": "CC33", "size_bytes": 5},
                "package": {"purl": "pkg:npm/evil@1.0.0", "ecosystem": "javascript"},
                "fetch": {
                    "collector": "forager",
                    "category": "submitted",
                    "at": "2026-09-08T10:06:05Z",
                    "url": "https://socket.dev/npm/package/evil/files/1.0.0",
                    "original_url": "https://registry.npmjs.org/evil/-/evil-1.0.0.tgz",
                },
            })
            .to_string(),
        )
        .expect("write sidecar");

        let arts = collect_upload_artifacts(&artifact, "cc33", 5, "scan+test", None, None);
        let art = &arts[0];
        assert!(art.backfill, "collected provenance is worth backfilling");
        let sidecar: serde_json::Value = serde_json::from_slice(&art.sidecar).unwrap();
        assert_eq!(sidecar["fetch"]["collector"], "forager");
        assert_eq!(
            sidecar["fetch"]["url"],
            "https://socket.dev/npm/package/evil/files/1.0.0"
        );
        assert_eq!(sidecar["package"]["purl"], "pkg:npm/evil@1.0.0");
        assert_eq!(sidecar["artifact"]["sha256"], "cc33");
        assert_eq!(sidecar["artifact"]["size_bytes"], 5);
    }

    #[test]
    fn local_root_without_a_sidecar_still_uploads_a_thin_record() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = dir.path().join("plain.bin");
        std::fs::write(&artifact, b"bytes").expect("write artifact");
        let arts = collect_upload_artifacts(&artifact, "dd44", 5, "scan+test", None, None);
        assert!(!arts[0].backfill);
        let sidecar: serde_json::Value = serde_json::from_slice(&arts[0].sidecar).unwrap();
        assert_eq!(sidecar["fetch"]["collector"], "scan+test");
        assert!(sidecar.get("package").is_none());
    }

    #[test]
    fn fetched_url_root_has_no_package_slot() {
        let rec = fetch_record(
            "https://example.com/tool.zip",
            "https://example.com/tool.zip",
        );
        let arts = collect_upload_artifacts(
            Path::new("https://example.com/tool.zip"),
            "bb22",
            7,
            "scan+test",
            None,
            Some(&rec),
        );
        let art = &arts[0];
        assert!(
            !art.backfill,
            "a bare URL fetch carries no registry identity"
        );
        assert_eq!(art.filename, "tool.zip");
        let sidecar: serde_json::Value = serde_json::from_slice(&art.sidecar).unwrap();
        assert!(
            sidecar.get("package").is_none(),
            "no PURL, no package claim"
        );
        assert_eq!(sidecar["fetch"]["url"], "https://example.com/tool.zip");
    }

    #[test]
    fn hostile_redirect_targets_cannot_shape_the_stored_filename() {
        use super::{MAX_ARTIFACT_FILENAME, artifact_filename};

        // A redirect picks the final URL, so every one of these is reachable by
        // an attacker who controls (or compromises) the server we fetch from.
        for (url, expect) in [
            // Encoded separators a consumer might decode back into a path.
            (
                "https://e.com/..%2f..%2fetc%2fpasswd",
                "_2f.._2fetc_2fpasswd",
            ),
            // Windows separators in the final segment.
            ("https://e.com/a\\..\\..\\system32\\x.dll", "x.dll"),
            // Bare path components.
            ("https://e.com/..", "artifact"),
            ("https://e.com/.", "artifact"),
            // A right-to-left override disguising the real extension.
            ("https://e.com/invoice\u{202e}gpj.exe", "invoice_gpj.exe"),
            // Control characters that would forge a log line.
            ("https://e.com/a\nb\rc", "a_b_c"),
            // Leading dash reads as a flag; leading dot hides the file.
            ("https://e.com/-rf", "rf"),
            ("https://e.com/.ssh", "ssh"),
            // Ordinary names are untouched.
            ("https://e.com/lodash-4.17.21.tgz", "lodash-4.17.21.tgz"),
            ("https://e.com/x_1+2~3.tar.gz", "x_1+2~3.tar.gz"),
        ] {
            assert_eq!(artifact_filename(url, "pkg:npm/x@1"), expect, "url {url:?}");
        }

        // Length is bounded regardless of what the server sends.
        let long = format!("https://e.com/{}", "a".repeat(4096));
        assert_eq!(artifact_filename(&long, "").len(), MAX_ARTIFACT_FILENAME);

        // A hostile locator gets the same treatment when no URL resolved.
        assert_eq!(artifact_filename("", "pkg:npm/../../etc@1"), "etc-1");
    }

    #[test]
    fn local_root_keeps_thin_sidecar_and_disk_bytes() {
        let arts = collect_upload_artifacts(
            Path::new("/tmp/sample.exe"),
            "cc33",
            1,
            "scan+test",
            None,
            None,
        );
        let art = &arts[0];
        assert!(!art.backfill);
        assert_eq!(art.filename, "sample.exe");
        assert!(matches!(&art.bytes, ArtifactBytes::File(p) if p == Path::new("/tmp/sample.exe")));
        let sidecar: serde_json::Value = serde_json::from_slice(&art.sidecar).unwrap();
        assert!(sidecar.get("package").is_none());
        assert_eq!(sidecar["fetch"]["url"], "");
    }
}
