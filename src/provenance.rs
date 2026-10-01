//! Reading the registry-metadata provenance hopper stores per sample.
//!
//! A collector (forager) runs `fletch registry <purl>` at fetch time and stores
//! its `{record, sources}` envelope in the sample's sidecar under `registry`.
//! Both the worker (over HTTP, from `/api/provenance/{sha256}`) and the CLI
//! (`--registry-map <file>`) read that sidecar to recover the normalized
//! [`fletch::Registry`] a scan reasons over, so a hopper-sourced scan sees the
//! same registry facts a live `pkg`/`url` scan fetches — without a refetch.

use fletch::Registry;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// The sidecar schema version hopper validates against ([`hopper.SidecarSchemaVersion`]).
const SCHEMA_VERSION: &str = "1.0";

/// Cap on `registry.raw`, matching hopper's `MaxRawBytes` (256 KiB). An upstream
/// document can be huge — a full npm packument carries every published version —
/// and embedding it verbatim would push the provenance part past hopper's 1 MiB
/// transport cap, where it is truncated into unparseable JSON and rejected. When
/// `raw` overflows the cap we drop it and downgrade the record to "partial", the
/// same trim hopper's `Sidecar.Finalize` performs on receipt.
const MAX_RAW_BYTES: usize = 256 << 10;

/// Suffix a collector writes beside every artifact it stores: forager's
/// `<artifact>.forage.json` capture record.
pub const COLLECTOR_SIDECAR_SUFFIX: &str = ".forage.json";

/// Largest collector sidecar worth reading. Hopper's provenance part shares a
/// 1 MiB transport budget with the rest of the upload, and `registry.raw` is
/// already capped well below that ([`MAX_RAW_BYTES`]), so anything larger is
/// malformed rather than merely generous.
const MAX_COLLECTOR_SIDECAR_BYTES: u64 = 1 << 20;

/// Read the provenance sidecar a collector wrote beside `artifact`, ready to
/// upload as this file's provenance.
///
/// The sidecar is capture-time evidence: the URL the bytes actually came from,
/// the package they were published as, and the registry facts that were true
/// when they were pulled — none of which a later local scan can reconstruct,
/// and all of which are lost if the scanner substitutes its own thin record.
/// A deleted package is exactly the case where that evidence is the only copy
/// left, so it is preserved and re-sent rather than rebuilt.
///
/// The artifact binding is re-stated from the bytes on disk, and a sidecar
/// claiming a different digest is refused outright: provenance describes
/// specific bytes, and attaching one file's history to another's is worse than
/// having none.
///
/// `None` when no sidecar is there, it is unreadable or oversized, or it
/// describes different bytes — provenance enriches an upload, never gates it.
#[must_use]
pub fn collector_sidecar(
    artifact: &Path,
    filename: &str,
    sha256: &str,
    size_bytes: u64,
) -> Option<Vec<u8>> {
    let document = collector_document(artifact, sha256)?;
    let artifact = ArtifactRef {
        filename,
        sha256,
        size_bytes,
    };
    rebind(document, artifact)
        .map_err(|e| tracing::warn!(%sha256, error = %e, "collector sidecar could not be rebound"))
        .ok()
}

/// The package coordinate a collector recorded for these exact bytes.
///
/// A collected sample is a package, but nothing in the file says so: the bytes
/// are a tarball whose name is a filename, and a deleted release cannot be
/// looked up. The capture record is the only place the coordinate survives, so
/// the terminal card reads it from there — it is what an operator needs in order
/// to say which package the verdict is about.
///
/// Canonicalized through fletch so a malformed or hostile `purl` string cannot
/// reach the display, and `None` whenever `collector_document` declines.
#[must_use]
pub fn collector_purl(artifact: &Path, sha256: &str) -> Option<String> {
    let document = collector_document(artifact, sha256)?;
    let purl = document.get("package")?.get("purl")?.as_str()?;
    fletch::purl::normalize(purl)
}

/// The registry's account of the package, as the collector recorded it.
///
/// A collector that enriched its capture stored the same normalized record a
/// live `pkg` scan looks up. Recovering it here means a collected sample
/// reasons over — and reads as — what the registry said at capture time, which
/// for a since-deleted release is the only version of that account left.
///
/// `None` when the sidecar carries no registry block, which is the ordinary
/// case for a collector that ran without its registry helper.
#[must_use]
pub fn collector_registry(artifact: &Path, sha256: &str) -> Option<RegistryProvenance> {
    let document = collector_document(artifact, sha256)?;
    RegistryProvenance::from_json(&serde_json::to_vec(&document).ok()?)
}

/// Read and authenticate the collector sidecar beside `artifact`.
///
/// Shared by the upload and display paths so both agree on what counts as a
/// usable capture record: present, small enough to be a record rather than a
/// payload, parseable, and bound to the digest of the bytes in hand.
fn collector_document(artifact: &Path, sha256: &str) -> Option<serde_json::Value> {
    let mut path = artifact.as_os_str().to_os_string();
    path.push(COLLECTOR_SIDECAR_SUFFIX);
    let path = std::path::PathBuf::from(path);
    if std::fs::metadata(&path).ok()?.len() > MAX_COLLECTOR_SIDECAR_BYTES {
        tracing::warn!("ignoring oversized provenance sidecar {}", path.display());
        return None;
    }
    let document: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).ok()?)
        .map_err(|e| {
            tracing::warn!(
                "ignoring unparseable provenance sidecar {}: {e}",
                path.display()
            )
        })
        .ok()?;
    if !document.is_object() {
        return None;
    }
    let recorded = document
        .get("artifact")
        .and_then(|a| a.get("sha256"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if !recorded.eq_ignore_ascii_case(sha256) {
        tracing::warn!(
            "ignoring provenance sidecar {}: it describes {recorded}, not {sha256}",
            path.display(),
        );
        return None;
    }
    Some(document)
}

#[cfg(test)]
mod collector_sidecar_tests {
    use super::collector_sidecar;

    #[test]
    fn refuses_a_sidecar_describing_other_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = dir.path().join("a.tgz");
        std::fs::write(&artifact, b"bytes").expect("write");
        std::fs::write(
            dir.path().join("a.tgz.forage.json"),
            br#"{"artifact":{"sha256":"beef"},"fetch":{"url":"https://x/"}}"#,
        )
        .expect("write");
        assert!(collector_sidecar(&artifact, "a.tgz", "cafe", 5).is_none());
        // Same bytes, different case: hex digests are compared, not strings.
        assert!(collector_sidecar(&artifact, "a.tgz", "BEEF", 5).is_some());
    }

    #[test]
    fn the_recorded_coordinate_is_canonicalized_before_display() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = dir.path().join("a.tgz");
        std::fs::write(&artifact, b"bytes").expect("write");
        let write = |package: serde_json::Value| {
            std::fs::write(
                dir.path().join("a.tgz.forage.json"),
                serde_json::json!({
                    "artifact": {"sha256": "cafe"},
                    "package": package,
                    "fetch": {"collector": "forager"},
                })
                .to_string(),
            )
            .expect("write");
        };
        write(serde_json::json!({"purl": "pkg:npm/blueai-cli@0.7.0"}));
        assert_eq!(
            super::collector_purl(&artifact, "cafe").as_deref(),
            Some("pkg:npm/blueai-cli@0.7.0")
        );
        // Nothing a collector wrote reaches the terminal unvalidated.
        write(serde_json::json!({"purl": "not a purl"}));
        assert!(super::collector_purl(&artifact, "cafe").is_none());
        write(serde_json::json!({"name": "blueai-cli"}));
        assert!(super::collector_purl(&artifact, "cafe").is_none());
        // The digest binding gates the display exactly as it gates the upload.
        write(serde_json::json!({"purl": "pkg:npm/blueai-cli@0.7.0"}));
        assert!(super::collector_purl(&artifact, "beef").is_none());
    }

    /// A collector that enriched its capture carries the registry's account of
    /// the package; recovering it is what lets a collected sample read like a
    /// live `purl` scan of the same release.
    #[test]
    fn the_recorded_registry_account_is_recovered() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = dir.path().join("a.tgz");
        std::fs::write(&artifact, b"bytes").expect("write");
        let sidecar = dir.path().join("a.tgz.forage.json");
        std::fs::write(
            &sidecar,
            serde_json::json!({
                "artifact": {"sha256": "cafe"},
                "fetch": {"collector": "forager"},
                "registry": {"record": {
                    "ecosystem": "npm", "name": "evil", "version": "1.0.0", "author": "mallory",
                }},
            })
            .to_string(),
        )
        .expect("write");
        let recovered = super::collector_registry(&artifact, "cafe").expect("registry record");
        assert_eq!(recovered.record.name, "evil");
        assert_eq!(recovered.record.author.as_deref(), Some("mallory"));
        // A capture record without one is the ordinary case, not a failure.
        std::fs::write(
            &sidecar,
            serde_json::json!({"artifact": {"sha256": "cafe"}, "fetch": {"collector": "forager"}})
                .to_string(),
        )
        .expect("write");
        assert!(super::collector_registry(&artifact, "cafe").is_none());
    }

    #[test]
    fn absent_or_unparseable_sidecars_are_not_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = dir.path().join("a.tgz");
        std::fs::write(&artifact, b"bytes").expect("write");
        assert!(collector_sidecar(&artifact, "a.tgz", "cafe", 5).is_none());
        std::fs::write(dir.path().join("a.tgz.forage.json"), b"{not json").expect("write");
        assert!(collector_sidecar(&artifact, "a.tgz", "cafe", 5).is_none());
    }

    /// A capture record and a registry lookup answer different questions, so a
    /// scan given both must not spend one to keep the other.
    #[test]
    fn registry_facts_join_the_capture_record_without_displacing_it() {
        let collected = serde_json::json!({
            "artifact": {"sha256": "cafe"},
            "package": {"purl": "pkg:npm/evil@1.0.0"},
            "fetch": {"collector": "forager", "url": "https://socket.dev/x"},
        })
        .to_string();
        let supplied = serde_json::json!({
            "artifact": {"sha256": "cafe"},
            "package": {"purl": "pkg:npm/other@2.0.0"},
            "fetch": {"collector": "scan+host", "url": ""},
            "registry": {"record": {"name": "evil"}, "status": "complete"},
        })
        .to_string();
        let merged: serde_json::Value = serde_json::from_slice(&super::with_registry_from(
            collected.as_bytes(),
            supplied.as_bytes(),
        ))
        .unwrap();
        assert_eq!(merged["fetch"]["collector"], "forager");
        assert_eq!(merged["fetch"]["url"], "https://socket.dev/x");
        assert_eq!(merged["package"]["purl"], "pkg:npm/evil@1.0.0");
        assert_eq!(merged["registry"]["record"]["name"], "evil");
    }

    #[test]
    fn preserves_collected_fields_and_rebinds_the_artifact() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifact = dir.path().join("a.tgz");
        std::fs::write(&artifact, b"bytes").expect("write");
        std::fs::write(
            dir.path().join("a.tgz.forage.json"),
            serde_json::json!({
                "schema_version": "0.9",
                "artifact": {"filename": "stale.tgz", "sha256": "cafe", "size_bytes": 1},
                "fetch": {"collector": "forager", "url": "https://x/", "original_url": "https://y/"},
            })
            .to_string(),
        )
        .expect("write");
        let bytes = collector_sidecar(&artifact, "a.tgz", "cafe", 5).expect("sidecar");
        let sidecar: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(sidecar["schema_version"], super::SCHEMA_VERSION);
        assert_eq!(sidecar["artifact"]["filename"], "a.tgz");
        assert_eq!(sidecar["artifact"]["size_bytes"], 5);
        assert_eq!(sidecar["fetch"]["collector"], "forager");
        assert_eq!(sidecar["fetch"]["original_url"], "https://y/");
    }
}

/// Fold supplied registry facts into a collector's capture record.
///
/// The two documents describe different things, so a scan handed both keeps
/// both. The capture record says where these exact bytes came from — a CDN
/// reconstruction, a mirror, an archive replay — and is the only account of it
/// once the package is gone; `--registry-map` (or a worker's hopper lookup) says
/// what the registry knew about the package. Identity, origin, and collector
/// therefore stay as captured, and only the registry block, plus a package slot
/// the collector did not record, are taken from `supplied`.
#[must_use]
pub fn with_registry_from(collected: &[u8], supplied: &[u8]) -> Vec<u8> {
    use serde_json::{Map, Value};
    let (Ok(mut merged), Ok(mut supplied)) = (
        serde_json::from_slice::<Map<String, Value>>(collected),
        serde_json::from_slice::<Map<String, Value>>(supplied),
    ) else {
        return collected.to_vec();
    };
    // The map was named explicitly, so its registry block wins outright — that
    // is the whole point of passing one. A package slot is different: the
    // collector recorded the coordinate it actually fetched, so the map only
    // fills that in when the capture record left it out.
    if let Some(registry) = supplied.remove("registry").filter(|r| !r.is_null()) {
        merged.insert("registry".into(), registry);
    }
    if let Some(package) = supplied.remove("package").filter(|p| !p.is_null())
        && merged.get("package").is_none_or(Value::is_null)
    {
        merged.insert("package".into(), package);
    }
    serde_json::to_vec(&merged).unwrap_or_else(|_| collected.to_vec())
}

/// The facts every sidecar states about the bytes it travels with: which bytes,
/// who pushed them, and where they came from. `purl` is the package's
/// canonical PURL, `None` for a scanned root file or a plain-URL fetch.
///
/// Builds the hopper `Sidecar` (`schema_version`, `artifact`, `fetch`, and —
/// for a fetched package — `package` + `registry`) field-for-field, so the
/// upload validator accepts it.
#[derive(Debug, Clone, Copy)]
pub struct Upload<'a> {
    /// Filename hopper stores and sniffs the type from.
    pub filename: &'a str,
    /// SHA-256 of the bytes.
    pub sha256: &'a str,
    /// Size of the bytes.
    pub size_bytes: u64,
    /// The collector name (`scan+<worker>`).
    pub collector: &'a str,
    /// When the bytes were collected, RFC 3339.
    pub at: &'a str,
    /// The URL the bytes came from, empty when there is none.
    pub url: &'a str,
    /// The package the bytes were fetched as.
    pub purl: Option<&'a str>,
}

impl Upload<'_> {
    /// A new sidecar. `registry` is the normalized record scan derived for a
    /// fetched dependency (`None` for the root file); `sources` are the raw
    /// provider documents its lookup read, archived under `registry.raw` as the
    /// re-parsing backup — the shape forager stores: a JSON body inline,
    /// anything else base64 in `body_b64`.
    ///
    /// # Errors
    /// Returns serde's error if the document cannot be serialized.
    pub fn sidecar(
        &self,
        registry: Option<&Registry>,
        sources: &[fletch::fetch::RecordedSource],
    ) -> Result<Vec<u8>, serde_json::Error> {
        // Measured once and embedded verbatim; an over-cap archive is dropped
        // rather than shipped in a part hopper cannot parse ([`MAX_RAW_BYTES`]).
        // The normalized `record` is small and bounded, so it always rides.
        let raw = match registry {
            Some(_) => Some(serde_json::to_string(&encoded_sources(sources))?)
                .filter(|raw| raw.len() <= MAX_RAW_BYTES)
                .map(serde_json::value::RawValue::from_string)
                .transpose()?,
            None => None,
        };
        serde_json::to_vec(&Sidecar {
            schema_version: SCHEMA_VERSION,
            artifact: self.artifact(),
            // A fetched dependency always carries its PURL, so hopper can
            // project the version-less form into the queryable purl_base
            // column — whether or not the registry lookup resolved.
            package: self.purl.map(|purl| Package {
                ecosystem: registry.map(|r| r.ecosystem.as_str()),
                name: registry.map(|r| r.name.as_str()),
                version: registry.map(|r| r.version.as_str()),
                purl,
            }),
            fetch: self.fetch(),
            registry: registry.map(|record| self.registry(record, raw.as_deref())),
        })
    }

    /// A sidecar that preserves provenance supplied by a worker, a
    /// `--registry-map`, or a live lookup.
    ///
    /// A complete hopper sidecar is reused as-is, rebound to these bytes. Bare
    /// fletch envelopes and legacy records are wrapped in a current sidecar;
    /// their raw provider data, when present, is copied into `registry.raw`
    /// without schema-specific parsing.
    ///
    /// # Errors
    /// Returns serde's error if the document cannot be parsed or serialized.
    pub fn sidecar_from_provenance(
        &self,
        provenance: &RegistryProvenance,
    ) -> Result<Vec<u8>, serde_json::Error> {
        if serde_json::from_slice::<CompleteSidecarProbe>(provenance.document()).is_ok() {
            return rebind(
                serde_json::from_slice(provenance.document())?,
                self.artifact(),
            );
        }
        let record = &provenance.record;
        // RawValue points directly into the compact document. Its length is a
        // safe cap check (conservative only when an external producer included
        // whitespace) and it serializes verbatim, with no Value tree.
        let raw =
            registry_raw_json(provenance.document()).filter(|raw| raw.get().len() <= MAX_RAW_BYTES);
        serde_json::to_vec(&Sidecar {
            schema_version: SCHEMA_VERSION,
            artifact: self.artifact(),
            package: self.purl.map(|purl| Package {
                ecosystem: Some(&record.ecosystem),
                name: Some(&record.name),
                version: Some(&record.version),
                purl,
            }),
            fetch: self.fetch(),
            registry: Some(self.registry(record, raw)),
        })
    }

    const fn artifact(&self) -> ArtifactRef<'_> {
        ArtifactRef {
            filename: self.filename,
            sha256: self.sha256,
            size_bytes: self.size_bytes,
        }
    }

    const fn fetch(&self) -> FetchRef<'_> {
        FetchRef {
            collector: self.collector,
            // A scan push is a discovered-by-us artifact, not a labeled feed
            // sample: hopper records it but derives the real label from
            // analysis, never from this claim.
            category: "submitted",
            at: self.at,
            url: self.url,
        }
    }

    fn registry<'r>(
        &'r self,
        record: &'r Registry,
        raw: Option<&'r serde_json::value::RawValue>,
    ) -> RegistryBlock<'r> {
        RegistryBlock {
            source_id: &record.ecosystem,
            ecosystem: &record.ecosystem,
            format: "fletch.registry",
            url: self.url,
            at: self.at,
            status: if raw.is_some() { "complete" } else { "partial" },
            record,
            raw,
        }
    }
}

/// Rebind an existing sidecar document to the bytes being uploaded. Its
/// provenance is historical; the artifact binding is a present-tense integrity
/// claim and must match what is uploaded, or hopper rejects otherwise valid
/// bytes. An over-cap raw archive is trimmed on the way out.
fn rebind(
    mut document: serde_json::Value,
    artifact: ArtifactRef<'_>,
) -> Result<Vec<u8>, serde_json::Error> {
    let Some(object) = document.as_object_mut() else {
        return Err(serde::de::Error::custom("a sidecar must be a JSON object"));
    };
    object.insert("schema_version".into(), SCHEMA_VERSION.into());
    object.insert("artifact".into(), serde_json::to_value(artifact)?);
    trim_registry_raw(&mut document);
    serde_json::to_vec(&document)
}

#[derive(Deserialize)]
struct CompleteSidecarProbe {
    #[serde(rename = "schema_version")]
    _schema_version: serde::de::IgnoredAny,
    #[serde(rename = "artifact")]
    _artifact: serde::de::IgnoredAny,
    #[serde(rename = "fetch")]
    _fetch: serde::de::IgnoredAny,
    #[serde(rename = "registry")]
    _registry: serde::de::IgnoredAny,
}

/// hopper's `Sidecar`, as this crate writes it.
#[derive(serde::Serialize)]
struct Sidecar<'a> {
    schema_version: &'static str,
    artifact: ArtifactRef<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    package: Option<Package<'a>>,
    fetch: FetchRef<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    registry: Option<RegistryBlock<'a>>,
}

#[derive(serde::Serialize)]
struct ArtifactRef<'a> {
    filename: &'a str,
    sha256: &'a str,
    size_bytes: u64,
}

/// Identity fields are filled from the registry record when one resolved.
#[derive(serde::Serialize)]
struct Package<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    ecosystem: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<&'a str>,
    purl: &'a str,
}

#[derive(serde::Serialize)]
struct FetchRef<'a> {
    collector: &'a str,
    category: &'static str,
    at: &'a str,
    url: &'a str,
}

#[derive(serde::Serialize)]
struct RegistryBlock<'a> {
    source_id: &'a str,
    ecosystem: &'a str,
    format: &'static str,
    url: &'a str,
    at: &'a str,
    status: &'static str,
    record: &'a Registry,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw: Option<&'a serde_json::value::RawValue>,
}

/// Remove an oversized raw provider payload using the same invariant hopper
/// enforces on receipt. Doing it here avoids allocating a multipart body hopper
/// will immediately discard.
fn trim_registry_raw(sidecar: &mut serde_json::Value) {
    let Some(registry) = sidecar
        .get_mut("registry")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };
    let over_cap = registry
        .get("raw")
        .and_then(|raw| serde_json::to_vec(raw).ok())
        .is_some_and(|raw| raw.len() > MAX_RAW_BYTES);
    if over_cap {
        registry.remove("raw");
        registry.insert(
            "status".to_string(),
            serde_json::Value::String("partial".to_string()),
        );
    }
}

/// Move the raw provider payload out of any accepted input shape.
fn take_registry_raw(document: &mut serde_json::Value) -> Option<serde_json::Value> {
    if let Some(registry) = document
        .get_mut("registry")
        .and_then(serde_json::Value::as_object_mut)
    {
        return registry
            .remove("raw")
            .or_else(|| registry.remove("sources"));
    }
    let document = document.as_object_mut()?;
    document
        .remove("raw")
        .or_else(|| document.remove("sources"))
}

/// Lossless registry provenance at scan's input boundary.
///
/// `record` is the small normalized view used for cleave facts, matching, and
/// terminal output. `document` is the complete opaque JSON scan received from
/// hopper, `--registry-map`, or a live fletch lookup. Keeping both prevents an
/// operational parse from silently discarding provider-specific or future
/// provenance fields.
///
/// The document stays serialized. Registry responses are often much larger as a
/// `serde_json::Value` tree (roughly 3–6× in practice), and most scans never need
/// to inspect their raw fields. Cloning this type is an atomic refcount bump; raw
/// JSON is parsed only for a selected interpret/terminal subject or an upload.
#[derive(Debug, Clone)]
pub struct RegistryProvenance {
    /// Normalized registry facts used by analysis and processed displays.
    pub record: Registry,
    document: bytes::Bytes,
}

impl RegistryProvenance {
    /// Preserve a provenance byte buffer and recover its normalized record
    /// without materializing ignored raw provider fields.
    #[must_use]
    pub fn from_bytes(document: bytes::Bytes) -> Option<Self> {
        let record = parse_registry_record(&document)?;
        Some(Self { record, document })
    }

    /// Preserve borrowed JSON. Prefer [`Self::from_bytes`] when the caller
    /// already owns a [`bytes::Bytes`] response to avoid a copy.
    #[must_use]
    pub fn from_json(document: &[u8]) -> Option<Self> {
        Self::from_bytes(bytes::Bytes::copy_from_slice(document))
    }

    /// Adopt an owned raw JSON value without copying its backing allocation.
    #[must_use]
    pub fn from_raw_value(document: Box<serde_json::value::RawValue>) -> Option<Self> {
        let document: Box<str> = document.into();
        Self::from_bytes(bytes::Bytes::from(String::from(document)))
    }

    /// Build the same lossless envelope for a live lookup.
    #[must_use]
    pub fn from_record_sources(
        record: Registry,
        sources: &[fletch::fetch::RecordedSource],
    ) -> Self {
        #[derive(serde::Serialize)]
        struct LiveDocument<'a> {
            record: &'a Registry,
            sources: Vec<EncodedSource<'a>>,
        }

        // Every field serializes infallibly; should that ever change, the
        // document stays empty — unparseable, so no consumer mistakes it for
        // provenance — and the failure is said here.
        let document = serde_json::to_vec(&LiveDocument {
            record: &record,
            sources: encoded_sources(sources),
        })
        .unwrap_or_else(|e| {
            tracing::error!(package = %record.name, error = %e, "registry provenance could not be serialized");
            Vec::new()
        });
        Self {
            record,
            document: bytes::Bytes::from(document),
        }
    }

    /// The complete provenance JSON as received or created at the boundary.
    #[must_use]
    pub fn document(&self) -> &[u8] {
        &self.document
    }

    /// Parse the complete document on demand.
    #[must_use]
    pub fn document_value(&self) -> Option<serde_json::Value> {
        serde_json::from_slice(&self.document).ok()
    }

    /// Raw provider data, regardless of whether it arrived in a hopper sidecar
    /// (`registry.raw`) or a live/bare fletch envelope (`sources`).
    #[must_use]
    pub fn raw(&self) -> Option<serde_json::Value> {
        take_registry_raw(&mut self.document_value()?)
    }

    /// Provider URLs carried by the preserved document, deduplicated in input
    /// order. Terminal rendering uses this instead of re-querying a registry.
    #[must_use]
    pub fn source_urls(&self) -> Vec<String> {
        let Ok(document) = serde_json::from_slice::<UrlDocument<'_>>(&self.document) else {
            return Vec::new();
        };
        let mut urls = Vec::new();
        let (registry_url, raw) = if let Some(registry) = document.registry {
            (registry.url, registry.raw.or(registry.sources))
        } else {
            (None, document.raw.or(document.sources))
        };
        if let Some(url) = registry_url {
            urls.push(url);
        }
        if let Some(raw) = raw
            && let Ok(sources) = serde_json::from_str::<Vec<SourceUrl>>(raw.get())
        {
            for source in sources {
                if let Some(url) = source.url
                    && !url.is_empty()
                    && !urls.iter().any(|existing| existing == &url)
                {
                    urls.push(url);
                }
            }
        }
        urls
    }
}

#[derive(serde::Serialize)]
#[serde(untagged)]
enum EncodedSource<'a> {
    Json {
        url: &'a str,
        status: u16,
        #[serde(skip_serializing_if = "Option::is_none")]
        content_type: Option<&'a str>,
        body: &'a serde_json::value::RawValue,
    },
    Binary {
        url: &'a str,
        status: u16,
        #[serde(skip_serializing_if = "Option::is_none")]
        content_type: Option<&'a str>,
        body_b64: String,
    },
    /// A document fletch recorded without its bytes — past a source limit,
    /// which scan never sets — so only what identifies it.
    Unkept {
        url: &'a str,
        status: u16,
        #[serde(skip_serializing_if = "Option::is_none")]
        content_type: Option<&'a str>,
        size: u64,
    },
}

/// Borrow valid JSON bodies directly from fletch's source buffers so building a
/// long-lived provenance document does not allocate a `Value` tree or reformat a
/// large packument. Non-JSON bodies pay only the unavoidable base64 allocation.
fn encoded_sources(sources: &[fletch::fetch::RecordedSource]) -> Vec<EncodedSource<'_>> {
    use base64::Engine as _;

    sources
        .iter()
        .map(|source| {
            let (url, status) = (source.url.as_str(), source.status);
            let content_type = source.content_type.as_deref();
            let Some(bytes) = source.bytes.as_deref() else {
                return EncodedSource::Unkept {
                    url,
                    status,
                    content_type,
                    size: source.size,
                };
            };
            match serde_json::from_slice::<&serde_json::value::RawValue>(bytes) {
                Ok(body) => EncodedSource::Json {
                    url,
                    status,
                    content_type,
                    body,
                },
                Err(_) => EncodedSource::Binary {
                    url,
                    status,
                    content_type,
                    body_b64: base64::engine::general_purpose::STANDARD.encode(bytes),
                },
            }
        })
        .collect()
}

#[derive(Deserialize)]
struct RecordSlot {
    record: Registry,
}

#[derive(Deserialize)]
struct SidecarRecord {
    #[serde(default)]
    registry: Option<RecordSlot>,
}

#[derive(Deserialize)]
struct UrlDocument<'a> {
    #[serde(default, borrow)]
    registry: Option<UrlRegistry<'a>>,
    #[serde(default, borrow)]
    raw: Option<&'a serde_json::value::RawValue>,
    #[serde(default, borrow)]
    sources: Option<&'a serde_json::value::RawValue>,
}

#[derive(Deserialize)]
struct UrlRegistry<'a> {
    #[serde(default)]
    url: Option<String>,
    #[serde(default, borrow)]
    raw: Option<&'a serde_json::value::RawValue>,
    #[serde(default, borrow)]
    sources: Option<&'a serde_json::value::RawValue>,
}

#[derive(Deserialize)]
struct SourceUrl {
    #[serde(default)]
    url: Option<String>,
}

fn registry_raw_json(json: &[u8]) -> Option<&serde_json::value::RawValue> {
    let document = serde_json::from_slice::<UrlDocument<'_>>(json).ok()?;
    if let Some(registry) = document.registry {
        registry.raw.or(registry.sources)
    } else {
        document.raw.or(document.sources)
    }
}

/// Parse only the normalized record, letting serde skip the potentially huge
/// raw subtree without allocating it.
fn parse_registry_record(json: &[u8]) -> Option<Registry> {
    if let Ok(sidecar) = serde_json::from_slice::<SidecarRecord>(json)
        && let Some(registry) = sidecar.registry
    {
        return Some(registry.record);
    }
    if let Ok(envelope) = serde_json::from_slice::<RecordSlot>(json) {
        return Some(envelope.record);
    }
    serde_json::from_slice::<Registry>(json).ok()
}

/// Preserve a provenance document and recover its normalized registry record.
/// Accepts
/// any of three shapes:
/// - a hopper sidecar (record nested under `registry`, as the worker receives
///   from `/api/provenance/{sha256}`),
/// - a bare `fletch registry` envelope (`{record, sources}`, straight from
///   fletch's stdout), or
/// - a bare normalized [`Registry`] record for legacy `--registry-map` callers.
///
/// `None` when the document is malformed or carries no record — registry
/// provenance enriches a scan but is never required, so absence is not an error.
#[must_use]
pub fn registry_provenance(json: &[u8]) -> Option<RegistryProvenance> {
    RegistryProvenance::from_json(json)
}

/// Parse a `--registry-map` without constructing a second full JSON value tree.
///
/// Each value is initially borrowed as [`serde_json::value::RawValue`], then
/// copied once into its long-lived compact buffer. Peak memory is therefore the
/// input file plus compact per-entry documents, rather than the input plus a
/// 3–6× parsed representation of every provider response.
///
/// # Errors
/// Returns serde's parse error when the top-level map is malformed.
pub fn registry_map(json: &[u8]) -> Result<HashMap<String, RegistryProvenance>, serde_json::Error> {
    let raw: HashMap<String, Box<serde_json::value::RawValue>> = serde_json::from_slice(json)?;
    Ok(raw
        .into_iter()
        .filter_map(|(sha, value)| {
            RegistryProvenance::from_raw_value(value).map(|provenance| (sha, provenance))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{registry_map, registry_provenance};

    // A minimal normalized record — the shape `fletch registry` emits under
    // `record`. Only the few fields the worker logs are asserted.
    const RECORD: &str = r#"{"ecosystem":"npm","name":"left-pad","version":"1.3.0"}"#;

    fn registry_record(json: &[u8]) -> Option<fletch::Registry> {
        registry_provenance(json).map(|provenance| provenance.record)
    }

    #[test]
    fn reads_record_from_hopper_sidecar() {
        // The full sidecar shape: registry record nested under `registry`.
        let json = format!(
            r#"{{"artifact":{{"sha256":"ab"}},"registry":{{"record":{RECORD},"sources":[]}}}}"#
        );
        let rec = registry_record(json.as_bytes()).expect("record present");
        assert_eq!(rec.ecosystem, "npm");
        assert_eq!(rec.name, "left-pad");
        assert_eq!(rec.version, "1.3.0");
    }

    #[test]
    fn preserves_complete_hopper_sidecar_and_exposes_raw() {
        let json = format!(
            r#"{{"schema_version":"1.0","future":{{"kept":true}},"registry":{{"url":"https://registry.example/left-pad","record":{RECORD},"raw":{{"provider_only":"value","nested":[1,2,3]}}}}}}"#
        );
        let provenance = registry_provenance(json.as_bytes()).expect("provenance present");
        assert_eq!(provenance.record.name, "left-pad");
        assert_eq!(provenance.document_value().unwrap()["future"]["kept"], true);
        assert_eq!(provenance.raw().unwrap()["provider_only"], "value");
        assert_eq!(
            provenance.source_urls(),
            vec!["https://registry.example/left-pad"]
        );
    }

    #[test]
    fn preserves_bare_fletch_sources() {
        let json = format!(
            r#"{{"record":{RECORD},"sources":[{{"url":"https://registry.example/left-pad","status":200,"body":{{"provider_only":42}}}}],"future":"kept"}}"#
        );
        let provenance = registry_provenance(json.as_bytes()).expect("provenance present");
        assert_eq!(provenance.document_value().unwrap()["future"], "kept");
        assert_eq!(provenance.raw().unwrap()[0]["body"]["provider_only"], 42);
    }

    #[test]
    fn wraps_bare_provenance_for_hopper_without_losing_raw() {
        let json = format!(
            r#"{{"record":{RECORD},"sources":[{{"url":"https://registry.example/left-pad","status":200,"body":{{"provider_only":42}}}}]}}"#
        );
        let provenance = registry_provenance(json.as_bytes()).expect("provenance present");
        let sidecar = super::Upload {
            filename: "left-pad.tgz",
            sha256: &"a".repeat(64),
            size_bytes: 123,
            collector: "scan+test",
            at: "2026-07-30T00:00:00Z",
            url: "",
            purl: None,
        }
        .sidecar_from_provenance(&provenance)
        .unwrap();
        let sidecar: serde_json::Value = serde_json::from_slice(&sidecar).unwrap();
        assert_eq!(sidecar["registry"]["raw"][0]["body"]["provider_only"], 42);
        assert_eq!(sidecar["registry"]["record"]["name"], "left-pad");
        assert_eq!(sidecar["artifact"]["sha256"], "a".repeat(64));
    }

    #[test]
    fn wrapping_bare_provenance_applies_hopper_raw_cap() {
        let large = format!(r#"{{"blob":"{}"}}"#, "x".repeat(super::MAX_RAW_BYTES)).into_bytes();
        let provenance = super::RegistryProvenance::from_record_sources(
            fletch::Registry {
                ecosystem: "npm".to_string(),
                name: "large".to_string(),
                version: "1".to_string(),
                ..fletch::Registry::default()
            },
            &[fletch::fetch::RecordedSource {
                url: "https://registry.example/large".to_string(),
                status: 200,
                content_type: Some("application/json".to_string()),
                size: large.len() as u64,
                bytes: Some(large),
            }],
        );
        let sidecar = super::Upload {
            filename: "large.tgz",
            sha256: &"a".repeat(64),
            size_bytes: 123,
            collector: "scan+test",
            at: "2026-07-30T00:00:00Z",
            url: "https://registry.example/large.tgz",
            purl: Some("pkg:npm/large@1"),
        }
        .sidecar_from_provenance(&provenance)
        .unwrap();
        let sidecar: serde_json::Value = serde_json::from_slice(&sidecar).unwrap();
        assert_eq!(sidecar["registry"]["status"], "partial");
        assert!(sidecar["registry"].get("raw").is_none());
        assert_eq!(sidecar["package"]["purl"], "pkg:npm/large@1");
    }

    #[test]
    fn reuses_complete_hopper_sidecar_semantically() {
        let json = format!(
            r#"{{"schema_version":"1.0","artifact":{{"sha256":"ab"}},"fetch":{{"collector":"forager","category":"malware","at":"2026-07-30T00:00:00Z","url":"https://example.test"}},"future":{{"kept":true}},"registry":{{"record":{RECORD},"raw":{{"provider_only":42}}}}}}"#
        );
        let provenance = registry_provenance(json.as_bytes()).expect("provenance present");
        let sidecar = super::Upload {
            filename: "ignored.tgz",
            sha256: "ignored",
            size_bytes: 0,
            collector: "scan+test",
            at: "2026-07-30T00:00:00Z",
            url: "",
            purl: None,
        }
        .sidecar_from_provenance(&provenance)
        .unwrap();
        let sidecar: serde_json::Value = serde_json::from_slice(&sidecar).unwrap();
        assert_eq!(sidecar["future"]["kept"], true);
        assert_eq!(sidecar["registry"]["raw"]["provider_only"], 42);
        assert_eq!(sidecar["fetch"]["collector"], "forager");
        assert_eq!(sidecar["artifact"]["sha256"], "ignored");
    }

    #[test]
    fn registry_map_preserves_raw_without_a_value_tree() {
        let sha = "a".repeat(64);
        let json = format!(
            r#"{{"{sha}":{{"record":{RECORD},"sources":[{{"url":"https://registry.example/left-pad","status":200,"body":{{"provider_only":42}}}}]}}}}"#
        );
        let map = registry_map(json.as_bytes()).expect("map parses");
        let provenance = map.get(&sha).expect("sha entry");
        assert_eq!(provenance.record.name, "left-pad");
        assert_eq!(provenance.raw().unwrap()[0]["body"]["provider_only"], 42);
        assert!(
            provenance.document().len() < json.len(),
            "entry stores only its own compact document, not the containing map"
        );
    }

    #[test]
    fn owned_worker_buffer_is_adopted_and_clones_share_it() {
        let bytes = bytes::Bytes::from_static(
            br#"{"record":{"ecosystem":"npm","name":"left-pad","version":"1.3.0"},"sources":[]}"#,
        );
        let ptr = bytes.as_ptr();
        let provenance =
            super::RegistryProvenance::from_bytes(bytes).expect("worker document parses");
        assert_eq!(provenance.document().as_ptr(), ptr);
        let cloned = provenance.clone();
        assert_eq!(cloned.document().as_ptr(), ptr);
        assert_eq!(provenance.document().as_ptr(), cloned.document().as_ptr());
    }

    #[test]
    fn reads_record_from_bare_fletch_envelope() {
        // Raw `fletch registry` stdout: the envelope itself, no sidecar wrapper.
        let json = format!(r#"{{"record":{RECORD},"sources":[]}}"#);
        let rec = registry_record(json.as_bytes()).expect("record present");
        assert_eq!(rec.name, "left-pad");
    }

    #[test]
    fn reads_bare_record() {
        // A bare normalized record — the per-sha value gauntlet puts in a
        // `--registry-map` after extracting just `registry.record`.
        let rec = registry_record(RECORD.as_bytes()).expect("record present");
        assert_eq!(rec.name, "left-pad");
        assert_eq!(rec.version, "1.3.0");
    }

    #[test]
    fn none_when_sidecar_has_no_registry_slot() {
        // A human upload / feed-only sidecar carries no registry record.
        let json = r#"{"artifact":{"sha256":"ab"},"feed":{"source_id":"npm"}}"#;
        assert!(registry_record(json.as_bytes()).is_none());
    }

    #[test]
    fn none_on_malformed_or_empty() {
        assert!(registry_record(b"not json at all").is_none());
        assert!(registry_record(b"").is_none());
        assert!(registry_record(b"{}").is_none());
    }

    #[test]
    fn build_sidecar_with_registry_round_trips_and_carries_schema() {
        let reg = fletch::Registry {
            ecosystem: "npm".into(),
            name: "left-pad".into(),
            version: "1.3.0".into(),
            ..Default::default()
        };
        let sources = vec![
            fletch::fetch::RecordedSource {
                url: "https://registry.npmjs.org/left-pad".to_string(),
                status: 200,
                content_type: Some("application/json".to_string()),
                size: 19,
                bytes: Some(br#"{"name":"left-pad"}"#.to_vec()),
            },
            fletch::fetch::RecordedSource {
                url: "https://chromewebstore.example/detail".to_string(),
                status: 200,
                content_type: Some("text/html".to_string()),
                size: 21,
                bytes: Some(b"<html>not json</html>".to_vec()),
            },
        ];
        let json = super::Upload {
            filename: "left-pad-1.3.0.tgz",
            sha256: &"a".repeat(64),
            size_bytes: 1234,
            collector: "scan+host",
            at: "2026-06-25T00:00:00Z",
            url: "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz",
            purl: Some("pkg:npm/left-pad@1.3.0"),
        }
        .sidecar(Some(&reg), &sources)
        .unwrap();

        // What scan writes, scan (and a worker) reads back as the same record.
        let rec = registry_record(&json).expect("registry record present");
        assert_eq!(rec.ecosystem, "npm");
        assert_eq!(rec.name, "left-pad");
        assert_eq!(rec.version, "1.3.0");

        // The fields hopper's sidecar validator requires / dispatches on.
        let v: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(v["schema_version"], "1.0");
        assert_eq!(v["artifact"]["sha256"], "a".repeat(64));
        assert_eq!(v["artifact"]["size_bytes"], 1234);
        assert_eq!(v["fetch"]["category"], "submitted");
        assert_eq!(v["fetch"]["collector"], "scan+host");
        assert_eq!(v["package"]["purl"], "pkg:npm/left-pad@1.3.0");
        assert_eq!(v["registry"]["format"], "fletch.registry");

        // Raw sources are archived with transport facts: a JSON body inline, a
        // non-JSON body as base64.
        let raw = v["registry"]["raw"].as_array().expect("raw array");
        assert_eq!(raw.len(), 2);
        assert_eq!(raw[0]["body"]["name"], "left-pad");
        assert_eq!(raw[0]["status"], 200);
        assert_eq!(raw[0]["content_type"], "application/json");
        assert!(raw[0].get("body_b64").is_none());
        assert!(raw[1]["body_b64"].is_string());
        assert!(raw[1].get("body").is_none());
    }

    #[test]
    fn build_sidecar_drops_over_cap_raw_and_downgrades_status() {
        // A packument larger than MAX_RAW_BYTES must not ride along verbatim: it
        // would push the provenance part past hopper's transport cap and be
        // rejected. Drop raw, mark the record partial, keep the normalized record.
        let reg = fletch::Registry {
            ecosystem: "npm".into(),
            name: "socket".into(),
            version: "1.1.137".into(),
            ..Default::default()
        };
        let huge = format!(
            r#"{{"name":"socket","blob":"{}"}}"#,
            "x".repeat(super::MAX_RAW_BYTES)
        );
        let sources = vec![fletch::fetch::RecordedSource {
            url: "https://registry.npmjs.org/socket".to_string(),
            status: 200,
            content_type: Some("application/json".to_string()),
            size: huge.len() as u64,
            bytes: Some(huge.into_bytes()),
        }];
        let json = super::Upload {
            filename: "socket-1.1.137.tgz",
            sha256: &"a".repeat(64),
            size_bytes: 5083868,
            collector: "scan+galadriel",
            at: "2026-07-03T16:34:16Z",
            url: "https://registry.npmjs.org/socket/-/socket-1.1.137.tgz",
            purl: Some("pkg:npm/socket@1.1.137"),
        }
        .sidecar(Some(&reg), &sources)
        .unwrap();

        assert!(
            json.len() < super::MAX_RAW_BYTES,
            "oversized raw was not dropped"
        );
        let v: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(v["registry"]["status"], "partial");
        assert!(v["registry"].get("raw").is_none());
        // The normalized record scan reasons over still round-trips.
        let rec = registry_record(&json).expect("registry record present");
        assert_eq!(rec.name, "socket");
        assert_eq!(rec.version, "1.1.137");
    }

    #[test]
    fn build_sidecar_dep_carries_purl_even_without_a_registry_record() {
        // A fetched dependency whose registry lookup didn't resolve still uploads
        // with its PURL, so hopper can populate purl_base.
        let json = super::Upload {
            filename: "assertion-error-2.0.1.tgz",
            sha256: &"d".repeat(64),
            size_bytes: 500,
            collector: "scan+host",
            at: "2026-06-25T00:00:00Z",
            url: "https://registry.npmjs.org/assertion-error/-/assertion-error-2.0.1.tgz",
            purl: Some("pkg:npm/assertion-error@2.0.1"),
        }
        .sidecar(None, &[])
        .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(v["package"]["purl"], "pkg:npm/assertion-error@2.0.1");
        // No record resolved, so no registry slot — but the PURL still rode along.
        assert!(v.get("registry").is_none());
    }

    /// A document that is not a sidecar is an error to the caller, never an
    /// empty or half-built upload.
    #[test]
    fn rebinding_something_other_than_a_sidecar_is_an_error() {
        let artifact = super::ArtifactRef {
            filename: "a.tgz",
            sha256: "cafe",
            size_bytes: 1,
        };
        assert!(super::rebind(serde_json::json!(["not", "a", "sidecar"]), artifact).is_err());
    }

    /// An upload names its package only when it has one; a PURL without a
    /// resolved record still rides, alone.
    #[test]
    fn an_upload_names_its_package_only_when_it_has_one() {
        let sha = "a".repeat(64);
        let upload = super::Upload {
            filename: "x.tgz",
            sha256: &sha,
            size_bytes: 1,
            collector: "scan+test",
            at: "2026-01-01T00:00:00Z",
            url: "",
            purl: None,
        };
        let plain: serde_json::Value =
            serde_json::from_slice(&upload.sidecar(None, &[]).unwrap()).unwrap();
        assert!(plain.get("package").is_none() && plain.get("registry").is_none());
        let named = super::Upload {
            purl: Some("pkg:npm/x@1"),
            ..upload
        };
        let named: serde_json::Value =
            serde_json::from_slice(&named.sidecar(None, &[]).unwrap()).unwrap();
        assert_eq!(named["package"], serde_json::json!({"purl": "pkg:npm/x@1"}));
    }

    #[test]
    fn build_sidecar_without_registry_omits_package_and_registry() {
        let json = super::Upload {
            filename: "mal.bin",
            sha256: &"b".repeat(64),
            size_bytes: 10,
            collector: "scan+host",
            at: "2026-06-25T00:00:00Z",
            url: "",
            purl: None,
        }
        .sidecar(None, &[])
        .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(v["artifact"]["filename"], "mal.bin");
        assert!(
            v.get("registry").is_none(),
            "no registry slot for a local file"
        );
        assert!(
            v.get("package").is_none(),
            "no package slot for a local file"
        );
    }
}
