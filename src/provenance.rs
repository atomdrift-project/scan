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

/// The sidecar schema version hopper validates against ([`hopper.SidecarSchemaVersion`]).
const SCHEMA_VERSION: &str = "1.0";

/// Cap on `registry.raw`, matching hopper's `MaxRawBytes` (256 KiB). An upstream
/// document can be huge — a full npm packument carries every published version —
/// and embedding it verbatim would push the provenance part past hopper's 1 MiB
/// transport cap, where it is truncated into unparseable JSON and rejected. When
/// `raw` overflows the cap we drop it and downgrade the record to "partial", the
/// same trim hopper's `Sidecar.Finalize` performs on receipt.
const MAX_RAW_BYTES: usize = 256 << 10;

/// Build a hopper provenance sidecar JSON for an artifact about to be uploaded
/// to `/api/upload`. Mirrors hopper's `Sidecar` Go struct field-for-field so the
/// upload validator accepts it: `schema_version`, `artifact`, `fetch`, and —
/// when the artifact is a fetched package — `package` + `registry{record}`.
///
/// `purl` is the package's canonical PURL (empty for the scanned root file or a
/// plain-URL fetch); `registry` is the normalized record scan derived for a
/// fetched dependency (`None` for the root file). `sources` are the raw provider
/// documents the registry lookup read (`(url, bytes)`), archived under
/// `registry.raw` as the re-parsing backup — the same shape forager stores: a
/// JSON body inline, anything else base64 in `body_b64`.
#[must_use]
#[allow(clippy::too_many_arguments)] // a flat sidecar record; a params struct would only indirect it
pub fn build_sidecar(
    filename: &str,
    sha256: &str,
    size_bytes: u64,
    collector: &str,
    now_rfc3339: &str,
    url: &str,
    purl: &str,
    registry: Option<&Registry>,
    sources: &[fletch::fetch::RecordedSource],
) -> Vec<u8> {
    let mut sidecar = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "artifact": { "filename": filename, "sha256": sha256, "size_bytes": size_bytes },
        // category "submitted": a scan push is a discovered-by-us artifact, not a
        // labeled feed sample. hopper records it but derives the real label from
        // analysis, never from this claim.
        "fetch": { "collector": collector, "category": "submitted", "at": now_rfc3339, "url": url },
    });
    // A fetched dependency always carries its PURL, so hopper can project the
    // version-less form into the queryable purl_base column — independent of
    // whether the registry lookup resolved. Identity fields are filled from the
    // record when present.
    if !purl.is_empty() {
        let mut package = serde_json::json!({ "purl": purl });
        if let Some(reg) = registry {
            package["ecosystem"] = serde_json::json!(reg.ecosystem);
            package["name"] = serde_json::json!(reg.name);
            package["version"] = serde_json::json!(reg.version);
        }
        sidecar["package"] = package;
    }
    if let Some(reg) = registry {
        // Drop an over-cap raw archive rather than ship an oversized part that
        // hopper cannot parse; see [`MAX_RAW_BYTES`]. The normalized `record`
        // is small and bounded, so the record scan reasons over is unaffected.
        let raw = raw_sources(sources);
        let over_cap = serde_json::to_vec(&raw).map_or(0, |b| b.len()) > MAX_RAW_BYTES;
        sidecar["registry"] = serde_json::json!({
            "source_id": reg.ecosystem,
            "ecosystem": reg.ecosystem,
            "format": "fletch.registry",
            "url": url,
            "at": now_rfc3339,
            "status": if over_cap { "partial" } else { "complete" },
            "record": reg,
        });
        if !over_cap {
            sidecar["registry"]["raw"] = raw;
        }
    }
    serde_json::to_vec(&sidecar).unwrap_or_default()
}

/// Produce an uploadable hopper sidecar while preserving provenance supplied by
/// a worker or `--registry-map`.
///
/// A complete hopper sidecar is reused as-is semantically. Bare fletch envelopes
/// and legacy records are wrapped in a current sidecar; their raw provider data,
/// when present, is copied into `registry.raw` without schema-specific parsing.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_sidecar_from_provenance(
    filename: &str,
    sha256: &str,
    size_bytes: u64,
    collector: &str,
    now_rfc3339: &str,
    url: &str,
    purl: &str,
    provenance: &RegistryProvenance,
) -> Vec<u8> {
    if serde_json::from_slice::<CompleteSidecarProbe>(provenance.document()).is_ok() {
        let mut document = provenance.document_value().unwrap_or_default();
        // The provenance wrapper is historical; the artifact binding is a
        // present-tense integrity claim and must match the bytes being uploaded.
        // Rebind it even for a complete sidecar so a stale/mistyped map entry
        // cannot make hopper reject otherwise valid bytes.
        document["schema_version"] = serde_json::json!(SCHEMA_VERSION);
        document["artifact"] = serde_json::json!({
            "filename": filename,
            "sha256": sha256,
            "size_bytes": size_bytes,
        });
        trim_registry_raw(&mut document);
        return serde_json::to_vec(&document).unwrap_or_default();
    }

    let record = &provenance.record;
    // RawValue points directly into the compact document. Its length is a safe
    // cap check (possibly conservative only when an external producer included
    // whitespace) and it serializes verbatim, avoiding a Value tree entirely.
    let raw =
        registry_raw_json(provenance.document()).filter(|raw| raw.get().len() <= MAX_RAW_BYTES);
    let sidecar = WrappedSidecar {
        schema_version: SCHEMA_VERSION,
        artifact: WrappedArtifact {
            filename,
            sha256,
            size_bytes,
        },
        package: (!purl.is_empty()).then_some(WrappedPackage {
            ecosystem: &record.ecosystem,
            name: &record.name,
            version: &record.version,
            purl,
        }),
        fetch: WrappedFetch {
            collector,
            category: "submitted",
            at: now_rfc3339,
            url,
        },
        registry: WrappedRegistry {
            source_id: &record.ecosystem,
            ecosystem: &record.ecosystem,
            format: "fletch.registry",
            url,
            at: now_rfc3339,
            status: if raw.is_some() { "complete" } else { "partial" },
            record,
            raw,
        },
    };
    serde_json::to_vec(&sidecar).unwrap_or_default()
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

#[derive(serde::Serialize)]
struct WrappedSidecar<'a> {
    schema_version: &'static str,
    artifact: WrappedArtifact<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    package: Option<WrappedPackage<'a>>,
    fetch: WrappedFetch<'a>,
    registry: WrappedRegistry<'a>,
}

#[derive(serde::Serialize)]
struct WrappedArtifact<'a> {
    filename: &'a str,
    sha256: &'a str,
    size_bytes: u64,
}

#[derive(serde::Serialize)]
struct WrappedPackage<'a> {
    ecosystem: &'a str,
    name: &'a str,
    version: &'a str,
    purl: &'a str,
}

#[derive(serde::Serialize)]
struct WrappedFetch<'a> {
    collector: &'a str,
    category: &'static str,
    at: &'a str,
    url: &'a str,
}

#[derive(serde::Serialize)]
struct WrappedRegistry<'a> {
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

/// Encode raw provider documents for the sidecar's `registry.raw`: each as
/// `{url, status, content_type?, body|body_b64}` — a JSON body inline (the
/// package-registry case), anything else base64. Matches the `sources` shape
/// `fletch registry` emits and forager stores, so a future re-parse handles both
/// producers uniformly.
fn raw_sources(sources: &[fletch::fetch::RecordedSource]) -> serde_json::Value {
    use base64::Engine as _;
    let entries: Vec<serde_json::Value> = sources
        .iter()
        .map(|s| {
            let mut entry = match serde_json::from_slice::<serde_json::Value>(&s.bytes) {
                Ok(body) => serde_json::json!({ "url": s.url, "body": body }),
                Err(_) => serde_json::json!({
                    "url": s.url,
                    "body_b64": base64::engine::general_purpose::STANDARD.encode(&s.bytes),
                }),
            };
            entry["status"] = serde_json::json!(s.status);
            if let Some(ct) = &s.content_type {
                entry["content_type"] = serde_json::json!(ct);
            }
            entry
        })
        .collect();
    serde_json::Value::Array(entries)
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

        let document = serde_json::to_vec(&LiveDocument {
            record: &record,
            sources: encoded_sources(sources),
        })
        .unwrap_or_default();
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
}

/// Borrow valid JSON bodies directly from fletch's source buffers so building a
/// long-lived provenance document does not allocate a `Value` tree or reformat a
/// large packument. Non-JSON bodies pay only the unavoidable base64 allocation.
fn encoded_sources(sources: &[fletch::fetch::RecordedSource]) -> Vec<EncodedSource<'_>> {
    use base64::Engine as _;

    sources
        .iter()
        .map(
            |source| match serde_json::from_slice::<&serde_json::value::RawValue>(&source.bytes) {
                Ok(body) => EncodedSource::Json {
                    url: &source.url,
                    status: source.status,
                    content_type: source.content_type.as_deref(),
                    body,
                },
                Err(_) => EncodedSource::Binary {
                    url: &source.url,
                    status: source.status,
                    content_type: source.content_type.as_deref(),
                    body_b64: base64::engine::general_purpose::STANDARD.encode(&source.bytes),
                },
            },
        )
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
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
        let sidecar = super::build_sidecar_from_provenance(
            "left-pad.tgz",
            &"a".repeat(64),
            123,
            "scan+test",
            "2026-07-30T00:00:00Z",
            "",
            "",
            &provenance,
        );
        let sidecar: serde_json::Value = serde_json::from_slice(&sidecar).unwrap();
        assert_eq!(sidecar["registry"]["raw"][0]["body"]["provider_only"], 42);
        assert_eq!(sidecar["registry"]["record"]["name"], "left-pad");
        assert_eq!(sidecar["artifact"]["sha256"], "a".repeat(64));
    }

    #[test]
    fn wrapping_bare_provenance_applies_hopper_raw_cap() {
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
                bytes: format!(r#"{{"blob":"{}"}}"#, "x".repeat(super::MAX_RAW_BYTES)).into_bytes(),
            }],
        );
        let sidecar = super::build_sidecar_from_provenance(
            "large.tgz",
            &"a".repeat(64),
            123,
            "scan+test",
            "2026-07-30T00:00:00Z",
            "https://registry.example/large.tgz",
            "pkg:npm/large@1",
            &provenance,
        );
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
        let sidecar = super::build_sidecar_from_provenance(
            "ignored.tgz",
            "ignored",
            0,
            "scan+test",
            "2026-07-30T00:00:00Z",
            "",
            "",
            &provenance,
        );
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
                bytes: br#"{"name":"left-pad"}"#.to_vec(),
            },
            fletch::fetch::RecordedSource {
                url: "https://chromewebstore.example/detail".to_string(),
                status: 200,
                content_type: Some("text/html".to_string()),
                bytes: b"<html>not json</html>".to_vec(),
            },
        ];
        let json = super::build_sidecar(
            "left-pad-1.3.0.tgz",
            &"a".repeat(64),
            1234,
            "scan+host",
            "2026-06-25T00:00:00Z",
            "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz",
            "pkg:npm/left-pad@1.3.0",
            Some(&reg),
            &sources,
        );

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
            bytes: huge.into_bytes(),
        }];
        let json = super::build_sidecar(
            "socket-1.1.137.tgz",
            &"a".repeat(64),
            5083868,
            "scan+galadriel",
            "2026-07-03T16:34:16Z",
            "https://registry.npmjs.org/socket/-/socket-1.1.137.tgz",
            "pkg:npm/socket@1.1.137",
            Some(&reg),
            &sources,
        );

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
        let json = super::build_sidecar(
            "assertion-error-2.0.1.tgz",
            &"d".repeat(64),
            500,
            "scan+host",
            "2026-06-25T00:00:00Z",
            "https://registry.npmjs.org/assertion-error/-/assertion-error-2.0.1.tgz",
            "pkg:npm/assertion-error@2.0.1",
            None,
            &[],
        );
        let v: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(v["package"]["purl"], "pkg:npm/assertion-error@2.0.1");
        // No record resolved, so no registry slot — but the PURL still rode along.
        assert!(v.get("registry").is_none());
    }

    #[test]
    fn build_sidecar_without_registry_omits_package_and_registry() {
        let json = super::build_sidecar(
            "mal.bin",
            &"b".repeat(64),
            10,
            "scan+host",
            "2026-06-25T00:00:00Z",
            "",
            "",
            None,
            &[],
        );
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
