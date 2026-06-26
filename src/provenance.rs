//! Reading the registry-metadata provenance hopper stores per sample.
//!
//! A collector (forager) runs `fletch registry <purl>` at fetch time and stores
//! its `{record, sources}` envelope in the sample's sidecar under `registry`.
//! Both the worker (over HTTP, from `/api/provenance/{sha256}`) and the CLI
//! (`--provenance <file>`) read that sidecar to recover the normalized
//! [`fletch::Registry`] a scan reasons over, so a hopper-sourced scan sees the
//! same registry facts a live `pkg`/`url` scan fetches — without a refetch.

use fletch::Registry;
use serde::Deserialize;

/// The sidecar schema version hopper validates against ([`hopper.SidecarSchemaVersion`]).
const SCHEMA_VERSION: &str = "1.0";

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
        sidecar["registry"] = serde_json::json!({
            "source_id": reg.ecosystem,
            "ecosystem": reg.ecosystem,
            "format": "fletch.registry",
            "url": url,
            "at": now_rfc3339,
            "status": "complete",
            "record": reg,
            "raw": raw_sources(sources),
        });
    }
    serde_json::to_vec(&sidecar).unwrap_or_default()
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

/// The provenance sidecar, of which only the registry slot matters to a scan.
/// Threat-feed and artifact fields are deliberately not modeled.
#[derive(Deserialize)]
struct Sidecar {
    #[serde(default)]
    registry: Option<RegistryProvenance>,
}

/// The `fletch registry` envelope: the normalized record scan consumes, plus the
/// raw provider documents it was derived from (archived in hopper, ignored here).
#[derive(Deserialize)]
struct RegistryProvenance {
    record: Registry,
}

/// Recover the normalized registry record from a provenance document. Accepts
/// either a hopper sidecar (the record nested under `registry`, as the worker
/// receives from `/api/provenance/{sha256}`) or a bare `fletch registry`
/// envelope (the `{record, sources}` document itself, as `--registry` may be
/// handed straight from fletch's stdout). `None` when the document is malformed
/// or carries no record — registry provenance enriches a scan but is never
/// required, so absence is not an error.
#[must_use]
pub fn registry_record(json: &[u8]) -> Option<Registry> {
    if let Ok(sidecar) = serde_json::from_slice::<Sidecar>(json)
        && let Some(provenance) = sidecar.registry
    {
        return Some(provenance.record);
    }
    serde_json::from_slice::<RegistryProvenance>(json)
        .ok()
        .map(|envelope| envelope.record)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::registry_record;

    // A minimal normalized record — the shape `fletch registry` emits under
    // `record`. Only the few fields the worker logs are asserted.
    const RECORD: &str = r#"{"ecosystem":"npm","name":"left-pad","version":"1.3.0"}"#;

    #[test]
    fn reads_record_from_hopper_sidecar() {
        // The full sidecar shape: registry record nested under `registry`.
        let json = format!(r#"{{"artifact":{{"sha256":"ab"}},"registry":{{"record":{RECORD},"sources":[]}}}}"#);
        let rec = registry_record(json.as_bytes()).expect("record present");
        assert_eq!(rec.ecosystem, "npm");
        assert_eq!(rec.name, "left-pad");
        assert_eq!(rec.version, "1.3.0");
    }

    #[test]
    fn reads_record_from_bare_fletch_envelope() {
        // Raw `fletch registry` stdout: the envelope itself, no sidecar wrapper.
        let json = format!(r#"{{"record":{RECORD},"sources":[]}}"#);
        let rec = registry_record(json.as_bytes()).expect("record present");
        assert_eq!(rec.name, "left-pad");
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
        assert!(v.get("registry").is_none(), "no registry slot for a local file");
        assert!(v.get("package").is_none(), "no package slot for a local file");
    }
}
