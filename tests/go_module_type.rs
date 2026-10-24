//! Go PURL bytes retain the package type through scan's model report path.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use cleave::types::compact_from_files;
use fletch::fetch::{BlobCache, Fixtures, Outcome, fetch_ref};
use fletch::{RefKind, RefLocator, Reference};

#[test]
fn go_module_purl_download_reaches_model_json_with_its_type() {
    let data = include_bytes!("data/go-module-stream.zip");
    let net = Fixtures::default()
        .with(
            "https://proxy.golang.org/example.com/hello/@v/v1.2.3.zip",
            data,
        )
        .with(
            "https://proxy.golang.org/example.com/hello/@v/v1.2.3.info",
            br#"{"Version":"v1.2.3"}"#,
        );
    let dir = tempfile::tempdir().unwrap();
    let cache = BlobCache::with_dir(dir.path().join("cache"));
    let reference = Reference::new(
        RefLocator::Purl("pkg:golang/example.com/hello@v1.2.3".into()),
        RefKind::Dependency,
        "test",
        "",
    );
    let record = fetch_ref(&reference, &net, &cache);
    assert!(matches!(record.outcome, Outcome::Ok), "{record:?}");
    let bytes = cache.load(&record.locator).expect("Go download");
    let name = scan::fetch::fetched_url(&record)
        .unwrap()
        .rsplit('/')
        .next()
        .unwrap();
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).unwrap();
    let mut report = cleave::Engine::empty()
        .with_compact_members(true)
        .analyze_file(&path, &cleave::AnalysisOptions::default())
        .unwrap();
    report.finalize();
    let compact = compact_from_files(&report.files);
    let json = serde_json::to_value(compact).unwrap();
    assert_eq!(json["files"][0]["type"], "go_module");
    assert!(
        json["files"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["type"] == "go")
    );
}
