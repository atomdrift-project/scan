//! Download bytes identify their registry ecosystem even when cached by digest.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use cleave::types::compact_from_files;
use fletch::fetch::{BlobCache, Fixtures, Outcome, fetch_ref};
use fletch::{RefKind, RefLocator, Reference};

#[test]
fn package_download_types_reach_model_json() {
    let cases: &[(&str, &str, &[u8])] = &[
        (
            "crate",
            "cargo/demo",
            include_bytes!("data/package-types/crate.tar.gz"),
        ),
        (
            "pub",
            "pub/demo",
            include_bytes!("data/package-types/pub.tar.gz"),
        ),
        (
            "cran",
            "cran/demo",
            include_bytes!("data/package-types/cran.tar.gz"),
        ),
        (
            "cpan",
            "cpan/AUTHOR/Demo",
            include_bytes!("data/package-types/cpan.tar.gz"),
        ),
        (
            "hackage",
            "hackage/demo",
            include_bytes!("data/package-types/hackage.tar.gz"),
        ),
        (
            "hex",
            "hex/demo",
            include_bytes!("data/package-types/hex.tar"),
        ),
        (
            "composer",
            "composer/vendor/demo",
            include_bytes!("data/package-types/composer.zip"),
        ),
    ];
    let dir = tempfile::tempdir().unwrap();
    let cache = BlobCache::with_dir(dir.path());
    for (label, coordinate, data) in cases {
        // The exact-download qualifier is supported for every ecosystem and
        // deliberately supplies a URL without a format-bearing suffix.
        let url = format!("https://packages.example.test/download/{label}");
        let reference = Reference::new(
            RefLocator::Purl(format!("pkg:{coordinate}@1.2.3?download_url={url}")),
            RefKind::Dependency,
            "test",
            "",
        );
        let record = fetch_ref(&reference, &Fixtures::default().with(&url, data), &cache);
        assert!(matches!(record.outcome, Outcome::Ok), "{record:?}");
        let bytes = cache.load(&record.locator).expect("download cached");
        let path = dir.path().join(format!("download-{label}"));
        std::fs::write(&path, bytes).unwrap();
        let mut report = cleave::Engine::empty()
            .with_compact_members(true)
            .analyze_file(&path, &cleave::AnalysisOptions::default())
            .unwrap();
        report.finalize();
        let json = serde_json::to_value(compact_from_files(&report.files)).unwrap();
        assert_eq!(json["files"][0]["type"], *label);
        assert!(
            json["files"].as_array().unwrap().len() > 1,
            "{label}: members retained"
        );
    }
}
