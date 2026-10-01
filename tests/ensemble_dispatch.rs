//! Integration tests for ensemble routing in `scan::model::Model`.
//!
//! Builds a temporary ensemble bundle (general/ + filegroups/native/ +
//! filetypes/elf/) from a real ONNX bundle and verifies:
//!   * `predict_report` on a report of a given file type consults the right
//!     specialists.
//!   * Files whose type is unmapped route to general only.
//!   * `required_routes` lists are enforced.
//!
//! Tests are `#[ignore]`d by default because they need a real bundle on
//! disk. Run with:
//!
//! ```sh
//! SCAN_ONNX_BUNDLE=/path/to/onnx-bundle \
//!     cargo test --test ensemble_dispatch -- --ignored
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use scan::model::Model;

fn onnx_bundle() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("SCAN_ONNX_BUNDLE")?);
    (p.join("model.onnx").is_file() && p.join("feature_spec.json").is_file()).then_some(p)
}

/// A full ensemble bundle (`general/` + `config.json` + `route_policies.json`).
fn ensemble_bundle() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("AZOTH_DIR")?);
    p.join("general").is_dir().then_some(p)
}

/// Loading the real bundle parses the full per-level grid from the multi-MB
/// `route_policies.json` and `config.json`, then calibrates every level. The
/// `l` sweep is level-independent, so the active level passed at load time must
/// not change whether the bundle loads. Gated on a real bundle:
///
/// ```sh
/// AZOTH_DIR=/path/to/ensemble-bundle cargo test --test ensemble_dispatch \
///     real_azoth_bundle_loads_grid_at_every_level -- --ignored
/// ```
#[test]
#[ignore = "needs AZOTH_DIR pointing at a real ensemble bundle"]
fn real_azoth_bundle_loads_grid_at_every_level() {
    let Some(dir) = ensemble_bundle() else {
        eprintln!("set AZOTH_DIR to a real ensemble bundle to run this test");
        return;
    };
    for level in [Some(0_u16), Some(50), Some(200), Some(1000), None] {
        Model::load(&dir, None, level)
            .unwrap_or_else(|e| panic!("azoth must load at level {level:?}: {e}"));
    }
}

/// Stage `general/` plus optionally `filegroups/<group>` and
/// `filetypes/<type>` from a single source bundle. All routes get the same
/// model, which is sufficient to exercise the routing decision (each route's
/// score is identical, so OR semantics are easy to verify).
fn stage_ensemble(
    src: &Path,
    groups: &[&str],
    types: &[&str],
    config_json: &str,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");

    let copy_bundle_to = |dest: &Path| {
        std::fs::create_dir_all(dest).unwrap();
        for f in &["model.onnx", "feature_spec.json"] {
            std::fs::copy(src.join(f), dest.join(f)).unwrap();
        }
    };

    copy_bundle_to(&dir.path().join("general"));
    for g in groups {
        copy_bundle_to(&dir.path().join("filegroups").join(g));
    }
    for t in types {
        copy_bundle_to(&dir.path().join("filetypes").join(t));
    }

    std::fs::write(dir.path().join("config.json"), config_json).unwrap();
    dir
}

/// A report whose primary file is of type `file_type`.
fn report_of(file_type: &str) -> cleave::types::CompactReport {
    let mut report = cleave::types::CompactReport::default();
    report.files.push(cleave::types::CompactFile {
        file_type: file_type.to_string(),
        ..Default::default()
    });
    report
}

/// The decided probability and the routes scored for a `file_type` report.
fn routes(model: &Model, file_type: &str) -> (f32, Vec<String>) {
    let (decision, scores, _) = model
        .predict_report(&report_of(file_type))
        .unwrap_or_else(|e| panic!("predict {file_type}: {e:#}"));
    (
        decision.probability,
        scores.into_iter().map(|s| s.model).collect(),
    )
}

/// An ensemble config registering every staged route at the default level.
const CALIBRATED: &str = r#"{
  "filetype_to_group": { "elf": "native", "pe": "native" },
  "levels": [{
    "level": 25,
    "hostile": {"thresholds": {
      "general": 0.99, "filegroups/native": 0.99, "filetypes/elf": 0.99
    }}
  }]
}"#;

#[test]
#[ignore = "needs SCAN_ONNX_BUNDLE pointing at a real ONNX bundle"]
fn ensemble_routes_to_filetype_specialist_when_present() {
    let Some(src) = onnx_bundle() else {
        panic!("SCAN_ONNX_BUNDLE not set or missing artifacts");
    };
    let dir = stage_ensemble(&src, &["native"], &["elf"], CALIBRATED);
    let model = Model::load(dir.path(), None, None).expect("load ensemble");

    // Files of type elf consult: general + filegroup(native) + filetype(elf).
    let (prob_elf, elf_routes) = routes(&model, "elf");
    assert_eq!(elf_routes, ["az", "az/native", "az/elf"]);

    // Files of type "python" are unmapped → route is general only.
    let (prob_py, py_routes) = routes(&model, "python");
    assert_eq!(py_routes, ["az"]);

    // Sanity: probabilities are finite.
    assert!(prob_elf.is_finite());
    assert!(prob_py.is_finite());
    // With identical models on every route, max-probability is the same
    // single-model probability for both file types.
    assert!((prob_elf - prob_py).abs() < 1e-6);
}

#[test]
#[ignore = "needs SCAN_ONNX_BUNDLE pointing at a real ONNX bundle"]
fn ensemble_routes_to_filegroup_when_filetype_absent() {
    let Some(src) = onnx_bundle() else {
        panic!("SCAN_ONNX_BUNDLE not set or missing artifacts");
    };
    // No filetypes/pe specialist; pe routes through filegroup(native) + general.
    let dir = stage_ensemble(&src, &["native"], &[], CALIBRATED);
    let model = Model::load(dir.path(), None, None).expect("load ensemble");

    let (prob, pe_routes) = routes(&model, "pe");
    assert_eq!(pe_routes, ["az", "az/native"]);
    assert!(prob.is_finite());
}

#[test]
#[ignore = "needs SCAN_ONNX_BUNDLE pointing at a real ONNX bundle"]
fn ensemble_with_only_general_falls_back_to_general() {
    let Some(src) = onnx_bundle() else {
        panic!("SCAN_ONNX_BUNDLE not set or missing artifacts");
    };
    let dir = stage_ensemble(&src, &[], &[], "{}");
    let model = Model::load(dir.path(), None, None).expect("load ensemble");

    // No specialists, so every file type scores on general alone.
    let (prob_general, general_routes) = routes(&model, "python");
    let (prob_elf, elf_routes) = routes(&model, "elf");
    assert_eq!(general_routes, ["az"]);
    assert_eq!(elf_routes, ["az"]);
    assert!((prob_general - prob_elf).abs() < 1e-6);
}

#[test]
#[ignore = "needs SCAN_ONNX_BUNDLE pointing at a real ONNX bundle"]
fn ensemble_required_route_missing_is_fatal() {
    let Some(src) = onnx_bundle() else {
        panic!("SCAN_ONNX_BUNDLE not set or missing artifacts");
    };
    let cfg = r#"{
      "required_routes": ["filetype:nonexistent"]
    }"#;
    let dir = stage_ensemble(&src, &[], &[], cfg);
    let err = Model::load(dir.path(), None, None).expect_err("missing required must fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("nonexistent") && msg.contains("required"),
        "error should call out the missing required route: {msg}"
    );
}
