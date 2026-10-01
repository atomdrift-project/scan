//! Setup shared by the worker integration tests.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

use scan::worker::{Rules, WorkerConfig, WorkerTuning};

/// The model bundle the worker tests load. They are `#[ignore]`d by default,
/// so a run that reaches here without one is a setup mistake: fail loudly.
pub(crate) fn models_dir() -> PathBuf {
    std::env::var_os("SCAN_MODELS_DIR")
        .map(PathBuf::from)
        .expect("set SCAN_MODELS_DIR to a model bundle to run this test")
}

/// A standalone two-slot worker polling the mock hopper at `hopper_url` every
/// second. It never updates rules and never exits on its own: the test ends it.
pub(crate) fn worker_config(
    name: &str,
    hopper_url: String,
    data_dir: Option<PathBuf>,
) -> WorkerConfig {
    WorkerConfig {
        renew_rules: false,
        hopper_url,
        name: name.into(),
        workers: NonZeroUsize::new(2).expect("2 workers"),
        poll_interval: Duration::from_secs(1),
        max_rss: None,
        data_dir,
        max_jobs: None,
        exit_if_empty: false,
        nice: 0,
        rules: Rules {
            model_dir: models_dir(),
            level: None,
            thresholds: None,
            slow_rule_ms: 4000,
            interpret: None,
            fetch: scan::fetch::FetchPolicy::default(),
            zip_passwords: scan::ArchivePasswords::default(),
        },
        tuning: WorkerTuning::default(),
    }
}
