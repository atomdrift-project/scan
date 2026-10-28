//! What a process runs with, settled once.
//!
//! A binary fills [`Settings`] from its flags and calls [`configure`] before
//! anything reads them; a library caller that never does gets the defaults.
//! Whichever comes first settles them for the life of the process, so no two
//! parts of one process can disagree about them.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result};

/// The process's settings.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    /// Wall-clock limit on each Rizin run; `None` keeps cleave's default.
    pub rizin_timeout: Option<Duration>,
    /// The larger limit for one retry of a native file whose Rizin run timed
    /// out; `None` does not retry.
    pub rizin_retry_timeout: Option<Duration>,
    /// What the fetch client and blob cache are built with.
    pub fetch: crate::fetch::Settings,
    /// A ceiling on everything one execution fetches; `None` is unlimited, as
    /// a server's is: each of its jobs is bounded on its own.
    pub fetch_total: Option<FetchTotal>,
    /// A file that keeps unfinished fetch work for a later run.
    pub fetch_backlog: Option<PathBuf>,
    /// Where scan's caches live instead of the platform cache directory.
    pub cache_dir: Option<PathBuf>,
    /// The terminal theme; `None` is dark.
    pub theme: Option<crate::output::Theme>,
}

/// A ceiling on what one execution fetches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FetchTotal {
    /// Live fetches.
    pub fetches: usize,
    /// Bytes fetched.
    pub bytes: u64,
}

/// The settings, with what [`configure`] built from them.
struct Settled {
    settings: Settings,
    backlog: Option<Arc<crate::fetch::Backlog>>,
}

static SETTLED: OnceLock<Settled> = OnceLock::new();

/// Settle the process's settings. Call it before anything reads them: the
/// first read settles the defaults.
///
/// # Errors
/// When the fetch backlog cannot be read, or the settings are already settled.
pub fn configure(settings: Settings) -> Result<()> {
    let backlog = match &settings.fetch_backlog {
        Some(path) => Some(Arc::new(
            crate::fetch::Backlog::open(path)
                .with_context(|| format!("opening fetch backlog {}", path.display()))?,
        )),
        None => None,
    };
    let total = settings.fetch_total;
    SETTLED
        .set(Settled { settings, backlog })
        .map_err(|_rejected| anyhow::anyhow!("process settings are already settled"))?;
    if let Some(total) = total {
        crate::fetch::set_total_budget(total.fetches, total.bytes);
    }
    Ok(())
}

fn settled() -> &'static Settled {
    SETTLED.get_or_init(|| Settled {
        settings: Settings::default(),
        backlog: None,
    })
}

/// The process's settings.
pub(crate) fn get() -> &'static Settings {
    &settled().settings
}

/// The fetch backlog [`configure`] opened, if one was named.
pub(crate) fn fetch_backlog() -> Option<&'static Arc<crate::fetch::Backlog>> {
    settled().backlog.as_ref()
}
