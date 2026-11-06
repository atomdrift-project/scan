//! What a process runs with, settled once.
//!
//! A binary fills [`Settings`] from its flags and calls [`configure`] before
//! anything reads them; a library caller that never does gets the defaults.
//! Whichever comes first settles them for the life of the process, so no two
//! parts of one process can disagree about them.

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::Result;

/// The process's settings.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    /// A ceiling on everything one execution fetches; `None` is unlimited, as
    /// a server's is: each of its jobs is bounded on its own.
    pub fetch_total: Option<FetchTotal>,
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

static SETTLED: OnceLock<Settings> = OnceLock::new();

/// Settle the process's settings. Call it before anything reads them: the
/// first read settles the defaults.
///
/// # Errors
/// When the settings are already settled.
pub fn configure(settings: Settings) -> Result<()> {
    let total = settings.fetch_total;
    SETTLED
        .set(settings)
        .map_err(|_rejected| anyhow::anyhow!("process settings are already settled"))?;
    if let Some(total) = total {
        crate::fetch::set_total_budget(total.fetches, total.bytes);
    }
    Ok(())
}

/// The process's settings.
pub(crate) fn get() -> &'static Settings {
    SETTLED.get_or_init(Settings::default)
}
