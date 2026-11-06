//! The server's settings: [`Startup`] as a command line hands them over, and
//! [`ServerConfig`] once every default is settled.

use std::net::SocketAddr;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;

use super::acl::{Cidr, TokenDigest, parse_cidr_list};
use crate::model::Thresholds;

/// How the HTTP API server runs, settled before it starts.
///
/// Plain data: build one with struct update syntax over [`Default`], or
/// resolve one from a command line with [`Startup::resolve`].
/// `build_app` validates it.
///
/// ```
/// use scan::server::ServerConfig;
///
/// let config = ServerConfig {
///     model_dir: "/path/to/models".into(),
///     workers: 2,
///     ..ServerConfig::default()
/// };
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to listen on.
    pub bind: SocketAddr,
    /// Largest request body accepted, in bytes.
    pub max_body_size: usize,
    /// RSS past which requests are refused. `None` disables in-process
    /// throttling, for when a supervisor such as systemd `MemoryMax=` enforces
    /// a hard cap already.
    pub max_rss_bytes: Option<NonZeroU64>,
    /// Model bundle directory.
    pub model_dir: PathBuf,
    /// Manual probability cutoffs. `None` takes the bundle's level grid.
    pub thresholds: Option<Thresholds>,
    /// The operating point (0..=10000) the verdict thresholds come from.
    /// `None` with manual thresholds.
    pub level: Option<u16>,
    /// Per-rule time budget before cleave logs a slow rule.
    pub slow_rule_ms: u64,
    /// Canonical directories `POST /v1/analyze?path=` may read files from.
    /// Empty refuses every `path=` request without touching the filesystem.
    pub local_paths: Vec<PathBuf>,
    /// Concurrent analyses; at least one. Past it requests are refused, not
    /// queued.
    pub workers: usize,
    /// Peer networks besides loopback allowed to connect.
    pub allow_cidrs: Vec<Cidr>,
    /// Digest of the bearer token required on every route but `/_/health`;
    /// `None` leaves the API open. Loopback is not exempt: behind a Cloudflare
    /// tunnel every remote request arrives over loopback.
    pub auth_digest: Option<TokenDigest>,
    /// Per-request analysis timeout in seconds. Zero disables it.
    pub analysis_timeout_secs: u64,
    /// The LLM second opinion, when one is configured.
    pub interpret: Option<crate::interpret::InterpretConfig>,
    /// Which references a sample declares are followed. Off by default: an
    /// upload server driving outbound fetches is an SSRF-shaped exposure, so
    /// turning it on is an explicit operator decision.
    pub fetch: crate::fetch::FetchPolicy,
    /// Hopper API root(s). Enables result renewal and corpus deferral.
    pub hopper: Option<String>,
    /// Passwords to try against encrypted archives.
    pub zip_passwords: crate::ArchivePasswords,
}

impl Default for ServerConfig {
    /// The `serve` command line's defaults, except `model_dir`, which has
    /// none: a caller always names one.
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 49999)),
            max_body_size: 100 * 1024 * 1024,
            max_rss_bytes: None,
            model_dir: PathBuf::new(),
            thresholds: None,
            level: None,
            slow_rule_ms: crate::cli::DEFAULT_SLOW_RULE_MS,
            local_paths: Vec::new(),
            workers: 1,
            allow_cidrs: Vec::new(),
            auth_digest: None,
            analysis_timeout_secs: DEFAULT_ANALYSIS_TIMEOUT_SECS,
            interpret: None,
            fetch: crate::fetch::FetchPolicy::default(),
            hopper: None,
            zip_passwords: crate::ArchivePasswords::default(),
        }
    }
}

impl ServerConfig {
    /// Check what plain data cannot enforce by itself.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid manual thresholds or zero workers.
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(t) = &self.thresholds {
            t.validate()
                .map_err(|error| anyhow::anyhow!("invalid thresholds: {error}"))?;
        }
        if self.workers == 0 {
            anyhow::bail!("workers must be >= 1");
        }
        // Reading local files is for a caller on this host, or for remote
        // callers the operator has both named and credentialed. Anything else
        // would hand a file-reading primitive to whoever reaches the socket.
        let loopback = self.bind.ip() == std::net::Ipv4Addr::LOCALHOST;
        let vetted = !self.allow_cidrs.is_empty() && self.auth_digest.is_some();
        if !self.local_paths.is_empty() && !loopback && !vetted {
            anyhow::bail!(
                "--allow-local-paths requires --bind 127.0.0.1, or both --allow-cidr and --token-file"
            );
        }
        Ok(())
    }

    /// The hopper root, or `None` when none is set or it is blank.
    pub(super) fn hopper(&self) -> Option<&str> {
        self.hopper.as_deref().filter(|s| !s.trim().is_empty())
    }
}

/// What a caller supplies to start a server, before resolution.
///
/// [`ServerConfig`] is the resolved form. This is the unresolved one — the
/// shape a command line hands over, with `None` meaning "take the default"
/// rather than "off". [`Startup::resolve`] settles every default in one
/// place, so two binaries serving this API serve it identically.
#[derive(Debug)]
pub struct Startup {
    /// Address to listen on.
    pub bind: SocketAddr,
    /// Largest request body accepted, in megabytes.
    pub max_size_mb: usize,
    /// The process RSS ceiling (`--max-rss-gb`).
    pub max_rss: crate::memory::MaxRssPolicy,
    /// Directories `POST /v1/analyze?path=` may read files from. Empty, the
    /// default, refuses every `path=` request.
    pub allow_local_paths: Vec<PathBuf>,
    /// Concurrent request slots. `None` sizes them from the core count; see
    /// [`crate::runtime::default_workers`].
    pub workers: Option<NonZeroUsize>,
    /// Comma-separated CIDRs allowed to reach non-loopback routes.
    pub allow_cidr: Option<String>,
    /// File holding the bearer token. `None` disables authentication.
    pub token_file: Option<PathBuf>,
    /// Hopper API root. Enables result renewal and corpus deferral.
    pub hopper: Option<String>,
    /// Per-request analysis timeout in seconds. Zero disables.
    pub analysis_timeout_secs: u64,
    /// Rules, model and analysis settings.
    /// Fetching is off by default: a server driving outbound fetches is an
    /// SSRF-shaped exposure, so turning it on is an operator decision.
    pub rules: crate::rules::RulesStartup,
}

impl Startup {
    /// Settle every default, read the bearer token, and validate the result.
    ///
    /// # Errors
    ///
    /// Returns an error when the model bundle cannot be resolved, a CIDR does
    /// not parse, the thresholds or worker count are invalid, or `token_file`
    /// is set but missing, empty or unreadable. That last one fails closed on
    /// purpose: an operator who asked for authentication must never get an
    /// open server because a file went away.
    pub fn resolve(self) -> anyhow::Result<ServerConfig> {
        let rules = self.rules.resolve()?;

        // Canonicalized here, so a request's canonical path can be checked
        // with a plain prefix match.
        let local_paths = self
            .allow_local_paths
            .iter()
            .map(|dir| {
                let canonical = dir
                    .canonicalize()
                    .map_err(|e| anyhow::anyhow!("--allow-local-paths {}: {e}", dir.display()))?;
                if !canonical.is_dir() {
                    anyhow::bail!("--allow-local-paths {}: not a directory", dir.display());
                }
                Ok(canonical)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let workers = self
            .workers
            .unwrap_or_else(crate::runtime::default_workers)
            .get();

        let allow_cidrs = match self.allow_cidr {
            Some(ref list) => {
                parse_cidr_list(list).map_err(|e| anyhow::anyhow!("--allow-cidr: {e}"))?
            }
            None => Vec::new(),
        };

        let auth_digest = match self.token_file {
            Some(ref path) => {
                let token = crate::interpret::read_token_file(path).ok_or_else(|| {
                    anyhow::anyhow!(
                        "--token-file {}: missing, empty, or unreadable",
                        path.display()
                    )
                })?;
                let digest = TokenDigest::new(&token)
                    .map_err(|e| anyhow::anyhow!("--token-file {}: {e}", path.display()))?;
                // Name the file the running process actually read: after a
                // rotation the token in a file and the token in memory can
                // differ, and the 401 that follows is otherwise unreadable.
                tracing::info!(
                    token_file = %path.display(),
                    "bearer authentication enabled (token is read once, at startup)",
                );
                Some(digest)
            }
            None => None,
        };

        let hopper = self.hopper.filter(|s| !s.trim().is_empty());

        let config = ServerConfig {
            bind: self.bind,
            max_body_size: self.max_size_mb.saturating_mul(1024 * 1024),
            max_rss_bytes: self.max_rss.process_ceiling(),
            model_dir: rules.model_dir,
            thresholds: rules.thresholds,
            level: rules.level,
            slow_rule_ms: rules.slow_rule_ms,
            local_paths,
            workers,
            allow_cidrs,
            auth_digest,
            analysis_timeout_secs: self.analysis_timeout_secs,
            interpret: rules.interpret,
            fetch: rules.fetch,
            hopper,
            zip_passwords: rules.zip_passwords,
        };
        config.validate()?;
        Ok(config)
    }
}

/// Default per-request analysis timeout: 34 minutes. Covers cold cleave scans
/// of large archives — and fetch-enabled scans whose dependency analysis can
/// far outlast the sample's own — while still preventing a pathological input
/// from pinning a slot forever. Override with `--analysis-timeout`.
pub const DEFAULT_ANALYSIS_TIMEOUT_SECS: u64 = 2040;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_config_rejects_invalid_thresholds() {
        let config = ServerConfig {
            thresholds: Some(Thresholds {
                suspicious: -0.1,
                hostile: 0.9,
            }),
            ..ServerConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn server_config_accepts_the_defaults() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.level.is_none());
        assert!(config.max_rss_bytes.is_none());
    }

    #[test]
    fn server_config_rejects_zero_workers() {
        let config = ServerConfig {
            workers: 0,
            ..ServerConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn a_blank_hopper_is_no_hopper() {
        let config = ServerConfig {
            hopper: Some("  ".into()),
            ..ServerConfig::default()
        };
        assert_eq!(config.hopper(), None);
    }
}
