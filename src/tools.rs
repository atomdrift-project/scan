//! Runtime tool availability checks.
//!
//! Cleave relies on several external binaries for full analysis coverage.
//! Missing tools do not prevent the service from starting but will cause
//! specific file types to fail analysis with a clear error message.

use std::collections::HashMap;
use std::fmt;
#[cfg(windows)]
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// An external analysis tool that was not found in PATH or a supported
/// platform-specific fallback directory.
#[derive(Debug, Clone)]
pub struct MissingTool {
    /// The command name (e.g. `"rizin"`).
    pub name: &'static str,
    /// What this tool enables (e.g. `"binary disassembly"`).
    pub purpose: &'static str,
    /// A concise install hint shown to the operator.
    pub install: &'static str,
}

impl fmt::Display for MissingTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({})\n  install: {}",
            self.name, self.purpose, self.install
        )
    }
}

/// Returns every required analysis tool that is absent from PATH and the
/// supported fallback locations.
///
/// An empty vec means the environment is fully equipped.
#[must_use]
pub fn missing() -> Vec<MissingTool> {
    let mut absent: Vec<MissingTool> = TOOLS
        .iter()
        .filter(|&&(name, _)| !bin_available(name))
        .map(|&(name, purpose)| MissingTool {
            name,
            purpose,
            install: install_hint(name),
        })
        .collect();

    if !sevenzip_available() {
        absent.push(MissingTool {
            name: "7z",
            purpose: "7-Zip / CAB / tar archive extraction",
            install: install_hint("7z"),
        });
    }

    absent
}

/// Returns the canonical names of available external analysis tools.
///
/// This is intentionally compact because workers send it on every `/api/next`
/// poll. The names are part of Hopper's scheduling contract.
#[must_use]
pub fn available_names() -> Vec<&'static str> {
    let mut available: Vec<&'static str> = TOOLS
        .iter()
        .filter(|&&(name, _)| bin_available(name))
        .map(|&(name, _)| name)
        .collect();

    if sevenzip_available() {
        available.push("7z");
    }

    available
}

/// Emit a structured warning for each tool that is missing from PATH.
///
/// Call this at service startup so operators see a clear diagnosis before
/// the first analysis request fails.
pub fn warn_missing() {
    for tool in missing() {
        tracing::warn!(
            tool = tool.name,
            purpose = tool.purpose,
            install = tool.install,
            "required analysis tool not found in PATH — {} will fail",
            tool.purpose,
        );
    }
}

/// If `err` originated from a missing external tool, returns a richer message
/// that names the tool and provides an install command appropriate for this
/// operating system.
///
/// Returns `None` when the error is unrelated to tool availability.
#[must_use]
pub fn enrich_error(err: &anyhow::Error) -> Option<String> {
    let msg = format!("{err:#}");
    let lower = msg.to_ascii_lowercase();
    let is_not_found = lower.contains("no such file or directory")
        || lower.contains("os error 2")
        || lower.contains("not found")
        || lower.contains("not in path")
        || lower.contains("not installed")
        || lower.contains("cannot find the file");
    if !is_not_found {
        return None;
    }

    for &(name, _) in TOOLS {
        if lower.contains(name) {
            return Some(format!(
                "{msg}\n  hint: {name} is not installed — {}",
                install_hint(name)
            ));
        }
    }
    for variant in SEVENZIP_BINS.iter().copied().chain(["p7zip"]) {
        if lower.contains(variant) {
            return Some(format!(
                "{msg}\n  hint: 7z is not installed — {}",
                install_hint("7z")
            ));
        }
    }
    None
}

const TOOLS: &[(&str, &str)] = &[
    ("rizin", "binary disassembly and import/export analysis"),
    ("upx", "UPX-packed executable detection"),
    ("innoextract", "Inno Setup installer extraction"),
];

static RESOLUTIONS: OnceLock<Mutex<HashMap<String, Option<PathBuf>>>> = OnceLock::new();

/// 7-Zip ships under several binary names depending on the platform.
const SEVENZIP_BINS: &[&str] = &["7z", "7za", "7zz", "7zr"];

/// Returns an install command for the host operating system.
fn install_hint(name: &str) -> &'static str {
    #[cfg(windows)]
    {
        match name {
            "rizin" => "winget install --exact --id Rizin.Rizin",
            "upx" => "winget install --exact --id UPX.UPX",
            "innoextract" => "winget install --exact --id dscharrer.innoextract",
            "7z" => "winget install --exact --id 7zip.7zip",
            _ => "winget search <tool>",
        }
    }

    #[cfg(not(windows))]
    {
        match name {
            "rizin" => "brew install rizin  |  pkg install rizin  |  https://rizin.re",
            "upx" => "brew install upx  |  apt-get install upx-ucl  |  apk add upx",
            "innoextract" => {
                "brew install innoextract  |  apt-get install innoextract  |  apk add innoextract"
            }
            "7z" => "brew install p7zip  |  apt-get install p7zip-full  |  apk add 7zip",
            _ => "use the package manager for your operating system",
        }
    }
}

/// Returns `true` if any 7-Zip binary variant is present in PATH or a
/// supported fallback location.
fn sevenzip_available() -> bool {
    SEVENZIP_BINS.iter().any(|&n| bin_available(n))
}

/// Returns `true` if `name` resolves to an executable file in PATH or a
/// supported fallback location.
///
/// Uses a filesystem lookup rather than spawning the binary, so this never
/// blocks even if the binary has a slow or hanging startup (e.g. rizin loading
/// plugins).
fn bin_available(name: &str) -> bool {
    resolved_binary(name).is_some()
}

/// Resolve each command at most once per process. The cached `PathBuf` is also
/// useful as a record of which installation won when both PATH and a fallback
/// location are present; `None` is cached too, so repeated warnings do not
/// repeatedly walk the filesystem.
fn resolved_binary(name: &str) -> Option<PathBuf> {
    let cache = RESOLUTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let Ok(mut cache) = cache.lock() else {
        return resolve_uncached(name);
    };
    if let Some(resolution) = cache.get(name) {
        return resolution.clone();
    }
    let resolution = resolve_uncached(name);
    cache.insert(name.to_string(), resolution.clone());
    resolution
}

fn resolve_uncached(name: &str) -> Option<PathBuf> {
    binary_in_path(name).or_else(|| fallback_binary(name))
}

fn binary_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        candidate_names(name)
            .iter()
            .map(|candidate| dir.join(candidate))
            .find(|candidate| candidate.is_file())
    })
}

fn fallback_binary(name: &str) -> Option<PathBuf> {
    let names = candidate_names(name);
    for root in fallback_roots() {
        if let Some(binary) = names
            .iter()
            .map(|candidate| root.join(candidate))
            .find(|candidate| candidate.is_file())
        {
            return Some(binary);
        }
    }

    #[cfg(windows)]
    {
        let winget_packages = windows_env_path("LOCALAPPDATA")?.join("Microsoft/WinGet/Packages");
        return find_in_tree(&winget_packages, &names, 5);
    }

    None
}

fn candidate_names(name: &str) -> Vec<String> {
    #[cfg(windows)]
    {
        let mut names = vec![name.to_string()];
        if Path::new(name).extension().is_none() {
            names.extend([
                format!("{name}.exe"),
                format!("{name}.cmd"),
                format!("{name}.bat"),
            ]);
        }
        names
    }
    #[cfg(not(windows))]
    {
        vec![name.to_string()]
    }
}

fn fallback_roots() -> Vec<PathBuf> {
    let mut roots = vec![
        PathBuf::from("/usr/local/bin"),
        PathBuf::from("/usr/bin"),
        PathBuf::from("/bin"),
    ];

    #[cfg(target_os = "macos")]
    roots.extend([
        PathBuf::from("/opt/homebrew/bin"),
        PathBuf::from("/opt/local/bin"),
    ]);

    #[cfg(windows)]
    {
        for variable in ["ProgramFiles", "ProgramFiles(x86)"] {
            if let Some(base) = windows_env_path(variable) {
                roots.extend([
                    base.join("7-Zip"),
                    base.join("Rizin"),
                    base.join("Rizin/bin"),
                    base.join("UPX"),
                    base.join("upx"),
                    base.join("innoextract"),
                    base.join("InnoExtract"),
                ]);
            }
        }
        if let Some(local) = windows_env_path("LOCALAPPDATA") {
            roots.extend([
                local.join("Programs/7-Zip"),
                local.join("Programs/Rizin"),
                local.join("Programs/Rizin/bin"),
                local.join("Programs/UPX"),
                local.join("Programs/upx"),
                local.join("Programs/innoextract"),
                local.join("Programs/InnoExtract"),
                local.join("Microsoft/WinGet/Links"),
                local.join("Microsoft/WinGet/Packages"),
            ]);
        }
    }

    roots
}

#[cfg(windows)]
fn windows_env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

#[cfg(windows)]
fn find_in_tree(root: &Path, names: &[String], depth: usize) -> Option<PathBuf> {
    if depth == 0 {
        return None;
    }
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path.file_name().is_some_and(|file_name| {
                names
                    .iter()
                    .any(|name| file_name.eq_ignore_ascii_case(name))
            })
        {
            return Some(path);
        }
        if entry.file_type().is_ok_and(|kind| kind.is_dir())
            && let Some(binary) = find_in_tree(&path, names, depth - 1)
        {
            return Some(binary);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{candidate_names, install_hint};

    #[test]
    fn install_hint_is_non_empty_for_every_supported_tool() {
        for name in ["rizin", "upx", "innoextract", "7z"] {
            assert!(!install_hint(name).is_empty());
        }
    }

    #[test]
    fn unix_candidate_names_are_not_augmented() {
        #[cfg(not(windows))]
        assert_eq!(candidate_names("7z"), vec!["7z"]);
    }
}
