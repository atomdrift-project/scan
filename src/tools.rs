//! Runtime tool availability checks.
//!
//! Cleave and filefacts own execution of the external analyzers. Their shared
//! resolver owns PATH/fallback discovery; this module only reports capability
//! status and provides operator-facing install guidance.

use std::fmt;

/// An external analysis tool that was not found by filefacts' resolver.
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
        .filter(|&&(name, _)| !tool_available(name))
        .map(|&(name, purpose)| MissingTool {
            name,
            purpose,
            install: install_hint(name),
        })
        .collect();

    if !SEVENZIP_BINS.iter().copied().any(tool_available) {
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
        .filter(|&&(name, _)| tool_available(name))
        .map(|&(name, _)| name)
        .collect();

    if SEVENZIP_BINS.iter().copied().any(tool_available) {
        available.push("7z");
    }

    available
}

/// Emit a structured warning for each tool that is unavailable to the shared
/// resolver, including supported fallback locations.
pub fn warn_missing() {
    for tool in missing() {
        tracing::warn!(
            tool = tool.name,
            purpose = tool.purpose,
            install = tool.install,
            "required analysis tool not found — {} will fail",
            tool.purpose,
        );
    }
}

/// If `err` originated from a missing external tool, returns a richer message
/// that names the tool and provides an install command appropriate for this
/// operating system.
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

const SEVENZIP_BINS: &[&str] = &["7z", "7za", "7zz", "7zr"];

fn tool_available(name: &str) -> bool {
    filefacts::tools::resolve(name).is_some()
}

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

#[cfg(test)]
mod tests {
    use super::install_hint;

    #[test]
    fn install_hint_is_non_empty_for_every_supported_tool() {
        for name in ["rizin", "upx", "innoextract", "7z"] {
            assert!(!install_hint(name).is_empty());
        }
    }
}
