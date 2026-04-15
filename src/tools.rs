//! Runtime tool availability checks.
//!
//! Cleave relies on several external binaries for full analysis coverage.
//! Missing tools do not prevent the service from starting but will cause
//! specific file types to fail analysis with a clear error message.

use std::fmt;

/// An external analysis tool that was not found in PATH.
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
        write!(f, "{} ({})\n  install: {}", self.name, self.purpose, self.install)
    }
}

/// Returns every required analysis tool that is absent from PATH.
///
/// An empty vec means the environment is fully equipped.
#[must_use]
pub fn missing() -> Vec<MissingTool> {
    let mut absent: Vec<MissingTool> = TOOLS
        .iter()
        .filter(|&&(name, _, _)| !bin_available(name))
        .map(|&(name, purpose, install)| MissingTool { name, purpose, install })
        .collect();

    // 7-Zip ships under several binary names depending on the platform.
    if !["7z", "7za", "7zz", "7zr"].iter().any(|&n| bin_available(n)) {
        absent.push(MissingTool {
            name: "7z",
            purpose: "7-Zip / CAB / tar archive extraction",
            install: "brew install p7zip  |  apt-get install p7zip-full  |  apk add 7zip",
        });
    }

    absent
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
/// that names the tool and provides install instructions.
///
/// Returns `None` when the error is unrelated to tool availability.
#[must_use]
pub fn enrich_error(err: &anyhow::Error) -> Option<String> {
    let msg = format!("{err:#}");
    let is_not_found = msg.contains("No such file or directory")
        || msg.contains("os error 2")
        || msg.contains("not found");
    if !is_not_found {
        return None;
    }
    for &(name, _, install) in TOOLS {
        if msg.contains(name) {
            return Some(format!("{msg}\n  hint: {name} is not installed — {install}"));
        }
    }
    for variant in ["7z", "7za", "7zz", "7zr", "p7zip"] {
        if msg.contains(variant) {
            return Some(format!(
                "{msg}\n  hint: 7z is not installed — brew install p7zip  |  apt-get install p7zip-full  |  apk add 7zip"
            ));
        }
    }
    None
}

const TOOLS: &[(&str, &str, &str)] = &[
    (
        "rizin",
        "binary disassembly and import/export analysis",
        "brew install rizin  |  pkg install rizin  |  https://rizin.re",
    ),
    (
        "upx",
        "UPX-packed executable detection",
        "brew install upx  |  apt-get install upx-ucl  |  apk add upx",
    ),
    (
        "innoextract",
        "Inno Setup installer extraction",
        "brew install innoextract  |  apt-get install innoextract  |  apk add innoextract",
    ),
];

/// Returns `true` if `name` resolves to an executable file in PATH.
///
/// Uses a PATH lookup rather than spawning the binary, so this never blocks
/// even if the binary has a slow or hanging startup (e.g. rizin loading plugins).
fn bin_available(name: &str) -> bool {
    let Ok(path_var) = std::env::var("PATH") else { return false };
    std::env::split_paths(&path_var).any(|dir| {
        let candidate = dir.join(name);
        candidate.is_file()
    })
}
