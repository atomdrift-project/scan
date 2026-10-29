//! The environment variables scan reads: every one is declared here, and read
//! only through here.
//!
//! Configuration belongs on the command line. What remains is what a flag
//! cannot carry: credentials, settings shared with sibling tools, and switches
//! an operator sets once for a whole host. `docs/ENVIRONMENT.md` documents each.

use std::path::PathBuf;
use std::str::FromStr;

/// One declared environment variable.
#[derive(Debug, Clone, Copy)]
pub struct Var {
    /// The variable's name.
    pub name: &'static str,
    /// What it does, in a phrase.
    pub what: &'static str,
    kind: Kind,
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    Text,
    Flag,
    Number,
}

macro_rules! vars {
    ($($id:ident: $kind:ident, $what:literal;)*) => {
        $(
            #[doc = $what]
            pub const $id: Var = Var { name: stringify!($id), what: $what, kind: Kind::$kind };
        )*
        /// Every declared variable.
        pub const ALL: &[Var] = &[$($id),*];
    };
}

vars! {
    SCAN_LLM: Text, "LLM endpoint list, when `--llm` is not given.";
    SCAN_LLM_MODEL: Text, "LLM model name, when `--llm-model` is not given.";
    SCAN_LLM_KEY: Text, "LLM bearer token, in place of `~/.tok/llm`.";
    SCAN_LLM_CONCURRENCY: Number, "LLM calls in flight per process.";
    SCAN_LLM_BACKGROUND_CONCURRENCY: Number, "Of those, how many work nobody waits on may hold.";
    SCAN_LLM_SYSTEM_PROMPT_FILE: Text, "Replace the built-in LLM grading prompt with this file.";
    SCAN_HOPPER: Text, "Hopper base URL(s), when `--hopper` is not given.";
    SCAN_NO_UPLOAD: Flag, "Disable CLI result and artifact uploads while retaining Hopper corpus lookups.";
    SCAN_MODELS_DIR: Text, "Use the model bundle in this directory, and never update it.";
    SCAN_NO_ANALYSIS_CACHE: Flag, "Disable every analysis cache in the stack, for benchmarks.";
    SCAN_NO_UPDATE_CHECK: Flag, "Do not check for a newer release.";
    CLEAVE_RAYON_THREADS: Number, "Width of the analysis thread pool.";
    CLEAVE_TRAITS_DIR: Text, "Read traits from this directory instead of the installed ones.";
    RUST_LOG: Text, "Log filter.";
    MALLOC_CONF: Text, "FreeBSD's allocator settings, checked for a known-bad option.";
}

/// The value, or `None` when unset or empty.
#[must_use]
#[expect(
    clippy::disallowed_methods,
    reason = "the one place the environment is read"
)]
pub fn text(var: Var) -> Option<String> {
    std::env::var(var.name)
        .ok()
        .filter(|value| !value.is_empty())
}

/// The value as a path, or `None` when unset or empty.
#[must_use]
pub fn path(var: Var) -> Option<PathBuf> {
    text(var).map(PathBuf::from)
}

/// Whether a flag is on. Unset, empty, `0`, `false` and `no` are off;
/// `1`, `true` and `yes` are on. Anything else is refused by [`check`] and
/// read as off.
#[must_use]
pub fn flag(var: Var) -> bool {
    text(var).is_some_and(|value| parse_flag(&value) == Some(true))
}

/// A number, or `None` when unset or empty. A value that does not parse is
/// refused by [`check`] and read as unset.
#[must_use]
pub fn number<T: FromStr>(var: Var) -> Option<T> {
    text(var).and_then(|value| value.trim().parse().ok())
}

fn parse_flag(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Some(true),
        "0" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// Refuse a flag or number that does not parse, rather than let it read as
/// unset.
///
/// # Errors
/// Names the first variable whose value is not one its kind accepts.
pub fn check() -> anyhow::Result<()> {
    for var in ALL {
        let Some(value) = text(*var) else { continue };
        let valid = match var.kind {
            Kind::Text => true,
            Kind::Flag => parse_flag(&value).is_some(),
            Kind::Number => value.trim().parse::<u64>().is_ok(),
        };
        anyhow::ensure!(
            valid,
            "{}={value:?} is not a valid value ({})",
            var.name,
            var.what
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_take_the_usual_spellings() {
        assert_eq!(parse_flag("1"), Some(true));
        assert_eq!(parse_flag(" TRUE "), Some(true));
        assert_eq!(parse_flag("no"), Some(false));
        assert_eq!(parse_flag("0"), Some(false));
        assert_eq!(parse_flag("on"), None);
        assert_eq!(parse_flag("2"), None);
    }

    /// Every variable scan reads is one an operator can look up.
    #[test]
    fn every_variable_is_documented() {
        let doc = include_str!("../docs/ENVIRONMENT.md");
        for var in ALL {
            assert!(
                doc.contains(&format!("`{}`", var.name)),
                "{} is not in docs/ENVIRONMENT.md",
                var.name
            );
        }
    }
}
