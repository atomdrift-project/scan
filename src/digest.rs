//! Content digests.

use sha2::{Digest, Sha256};
use std::path::Path;

/// Lowercase hex SHA-256 of the file at `path`, streamed.
///
/// # Errors
/// When the file cannot be opened or read.
pub(crate) fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}
