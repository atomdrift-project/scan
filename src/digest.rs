//! Content digests.

use sha2::{Digest, Sha256};
use std::path::Path;

/// Lowercase hex SHA-256 of the regular file at `path`, streamed.
///
/// Opened without blocking and checked on the open handle, so a FIFO or a
/// device swapped in at `path` (`/dev/zero`) is refused rather than read
/// forever.
///
/// # Errors
/// When the file cannot be opened or read, or is not a regular file.
pub(crate) fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut options, libc::O_NONBLOCK);
    let mut file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a regular file",
        ));
    }
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}
