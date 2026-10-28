//! Reading a file on this host for `POST /v1/analyze?path=`.
//!
//! The file is opened once and read from that handle, so the bytes analyzed are
//! the bytes that passed the checks: a path swapped for a symlink afterwards
//! cannot redirect the read. Every refusal looks the same to the caller, so the
//! route cannot be used to learn which paths exist.

use std::io::Read as _;
use std::path::{Path, PathBuf};

use axum::http::StatusCode;
use sha2::{Digest, Sha256};

use super::error::ApiError;
use super::input::sanitize_upload_filename;

/// A local file, read and hashed.
#[derive(Debug)]
pub(super) struct LocalFile {
    pub(super) bytes: bytes::Bytes,
    /// Lowercase hex SHA-256 of `bytes`.
    pub(super) sha256: String,
    /// The file's name, sanitized, so cleave can type it by extension.
    pub(super) filename: String,
}

/// Read `path` if it is a regular file under one of `allowed` (canonical
/// directories) and at most `limit` bytes. Blocking.
pub(super) fn read(path: &Path, allowed: &[PathBuf], limit: usize) -> Result<LocalFile, ApiError> {
    // `?path` rather than `%path`: Debug escapes the control characters a
    // caller could otherwise use to forge log lines.
    let refuse = |reason: &dyn std::fmt::Display| {
        tracing::warn!(path = ?path, %reason, "path= refused");
        ApiError::new(
            StatusCode::FORBIDDEN,
            "path_not_allowed",
            "The path is not a file this server may read.",
        )
    };
    let canonical = path.canonicalize().map_err(|e| refuse(&e))?;
    if !allowed.iter().any(|dir| canonical.starts_with(dir)) {
        return Err(refuse(&"outside --allow-local-paths"));
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    // No following a symlink planted after canonicalization, and no blocking
    // on a FIFO before the regular-file check below can refuse it.
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(
        &mut options,
        libc::O_NOFOLLOW | libc::O_NONBLOCK,
    );
    let file = options.open(&canonical).map_err(|e| refuse(&e))?;
    let meta = file.metadata().map_err(|e| refuse(&e))?;
    if !meta.is_file() {
        return Err(refuse(&"not a regular file"));
    }
    // A directory above the file swapped for a symlink between canonicalizing
    // and opening would have opened something else under the same name.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        let named = std::fs::metadata(&canonical).map_err(|e| refuse(&e))?;
        if (named.dev(), named.ino()) != (meta.dev(), meta.ino()) {
            return Err(refuse(&"replaced while opening"));
        }
    }

    let too_large = || {
        ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "artifact_too_large",
            format!("The artifact exceeds the {limit} byte limit."),
        )
    };
    if meta.len() > limit as u64 {
        return Err(too_large());
    }
    let mut buf = Vec::with_capacity(usize::try_from(meta.len()).unwrap_or(limit));
    // One byte past the limit is enough to see a file that grew after the
    // size check.
    file.take(limit as u64 + 1)
        .read_to_end(&mut buf)
        .map_err(|e| refuse(&e))?;
    if buf.len() > limit {
        return Err(too_large());
    }
    let filename = canonical
        .file_name()
        .map(|name| sanitize_upload_filename(&name.to_string_lossy()))
        .unwrap_or_default();
    Ok(LocalFile {
        sha256: format!("{:x}", Sha256::digest(&buf)),
        bytes: buf.into(),
        filename,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed(dir: &tempfile::TempDir) -> Vec<PathBuf> {
        vec![dir.path().canonicalize().expect("canonical tempdir")]
    }

    #[test]
    fn reads_a_file_under_an_allowed_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("left-pad-1.3.0.tgz");
        std::fs::write(&path, b"bytes").expect("write");
        let file = read(&path, &allowed(&dir), 1024).expect("allowed");
        assert_eq!(&file.bytes[..], b"bytes");
        assert_eq!(file.filename, "left-pad-1.3.0.tgz");
        assert_eq!(file.sha256, format!("{:x}", Sha256::digest(b"bytes")));
    }

    /// A missing file, a file outside the allowed directories, and something
    /// that is not a file must be indistinguishable to the caller.
    #[test]
    fn every_refusal_reads_the_same() {
        let dir = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("tempdir");
        let elsewhere = outside.path().join("secret");
        std::fs::write(&elsewhere, b"secret").expect("write");
        let refusals = [
            read(&dir.path().join("missing"), &allowed(&dir), 1024),
            read(&elsewhere, &allowed(&dir), 1024),
            read(dir.path(), &allowed(&dir), 1024),
        ];
        for refusal in refusals {
            let error = refusal.expect_err("refused");
            assert_eq!(error.status, StatusCode::FORBIDDEN);
            assert_eq!(error.code, "path_not_allowed");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_the_allowed_directory_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("tempdir");
        let secret = outside.path().join("secret");
        std::fs::write(&secret, b"secret").expect("write");
        let link = dir.path().join("innocent.txt");
        std::os::unix::fs::symlink(&secret, &link).expect("symlink");
        let error = read(&link, &allowed(&dir), 1024).expect_err("refused");
        assert_eq!(error.status, StatusCode::FORBIDDEN);
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_is_refused_without_blocking() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fifo = dir.path().join("pipe");
        let c_path = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).expect("path");
        // SAFETY: `c_path` is a valid NUL-terminated path for the call's duration.
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        let error = read(&fifo, &allowed(&dir), 1024).expect_err("refused");
        assert_eq!(error.status, StatusCode::FORBIDDEN);
    }

    #[test]
    fn a_file_past_the_limit_is_too_large() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("big.bin");
        std::fs::write(&path, vec![0u8; 16]).expect("write");
        let error = read(&path, &allowed(&dir), 15).expect_err("refused");
        assert_eq!(error.status, StatusCode::PAYLOAD_TOO_LARGE);
    }
}
