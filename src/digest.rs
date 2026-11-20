//! Content digests.

use sha2::{Digest, Sha256};
use std::io::{self, Read as _};
use std::path::Path;

/// Lowercase hex of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// A writer that hashes what is written to it, for a serializer to feed.
pub(crate) struct Hashing<'a>(pub(crate) &'a mut Sha256);

impl io::Write for Hashing<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Lowercase hex SHA-256 of the regular file at `path`, streamed.
///
/// Opened without blocking and checked on the open handle, so a FIFO or a
/// device swapped in at `path` (`/dev/zero`) is refused rather than read
/// forever.
///
/// # Errors
/// When the file cannot be opened or read, or is not a regular file.
pub(crate) fn sha256_file(path: &Path) -> io::Result<String> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut options, libc::O_NONBLOCK);
    let mut file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not a regular file",
        ));
    }
    // A quarter-megabyte at a time: few system calls beside a hash that runs
    // at gigabytes a second.
    let mut hasher = Sha256::new();
    let mut piece = vec![0; 1 << 18];
    loop {
        match file.read(&mut piece) {
            Ok(0) => return Ok(hex(&hasher.finalize())),
            Ok(n) => hasher.update(&piece[..n]),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digest of a file is the digest of its bytes, whether it is empty,
    /// shorter than one read or several reads long.
    #[test]
    fn a_file_hashes_to_the_digest_of_its_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sample.bin");
        std::fs::write(&path, b"abc").expect("write");
        assert_eq!(
            sha256_file(&path).expect("hash"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        for len in [0, 1, (1 << 18) - 1, 1 << 18, (1 << 19) + 7] {
            let bytes: Vec<u8> = (0..=250u8).cycle().take(len).collect();
            std::fs::write(&path, &bytes).expect("write");
            let hashed = sha256_file(&path).expect("hash");
            assert_eq!(hashed, hex(&Sha256::digest(&bytes)), "{len} bytes");
        }
    }

    /// A directory is refused, not read.
    #[test]
    fn only_a_regular_file_is_hashed() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(sha256_file(dir.path()).is_err());
        assert!(sha256_file(&dir.path().join("absent")).is_err());
    }

    #[test]
    fn hex_is_lowercase_and_two_digits_a_byte() {
        assert_eq!(hex(&[]), "");
        assert_eq!(hex(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
    }

    /// What a serializer writes through [`Hashing`] hashes as if it had been
    /// written to a buffer and hashed whole.
    #[test]
    fn hashing_a_writer_matches_hashing_the_bytes() {
        use std::io::Write as _;
        let mut streamed = Sha256::new();
        let mut writer = Hashing(&mut streamed);
        writer.write_all(b"one ").expect("write");
        writer.write_all(b"two").expect("write");
        writer.flush().expect("flush");
        assert_eq!(streamed.finalize(), Sha256::digest(b"one two"));
    }
}
