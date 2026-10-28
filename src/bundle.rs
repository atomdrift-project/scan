//! Downloading and installing a rules bundle — the model bundle, the bloom
//! filters, the NSRL filters — the one way all three are installed.
//!
//! Each bundle says how to read its manifest, whether an install already
//! matches it, whether installing it would go backwards, and how to stage it;
//! [`update`] does the rest. It leaves pinned and git-managed trees alone,
//! locks the target, stages beside it, and swaps it in, so a reader never sees
//! a partial install.
//!
//! Nothing here is signed. A manifest's sha256 catches a corrupt or truncated
//! download, not a compromised bucket: trust is HTTPS to our own bucket.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

/// The public update bucket (`<base>/versions.toml`, `<base>/models/...`,
/// `<base>/bloom/v<N>/...`, `<base>/nsrl/...`).
const BASE_URL: &str = "https://updates.atomdrift.org/litmus";

/// One kind of installable bundle.
pub(crate) trait Bundle {
    /// What a line about it calls it ("Models", "Bloom filters").
    const NAME: &'static str;
    /// What an install leaves behind to compare against, read from `dir`.
    type Installed;
    /// The remote manifest, read and validated.
    type Remote;

    /// What `dir` holds, if an install completed there.
    fn installed(dir: &Path) -> Option<Self::Installed>;
    /// Read and validate the remote manifest.
    fn fetch(client: &Client) -> Result<Self::Remote>;
    /// Whether `have`, installed in `dir`, is already what `remote` offers.
    fn is_current(dir: &Path, have: &Self::Installed, remote: &Self::Remote) -> bool;
    /// Why installing `remote` over `have` would go backwards, if it would.
    ///
    /// The manifest is unsigned, so a replayed older one — whose files are
    /// still hosted and still hash correctly — would otherwise reinstate
    /// verdicts since revoked.
    fn rollback(have: &Self::Installed, remote: &Self::Remote) -> Option<String>;
    /// Write `remote` into the empty `staging`, which then replaces `dir`
    /// whole. A bundle that fails validation here never goes live.
    fn stage(staging: &Path, client: &Client, remote: &Self::Remote) -> Result<()>;
    /// `remote` as a line names it.
    fn describe(remote: &Self::Remote) -> String;
    /// `have` as a line names it.
    fn describe_installed(have: &Self::Installed) -> String;
    /// Anything else worth saying once a run settles.
    fn note(_remote: &Self::Remote) -> Option<String> {
        None
    }
}

/// Install or refresh bundle `B` in `dir`; `true` when something was
/// installed. `force` reinstalls a current bundle and allows a rollback.
/// `quiet` is a background run: lines go to the log only, and an
/// unreachable bucket fails fast instead of stalling the caller.
///
/// # Errors
/// When the manifest or bundle cannot be fetched or verified, the install
/// would roll back without `force`, or the install itself fails.
pub(crate) fn update<B: Bundle>(dir: &Path, force: bool, quiet: bool) -> Result<bool> {
    let say = |line: &str| {
        tracing::info!(bundle = B::NAME, dir = %dir.display(), "{line}");
        if !quiet {
            eprintln!("{line}");
        }
    };
    if let Some(why) = hands_off(B::NAME, dir) {
        say(&why);
        return Ok(false);
    }
    let client = client(quiet && B::installed(dir).is_some())?;
    let remote = B::fetch(&client)?;
    let current = |have: &Option<B::Installed>| {
        !force
            && have
                .as_ref()
                .is_some_and(|have| B::is_current(dir, have, &remote))
    };
    let settled = |remote: &B::Remote, line: String| {
        say(&line);
        if let Some(note) = B::note(remote) {
            say(&note);
        }
    };
    if current(&B::installed(dir)) {
        settled(
            &remote,
            format!("{} already up to date: {}", B::NAME, B::describe(&remote)),
        );
        return Ok(false);
    }
    let installer = Installer::lock(dir)?;
    // Another updater may have installed while we waited for the lock.
    let have = B::installed(dir);
    if current(&have) {
        return Ok(false);
    }
    if !force
        && let Some(have) = &have
        && let Some(why) = B::rollback(have, &remote)
    {
        bail!(
            "refusing to roll {} back: {why} (pass --force to install it anyway)",
            B::NAME
        );
    }
    say(&format!(
        "Installing {} {}...",
        B::NAME,
        B::describe(&remote)
    ));
    installer.install(|staging| B::stage(staging, &client, &remote))?;
    settled(
        &remote,
        format!(
            "{} updated to {} at {}",
            B::NAME,
            B::describe(&remote),
            dir.display()
        ),
    );
    Ok(true)
}

/// Report what [`update`] would install into `dir`, changing nothing.
///
/// # Errors
/// When the manifest cannot be fetched or validated.
pub(crate) fn check<B: Bundle>(dir: &Path) -> Result<()> {
    if let Some(why) = hands_off(B::NAME, dir) {
        eprintln!("{why}");
        return Ok(());
    }
    let remote = B::fetch(&client(false)?)?;
    let available = B::describe(&remote);
    match B::installed(dir) {
        Some(have) if B::is_current(dir, &have, &remote) => {
            eprintln!("{} up to date: {available}", B::NAME);
        }
        Some(have) => eprintln!(
            "{} update available: {available} (installed: {})",
            B::NAME,
            B::describe_installed(&have)
        ),
        None => eprintln!("{} not installed; available: {available}", B::NAME),
    }
    if let Some(note) = B::note(&remote) {
        eprintln!("{note}");
    }
    Ok(())
}

/// Stall budget: how long a connect, the request, or any one read of a body
/// may take. A download as a whole may take longer; it is bounded by size.
const TIMEOUT: Duration = Duration::from_secs(120);
/// Connect budget for a quiet refresh of an installed bundle, so the default
/// auto-update can't stall a scan on an offline host.
const QUICK_CONNECT_TIMEOUT: Duration = Duration::from_secs(4);
/// The largest manifest any bundle publishes, with ample headroom.
const MAX_MANIFEST_BYTES: u64 = 1 << 20;
/// The largest single file any bundle publishes, with ample headroom.
const MAX_FILE_BYTES: u64 = 4 << 30;

/// The HTTP client for one update run, shared by its manifest and bundle
/// fetches. `quick` fails fast when the bucket is unreachable; a first install
/// or an explicit update stays patient so a slow first fetch still completes.
fn client(quick: bool) -> Result<Client> {
    let mut builder = Client::builder().timeout(TIMEOUT);
    if quick {
        builder = builder.connect_timeout(QUICK_CONNECT_TIMEOUT);
    }
    builder.build().context("building update http client")
}

/// The URL of `path` in the update bucket.
pub(crate) fn url(path: &str) -> String {
    format!("{BASE_URL}/{path}")
}

/// GET a manifest. An HTTP error status is an error whose chain carries the
/// [`reqwest::Error`], so a caller can test for a 404.
pub(crate) fn manifest(client: &Client, url: &str) -> Result<String> {
    let mut text = Vec::new();
    copy_capped(client, url, MAX_MANIFEST_BYTES, &mut text)?;
    String::from_utf8(text).with_context(|| format!("{url} is not valid UTF-8"))
}

/// Stream `url` into `out`, checking it against `sha256` (lowercase hex). That
/// catches a corrupt or truncated download, not a hostile bucket: the manifest
/// the digest comes from is unsigned.
// SIGNING: manifests are unsigned. A signature check belongs in `manifest`,
// the one way every manifest — and so every digest checked here — enters.
pub(crate) fn download(
    client: &Client,
    url: &str,
    sha256: &str,
    out: &mut impl Write,
) -> Result<()> {
    let mut hashed = Hashed {
        out,
        digest: Sha256::new(),
    };
    copy_capped(client, url, MAX_FILE_BYTES, &mut hashed)?;
    let got = format!("{:x}", hashed.digest.finalize());
    if got != sha256 {
        bail!("sha256 mismatch for {url}: got {got}, manifest says {sha256}");
    }
    Ok(())
}

/// [`download`] into a new file at `path`, flushed before it returns.
pub(crate) fn download_to(client: &Client, url: &str, sha256: &str, path: &Path) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    download(client, url, sha256, &mut file)?;
    file.flush()
        .with_context(|| format!("writing {}", path.display()))
}

fn copy_capped(client: &Client, url: &str, max: u64, out: &mut impl Write) -> Result<()> {
    tracing::debug!("fetching {url}");
    let resp = client
        .get(url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .with_context(|| format!("GET {url}"))?;
    let copied =
        io::copy(&mut resp.take(max + 1), out).with_context(|| format!("reading {url}"))?;
    if copied > max {
        bail!("{url} is larger than {max} bytes");
    }
    Ok(())
}

/// A writer that hashes what passes through it.
struct Hashed<'a, W> {
    out: &'a mut W,
    digest: Sha256,
}

impl<W: Write> Write for Hashed<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.out.write(buf)?;
        self.digest.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

/// Why the updater must leave `dir` alone, as a line for the user; `None` when
/// it may install there. `what` names the bundle ("Models").
///
/// `dir` is left alone when `SCAN_MODELS_DIR` names it — "use exactly this
/// bundle": a deploy validating a candidate in a `mktemp -d` with no `.git`, a
/// test pinning a fixture — or when it is a git checkout, which git updates.
fn hands_off(what: &str, dir: &Path) -> Option<String> {
    let var = crate::env::SCAN_MODELS_DIR;
    if crate::env::path(var).is_some_and(|pin| same_dir(&pin, dir)) {
        return Some(format!(
            "{what} at {} come from {}; leaving them untouched.\nUnset {} to update the installed copy.",
            dir.display(),
            var.name,
            var.name,
        ));
    }
    // `exists` follows a symlinked `dir`, so a link into a checkout counts.
    if dir.join(".git").exists() {
        return Some(format!(
            "{what} at {} are git-managed (a checkout or symlink to one); leaving them untouched.\nUse 'git pull' there to update, or remove the directory to switch to bundle updates.",
            dir.display()
        ));
    }
    None
}

/// Same directory, comparing symlink-resolved paths when both exist.
fn same_dir(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// The right to replace one install directory, held across processes.
///
/// Several updaters can target one directory at once (a scan's auto-update,
/// `update-rules`, a server's `/_/update`, postdoc's worker). The lock, a
/// `.<name>.lock` file beside the target, serializes their staging and swap.
/// Readers don't take it; see [`settle`].
#[derive(Debug)]
struct Installer {
    target: PathBuf,
    /// `None` where the platform has no file locking; updaters then race as
    /// they did before this lock existed.
    lock: Option<File>,
}

impl Installer {
    /// Wait for, then take, `target`'s install lock.
    fn lock(target: &Path) -> Result<Self> {
        let (parent, _) = split(target)?;
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        let path = lock_path(target)?;
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let lock = match file.lock() {
            Ok(()) => Some(file),
            Err(e) if e.kind() == io::ErrorKind::Unsupported => {
                tracing::debug!(path = %path.display(), "no file locking here; installing unlocked");
                None
            }
            Err(e) => return Err(e).with_context(|| format!("locking {}", path.display())),
        };
        Ok(Self {
            target: target.to_owned(),
            lock,
        })
    }

    /// Replace the target with the bundle `fill` writes into an empty staging
    /// directory beside it: fill, flush to disk, swap. On any failure the
    /// target keeps its old contents.
    ///
    /// The swap is two renames (old out, new in), so for an instant the target
    /// does not exist. Moving to a `current` symlink would close that, but the
    /// readers in other modules open the directory by path, and symlinks need
    /// privileges on Windows. Readers that find nothing call [`settle`].
    fn install(self, fill: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
        let (parent, name) = split(&self.target)?;
        let prefix = format!(".{name}.staging-");
        if self.lock.is_some() {
            // Holding the lock, no other updater is staging: anything with our
            // prefix was left by one that died mid-install (SIGKILL, Ctrl-C).
            for entry in fs::read_dir(parent).into_iter().flatten().flatten() {
                if entry.file_name().to_string_lossy().starts_with(&prefix) {
                    let _ = fs::remove_dir_all(entry.path());
                }
            }
        }
        let tempdir = || {
            tempfile::Builder::new()
                .prefix(&prefix)
                .tempdir_in(parent)
                .with_context(|| format!("creating a staging dir in {}", parent.display()))
        };

        let staging = tempdir()?;
        fill(staging.path())?;
        sync_tree(staging.path()).context("flushing the staged bundle to disk")?;

        // The old bundle moves into `trash`, which deletes it on drop.
        let trash = tempdir()?;
        let old = trash.path().join(name);
        let had_old = match fs::rename(&self.target, &old) {
            Ok(()) => true,
            Err(e) if e.kind() == io::ErrorKind::NotFound => false,
            Err(e) => {
                return Err(e).with_context(|| format!("moving {} aside", self.target.display()));
            }
        };
        if let Err(e) = fs::rename(staging.path(), &self.target) {
            if had_old {
                let _ = fs::rename(&old, &self.target);
            }
            return Err(e).with_context(|| format!("installing {}", self.target.display()));
        }
        let _ = staging.keep(); // it is the target now
        fsync(parent).with_context(|| format!("flushing {}", parent.display()))
    }
}

/// Wait out any install in progress at `dir`; `false` when no updater has
/// ever run there. A reader that finds `dir` missing calls this and looks
/// again, so the instant between [`Installer::install`]'s two renames never
/// reads as "not installed".
pub(crate) fn settle(dir: &Path) -> bool {
    let Ok(path) = lock_path(dir) else {
        return false;
    };
    File::open(path).is_ok_and(|file| file.lock_shared().is_ok())
}

/// `target`'s parent directory and file name.
fn split(target: &Path) -> Result<(&Path, String)> {
    let name = target
        .file_name()
        .with_context(|| format!("install target {} has no name", target.display()))?;
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok((parent, name.to_string_lossy().into_owned()))
}

fn lock_path(target: &Path) -> Result<PathBuf> {
    let (parent, name) = split(target)?;
    Ok(parent.join(format!(".{name}.lock")))
}

/// Flush every file and directory under `dir` to disk, so a crash after the
/// swap can't leave an installed bundle of empty or torn files.
fn sync_tree(dir: &Path) -> Result<()> {
    for entry in walkdir::WalkDir::new(dir) {
        let entry = entry?;
        if entry.file_type().is_file() || entry.file_type().is_dir() {
            fsync(entry.path())?;
        }
    }
    Ok(())
}

/// fsync a file or directory. Unix only: on Windows, flushing needs a writable
/// handle, which a read-only extracted file refuses, and a directory can't be
/// opened as a file.
fn fsync(path: &Path) -> io::Result<()> {
    if cfg!(unix) {
        File::open(path)?.sync_all()
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_dir_matches_identical_paths() {
        assert!(same_dir(Path::new("/tmp/bundle"), Path::new("/tmp/bundle")));
    }

    #[test]
    fn same_dir_rejects_unrelated_paths() {
        assert!(!same_dir(Path::new("/tmp/bundle"), Path::new("/tmp/other")));
    }

    #[test]
    #[cfg(unix)] // std::os::unix symlink; Windows symlink creation also needs privileges
    fn same_dir_resolves_symlinks() {
        // The deploy dir is a symlink to a checkout; a pin naming either spelling
        // must resolve to the same bundle.
        let root = tempfile::tempdir().expect("tempdir");
        let real = root.path().join("real");
        fs::create_dir(&real).expect("mkdir");
        let link = root.path().join("link");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        assert!(same_dir(&link, &real));
    }

    #[test]
    fn same_dir_is_lexical_when_paths_do_not_exist() {
        // Nonexistent paths can't be canonicalized; equal spellings still match,
        // unequal ones stay distinct rather than erroring.
        let a = Path::new("/nonexistent/a");
        assert!(same_dir(a, Path::new("/nonexistent/a")));
        assert!(!same_dir(a, Path::new("/nonexistent/b")));
    }

    #[test]
    fn a_git_checkout_is_left_alone() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("bundle");
        fs::create_dir_all(dir.join(".git")).unwrap();
        let why = hands_off("Models", &dir).unwrap();
        assert!(why.contains("git-managed"), "{why}");
        fs::remove_dir(dir.join(".git")).unwrap();
        assert_eq!(hands_off("Models", &dir), None);
    }

    fn files(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn install_replaces_the_target_and_leaves_only_the_lock_behind() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("bundle");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("old.bin"), b"old").unwrap();
        // A staging dir abandoned by an updater that was killed mid-install.
        fs::create_dir(root.path().join(".bundle.staging-dead")).unwrap();

        Installer::lock(&target)
            .unwrap()
            .install(|staging| Ok(fs::write(staging.join("new.bin"), b"new")?))
            .unwrap();

        assert_eq!(files(&target), ["new.bin"]);
        assert_eq!(files(root.path()), [".bundle.lock", "bundle"]);
        assert!(settle(&target), "an updater ran here");
    }

    #[test]
    fn a_failed_fill_keeps_the_old_bundle() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("bundle");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("old.bin"), b"old").unwrap();

        let err = Installer::lock(&target)
            .unwrap()
            .install(|staging| {
                fs::write(staging.join("half.bin"), b"half")?;
                bail!("validation failed")
            })
            .unwrap_err();

        assert!(format!("{err:#}").contains("validation failed"));
        assert_eq!(files(&target), ["old.bin"]);
        assert_eq!(files(root.path()), [".bundle.lock", "bundle"]);
    }

    #[test]
    fn a_first_install_creates_the_target() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("nested").join("bundle");
        assert!(!settle(&target), "no updater has run here yet");
        Installer::lock(&target)
            .unwrap()
            .install(|staging| Ok(fs::write(staging.join("f"), b"x")?))
            .unwrap();
        assert_eq!(files(&target), ["f"]);
    }

    /// A bundle whose "manifest" is a version number, published in a thread
    /// local so each test sets its own.
    struct Fake;

    thread_local! {
        static PUBLISHED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    impl Bundle for Fake {
        const NAME: &'static str = "Fake";
        type Installed = u32;
        type Remote = u32;

        fn installed(dir: &Path) -> Option<u32> {
            fs::read_to_string(dir.join("version")).ok()?.parse().ok()
        }
        fn fetch(_client: &Client) -> Result<u32> {
            Ok(PUBLISHED.get())
        }
        fn is_current(_dir: &Path, have: &u32, remote: &u32) -> bool {
            have == remote
        }
        fn rollback(have: &u32, remote: &u32) -> Option<String> {
            (remote < have).then(|| format!("{remote} is older than {have}"))
        }
        fn stage(staging: &Path, _client: &Client, remote: &u32) -> Result<()> {
            Ok(fs::write(staging.join("version"), remote.to_string())?)
        }
        fn describe(remote: &u32) -> String {
            remote.to_string()
        }
        fn describe_installed(have: &u32) -> String {
            have.to_string()
        }
    }

    #[test]
    fn update_installs_skips_when_current_and_refuses_a_rollback() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("fake");

        PUBLISHED.set(2);
        assert!(update::<Fake>(&dir, false, true).unwrap(), "first install");
        assert!(
            !update::<Fake>(&dir, false, true).unwrap(),
            "already current"
        );
        assert!(
            update::<Fake>(&dir, true, true).unwrap(),
            "force reinstalls"
        );

        PUBLISHED.set(1);
        let err = update::<Fake>(&dir, false, true).unwrap_err();
        assert!(
            format!("{err:#}").contains("refusing to roll Fake back"),
            "{err:#}"
        );
        assert_eq!(
            Fake::installed(&dir),
            Some(2),
            "the refusal changed nothing"
        );
        assert!(update::<Fake>(&dir, true, true).unwrap(), "force allows it");
        assert_eq!(Fake::installed(&dir), Some(1));

        PUBLISHED.set(3);
        assert!(
            update::<Fake>(&dir, false, true).unwrap(),
            "a newer one installs"
        );
        assert_eq!(Fake::installed(&dir), Some(3));
    }

    #[test]
    fn update_leaves_a_git_checkout_alone() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("fake");
        fs::create_dir_all(dir.join(".git")).unwrap();
        PUBLISHED.set(1);
        assert!(!update::<Fake>(&dir, true, true).unwrap());
        assert_eq!(Fake::installed(&dir), None);
    }

    /// Serve `body` once on a local port; the URL to fetch it from.
    fn serve_once(body: Vec<u8>) -> (String, std::thread::JoinHandle<()>) {
        use std::io::{BufRead, BufReader};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/m", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            let mut request = String::new();
            let mut reader = BufReader::new(&mut connection);
            while !request.ends_with("\r\n\r\n") {
                assert!(reader.read_line(&mut request).unwrap() > 0);
            }
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            // The client may hang up once it has read past the cap.
            let _ = connection.write_all(head.as_bytes());
            let _ = connection.write_all(&body);
        });
        (url, server)
    }

    #[test]
    fn a_manifest_past_the_cap_is_refused() {
        let cap = usize::try_from(MAX_MANIFEST_BYTES).unwrap();
        let (url, server) = serve_once(vec![b'#'; cap + 1]);
        let err = manifest(&Client::new(), &url).unwrap_err();
        assert!(format!("{err:#}").contains("larger than"), "{err:#}");
        server.join().unwrap();

        let (url, server) = serve_once(vec![b'#'; cap]);
        assert_eq!(manifest(&Client::new(), &url).unwrap().len(), cap);
        server.join().unwrap();
    }
}
