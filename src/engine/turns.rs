//! Sharing work between the files of one scan that follow the same things.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Mutex, PoisonError};

use cleave::AnalysisReport;

/// Work that belongs together, done a turn at a time.
///
/// The files of one release — a wheel and its sdist, a download kept twice —
/// resolve to the same dependencies. Recorded at once, each analyzed every
/// one of them for itself, because none had finished for the others to find
/// in the cache. Here the first to arrive goes ahead and the rest are left
/// with it: when it is done it does theirs, one after another, on its own
/// thread.
///
/// Nobody waits. A rayon worker blocked on another's work can be holding the
/// very job that work needs, and then neither finishes.
pub(super) struct Turns<T>(Mutex<HashMap<String, VecDeque<T>>>);

impl<T> Default for Turns<T> {
    fn default() -> Self {
        Self(Mutex::new(HashMap::new()))
    }
}

impl<T> Turns<T> {
    /// Do `work` on `item` now, or leave `item` with whoever has the turn for
    /// `key`. Whoever has it does every item left with it before giving it up.
    pub(super) fn take(&self, key: &str, mut item: T, work: impl Fn(T)) {
        {
            let mut turns = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(left) = turns.get_mut(key) {
                left.push_back(item);
                return;
            }
            turns.insert(key.to_string(), VecDeque::new());
        }
        loop {
            work(item);
            let mut turns = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            let Some(next) = turns.get_mut(key).and_then(VecDeque::pop_front) else {
                turns.remove(key);
                return;
            };
            drop(turns);
            item = next;
        }
    }
}

/// The release the archive in `report` claims to be, when it names a package
/// and a version. A lone file is no release: the binaries of one product
/// share a name and a version, and nothing they follow.
pub(super) fn release(report: &AnalysisReport) -> Option<String> {
    if report.files.len() < 2 {
        return None;
    }
    let identity = report.identity.as_ref()?;
    let (name, version) = (identity.name.as_ref()?, identity.version.as_ref()?);
    Some(format!("{}@{}", name.value, version.value))
}

/// `files` in two waves: the first file of each content, then the files
/// whose bytes one of the first wave has (`shas` names the content of each
/// file that was hashed). A file that was not hashed is in the first.
pub(super) fn first_of_each_content(
    files: Vec<PathBuf>,
    shas: &HashMap<PathBuf, String>,
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut seen = HashSet::new();
    files
        .into_iter()
        .partition(|path| shas.get(path).is_none_or(|sha| seen.insert(sha.as_str())))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    /// The first file of each content is in the first wave, and so is a file
    /// that was never hashed; the files that repeat one are in the second.
    #[test]
    fn files_of_the_same_bytes_wait_for_the_first_of_them() {
        let path = |name: &str| PathBuf::from(name);
        let shas: HashMap<PathBuf, String> =
            [("a", "1"), ("b", "2"), ("a-copy", "1"), ("a-again", "1")]
                .into_iter()
                .map(|(name, sha)| (path(name), sha.to_string()))
                .collect();
        let files = ["a", "unhashed", "a-copy", "b", "a-again", "unhashed-too"];
        let (first, again) =
            first_of_each_content(files.iter().map(|name| path(name)).collect(), &shas);
        assert_eq!(first, ["a", "unhashed", "b", "unhashed-too"].map(path));
        assert_eq!(again, ["a-copy", "a-again"].map(path));
    }

    /// Archives that claim the same name and version are one release; a lone
    /// file, or an archive that leaves either out, is none.
    #[test]
    fn only_an_archive_naming_a_package_and_a_version_is_a_release() {
        let claim = |value: &str| {
            let json = format!(r#"{{"value":"{value}","source":"wheel.metadata"}}"#);
            Some(serde_json::from_str(&json).expect("a claim"))
        };
        let claims = |version: Option<&str>, members: usize| {
            let mut report = AnalysisReport::new(cleave::TargetInfo::default());
            let identity = report.identity.get_or_insert_default();
            identity.name = claim("litellm");
            identity.version = version.and_then(claim);
            report.files.resize_with(members, Default::default);
            release(&report)
        };
        assert_eq!(
            claims(Some("1.107.0"), 2).as_deref(),
            Some("litellm@1.107.0")
        );
        assert_eq!(claims(Some("1.107.0"), 1), None, "a lone file");
        assert_eq!(claims(None, 2), None, "no version");
        let mut anonymous = AnalysisReport::new(cleave::TargetInfo::default());
        anonymous.files.resize_with(2, Default::default);
        assert_eq!(release(&anonymous), None);
    }

    /// Items under one key are never worked on at once, each is worked on
    /// exactly once, and nobody waits: an item that arrives while its key is
    /// taken is done by whoever took it. Other keys go ahead meanwhile.
    #[test]
    fn work_under_one_key_takes_turns_and_none_is_lost() {
        const THREADS: usize = 8;
        const ITEMS: usize = 200;
        let turns = Turns::default();
        let busy = [const { AtomicU32::new(0) }; 2];
        let done = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for thread in 0..THREADS {
                let (turns, busy, done) = (&turns, &busy, &done);
                scope.spawn(move || {
                    for item in 0..ITEMS {
                        let key = (thread + item) % 2;
                        turns.take(
                            &key.to_string(),
                            (key, thread, item),
                            |(key, thread, item)| {
                                assert_eq!(
                                    busy[key].fetch_add(1, Ordering::SeqCst),
                                    0,
                                    "two at once"
                                );
                                std::thread::yield_now();
                                done.lock().unwrap().push((thread, item));
                                busy[key].fetch_sub(1, Ordering::SeqCst);
                            },
                        );
                    }
                });
            }
        });
        let mut done = done.into_inner().unwrap();
        done.sort_unstable();
        let every: Vec<_> = (0..THREADS)
            .flat_map(|thread| (0..ITEMS).map(move |item| (thread, item)))
            .collect();
        assert_eq!(done, every);
        assert!(
            turns.0.lock().unwrap().is_empty(),
            "every turn was given up"
        );
    }
}
