//! Durable, opt-in fetch backlog. Entries retain provenance, pins and hop depth.
use fletch::Reference;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const MAX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ENTRIES: usize = 100_000;
/// Most entries one root may hold. The backlog is shared, and past
/// [`MAX_ENTRIES`] every checkpoint fails for *every* root; one hostile root
/// naming a hundred thousand URLs must not wedge the rest. Additions past this
/// are dropped (and logged), never the backlog.
const MAX_ROOT_ENTRIES: usize = 10_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Entry {
    pub root_sha: String,
    pub source_sha: String,
    pub hop: u8,
    pub reference: Reference,
    pub reason: String,
    #[serde(default)]
    pub redirect_credit: u8,
}
#[derive(Serialize, Deserialize)]
struct State {
    version: u32,
    entries: Vec<Entry>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            version: 1,
            entries: Vec::new(),
        }
    }
}

pub(super) struct Store {
    path: PathBuf,
}
impl Store {
    pub(super) fn open(path: &Path) -> io::Result<Self> {
        let store = Self {
            path: path.to_owned(),
        };
        let _ = store.read()?;
        Ok(store)
    }
    fn read(&self) -> io::Result<State> {
        let file = match File::open(&self.path) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(State::default()),
            Err(e) => return Err(e),
        };
        let mut bytes = Vec::new();
        file.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err(io::Error::other("fetch backlog exceeds size limit"));
        }
        let state: State = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        if state.version != 1
            || state.entries.len() > MAX_ENTRIES
            || state
                .entries
                .iter()
                .any(|entry| entry.redirect_credit > super::MAX_REDIRECT_HOPS)
        {
            return Err(io::Error::other("unsupported or oversized fetch backlog"));
        }
        Ok(state)
    }
    pub(super) fn entries(&self, root_sha: &str) -> io::Result<Vec<Entry>> {
        Ok(self
            .read()?
            .entries
            .into_iter()
            .filter(|e| e.root_sha == root_sha)
            .collect())
    }
    /// Reload under an OS file lock: independent batch workers and processes
    /// never overwrite another root's pending work. A failed write leaves the
    /// previous state intact. Corrupt/unknown state is never silently replaced.
    pub(super) fn update(&self, additions: &[Entry], completed: &[Entry]) -> io::Result<()> {
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let mut lock_name = self.path.as_os_str().to_os_string();
        lock_name.push(".lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(PathBuf::from(lock_name))?;
        lock.lock()?;
        let mut state = self.read()?;
        let key = |entry: &Entry| {
            (
                entry.root_sha.clone(),
                entry.source_sha.clone(),
                entry.hop,
                entry.reference.clone(),
            )
        };
        // One complete analysis satisfies duplicate declarations of the same
        // pinned target. Offsets or scope can change between detector builds;
        // they must not strand an older backlog entry forever. Different pins
        // remain independent work, even when they name the same package.
        let completion_key = |entry: &Entry| {
            (
                entry.root_sha.clone(),
                entry.reference.locator.clone(),
                entry.reference.pinned_hash.clone(),
            )
        };
        let done: std::collections::HashSet<_> = completed.iter().map(completion_key).collect();
        state
            .entries
            .retain(|entry| !done.contains(&completion_key(entry)));
        let mut indices: std::collections::HashMap<_, _> = state
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| (key(entry), i))
            .collect();
        let mut per_root: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for entry in &state.entries {
            *per_root.entry(entry.root_sha.clone()).or_default() += 1;
        }
        let mut dropped = 0usize;
        for entry in additions {
            let signature = key(entry);
            if let Some(&i) = indices.get(&signature) {
                state.entries[i].reason.clone_from(&entry.reason);
                state.entries[i].redirect_credit =
                    state.entries[i].redirect_credit.max(entry.redirect_credit);
                continue;
            }
            let held = per_root.entry(entry.root_sha.clone()).or_default();
            if *held >= MAX_ROOT_ENTRIES {
                dropped += 1;
                continue;
            }
            *held += 1;
            indices.insert(signature, state.entries.len());
            state.entries.push(entry.clone());
        }
        if dropped > 0 {
            tracing::warn!(
                dropped,
                cap = MAX_ROOT_ENTRIES,
                "fetch backlog: root at its entry cap; additions dropped"
            );
        }
        if state.entries.len() > MAX_ENTRIES {
            return Err(io::Error::other("fetch backlog exceeds entry limit"));
        }
        let bytes = serde_json::to_vec(&state).map_err(io::Error::other)?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err(io::Error::other("fetch backlog exceeds size limit"));
        }
        let mut temp = tempfile::NamedTempFile::new_in(parent)?;
        temp.write_all(&bytes)?;
        temp.as_file().sync_all()?;
        temp.persist(&self.path).map_err(|e| e.error)?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fletch::{RefKind, RefLocator};
    fn entry(root: &str, hop: u8) -> Entry {
        let mut reference = Reference::new(
            RefLocator::Purl("pkg:npm/example@1.2.3".into()),
            RefKind::Dependency,
            "packages.node_modules/alias",
            "node_modules/alias",
        );
        reference.offset = Some(71);
        reference.pinned_hash = Some(filefacts::PinnedHash {
            algo: filefacts::HashAlgo::Sha512,
            value: "EXACT_PIN".into(),
        });
        Entry {
            root_sha: root.into(),
            source_sha: "source".into(),
            hop,
            reference,
            reason: "depth limit".into(),
            redirect_credit: 0,
        }
    }
    #[test]
    fn roundtrip_preserves_pin_offset_and_depth_and_isolates_roots() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("queue.json")).unwrap();
        let mut a = entry("a", 3);
        a.redirect_credit = 2;
        let b = entry("b", 1);
        store.update(&[a.clone(), b.clone()], &[]).unwrap();
        assert_eq!(store.entries("a").unwrap(), vec![a.clone()]);
        assert!(store.entries("changed-root").unwrap().is_empty());
        store.update(&[], &[a]).unwrap();
        assert!(store.entries("a").unwrap().is_empty());
        assert_eq!(store.entries("b").unwrap(), vec![b]);
    }
    #[test]
    fn duplicate_update_changes_reason_and_conflicting_pins_stay_distinct() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("queue")).unwrap();
        let a = entry("a", 2);
        let mut b = a.clone();
        b.reason = "timeout".into();
        store.update(&[a, b.clone()], &[]).unwrap();
        assert_eq!(store.entries("a").unwrap(), vec![b.clone()]);
        b.reference.pinned_hash.as_mut().unwrap().value = "DIFFERENT".into();
        store.update(&[b], &[]).unwrap();
        assert_eq!(store.entries("a").unwrap().len(), 2);
    }
    #[test]
    fn corrupt_and_future_state_is_not_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("queue");
        for bytes in [b"broken".as_slice(), br#"{"version":2,"entries":[]}"#] {
            std::fs::write(&path, bytes).unwrap();
            assert!(Store::open(&path).is_err());
            let store = Store { path: path.clone() };
            assert!(store.update(&[entry("a", 0)], &[]).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
        }
    }
    #[test]
    fn old_entries_default_redirect_credit_and_excess_credit_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("queue");
        let mut value = serde_json::to_value(entry("a", 2)).unwrap();
        value.as_object_mut().unwrap().remove("redirect_credit");
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({"version":1,"entries":[value]})).unwrap(),
        )
        .unwrap();
        assert_eq!(
            Store::open(&path).unwrap().entries("a").unwrap()[0].redirect_credit,
            0
        );
        let mut invalid = entry("a", 2);
        invalid.redirect_credit = super::super::MAX_REDIRECT_HOPS + 1;
        let bytes =
            serde_json::to_vec(&serde_json::json!({"version":1,"entries":[invalid]})).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        assert!(Store::open(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }
    #[test]
    fn one_root_cannot_fill_the_shared_backlog() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("pending.json")).unwrap();
        let flood: Vec<Entry> = (0..=MAX_ROOT_ENTRIES)
            .map(|i| {
                let mut e = entry("hostile", 0);
                e.reference.locator = RefLocator::Purl(format!("pkg:npm/flood-{i}@1.0.0"));
                e
            })
            .collect();
        store.update(&flood, &[]).unwrap();
        assert_eq!(store.entries("hostile").unwrap().len(), MAX_ROOT_ENTRIES);
        // Another root still checkpoints.
        store.update(&[entry("other", 0)], &[]).unwrap();
        assert_eq!(store.entries("other").unwrap().len(), 1);
    }

    #[test]
    fn concurrent_updates_preserve_every_root() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("queue");
        std::thread::scope(|scope| {
            for n in 0..8 {
                let path = path.clone();
                scope.spawn(move || {
                    Store::open(&path)
                        .unwrap()
                        .update(&[entry(&n.to_string(), 1)], &[])
                        .unwrap();
                });
            }
        });
        assert_eq!(Store::open(&path).unwrap().read().unwrap().entries.len(), 8);
    }
}

#[cfg(test)]
mod duplicate_completion_tests {
    use super::*;
    #[test]
    fn complete_target_clears_duplicate_origins_but_not_conflicting_pins() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("queue")).unwrap();
        let first = Entry {
            root_sha: "root".into(),
            source_sha: "old-parent".into(),
            hop: 3,
            reference: Reference::new(
                fletch::RefLocator::Purl("pkg:npm/pkg@1.0.0".into()),
                fletch::RefKind::Dependency,
                "lock",
                "pkg",
            ),
            reason: "in progress".into(),
            redirect_credit: 0,
        };
        let mut duplicate = first.clone();
        duplicate.reference.offset = Some(200);
        duplicate.source_sha = "new-parent".into();
        duplicate.hop = 1;
        let mut conflicting = first.clone();
        conflicting.reference.pinned_hash = Some(filefacts::PinnedHash {
            algo: filefacts::HashAlgo::Sha512,
            value: "OTHER".into(),
        });
        store
            .update(&[first.clone(), duplicate, conflicting.clone()], &[])
            .unwrap();
        store.update(&[], &[first]).unwrap();
        assert_eq!(store.entries("root").unwrap(), vec![conflicting]);
    }
}
