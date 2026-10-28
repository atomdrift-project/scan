//! FIFO handoff with a bounded number of resident payloads. Overflow lives in
//! private temporary files, so network backpressure never parks an analyzer.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};

use serde::{Serialize, de::DeserializeOwned};

#[derive(Debug)]
pub(super) struct Sender<T> {
    tx: mpsc::Sender<Entry<T>>,
    resident: Arc<AtomicUsize>,
    capacity: usize,
}

pub(super) struct Receiver<T>(mpsc::Receiver<Entry<T>>);

#[derive(Debug)]
enum Entry<T> {
    Memory(T, Slot),
    Disk(tempfile::TempPath),
}

#[derive(Debug)]
struct Slot(Arc<AtomicUsize>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

pub(super) fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let (tx, rx) = mpsc::channel();
    (
        Sender {
            tx,
            resident: Arc::default(),
            capacity,
        },
        Receiver(rx),
    )
}

impl<T: Serialize> Sender<T> {
    /// On failure return the original job so the caller can reclaim any
    /// artifact staging it owns. Dropping an unsent entry also reclaims its
    /// resident slot or temporary file.
    pub(super) fn send(&self, job: T) -> Result<(), (T, String)> {
        if self
            .resident
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n < self.capacity).then_some(n + 1)
            })
            .is_ok()
        {
            let entry = Entry::Memory(job, Slot(Arc::clone(&self.resident)));
            return self.tx.send(entry).map_err(|mpsc::SendError(entry)| {
                let Entry::Memory(job, _slot) = entry else {
                    unreachable_entry()
                };
                (job, "uploader stopped".to_owned())
            });
        }

        let path = match spool(&job) {
            Ok(path) => path,
            Err(error) => return Err((job, format!("spool upload: {error}"))),
        };
        self.tx
            .send(Entry::Disk(path))
            .map_err(|_closed| (job, "uploader stopped".to_owned()))
    }
}

// This helper avoids an unchecked extraction at the sole typed send site.
// The channel returns exactly the Entry it was given.
#[allow(clippy::panic)]
fn unreachable_entry() -> ! {
    panic!("channel returned a different entry variant")
}

fn spool<T: Serialize>(job: &T) -> Result<tempfile::TempPath, Box<dyn std::error::Error>> {
    let mut file = tempfile::Builder::new()
        .prefix("scan-upload-queue-")
        .tempfile()?;
    {
        let mut writer = zstd::stream::write::Encoder::new(file.as_file_mut(), 1)?;
        serde_json::to_writer(&mut writer, job)?;
        writer.finish()?;
    }
    Ok(file.into_temp_path())
}

impl<T: DeserializeOwned> Iterator for Receiver<T> {
    type Item = Result<T, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.0.recv().ok()?;
        Some(match entry {
            Entry::Memory(job, _slot) => Ok(job),
            Entry::Disk(path) => (|| {
                let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
                let reader = zstd::stream::read::Decoder::new(file).map_err(|e| e.to_string())?;
                serde_json::from_reader(reader).map_err(|e| e.to_string())
            })(),
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn overflow_preserves_fifo_and_reclaims_files() {
        let (tx, rx) = channel(2);
        for n in 0..40 {
            tx.send(vec![n; 1024]).expect("enqueue without a consumer");
        }
        assert_eq!(tx.resident.load(Ordering::Relaxed), 2);
        let mut paths = Vec::new();
        let mut entries = Vec::new();
        for _ in 0..40 {
            let entry = rx.0.recv().expect("queued entry");
            if let Entry::Disk(path) = &entry {
                assert!(path.exists());
                paths.push(path.to_path_buf());
            }
            entries.push(entry);
        }
        assert_eq!(paths.len(), 38);
        for entry in entries {
            tx.tx.send(entry).expect("restore queue for decode");
        }
        drop(tx);
        for (n, job) in rx.enumerate() {
            assert_eq!(job.expect("decode"), vec![n; 1024]);
        }
        assert!(paths.iter().all(|p| !p.exists()));
    }

    #[test]
    fn closed_queue_returns_the_job_and_releases_the_slot() {
        for capacity in [0, 2] {
            let (tx, rx) = channel(capacity);
            drop(rx);
            assert_eq!(tx.send(vec![7]).expect_err("closed").0, vec![7]);
            assert_eq!(tx.resident.load(Ordering::Relaxed), 0);
        }
    }
    #[test]
    fn spool_failure_returns_ownership_instead_of_dropping_the_job() {
        #[derive(Debug)]
        struct Unserializable;
        impl Serialize for Unserializable {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("injected serialization failure"))
            }
        }
        let (tx, _rx) = channel(0);
        let (Unserializable, error) = tx.send(Unserializable).expect_err("spooling must fail");
        assert!(error.contains("injected serialization failure"), "{error}");
        assert_eq!(tx.resident.load(Ordering::Relaxed), 0);
    }
}
