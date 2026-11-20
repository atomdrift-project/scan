//! Member scores remembered across scans of the same report.
//!
//! What the model says of a member is a pure function of the member's compact
//! record, the model, and the options a score depends on. A repeat scan of an
//! artifact whose dependencies are already analyzed spent most of its time
//! recomputing those answers: tens of thousands of members, three routes each.
//!
//! Each report — the root with what was grafted onto it, and each dependency
//! on its own — keeps a table of its members' scores, named by the sha256 of
//! the report's root file and stored in the analysis cache's namespace, so a
//! rules, engine or model change starts from nothing. An entry is keyed by a
//! digest of the member's whole serialized record: a member that changed in
//! any way, a grafted finding included, misses and is scored again.
//!
//! Only the model's reading is kept. The trait floor reads the findings
//! directly and is applied on every scan.

use std::collections::HashMap;

use cleave::types::CompactFile;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::analysis_cache::AnalysisCache;
use crate::features::RawNeeds;
use crate::model::{Classification, Decision, Level, Model, RouteScore, SkippedRoute};

/// What the model said of one member: its decision, each route's score, and
/// the applicable routes it could not use.
pub(super) type Reading = (Decision, Vec<RouteScore>, Vec<SkippedRoute>);

/// The first 128 bits of a member's digest, in two halves.
pub(super) type Key = (u64, u64);

/// A [`Reading`] as stored: the wire types serialize for consumers and cannot
/// all be read back (a skip reason is a static string).
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Stored {
    class: Classification,
    probability: f32,
    threshold: f32,
    level: Level,
    /// Route name, probability, class.
    routes: Vec<(String, f32, Classification)>,
    /// Route name, reason.
    skipped: Vec<(String, String)>,
}

impl Stored {
    pub(super) fn new((decision, routes, skipped): &Reading) -> Self {
        Self {
            class: decision.class,
            probability: decision.probability,
            threshold: decision.threshold,
            level: decision.level,
            routes: routes
                .iter()
                .map(|r| (r.model.clone(), r.probability, r.classification))
                .collect(),
            skipped: skipped
                .iter()
                .map(|s| (s.model.clone(), s.reason.to_string()))
                .collect(),
        }
    }

    /// The reading this entry holds, or `None` when it names a skip reason
    /// this build does not have; the member is then scored again.
    fn reading(&self) -> Option<Reading> {
        let decision = Decision {
            class: self.class,
            probability: self.probability,
            threshold: self.threshold,
            level: self.level,
        };
        let routes = self
            .routes
            .iter()
            .map(|(model, probability, classification)| RouteScore {
                model: model.clone(),
                probability: *probability,
                raw: *probability,
                classification: *classification,
            })
            .collect();
        let skipped = self
            .skipped
            .iter()
            .map(|(model, reason)| {
                let reason = SkippedRoute::REASONS
                    .into_iter()
                    .find(|known| *known == reason.as_str())?;
                Some(SkippedRoute {
                    model: model.clone(),
                    reason,
                })
            })
            .collect::<Option<_>>()?;
        Some((decision, routes, skipped))
    }
}

/// One report's remembered member scores.
pub(super) struct ScoreTable {
    cache: &'static AnalysisCache,
    name: String,
    /// A digest already fed everything but the member that a score depends
    /// on; each member's key continues a copy of it.
    seed: Sha256,
    prior: HashMap<Key, Stored>,
}

impl ScoreTable {
    /// The table for the report rooted at `root_sha`, as `model` scores it
    /// with `needs` parsed. `None` when the analysis cache is off or the
    /// report has no content hash to name the table by.
    pub(super) fn open(root_sha: &str, model: &Model, needs: RawNeeds) -> Option<Self> {
        // The sha names a file: nothing but hex digits may reach the path.
        if root_sha.len() != 64 || !root_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let cache = AnalysisCache::shared()?;
        let name = format!("{root_sha}.scores");
        // The deploy level and the thresholds decide the class a probability
        // maps to, and the parsed inputs decide what a route can read; the
        // model bundle itself is in the cache namespace.
        let seed = Sha256::new_with_prefix(format!(
            "{needs:?}|{:?}|{:?}|",
            model.active_level(),
            model.thresholds()
        ));
        let prior = cache
            .read::<Vec<(Key, Stored)>>(&name)
            .map(|entries| entries.into_iter().collect())
            .unwrap_or_default();
        Some(Self {
            cache,
            name,
            seed,
            prior,
        })
    }

    /// The key `file` is remembered under: a digest of its serialized record.
    /// `None` when the record cannot be serialized; it is then scored and not
    /// remembered.
    pub(super) fn key(&self, file: &CompactFile) -> Option<Key> {
        let mut digest = self.seed.clone();
        serde_json::to_writer(&mut digest, file).ok()?;
        let digest = digest.finalize();
        let half = |at: usize| digest[at..at + 8].try_into().ok().map(u64::from_be_bytes);
        Some((half(0)?, half(8)?))
    }

    /// The entry and reading remembered under `key`.
    pub(super) fn get(&self, key: Key) -> Option<(&Stored, Reading)> {
        let stored = self.prior.get(&key)?;
        Some((stored, stored.reading()?))
    }

    /// Replace the table with `entries`: this scan's members, so a table
    /// never outgrows its report.
    pub(super) fn store(&self, entries: &[(Key, Stored)]) {
        self.cache.write(&self.name, &entries);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading() -> Reading {
        let decision = Decision {
            class: Classification::Suspicious,
            probability: 0.123_456_79,
            threshold: 0.987_654_3,
            level: Level::At(40),
        };
        let routes = vec![RouteScore {
            model: "az/python".to_string(),
            probability: 0.333_333_34,
            raw: 0.333_333_34,
            classification: Classification::Hostile,
        }];
        let skipped = vec![SkippedRoute {
            model: "az/native".to_string(),
            reason: SkippedRoute::UNAVAILABLE,
        }];
        (decision, routes, skipped)
    }

    /// A stored reading comes back bit for bit, through the same JSON the
    /// table is written as.
    #[test]
    fn a_reading_survives_storage_exactly() {
        let original = reading();
        let json = serde_json::to_string(&Stored::new(&original)).unwrap();
        let stored: Stored = serde_json::from_str(&json).unwrap();
        let (decision, routes, skipped) = stored.reading().expect("every value is known");
        assert_eq!(decision, original.0);
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].model, "az/python");
        assert_eq!(routes[0].probability.to_bits(), 0.333_333_34_f32.to_bits());
        assert_eq!(routes[0].raw.to_bits(), routes[0].probability.to_bits());
        assert_eq!(routes[0].classification, Classification::Hostile);
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].model, "az/native");
        assert_eq!(skipped[0].reason, SkippedRoute::UNAVAILABLE);
    }

    /// An entry naming a skip reason this build lacks is a miss, never a
    /// guess, and one naming an unknown class does not parse at all.
    #[test]
    fn an_entry_with_an_unknown_value_is_a_miss() {
        let mut stored = Stored::new(&reading());
        stored.skipped[0].1 = "retired".to_string();
        assert!(stored.reading().is_none());

        let json = serde_json::to_string(&Stored::new(&reading())).unwrap();
        let unknown_class = json.replacen("\"class\":1", "\"class\":3", 1);
        assert_ne!(json, unknown_class);
        assert!(serde_json::from_str::<Stored>(&unknown_class).is_err());
    }
}
