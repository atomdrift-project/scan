//! What a model says about a sample: its classification, the level it fires
//! at, and the thresholds that turn scores into both.

use std::fmt;

/// The suspicious ceiling never sits below this level, whatever the deploy.
///
/// Without a floor the strictest deploys would have no suspicious band at all
/// (`100 × 0 = 0`). L500 sits just above where interpreted and floor-placed
/// suspicious verdicts land (see `engine::suspicious_band_level`), so even an
/// `-l 0` deploy keeps a review band for them.
pub(super) const SUSPICIOUS_CEILING_FLOOR: u16 = 500;

/// The suspicious ceiling as a multiple of the active hostile level.
///
/// Suspicious means "up to 100× the false-positive budget you chose for
/// blocking": the default `-l 25` reviews through L2500, and a deploy loose
/// enough to block at L250 reviews the whole grid.
pub(super) const SUSPICIOUS_CEILING_MULTIPLE: u16 = 100;

/// The suspicious ceiling for a deploy at `active`:
/// `min(max_grid_level, max(SUSPICIOUS_CEILING_FLOOR, 100 × active))`.
///
/// Any file that fires at a level looser than `active` but not above this is
/// classified as suspicious. Not necessarily a grid row (`-l 70` gives 7000):
/// fired levels are grid rows, so comparing against it reads as "the loosest
/// row at or below it", which is how the per-route lookup in `policy` resolves
/// it too.
#[must_use]
pub(crate) fn suspicious_ceiling(active: u16, max_grid_level: u16) -> u16 {
    active
        .saturating_mul(SUSPICIOUS_CEILING_MULTIPLE)
        .max(SUSPICIOUS_CEILING_FLOOR)
        .min(max_grid_level)
}

/// Classification outcome, ordered by severity (`Benign < Suspicious < Hostile`).
///
/// Serializes as an integer: 0 = benign, 1 = suspicious, 2 = hostile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum Classification {
    /// File shows no significant malicious indicators.
    Benign = 0,
    /// File has notable suspicious indicators.
    Suspicious = 1,
    /// File is likely malicious.
    Hostile = 2,
}

impl serde::Serialize for Classification {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u8(*self as u8)
    }
}

impl std::fmt::Display for Classification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Benign => write!(f, "benign"),
            Self::Suspicious => write!(f, "suspicious"),
            Self::Hostile => write!(f, "hostile"),
        }
    }
}

/// One route score from a routed ensemble decision.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RouteScore {
    /// Compact model route name, e.g. `az`, `az/native`, `az/elf`.
    #[serde(rename = "rte")]
    pub model: String,
    /// Probability emitted by this route's model, in the space thresholds
    /// live in; it drives the verdict.
    #[serde(rename = "prob")]
    pub probability: f32,
    /// The model's own score. Scan applies no transform to it, so this
    /// equals `probability`; the field stays on the wire for the consumers
    /// that read it.
    #[serde(rename = "raw")]
    pub raw: f32,
    /// Classification after applying this route's thresholds.
    #[serde(rename = "cls")]
    pub classification: Classification,
}

/// One applicable route that was not scored.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SkippedRoute {
    /// Compact model route name, e.g. `az/native`.
    #[serde(rename = "rte")]
    pub model: String,
    /// Short reason the route was not used.
    #[serde(rename = "why")]
    pub reason: &'static str,
}

/// Probability cutoffs used to map model output into a [`Classification`].
///
/// Invariants:
/// - `suspicious` must be within `0.0..=1.0`
/// - `hostile` must be within `0.0..=1.0`
/// - `suspicious <= hostile`
///
/// # Example
/// ```
/// use scan::{Classification, Thresholds};
///
/// let thresholds = Thresholds {
///     suspicious: 0.8,
///     hostile: 0.95,
/// };
/// thresholds.validate()?;
///
/// assert_eq!(thresholds.classify(0.2), Classification::Benign);
/// assert_eq!(thresholds.classify(0.85), Classification::Suspicious);
/// assert_eq!(thresholds.classify(0.99), Classification::Hostile);
/// # Ok::<(), scan::model::ThresholdValidationError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Thresholds {
    /// Minimum probability to classify as suspicious.
    pub suspicious: f32,
    /// Minimum probability to classify as hostile.
    pub hostile: f32,
}

impl Thresholds {
    /// Validate the threshold invariants.
    ///
    /// Callers constructing thresholds dynamically should validate once at the
    /// boundary, then pass the value through the rest of the system unchanged.
    ///
    /// # Errors
    /// When a threshold is outside `[0, 1]` or suspicious exceeds hostile.
    pub fn validate(&self) -> std::result::Result<(), ThresholdValidationError> {
        if !(0.0..=1.0).contains(&self.suspicious) {
            return Err(ThresholdValidationError::OutOfRange {
                name: "suspicious",
                value: self.suspicious,
            });
        }
        if !(0.0..=1.0).contains(&self.hostile) {
            return Err(ThresholdValidationError::OutOfRange {
                name: "hostile",
                value: self.hostile,
            });
        }
        if self.suspicious > self.hostile {
            return Err(ThresholdValidationError::Misordered {
                suspicious: self.suspicious,
                hostile: self.hostile,
            });
        }
        Ok(())
    }

    /// Classify a raw model probability into a [`Classification`].
    ///
    /// This method assumes the thresholds are already valid.
    #[must_use]
    pub fn classify(&self, probability: f32) -> Classification {
        if probability >= self.hostile {
            Classification::Hostile
        } else if probability >= self.suspicious {
            Classification::Suspicious
        } else {
            Classification::Benign
        }
    }

    /// Decide a verdict from a probability, returning the (class, prob,
    /// threshold) triple that the decision was made against.
    ///
    /// The reported `threshold` is the cutoff defining the verdict band:
    /// - Hostile: `hostile` (the cutoff `prob` met or exceeded)
    /// - Suspicious: `suspicious` (the cutoff `prob` met or exceeded)
    /// - Benign: `suspicious` (the cutoff `prob` did not reach)
    #[must_use]
    pub fn decide(&self, probability: f32) -> Decision {
        // The raw threshold path carries no level table (manual `--threshold-*`
        // mode, or the no-grid fallback), so `level` is `Manual`.
        let class = self.classify(probability);
        Decision {
            class,
            probability,
            threshold: if class == Classification::Hostile {
                self.hostile
            } else {
                self.suspicious
            },
            level: Level::Manual,
        }
    }
}

/// The outcome of a routed/threshold decision, carrying the probability the
/// verdict was based on and the threshold it was compared against.
///
/// Invariant: `class = Hostile` iff `probability >= threshold` for the Hostile
/// cutoff; `class = Suspicious` iff `probability >= threshold` for the
/// Suspicious cutoff; `class = Benign` iff `probability < threshold`, where
/// `threshold` is the Suspicious cutoff the score did not reach.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    /// Classification outcome.
    pub class: Classification,
    /// Probability the decision was made on.
    pub probability: f32,
    /// Cutoff defining the verdict band.
    pub threshold: f32,
    /// Level-independent envelope marker (serialized as JSON `lvl`): where this
    /// file's hostile decision fires on the grid. Independent of the deploy
    /// `-l`, so the serialized envelope is identical across levels and
    /// cache-shareable — `-l` only moves the hostile/suspicious cutoffs applied
    /// to `level` to produce `class`.
    pub level: Level,
}

/// The lowest pool-adjusted false-positive level at which a file's
/// hostile decision fires: a property of the file and the model, never of a
/// caller's budget.
///
/// On the wire (`lvl`, `fires_at`) it is `null`, `-1` or the level. That
/// encoding invites comparing `-1` as the tightest level of all, so in code
/// each case is its own variant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Level {
    /// No level applies: manual-threshold mode, or a record that carries none
    /// (a corpus row written before levels, a decision with no answer).
    /// Serialized as `null`.
    #[default]
    Manual,
    /// Fires at no level on the grid. Serialized as `-1`.
    Clean,
    /// Fires at this level and every looser one. Synthesized verdicts (LLM,
    /// trait floor) are placed on the same axis so they decode to their class.
    At(u16),
}

impl Level {
    /// Whether no level table applies; omits `lvl` where the wire does.
    #[must_use]
    pub const fn is_manual(&self) -> bool {
        matches!(self, Self::Manual)
    }
}

impl fmt::Display for Level {
    /// `L50`, `clean` or `manual`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manual => f.write_str("manual"),
            Self::Clean => f.write_str("clean"),
            Self::At(n) => write!(f, "L{n}"),
        }
    }
}

impl serde::Serialize for Level {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            Self::Manual => None,
            Self::Clean => Some(-1),
            Self::At(n) => Some(i32::from(*n)),
        };
        serde::Serialize::serialize(&wire, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Level {
    /// Through `Option`, so an absent field reads as [`Self::Manual`] exactly as
    /// the `Option<i32>` it replaces did.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match <Option<i32> as serde::Deserialize>::deserialize(deserializer)? {
            None => Ok(Self::Manual),
            Some(-1) => Ok(Self::Clean),
            Some(n) => u16::try_from(n).ok().map(Self::At).ok_or_else(|| {
                serde::de::Error::invalid_value(
                    serde::de::Unexpected::Signed(n.into()),
                    &"null, -1, or a level in 0..=65535",
                )
            }),
        }
    }
}

/// Validation error for [`Thresholds`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ThresholdValidationError {
    /// A threshold was outside the inclusive `[0.0, 1.0]` range.
    OutOfRange {
        /// Threshold field name.
        name: &'static str,
        /// Invalid threshold value.
        value: f32,
    },
    /// The suspicious threshold was greater than the hostile threshold.
    Misordered {
        /// Suspicious threshold value.
        suspicious: f32,
        /// Hostile threshold value.
        hostile: f32,
    },
}

impl fmt::Display for ThresholdValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { name, value } => {
                write!(f, "{name} threshold {value} is outside [0.0, 1.0]")
            }
            Self::Misordered {
                suspicious,
                hostile,
            } => write!(
                f,
                "suspicious threshold ({suspicious}) must be less than or equal to hostile threshold ({hostile})"
            ),
        }
    }
}

impl std::error::Error for ThresholdValidationError {}

/// Stable metadata about the loaded model, computed once at startup.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ModelInfo {
    /// Feature spec version (e.g. 20).
    pub version: u32,
    /// Stable preprocessing/inference ABI version.
    pub abi_version: u32,
}
