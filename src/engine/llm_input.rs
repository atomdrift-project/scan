//! A deferred LLM pass, stored so another host can run it.
//!
//! A worker that cannot finish a sample's LLM pass — it has no interpreter,
//! the pass was skipped under load, or the call failed — still posts the ML
//! verdict, then stores an [`LlmInput`]: exactly what
//! [`apply_pending_interpretation`](super::apply_pending_interpretation) reads
//! and what the amendment it produces re-posts. A worker with an interpreter
//! later runs only that pass from it, never the analysis.
//!
//! The encoding is [`MAGIC`], one format byte ([`LLM_INPUT_FORMAT`]), then a
//! zstd frame of JSON. JSON because the upload document holds raw JSON
//! objects (jurors, scanners, identities) that only a JSON serializer can
//! carry verbatim; zstd because the reports in it are repetitive text.

use std::borrow::Cow;
use std::io::Read as _;

use serde::{Deserialize, Serialize};

use super::{
    DependencyAttention, FindingCounts, HopperRoute, MemberEvals, PendingLlm, ScanResult, UploadDoc,
};
use crate::interpret::{Admitted, LlmAdmission};
use crate::model::{Classification, Decision, Level, Model};

/// The format [`LlmInput::encode`] writes and the only one
/// [`LlmInput::decode`] reads. Bump it whenever a stored field changes, in
/// this module or in the types it stores ([`UploadDoc`], [`Admitted`], and
/// those inside them): a blob of any other format decodes as absent, and the
/// worker that reads it falls back to a full analysis.
pub const LLM_INPUT_FORMAT: u8 = 1;

/// The largest input [`LlmInput::encode`] returns, half of hopper's 16 MiB
/// cap on a stored input.
pub const LLM_INPUT_MAX_BYTES: usize = 8 << 20;

/// The largest blob [`LlmInput::decode`] reads: hopper's cap.
const MAX_STORED_BYTES: usize = 16 << 20;

/// The most a blob may decompress to. Real inputs compress 5-15x, so this is
/// far past anything [`LLM_INPUT_MAX_BYTES`] admits; it bounds what a hostile
/// or corrupt blob can make a reader allocate.
const MAX_DECODED_BYTES: usize = 256 << 20;

/// Leads every encoded input, so a blob of anything else is told apart
/// before it is decompressed.
const MAGIC: &[u8; 4] = b"ADLI";

/// The level the upload body is compressed at, for the same kind of text.
const ZSTD_LEVEL: i32 = 3;

/// Why [`LlmInput::encode`] wrote nothing.
#[derive(Debug)]
#[non_exhaustive]
pub enum LlmInputError {
    /// The result was not classified for upload, so an amendment would have
    /// no body to re-post.
    NotForUpload,
    /// Encoded, the input is this many bytes, over [`LLM_INPUT_MAX_BYTES`].
    TooLarge(usize),
    /// It could not be serialized or compressed.
    Encode(std::io::Error),
}

impl std::fmt::Display for LlmInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotForUpload => f.write_str("result was not classified for upload"),
            Self::TooLarge(bytes) => write!(
                f,
                "encoded LLM input is {bytes} bytes; the limit is {LLM_INPUT_MAX_BYTES}"
            ),
            Self::Encode(e) => write!(f, "encoding the LLM input: {e}"),
        }
    }
}

impl std::error::Error for LlmInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encode(e) => Some(e),
            Self::NotForUpload | Self::TooLarge(_) => None,
        }
    }
}

/// What is stored: the fields of a [`ScanResult`] the LLM pass reads or the
/// amendment re-posts, the analysis time the amendment is stamped with, and
/// the pending pass itself. Borrowed to encode, owned once decoded.
#[derive(Debug, Serialize, Deserialize)]
struct Stored<'a> {
    sha256: Cow<'a, str>,
    path: Cow<'a, str>,
    file_type: Cow<'a, str>,
    size_bytes: u64,
    /// The model version the verdict was reached with.
    version: Cow<'a, str>,
    analyzed_at: Cow<'a, str>,
    analysis_ms: u64,
    classification: Classification,
    probability: f32,
    threshold: f32,
    level: Level,
    /// [`PendingLlm::ctx`]: the sanitized render the model reads.
    ctx: Cow<'a, str>,
    admitted: Admitted,
    upload: Cow<'a, UploadDoc>,
}

/// A deferred LLM pass, decoded: run it by turning it back into a result with
/// [`into_result`](Self::into_result) and handing that to
/// [`apply_pending_interpretation`](super::apply_pending_interpretation).
#[derive(Debug)]
pub struct LlmInput {
    stored: Stored<'static>,
}

impl LlmInput {
    /// Encode `result`'s pending LLM pass for a host that can run it.
    /// `pending` is passed apart from the result because a failed call has
    /// already taken it out; `analysis_ms` is what the amendment's body is
    /// stamped with, as the original post was.
    ///
    /// # Errors
    ///
    /// [`LlmInputError::NotForUpload`] when the result has no upload body,
    /// [`LlmInputError::TooLarge`] past [`LLM_INPUT_MAX_BYTES`], and
    /// [`LlmInputError::Encode`] when serializing or compressing fails.
    pub fn encode(
        result: &ScanResult,
        pending: &PendingLlm,
        analysis_ms: u64,
    ) -> Result<Vec<u8>, LlmInputError> {
        let upload = result.upload.as_ref().ok_or(LlmInputError::NotForUpload)?;
        let stored = Stored {
            sha256: Cow::Borrowed(&result.sha256),
            path: Cow::Borrowed(&result.path),
            file_type: Cow::Borrowed(&result.file_type),
            size_bytes: result.size_bytes,
            version: Cow::Borrowed(&result.version),
            analyzed_at: Cow::Borrowed(&result.analyzed_at),
            analysis_ms,
            classification: result.classification,
            probability: result.probability,
            threshold: result.threshold,
            level: result.level,
            ctx: Cow::Borrowed(&pending.ctx),
            admitted: pending.admitted,
            upload: Cow::Borrowed(upload),
        };
        let mut out = Vec::with_capacity(64 << 10);
        out.extend_from_slice(MAGIC);
        out.push(LLM_INPUT_FORMAT);
        // Streamed through the compressor, so the JSON is never whole in memory.
        let mut encoder =
            zstd::stream::write::Encoder::new(out, ZSTD_LEVEL).map_err(LlmInputError::Encode)?;
        serde_json::to_writer(&mut encoder, &stored)
            .map_err(|e| LlmInputError::Encode(e.into()))?;
        let out = encoder.finish().map_err(LlmInputError::Encode)?;
        if out.len() > LLM_INPUT_MAX_BYTES {
            return Err(LlmInputError::TooLarge(out.len()));
        }
        Ok(out)
    }

    /// Decode a blob [`encode`](Self::encode) wrote. `None` — logged at debug,
    /// with why — for anything else: another format, a corrupt or truncated
    /// blob, or one past the size limits. A reader treats that as no input.
    #[must_use]
    pub fn decode(blob: &[u8]) -> Option<Self> {
        let refuse = |why: &str| {
            tracing::debug!(bytes = blob.len(), why, "LLM input not decoded");
            None
        };
        if blob.len() > MAX_STORED_BYTES {
            return refuse("over the stored size limit");
        }
        let Some(body) = blob.strip_prefix(MAGIC.as_slice()) else {
            return refuse("not an LLM input");
        };
        let Some((&format, frame)) = body.split_first() else {
            return refuse("truncated");
        };
        if format != LLM_INPUT_FORMAT {
            tracing::debug!(
                format,
                expected = LLM_INPUT_FORMAT,
                "LLM input not decoded: another format"
            );
            return None;
        }
        let Ok(decoder) = zstd::stream::read::Decoder::new(frame) else {
            return refuse("decompressor failed to start");
        };
        let mut json = Vec::new();
        if decoder
            .take(MAX_DECODED_BYTES as u64 + 1)
            .read_to_end(&mut json)
            .is_err()
        {
            return refuse("corrupt compressed frame");
        }
        if json.len() > MAX_DECODED_BYTES {
            return refuse("over the decompressed size limit");
        }
        match serde_json::from_slice::<Stored<'static>>(&json) {
            Ok(stored) => Some(Self { stored }),
            Err(e) => {
                tracing::debug!(error = %e, "LLM input not decoded: malformed");
                None
            }
        }
    }

    /// The sample the input is for.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.stored.sha256
    }

    /// Whether the pass may be skipped under load.
    #[must_use]
    pub const fn tier(&self) -> LlmAdmission {
        self.stored.admitted.tier
    }

    /// The analysis time the original verdict was posted with.
    #[must_use]
    pub const fn analysis_ms(&self) -> u64 {
        self.stored.analysis_ms
    }

    /// Whether `model` blends a grade into this verdict exactly as the model
    /// that reached it would: the same deploy level on the same grid, the
    /// only model facts the blend reads. A host running another bundle must
    /// not amend a verdict it would have placed differently.
    #[must_use]
    pub fn fits(&self, model: &Model) -> bool {
        let levels = self.stored.admitted.levels();
        levels.active == model.active_level() && levels.grid_max == model.grid_max()
    }

    /// The result the pass runs on: everything
    /// [`apply_pending_interpretation`](super::apply_pending_interpretation)
    /// reads, with the pass pending, and the upload body its amendment
    /// re-posts. Nothing else of the original result is kept: no report,
    /// members, findings or renders.
    #[must_use]
    pub fn into_result(self) -> ScanResult {
        let Stored {
            sha256,
            path,
            file_type,
            size_bytes,
            version,
            analyzed_at,
            analysis_ms: _,
            classification,
            probability,
            threshold,
            level,
            ctx,
            admitted,
            upload,
        } = self.stored;
        ScanResult {
            classification,
            dependency_attention: DependencyAttention::default(),
            probability,
            threshold,
            level,
            version: version.into_owned(),
            analyzed_at: analyzed_at.into_owned(),
            cleave: None,
            pids: None,
            deleted: None,
            path: path.into_owned(),
            finding_counts: FindingCounts::default(),
            formula: String::new(),
            reasons: Vec::new(),
            top_findings: Vec::new(),
            file_type: file_type.into_owned(),
            size_bytes,
            sha256: sha256.into_owned(),
            embedded_files: MemberEvals::new(),
            model_scores: Vec::new(),
            skipped_models: Vec::new(),
            rendered_context: String::new(),
            interpretation: None,
            pending_llm: Some(PendingLlm {
                ctx: ctx.into_owned(),
                admitted,
            }),
            analysis_cached: false,
            interpret_ms: 0,
            dependency_results: Vec::new(),
            bloom_mark: None,
            uplink: None,
            uplink_reason: None,
            hopper_route: HopperRoute::Normal,
            // Not stored: the amendment reads the verdict, not how the model
            // and the floor shared it.
            model: Decision {
                class: classification,
                probability,
                threshold,
                level,
            },
            floor: None,
            upload: Some(upload.into_owned()),
        }
    }
}

#[cfg(test)]
mod tests;
