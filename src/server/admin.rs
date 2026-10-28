//! The admin routes: `/_/reload` swaps in the model bundle on disk, and
//! `/_/update` pulls the latest models and traits first.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::AppState;
use super::error::ApiError;
use crate::model::Model;

/// How long `/_/reload` waits before answering; the reload itself runs on.
const RELOAD_TIMEOUT: Duration = Duration::from_secs(120);

/// How long `/_/update` waits: a backstop around the sequential models and
/// traits pulls, then the reload.
const UPDATE_TIMEOUT: Duration = Duration::from_secs(21 * 60 + 120);

/// What a reload installed.
struct Reloaded {
    elapsed_ms: u128,
    /// Why the cleave traits did not reload; the previous traits stay.
    traits_reload_error: Option<String>,
}

impl AppState {
    /// Reload the cleave traits and the model bundle from disk and serve them.
    /// Blocking. On failure the previous model stays in service.
    fn reload_bundle(&self) -> anyhow::Result<Reloaded> {
        let start = Instant::now();
        // Traits first, so the new model runs against fresh rules.
        let traits_reload_error = match cleave::reload_capability_mapper() {
            Err(e) => {
                tracing::warn!("cleave trait reload failed (previous traits retained): {e:#}");
                Some(e)
            }
            Ok(_) => {
                tracing::info!("cleave traits reloaded");
                None
            }
        };
        cleave::clear_all_thread_caches();

        let config = &self.config;
        let model = Model::load(&config.model_dir, config.thresholds, config.level)?;
        let shap = super::load_shap(&config.model_dir)?;
        let elapsed_ms = start.elapsed().as_millis();
        let spec_version = model.spec().version();
        let features = model.spec().total_features();
        let shap_loaded = shap.is_some();
        if self.install(self.bundle(model, shap)) {
            tracing::info!(
                elapsed_ms,
                spec_version,
                features,
                shap_loaded,
                "model reloaded"
            );
        } else {
            tracing::info!(
                elapsed_ms,
                spec_version,
                features,
                shap_loaded,
                "model loaded via reload — server now ready"
            );
        }
        Ok(Reloaded {
            elapsed_ms,
            traits_reload_error,
        })
    }
}

fn reload_in_progress() -> Response {
    ApiError::new(
        StatusCode::CONFLICT,
        "reload_in_progress",
        "Reload already in progress",
    )
    .into_response()
}

/// POST /_/reload — reload the model bundle from disk and swap it in.
///
/// One reload at a time; concurrent calls receive 409. The lock travels with
/// the blocking work, so a reload that outlives the response still holds it
/// until it finishes — and still installs what it loaded.
pub(super) async fn reload(State(state): State<Arc<AppState>>) -> Response {
    tracing::info!("POST /_/reload");
    // Each load allocates significant memory; never two at once.
    let Ok(held) = Arc::clone(&state.reload_lock).try_lock_owned() else {
        tracing::warn!("reload rejected: already in progress");
        return reload_in_progress();
    };
    let reloading = Arc::clone(&state);
    let job = tokio::task::spawn_blocking(move || {
        let _held = held;
        reloading.reload_bundle()
    });
    match tokio::time::timeout(RELOAD_TIMEOUT, job).await {
        Ok(Ok(Ok(reloaded))) => {
            let mut body = serde_json::json!({
                "status": "ok",
                "elapsed_ms": reloaded.elapsed_ms,
            });
            if let Some(err) = &reloaded.traits_reload_error {
                body["traits_reload_error"] = serde_json::json!(err);
            }
            Json(body).into_response()
        }
        Ok(Ok(Err(e))) => {
            // Logged here; the caller is not told filesystem paths or model internals.
            tracing::warn!("reload failed (previous model retained): {e:#}");
            ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "reload_failed",
                "Failed to load model",
            )
            .into_response()
        }
        Ok(Err(e)) => {
            tracing::warn!("reload task join error: {e}");
            ApiError::internal().into_response()
        }
        Err(_elapsed) => {
            tracing::warn!(
                "reload still running after {}s; it keeps the lock and installs when done",
                RELOAD_TIMEOUT.as_secs()
            );
            ApiError::new(
                StatusCode::GATEWAY_TIMEOUT,
                "reload_timeout",
                "Reload timed out",
            )
            .into_response()
        }
    }
}

/// Pull the latest models and traits. Both pulls are non-fatal: each failure
/// is reported, and the reload that follows runs against whatever is on disk.
/// `model_update` validates a bundle (`Model::load`) before swapping it in, so
/// a broken bundle never lands and there is nothing to roll back. Blocking.
fn pull_updates() -> (Option<String>, Option<String>) {
    let models_err = crate::models_repo::install_target()
        .and_then(|dir| crate::model_update::update(&dir, false, false))
        .err()
        .map(|e| {
            tracing::warn!("models update failed: {e:#}");
            e.to_string()
        });
    let traits_err = crate::traits_repo::update(false, false).err().map(|e| {
        tracing::warn!("traits update failed: {e:#}");
        e.to_string()
    });
    (models_err, traits_err)
}

/// POST /_/update — pull the latest models and traits, then reload.
///
/// Shares the reload lock with `/_/reload`, so concurrent calls receive 409,
/// and holds it across the pulls and the reload for as long as they run.
pub(super) async fn update(State(state): State<Arc<AppState>>) -> Response {
    tracing::info!("POST /_/update");
    let Ok(held) = Arc::clone(&state.reload_lock).try_lock_owned() else {
        tracing::warn!("update rejected: reload already in progress");
        return reload_in_progress();
    };
    let updating = Arc::clone(&state);
    let job = tokio::task::spawn_blocking(move || {
        let _held = held;
        let pulled = pull_updates();
        (pulled, updating.reload_bundle())
    });
    let ((models_err, traits_err), reloaded) = match tokio::time::timeout(UPDATE_TIMEOUT, job).await
    {
        Ok(Ok(done)) => done,
        Ok(Err(e)) => {
            tracing::warn!("update task join error: {e}");
            return ApiError::internal().into_response();
        }
        Err(_elapsed) => {
            tracing::warn!(
                "update still running after {}s; it keeps the lock and installs when done",
                UPDATE_TIMEOUT.as_secs()
            );
            return ApiError::new(
                StatusCode::GATEWAY_TIMEOUT,
                "update_timeout",
                "Update timed out",
            )
            .into_response();
        }
    };
    match reloaded {
        Ok(reloaded) => {
            let mut body = serde_json::json!({
                "status": "ok",
                "elapsed_ms": reloaded.elapsed_ms,
                "models_updated": models_err.is_none(),
                "traits_updated": traits_err.is_none(),
                "models_error": models_err,
                "traits_error": traits_err,
                "version": env!("CARGO_PKG_VERSION"),
                "model_commit": crate::models_repo::version(),
                "traits_commit": cleave::traits_repo::version(),
            });
            if let Some(err) = &reloaded.traits_reload_error {
                body["traits_reload_error"] = serde_json::json!(err);
            }
            Json(body).into_response()
        }
        Err(e) => {
            // The in-memory model was not swapped, so requests continue against
            // the previous model; the on-disk bundle was validated before
            // install, so there is nothing to roll back.
            tracing::error!("model reload failed after update: {e:#}");
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({
                    "status": "reload_failed",
                    "error": "Failed to load model",
                    "models_updated": models_err.is_none(),
                    "traits_updated": traits_err.is_none(),
                    "models_error": models_err,
                    "traits_error": traits_err,
                })),
            )
                .into_response()
        }
    }
}
