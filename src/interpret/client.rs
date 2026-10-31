//! The chat-completions wire: requests, retries, model discovery, and the
//! on-disk verdict cache.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

#[derive(Serialize)]
pub(super) struct ChatRequest<'a> {
    pub(super) model: &'a str,
    pub(super) messages: [ChatMessage<'a>; 2],
    pub(super) temperature: f32,
    pub(super) max_tokens: u32,
    pub(super) stream: bool,
    /// vLLM extension to disable Qwen-style chain-of-thought: without it the
    /// model spends the token budget "thinking" and returns a null `content`.
    /// Omitted for OpenRouter, which rejects or ignores unknown vLLM fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// OpenRouter reasoning toggle. vLLM does not take this field; OpenRouter
    /// thinking models otherwise burn [`MAX_TOKENS`] on chain-of-thought and
    /// return empty `content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reasoning: Option<ReasoningParam>,
    /// vLLM queue priority (see [`LlmCaller::priority`]). Omitted at 0, and
    /// never sent to OpenRouter, which does not know the field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) priority: Option<u8>,
}

/// Either disables reasoning outright (for a pinned model we know accepts
/// that) or, when the underlying model isn't known in advance, caps it to
/// `low` effort instead: some providers reject `enabled: false` ("reasoning
/// is mandatory for this endpoint"), but all observed ones accept `effort`
/// and it still leaves the completion's token budget mostly for the answer.
#[derive(Serialize, Clone, Copy)]
pub(super) struct ReasoningParam {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) effort: Option<&'static str>,
}

#[derive(Serialize)]
pub(super) struct ChatTemplateKwargs {
    pub(super) enable_thinking: bool,
}

#[derive(Serialize)]
pub(super) struct ChatMessage<'a> {
    pub(super) role: &'a str,
    pub(super) content: &'a str,
}

/// The `GET {base}/models` listing (OpenAI shape: `{"data": [{"id": …}]}`).
#[derive(serde::Deserialize)]
pub(super) struct ModelList {
    #[serde(default)]
    pub(super) data: Vec<ModelEntry>,
}

#[derive(serde::Deserialize)]
pub(super) struct ModelEntry {
    pub(super) id: String,
}

#[derive(serde::Deserialize)]
pub(super) struct ChatResponse {
    #[serde(default)]
    pub(super) choices: Vec<Choice>,
    /// Token accounting as the endpoint reports it (OpenAI-compatible
    /// servers, vLLM included, fill this in); absent on the odd proxy.
    #[serde(default)]
    pub(super) usage: Option<Usage>,
}

#[derive(serde::Deserialize, Default)]
pub(super) struct Usage {
    #[serde(default)]
    pub(super) prompt_tokens: u64,
    #[serde(default)]
    pub(super) completion_tokens: u64,
}

#[derive(serde::Deserialize)]
pub(super) struct Choice {
    pub(super) message: RespMessage,
}

#[derive(serde::Deserialize)]
pub(super) struct RespMessage {
    /// `null` when the model returned only chain-of-thought; tolerate it.
    #[serde(default)]
    pub(super) content: Option<String>,
    /// Fallback used by "thinking" models that put the answer here.
    #[serde(default)]
    pub(super) reasoning_content: Option<String>,
}

impl RespMessage {
    /// The usable reply text: the answer `content`, else `reasoning_content`.
    pub(super) fn text(self) -> String {
        self.content
            .filter(|s| !s.trim().is_empty())
            .or(self.reasoning_content)
            .unwrap_or_default()
    }
}

#[derive(serde::Deserialize)]
pub(super) struct GradeReason {
    pub(super) grade: String,
    #[serde(default)]
    pub(super) reason: String,
}

/// A failed LLM call, classified by whether it implies the endpoint is unhealthy.
#[derive(Debug)]
pub(super) enum CallError {
    /// Unreachable, timed out, or a 5xx — a health problem; flips the breaker.
    Transport(anyhow::Error),
    /// The endpoint answered but the reply was unusable (4xx, undecodable, no
    /// grade) — not a health problem; leaves the breaker closed.
    BadReply(anyhow::Error),
}

impl CallError {
    pub(super) fn into_inner(self) -> anyhow::Error {
        match self {
            Self::Transport(e) | Self::BadReply(e) => e,
        }
    }

    /// The underlying error, for logging a failure that is being retried
    /// elsewhere rather than reported.
    pub(super) fn inner(&self) -> &anyhow::Error {
        match self {
            Self::Transport(e) | Self::BadReply(e) => e,
        }
    }

    /// Which half of the split this failure fell on — the one thing a reader
    /// of a fall-over warning cannot infer from the message: `transport` trips
    /// the endpoint's breaker, `bad-reply` leaves it closed.
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Transport(_) => "transport",
            Self::BadReply(_) => "bad-reply",
        }
    }
}

/// POST the prebuilt user message with scan's grading prompt and parse
/// `{grade, reason}`. Also returns the model that answered, which is not
/// necessarily the primary's: see [`chat_raw`].
pub(super) fn request(
    cfg: &InterpretConfig,
    user: &str,
    priority: u8,
) -> std::result::Result<(LlmGrade, String, String), CallError> {
    let (content, model) = chat_raw(cfg, &cfg.system_prompt, user, MAX_TOKENS, priority)?;
    let (grade, reason) = parse_grade_reason(&content)
        .ok_or_else(|| CallError::BadReply(anyhow!("no parseable grade in reply: {content:?}")))?;
    Ok((grade, reason, model))
}

/// Grade an interpret render (`--format interpret`) as a live second opinion
/// asks — same prompt, same endpoint chain — but with no ML admission gate or
/// blend, queued as background work, for checking a saved report against the
/// grade a live scan gave it. Returns `{grade, model, reason}`.
///
/// # Errors
/// Every endpoint refused, or none replied with a parseable grade.
pub fn grade_render(cfg: &InterpretConfig, context: &str) -> Result<serde_json::Value> {
    // A render is normally recategorized already, and recategorizing is
    // idempotent; doing it here keeps this the live path for any render.
    let user = crate::engine::recategorize_annotations(context);
    let priority = LlmCaller::Background.priority(user.len());
    let (grade, reason, model) = request(cfg, &user, priority).map_err(CallError::into_inner)?;
    Ok(serde_json::json!({"grade": grade.as_str(), "model": model, "reason": reason}))
}

/// Send a `system` + `user` prompt to the configured endpoint and return the
/// model's reply text. This is the shared transport used by scan's grader and
/// by external callers (e.g. isomer's diff interpreter) that reuse scan's
/// client, health/HTML handling, and model config with a *different* prompt.
///
/// # Errors
/// Propagates transport failures (unreachable, timeout, 5xx) and bad replies
/// (4xx, HTML page, undecodable body) as a flat [`anyhow::Error`].
pub fn chat(
    cfg: &InterpretConfig,
    system: &str,
    user: &str,
    max_tokens: u32,
) -> anyhow::Result<String> {
    chat_raw(cfg, system, user, max_tokens, 0)
        .map(|(content, _)| content)
        .map_err(CallError::into_inner)
}

/// POST a system+user prompt down the endpoint chain and return the reply text
/// together with the model that produced it.
///
/// Each endpoint in the `--llm` list is tried in order and the first answer
/// wins. A failure is *any* refusal from that host — unreachable, timeout, 5xx,
/// or a 4xx such as the 401 a host we hold no token for returns — because from
/// here they are the same event: this endpoint will not grade this sample, and
/// the next one might. The last endpoint's error is the one reported, since by
/// then nothing is left to try.
///
/// The returned model name is the one that answered, not the one configured
/// first, so the `llm` JSON section names the model that actually graded.
pub(super) fn chat_raw(
    cfg: &InterpretConfig,
    system: &str,
    user: &str,
    max_tokens: u32,
    priority: u8,
) -> std::result::Result<(String, String), CallError> {
    let client = http_client().map_err(CallError::Transport)?;
    chat_raw_with(
        BREAKERS.get_or_init(|| Breakers::new(BREAKER_COOLDOWN)),
        cfg,
        client,
        system,
        user,
        max_tokens,
        priority,
    )
}

/// The process's one HTTP client for LLM endpoints, built on first use.
/// It carries only what is the same for every request — the connect timeout
/// and the user agent; each request sets its own total timeout.
pub(super) fn http_client() -> Result<&'static reqwest::blocking::Client> {
    static CLIENT: OnceLock<std::result::Result<reqwest::blocking::Client, String>> =
        OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::blocking::Client::builder()
                .connect_timeout(LLM_CONNECT_TIMEOUT)
                .user_agent(concat!("scan/", env!("CARGO_PKG_VERSION")))
                .build()
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| anyhow!("building LLM HTTP client: {e}"))
}

pub(super) fn chat_raw_with(
    breakers: &Breakers,
    cfg: &InterpretConfig,
    client: &reqwest::blocking::Client,
    system: &str,
    user: &str,
    max_tokens: u32,
    priority: u8,
) -> std::result::Result<(String, String), CallError> {
    let attempts = cfg.attempts();
    let last = attempts.len() - 1;
    let mut err = None;
    for (i, at) in attempts.iter().enumerate() {
        let admit = breakers.admit(at.base_url);
        if admit == Admit::Skip {
            // Debug, not warn: this fires on every call for the whole cooldown,
            // and the opening already said what happened.
            tracing::debug!(endpoint = %at.base_url, "LLM endpoint skipped: breaker open");
            err = Some(CallError::Transport(anyhow!(
                "{} skipped: breaker open after repeated transport failures",
                at.base_url
            )));
            continue;
        }
        match chat_once(
            client,
            *at,
            attempt_timeout(cfg, at.base_url),
            system,
            user,
            max_tokens,
            priority,
        ) {
            Ok(content) => {
                if breakers.answered(at.base_url) {
                    tracing::info!(endpoint = %at.base_url, "LLM endpoint breaker closed");
                }
                return Ok((content, at.model.to_string()));
            }
            Err(e) => {
                match e {
                    CallError::Transport(_) => {
                        if breakers.failed(at.base_url) {
                            tracing::warn!(
                                endpoint = %at.base_url,
                                cooldown_secs = breakers.cooldown.as_secs(),
                                "LLM endpoint breaker opened after {BREAKER_TRIP_AFTER} \
                                 consecutive transport failures; skipping it until a probe \
                                 succeeds"
                            );
                        } else if admit == Admit::Probe {
                            tracing::info!(
                                endpoint = %at.base_url,
                                "LLM endpoint probe failed; breaker stays open"
                            );
                        }
                    }
                    CallError::BadReply(_) => {
                        breakers.answered(at.base_url);
                    }
                }
                if i < last {
                    // Worth a warning, not an error: the pass has not failed
                    // yet. Silence here would make a fleet quietly billing
                    // OpenRouter look identical to one served by its own box.
                    tracing::warn!(
                        endpoint = %at.base_url,
                        next = %attempts[i + 1].base_url,
                        kind = e.kind(),
                        // `{:#}` walks the source chain: plain `{}` prints only
                        // the outermost context ("posting to <url>") and drops
                        // the cause that says *why*, which is the whole reason
                        // this line exists.
                        "LLM endpoint failed, falling over: {:#}",
                        e.inner(),
                    );
                }
                err = Some(e);
            }
        }
    }
    // `attempts` is never empty, so the loop always sets this.
    Err(err.unwrap_or_else(|| CallError::Transport(anyhow!("no LLM endpoint configured"))))
}

/// How long one attempt against `base_url` may take: the configured timeout
/// for the local origin, and at least [`OPENROUTER_MIN_TIMEOUT_SECS`] for a
/// hosted route that cannot answer on a short origin budget.
pub(super) fn attempt_timeout(cfg: &InterpretConfig, base_url: &str) -> Duration {
    if is_openrouter_endpoint(base_url) {
        cfg.timeout
            .max(Duration::from_secs(OPENROUTER_MIN_TIMEOUT_SECS))
    } else {
        cfg.timeout
    }
}

/// POST a system+user prompt to one endpoint and return the model's reply text,
/// classifying failures for the caller's health accounting.
pub(super) fn chat_once(
    client: &reqwest::blocking::Client,
    cfg: Attempt<'_>,
    timeout: Duration,
    system: &str,
    user: &str,
    max_tokens: u32,
    priority: u8,
) -> std::result::Result<String, CallError> {
    let openrouter = is_openrouter_endpoint(cfg.base_url);
    // A pinned model is one we know accepts disabling reasoning outright, so
    // it gets `enabled: false` as before. `openrouter/auto` (the default when
    // none is pinned) routes each request to a model we can't predict, and
    // some providers reject `enabled: false` outright ("Reasoning is
    // mandatory for this endpoint and cannot be disabled") rather than
    // ignoring it — but capping it to `effort: "low"` instead is accepted
    // everywhere observed, and (unlike sending no `reasoning` field at all)
    // keeps the model from spending the whole completion budget thinking and
    // returning truncated or empty `content`.
    let auto_routed = cfg.model == OPENROUTER_DEFAULT_MODEL;
    let reasoning = openrouter.then_some(if auto_routed {
        ReasoningParam {
            enabled: None,
            effort: Some("low"),
        }
    } else {
        ReasoningParam {
            enabled: Some(false),
            effort: None,
        }
    });
    // `effort: "low"` above is only a hint some providers ignore, so also
    // widen the budget itself — never shrink a caller's larger request — so
    // an auto-routed model that reasons anyway still has room left to answer.
    let max_tokens = if auto_routed {
        max_tokens.max(OPENROUTER_AUTO_MIN_TOKENS)
    } else {
        max_tokens
    };
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let prompt_bytes = system.len() + user.len();
    tracing::debug!(
        model = %cfg.model,
        url = %url,
        prompt_bytes,
        "LLM request\n--- system ---\n{system}\n--- user ---\n{user}",
    );
    // One line per call at info, whatever the outcome: what was sent (size),
    // what came back (tokens), and how long it took. A p90 that is "the LLM"
    // is only actionable when the log says which calls were slow and why —
    // a 50k-token prompt and a queued request look the same from outside.
    let started = std::time::Instant::now();
    let call_log = |outcome: &str, usage: Option<&Usage>| {
        let latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        match usage {
            Some(u) => tracing::info!(
                model = %cfg.model,
                endpoint = %cfg.base_url,
                prompt_bytes,
                prompt_tokens = u.prompt_tokens,
                completion_tokens = u.completion_tokens,
                latency_ms,
                outcome,
                "LLM call"
            ),
            None => tracing::info!(
                model = %cfg.model,
                endpoint = %cfg.base_url,
                prompt_bytes,
                latency_ms,
                outcome,
                "LLM call"
            ),
        }
    };
    let post =
        |priority: Option<u8>| -> std::result::Result<reqwest::blocking::Response, CallError> {
            let body = ChatRequest {
                model: cfg.model,
                messages: [
                    ChatMessage {
                        role: "system",
                        content: system,
                    },
                    ChatMessage {
                        role: "user",
                        content: user,
                    },
                ],
                temperature: 0.0,
                max_tokens,
                stream: false,
                chat_template_kwargs: (!openrouter).then_some(ChatTemplateKwargs {
                    enable_thinking: false,
                }),
                reasoning,
                priority,
            };
            // Per request, not per client: the client is shared across the chain
            // and its own timeout is the origin's, which a hosted fallback outlives.
            let mut req = client.post(&url).json(&body).timeout(timeout);
            if let Some(key) = cfg.api_key.filter(|k| !k.is_empty()) {
                req = req.bearer_auth(key);
            }
            req.send()
                .with_context(|| format!("posting to {url}"))
                .map_err(CallError::Transport)
        };
    // Every attempt ends in exactly one call line: content, or an error with
    // the outcome it is logged under.
    let attempt = || -> std::result::Result<(String, Option<Usage>), (CallError, String)> {
        let transport = |e: CallError| {
            let timed_out = matches!(&e, CallError::Transport(t)
                if t.root_cause().downcast_ref::<reqwest::Error>().is_some_and(reqwest::Error::is_timeout));
            let outcome = if timed_out {
                "timeout"
            } else {
                "transport-error"
            };
            (e, outcome.to_owned())
        };
        let bad_reply = |e: anyhow::Error| (CallError::BadReply(e), "bad-reply".to_owned());
        // The queue priority goes only to our own vLLM (OpenRouter does not know
        // the field), and only when it says something: 0 is vLLM's default.
        let mut priority = (!openrouter && priority > 0).then_some(priority);
        let mut resp = post(priority).map_err(transport)?;
        if priority.is_some() && resp.status() == reqwest::StatusCode::BAD_REQUEST {
            // A vLLM not started with `--scheduling-policy priority` refuses any
            // non-zero priority outright. That is a deployment mismatch, not a
            // reason to give up on the sample — and certainly not one to fall
            // over to a paid endpoint for — so the request goes again without
            // it. Any other 400 is the usual bad request.
            let raw = resp.text().unwrap_or_default();
            if !raw.contains("priority") {
                return Err((
                    CallError::BadReply(anyhow!(
                        "LLM endpoint returned 400 Bad Request: {:?}",
                        body_snippet(raw.trim())
                    )),
                    "http-400".to_owned(),
                ));
            }
            tracing::warn!(
                endpoint = %cfg.base_url,
                "LLM endpoint refused the request priority (not running \
                 --scheduling-policy priority); resending without it"
            );
            priority = None;
            resp = post(priority).map_err(transport)?;
        }
        // 5xx, 408 and 429 are the endpoint's problem (unhealthy); any other
        // 4xx is ours (bad request or auth). On a 4xx the body is the whole diagnosis — an over-budget
        // prompt, an unknown model and a rejected key all arrive as a bare
        // 400/404/401, and the endpoint explained which in the body.
        let status = resp.status();
        if !status.is_success() {
            let raw = resp.text().unwrap_or_default();
            let e = anyhow!(
                "LLM endpoint returned {status}: {:?}",
                body_snippet(raw.trim())
            );
            // 408 and 429 are the endpoint shedding load, not our request
            // being wrong: count them against its health like a 5xx.
            let e = if status.is_server_error()
                || status == reqwest::StatusCode::REQUEST_TIMEOUT
                || status == reqwest::StatusCode::TOO_MANY_REQUESTS
            {
                CallError::Transport(e)
            } else {
                CallError::BadReply(e)
            };
            return Err((e, format!("http-{}", status.as_u16())));
        }
        let html_content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("text/html"));
        // Buffer the body as text before parsing so an undecodable reply can
        // show what the endpoint actually sent (empty body, an SSE stream, a
        // proxy's HTML error page, a "model loading" notice) instead of a bare
        // serde position. A read failure is a dropped connection.
        let raw = resp
            .text()
            .with_context(|| format!("reading response body from {url}"))
            .map_err(|e| transport(CallError::Transport(e)))?;
        // A 200 carrying an HTML page means the base URL points at a web UI (a
        // dashboard, a reverse-proxy landing page), not an OpenAI-compatible
        // API. Say so plainly rather than as a serde error over markup.
        if html_content_type || raw.trim_start().starts_with('<') {
            return Err(bad_reply(anyhow!(
                "endpoint at {url} returned an HTML page, not JSON — the base URL \
                 likely points at a web UI, not an OpenAI-compatible API; body \
                 starts: {:?}",
                body_snippet(&raw),
            )));
        }
        let parsed: ChatResponse = serde_json::from_str(&raw).map_err(|e| {
            bad_reply(anyhow!(
                "decoding LLM response ({} bytes): {e}; body starts: {:?}",
                raw.len(),
                body_snippet(&raw),
            ))
        })?;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.text())
            .unwrap_or_default();
        Ok((content, parsed.usage))
    };
    match attempt() {
        Ok((content, usage)) => {
            call_log("ok", usage.as_ref());
            tracing::debug!(model = %cfg.model, "LLM response\n{content}");
            Ok(content)
        }
        Err((e, outcome)) => {
            call_log(&outcome, None);
            Err(e)
        }
    }
}

/// Lightweight health probe: `GET {base}/models` (the standard OpenAI listing,
/// which vLLM answers when the model is loaded). Healthy iff *some* endpoint in
/// the failover list returns 2xx — one reachable grader is all the pass needs,
/// so a dead primary behind a live fallback must not gate the whole scan.
pub(super) fn probe_endpoint(cfg: &InterpretConfig) -> bool {
    let Ok(client) = http_client() else {
        return false;
    };
    let timeout = cfg.timeout.min(HEALTH_PROBE_TIMEOUT);
    cfg.attempts().iter().any(|at| {
        let url = format!("{}/models", at.base_url.trim_end_matches('/'));
        let mut req = client.get(&url).timeout(timeout);
        if let Some(key) = at.api_key.filter(|k| !k.is_empty()) {
            req = req.bearer_auth(key);
        }
        req.send().is_ok_and(|r| r.status().is_success())
    })
}

/// Discover the model to grade with by listing `{base}/models` (the standard
/// OpenAI endpoint): pick the served model whose id implies the most parameters
/// (`…-32B` over `…-8B`; MoE `8x7B` counted as 56B), keeping the first-listed on
/// a tie and falling back to the first entry when no id encodes a size. The
/// choice is logged at INFO. Skipped entirely when the user pins `--llm-model`.
///
/// # Errors
///
/// When the endpoint cannot be reached, answers with a non-2xx status, does not
/// speak the OpenAI model-list shape, or serves nothing. There is no hardcoded
/// fallback, so the error says which of those it was — the three have different
/// fixes (start the server, add a key or the missing `/v1`, pin `--llm-model`)
/// and a caller cannot distinguish them from a bare `None`.
pub fn discover_model(base_url: &str, api_key: Option<&str>) -> Result<String> {
    let client = http_client()?;
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let key = api_key.filter(|k| !k.is_empty());
    let mut req = client.get(&url).timeout(HEALTH_PROBE_TIMEOUT);
    if let Some(key) = key {
        req = req.bearer_auth(key);
    }
    let resp = req
        .send()
        .map_err(|e| anyhow!("GET {url} failed ({e}); is the endpoint running?"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(anyhow!(
            "GET {url} returned {status}{}",
            status_hint(status, base_url, key.is_some())
        ));
    }
    let list: ModelList = resp
        .json()
        .map_err(|e| anyhow!("GET {url} did not answer with an OpenAI model list ({e})"))?;
    // Only replace on a strictly larger inferred size, so equal-size models keep
    // the endpoint's listing order (first wins).
    let mut best: Option<(f64, String)> = None;
    for entry in list.data {
        let size = param_billions(&entry.id).unwrap_or(-1.0);
        if best.as_ref().is_none_or(|(b, _)| size > *b) {
            best = Some((size, entry.id));
        }
    }
    let chosen = best
        .map(|(_, id)| id)
        .ok_or_else(|| anyhow!("GET {url} listed no models"))?;
    tracing::info!(model = %chosen, endpoint = %base_url, "selected LLM model");
    Ok(chosen)
}

/// The actionable half of a failed `{base}/models`: what an operator would have
/// to change to get a 200. Only the two statuses that have one fix each — a
/// missing `/v1` on the base URL, and a missing or rejected key — earn a hint;
/// anything else stands on the status code alone rather than guessing.
pub(super) fn status_hint(status: reqwest::StatusCode, base_url: &str, keyed: bool) -> String {
    let base = base_url.trim_end_matches('/');
    match status {
        reqwest::StatusCode::NOT_FOUND if !base.ends_with("/v1") => {
            format!(" — an OpenAI-compatible base URL usually ends in /v1; try --llm {base}/v1")
        }
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
            if keyed {
                " — the endpoint rejected the key (SCAN_LLM_KEY or ~/.tok/llm)".to_string()
            } else {
                " — the endpoint wants a key: SCAN_LLM_KEY or ~/.tok/llm".to_string()
            }
        }
        _ => String::new(),
    }
}

/// Infer a model's parameter count in billions from its id, for choosing the
/// largest of several served models. Reads size tokens like `27B`, `8b`, or MoE
/// `8x7B` (→ 56); a bare `1.5B` counts as 1.5. Version numbers not followed by a
/// `b` suffix (the `3.8` in `Qwen3.8-27B`) are ignored. `None` when the id
/// encodes no size (`gpt-4`, `phi-3-mini`).
pub(super) fn param_billions(id: &str) -> Option<f64> {
    let bytes = id.as_bytes();
    let mut best: Option<f64> = None;
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let (mut value, mut j) = parse_decimal(bytes, i);
        // MoE `NxM` — the expert count times the per-expert size.
        if matches!(bytes.get(j), Some(b'x' | b'X'))
            && bytes.get(j + 1).is_some_and(u8::is_ascii_digit)
        {
            let (second, k) = parse_decimal(bytes, j + 1);
            value *= second;
            j = k;
        }
        // Read it as a size only with a `b`/`B` suffix on a word boundary.
        if matches!(bytes.get(j), Some(b'b' | b'B'))
            && bytes.get(j + 1).is_none_or(|c| !c.is_ascii_alphanumeric())
        {
            best = Some(best.map_or(value, |b| b.max(value)));
            i = j + 1;
        } else {
            i = j;
        }
    }
    best
}

/// Parse the decimal starting at `start` (digits, with an optional single `.`
/// fraction). Returns the value and the index just past it. The caller
/// guarantees `bytes[start]` is a digit, so `end > start`.
pub(super) fn parse_decimal(bytes: &[u8], start: usize) -> (f64, usize) {
    let mut end = start;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if bytes.get(end) == Some(&b'.') && bytes.get(end + 1).is_some_and(u8::is_ascii_digit) {
        end += 1;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
    }
    let value = std::str::from_utf8(&bytes[start..end])
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    (value, end)
}

// ── verdict cache ───────────────────────────────────────────────────────────
// Cache the LLM verdict keyed by the prompt's content hash, so a verdict is
// reused iff the model would see byte-identical input (model + system prompt +
// analysis). Editing the prompt, the model, or the analysis invalidates it; a
// plain rebuild does not. Only successful verdicts are cached — failures retry.

#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct CachedVerdict {
    pub(super) grade: String,
    pub(super) reason: String,
}

/// SHA-256 of the exact prompt (model + system + user), as a hex string. The
/// system prompt is part of the key, so editing [`SYSTEM_PROMPT`] keys fresh
/// verdicts rather than replaying ones graded under a different framing.
pub(super) fn prompt_hash(system: &str, model: &str, user: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(model.as_bytes());
    h.update(b"\0");
    h.update(system.as_bytes());
    h.update(b"\0");
    h.update(user.as_bytes());
    format!("{:x}", h.finalize())
}

/// Root of the interpret verdict cache (`…/atomdrift/scan/interpret`), or `None`
/// when no OS cache dir is available. Used by [`crate::cache_cleanup`] to reclaim
/// the store; entries live at `interpret/<hash>.json` directly below this root.
pub(crate) fn cache_base() -> Option<PathBuf> {
    Some(crate::cache_dir()?.join("interpret"))
}

/// Cache file path for a prompt hash, or `None` when no cache dir is available.
pub(super) fn cache_path(hash: &str) -> Option<PathBuf> {
    Some(cache_base()?.join(format!("{hash}.json")))
}

/// Atomic cache write (temp file + rename).
pub(super) fn cache_put(path: &Path, verdict: &CachedVerdict) -> Result<()> {
    use std::io::Write;
    let parent = path.parent().context("cache path has no directory")?;
    std::fs::create_dir_all(parent).context("creating the cache directory")?;
    let json = serde_json::to_vec(verdict).context("serializing the verdict")?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent).context("creating a temporary file")?;
    tmp.write_all(&json).context("writing the verdict")?;
    tmp.persist(path).context("moving the verdict into place")?;
    Ok(())
}

/// Longest body prefix included verbatim when a reply fails to decode — enough
/// to recognize an HTML page or a stray streaming chunk without flooding the log.
pub(super) const BODY_SNIPPET: usize = 512;

/// A char-boundary-safe leading slice of an undecodable body, for diagnostics.
pub(super) fn body_snippet(s: &str) -> &str {
    // A char boundary ≤ len, so the slice never misses.
    s.get(..s.floor_char_boundary(BODY_SNIPPET)).unwrap_or(s)
}

/// Longest `reason` kept from a reply. The prompt asks for five words; anything
/// beyond this is the model having been talked into monologuing.
pub(super) const MAX_REASON_CHARS: usize = 120;

/// Extract `{"grade":...,"reason":...}` from a reply that may be wrapped in
/// prose or code fences. Returns `None` if no valid trinary grade is found.
///
/// The reason is model-generated text derived from an attacker-controlled render
/// and lands in an operator's terminal, so it is stripped of ANSI escapes and
/// clamped before it goes anywhere — the render is sanitized on the way in, and
/// this closes the same hole on the way out.
pub(super) fn parse_grade_reason(content: &str) -> Option<(LlmGrade, String)> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    let slice = content.get(start..=end)?;
    let gr: GradeReason = serde_json::from_str(slice).ok()?;
    let grade = LlmGrade::parse(&gr.grade)?;
    let mut reason = strip_ansi(gr.reason.trim());
    reason.retain(|c| c == '\t' || !c.is_control());
    if let Some((cut, _)) = reason.char_indices().nth(MAX_REASON_CHARS) {
        reason.truncate(cut);
    }
    Some((grade, reason.trim_end().to_string()))
}
