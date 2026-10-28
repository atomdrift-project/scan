//! Writers for the finding, breadth, volume and n-gram features.

#![expect(
    clippy::cast_possible_truncation,
    reason = "feature values are narrowed to the model's f32 input dtype"
)]

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct FileRiskStats {
    pub(super) suspicious_ratio: f32,
    pub(super) hostile_ratio: f32,
    pub(super) suspicious_findings: u32,
    pub(super) hostile_findings: u32,
    pub(super) suspicious_density: f32,
    pub(super) hostile_density: f32,
    pub(super) suspicious_category_breadth: usize,
    pub(super) hostile_category_breadth: usize,
    pub(super) max_crit: u32,
}

/// G3: report-level aggregates.
/// Optional finding features consume the retained raw fields without another parse.
pub(super) fn write_optional_finding_features(parsed: &ParsedReport, w: &mut FeatureWriter<'_>) {
    for code in &parsed.mbc_codes {
        w.set_token(&format!("mbc:{code}"), 1.0);
    }
    let needs_moments = [
        "agg:confidence_mean",
        "agg:confidence_std",
        "agg:confidence_skew",
        "agg:confidence_kurtosis",
    ]
    .iter()
    .any(|name| w.lookup.contains_key(*name));
    if needs_moments {
        let values: Vec<f64> = parsed
            .summaries
            .iter()
            .flat_map(|s| &s.raw_findings)
            .filter(|f| !f.id.is_empty() && f.conf >= MIN_CONFIDENCE)
            .map(|f| f.conf)
            .collect();
        if !values.is_empty() {
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let variance = values.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt();
            let (skew, kurtosis) = if std > 1e-9 {
                (
                    values.iter().map(|c| (c - mean).powi(3)).sum::<f64>() / n / std.powi(3),
                    values.iter().map(|c| (c - mean).powi(4)).sum::<f64>() / n / std.powi(4) - 3.0,
                )
            } else {
                (0.0, 0.0)
            };
            w.set("agg:confidence_mean", mean as f32);
            w.set("agg:confidence_std", std as f32);
            w.set("agg:confidence_skew", skew as f32);
            w.set("agg:confidence_kurtosis", kurtosis as f32);
        }
    }
    if w.lookup.contains_key("agg:trait_id_lexical_distance") {
        let mut ids: Vec<&str> = parsed
            .summaries
            .iter()
            .flat_map(|s| &s.raw_findings)
            .map(|f| f.id.as_str())
            .filter(|s| !s.is_empty())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let total: usize = ids
            .windows(2)
            .map(|pair| lexical_distance(pair[0], pair[1]))
            .sum();
        let average = if ids.len() > 1 {
            total as f64 / (ids.len() - 1) as f64
        } else {
            0.0
        };
        w.set("agg:trait_id_lexical_distance", average as f32);
    }
    if [
        "agg:docobf_obfuscation_count",
        "agg:docobf_eval_count",
        "agg:docobf_lure_count",
        "agg:docobf_total_count",
        "agg:docobf_has_any",
    ]
    .iter()
    .any(|name| w.lookup.contains_key(*name))
    {
        let mut counts = [0u32; 3];
        let prefixes = [
            "objectives/anti-static/obfuscation/document",
            "objectives/execution/interpreter/eval",
            "objectives/execution/lure/document",
        ];
        for finding in parsed.summaries.iter().flat_map(|s| &s.raw_findings) {
            if let Some(index) = prefixes.iter().position(|p| finding.id.starts_with(p)) {
                counts[index] += 1;
            }
        }
        let total = counts.iter().sum::<u32>();
        w.set("agg:docobf_obfuscation_count", counts[0] as f32);
        w.set("agg:docobf_eval_count", counts[1] as f32);
        w.set("agg:docobf_lure_count", counts[2] as f32);
        w.set("agg:docobf_total_count", total as f32);
        w.set("agg:docobf_has_any", f32::from(total > 0));
    }
}

pub(super) fn lexical_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let previous = row[j + 1];
            row[j + 1] = (row[j] + 1)
                .min(previous + 1)
                .min(diagonal + usize::from(ca != cb));
            diagonal = previous;
        }
    }
    row[b.len()]
}

pub(super) fn write_aggregate_features(parsed: &ParsedReport, w: &mut FeatureWriter<'_>) {
    let summary = &parsed.combined;
    let summaries = parsed.summaries.as_slice();
    let breadth = PathBreadth::new(summary);
    write_breadth_features(summary, &breadth, w);
    write_finding_volume_features(summary, summaries, w);
    write_top_file_features(summaries, w);
    write_file_tier_features(summaries, w);
    write_code_features(summary, &parsed.attack_codes, &parsed.mbc_codes, w);
}

/// How widely the sample's findings spread over the trait taxonomy, counted
/// over distinct sample paths at each criticality tier.
pub(super) struct PathBreadth {
    pub(super) max_crit: u32,
    /// Top-level categories reached at baseline or above.
    pub(super) categories: usize,
    /// Paths at depth >= 2 reached at baseline or above.
    pub(super) any: u32,
    /// Paths at depth >= 2 reached at notable or above, and the split of
    /// those by their highest tier.
    pub(super) notable: u32,
    pub(super) suspicious: u32,
    pub(super) hostile: u32,
    pub(super) notable_only: u32,
}

impl PathBreadth {
    pub(super) fn new(summary: &FindingSummary) -> Self {
        let mut breadth = Self {
            max_crit: 0,
            categories: 0,
            any: 0,
            notable: 0,
            suspicious: 0,
            hostile: 0,
            notable_only: 0,
        };
        let mut categories: HashSet<&str> = HashSet::new();
        for (path, &max_ord) in &summary.sample_paths {
            let path_depth = path.chars().filter(|&c| c == '/').count();
            if max_ord >= CRIT_BASELINE {
                categories.insert(path.split('/').next().unwrap_or(""));
                if path_depth >= 2 {
                    breadth.any += 1;
                }
            }
            if path_depth < 2 || max_ord < CRIT_NOTABLE {
                continue;
            }
            breadth.notable += 1;
            breadth.max_crit = breadth.max_crit.max(max_ord);
            if max_ord >= CRIT_SUSPICIOUS {
                breadth.suspicious += 1;
            }
            if max_ord >= CRIT_HOSTILE {
                breadth.hostile += 1;
            } else if max_ord == CRIT_NOTABLE {
                breadth.notable_only += 1;
            }
        }
        breadth.categories = categories.len();
        breadth
    }
}

/// Breadth over the taxonomy: categories, active paths, and how far their
/// criticality escalates.
pub(super) fn write_breadth_features(
    summary: &FindingSummary,
    breadth: &PathBreadth,
    w: &mut FeatureWriter<'_>,
) {
    w.set("agg:max_crit", breadth.max_crit as f32);
    w.set("agg:category_breadth", breadth.categories as f32);
    w.set("agg:path_breadth_any", (breadth.any as f32).ln_1p());
    w.set("agg:total_active_paths", (breadth.notable as f32).ln_1p());
    w.set(
        "agg:suspicious_concentration",
        breadth.suspicious as f32 / breadth.any.max(1) as f32,
    );
    w.set(
        "agg:hostile_concentration",
        breadth.hostile as f32 / breadth.any.max(1) as f32,
    );
    w.set(
        "agg:escalation_rate",
        breadth.suspicious as f32 / breadth.notable.max(1) as f32,
    );
    w.set(
        "agg:notable_only_fraction",
        breadth.notable_only as f32 / breadth.notable.max(1) as f32,
    );
    w.set(
        "agg:hostile_escalation_rate",
        breadth.hostile as f32 / breadth.notable.max(1) as f32,
    );
    w.set(
        "agg:hostile_share_of_suspicious",
        breadth.hostile as f32 / breadth.suspicious.max(1) as f32,
    );
    let category_denom = breadth.categories.max(1) as f32;
    w.set(
        "agg:suspicious_category_density",
        summary.suspicious_category_breadth as f32 / category_denom,
    );
    w.set(
        "agg:hostile_category_density",
        summary.hostile_category_breadth as f32 / category_denom,
    );

    // 2-level breadth features.
    let mut suspicious_2level: HashSet<String> = HashSet::new();
    let mut hostile_2level: HashSet<String> = HashSet::new();
    let mut objectives_2level: HashSet<String> = HashSet::new();
    for (path, &max_ord) in &summary.sample_paths {
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() >= 2 {
            let two_level = format!("{}/{}", parts[0], parts[1]);
            if max_ord >= CRIT_SUSPICIOUS {
                suspicious_2level.insert(two_level.clone());
            }
            if max_ord >= CRIT_HOSTILE {
                hostile_2level.insert(two_level.clone());
            }
            if parts[0] == "objectives" && max_ord >= CRIT_BASELINE {
                objectives_2level.insert(two_level);
            }
        }
    }
    w.set(
        "agg:suspicious_2level_breadth",
        suspicious_2level.len() as f32,
    );
    w.set("agg:hostile_2level_breadth", hostile_2level.len() as f32);
    w.set("agg:objectives_breadth", objectives_2level.len() as f32);
}

/// Finding counts at each tier: as logs, per KB, as shares of each other,
/// and how often the same id or category repeats.
pub(super) fn write_finding_volume_features(
    summary: &FindingSummary,
    summaries: &[FileSummary],
    w: &mut FeatureWriter<'_>,
) {
    let total_size_bytes: f64 = summaries.iter().map(|s| s.size_bytes).sum();
    let total_kb_raw = (total_size_bytes / 1024.0) as f32;
    let total_kb_p1 = total_kb_raw.max(0.1);
    let total_kb_1 = total_kb_raw.max(1.0);

    w.set(
        "agg:notable_findings_log",
        (summary.notable_finding_count as f32).ln_1p(),
    );
    w.set(
        "agg:suspicious_findings_log",
        (summary.suspicious_finding_count as f32).ln_1p(),
    );
    w.set(
        "agg:hostile_findings_log",
        (summary.hostile_finding_count as f32).ln_1p(),
    );
    w.set(
        "agg:notable_finding_ratio",
        summary.notable_finding_count as f32 / total_kb_p1,
    );
    w.set(
        "agg:suspicious_finding_ratio",
        summary.suspicious_finding_count as f32 / total_kb_p1,
    );
    w.set(
        "agg:hostile_finding_ratio",
        summary.hostile_finding_count as f32 / total_kb_p1,
    );
    let log_kb_p1 = total_kb_p1.ln_1p();
    w.set(
        "agg:unique_suspicious_ids_log",
        (summary.unique_suspicious_ids as f32).ln_1p() / log_kb_p1,
    );
    w.set(
        "agg:unique_hostile_ids_log",
        (summary.unique_hostile_ids as f32).ln_1p() / log_kb_p1,
    );

    // Size-invariant crit-tier severity fractions — mirror of collimator's
    // include_severity_fractions block (count of crit>=N findings as a share of
    // ALL findings, not per-KB).
    let total_findings = summary.filtered_finding_count.max(1) as f32;
    let mundane = summary
        .filtered_finding_count
        .saturating_sub(summary.notable_finding_count)
        .max(1) as f32;
    w.set(
        "agg:crit3_finding_fraction",
        summary.notable_finding_count as f32 / total_findings,
    );
    w.set(
        "agg:crit4_finding_fraction",
        summary.suspicious_finding_count as f32 / total_findings,
    );
    w.set(
        "agg:hostile_finding_fraction",
        summary.hostile_finding_count as f32 / total_findings,
    );
    w.set(
        "agg:severe_to_mundane_ratio",
        summary.notable_finding_count as f32 / mundane,
    );
    w.set(
        "agg:crit4_present",
        f32::from(summary.suspicious_finding_count > 0),
    );

    w.set(
        "agg:suspicious_category_breadth",
        summary.suspicious_category_breadth as f32,
    );
    w.set(
        "agg:hostile_category_breadth",
        summary.hostile_category_breadth as f32,
    );
    w.set(
        "agg:suspicious_findings_per_kb",
        summary.suspicious_finding_count as f32 / total_kb_1,
    );
    w.set(
        "agg:hostile_findings_per_kb",
        summary.hostile_finding_count as f32 / total_kb_1,
    );
    w.set(
        "agg:suspicious_categories_per_kb",
        summary.suspicious_category_breadth as f32 / total_kb_1,
    );
    w.set(
        "agg:hostile_categories_per_kb",
        summary.hostile_category_breadth as f32 / total_kb_1,
    );

    w.set(
        "agg:suspicious_finding_escalation_rate",
        summary.suspicious_finding_count as f32 / summary.notable_finding_count.max(1) as f32,
    );
    w.set(
        "agg:hostile_finding_escalation_rate",
        summary.hostile_finding_count as f32 / summary.notable_finding_count.max(1) as f32,
    );
    w.set(
        "agg:hostile_share_of_suspicious_findings",
        summary.hostile_finding_count as f32 / summary.suspicious_finding_count.max(1) as f32,
    );

    let host_density_global = summary.hostile_finding_count as f32 / total_kb_1;
    let susp_density_global = summary.suspicious_finding_count as f32 / total_kb_1;
    w.set(
        "agg:hostile_weighted_density",
        host_density_global + 0.25 * susp_density_global,
    );

    w.set(
        "agg:suspicious_id_repeat_ratio",
        1.0 - (summary.unique_suspicious_ids as f32
            / summary.suspicious_finding_count.max(1) as f32),
    );
    w.set(
        "agg:hostile_id_repeat_ratio",
        1.0 - (summary.unique_hostile_ids as f32 / summary.hostile_finding_count.max(1) as f32),
    );
    w.set(
        "agg:suspicious_category_repeat_ratio",
        1.0 - (summary.suspicious_category_breadth as f32
            / summary.suspicious_finding_count.max(1) as f32),
    );
    w.set(
        "agg:hostile_category_repeat_ratio",
        1.0 - (summary.hostile_category_breadth as f32
            / summary.hostile_finding_count.max(1) as f32),
    );
}

/// The `top1` block: the single riskiest file by each ordering.
pub(super) fn write_top_file_features(summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let topk = topk_file_risk_features_from_summaries(summaries);
    w.set("agg:top1_file_suspicious_ratio_sum", topk[0]);
    w.set("agg:top1_file_hostile_ratio_sum", topk[1]);
    w.set("agg:top1_file_suspicious_findings_log", topk[2]);
    w.set("agg:top1_file_hostile_findings_log", topk[3]);
    w.set("agg:top1_file_suspicious_density_sum", topk[4]);
    w.set("agg:top1_file_hostile_density_sum", topk[5]);
    w.set("agg:top1_file_suspicious_category_breadth_sum", topk[6]);
    w.set("agg:top1_file_hostile_category_breadth_sum", topk[7]);

    // The "top-k weighted sum" is the single max; min_by over the same
    // descending comparator picks the same first-maximum file a stable
    // sort + take(1) would.
    let top_weighted = summaries
        .iter()
        .map(|s| &s.risk)
        .min_by(|a, b| {
            let weighted = |s: &FileRiskStats| s.hostile_density + 0.25 * s.suspicious_density;
            weighted(b)
                .total_cmp(&weighted(a))
                .then(b.hostile_density.total_cmp(&a.hostile_density))
                .then(b.suspicious_density.total_cmp(&a.suspicious_density))
        })
        .map_or(0.0, |s| s.hostile_density + 0.25 * s.suspicious_density);
    w.set("agg:top1_file_hostile_weighted_density_sum", top_weighted);
}

/// How many files sit in each criticality tier.
pub(super) fn write_file_tier_features(summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let n_files = summaries.len().max(1) as f32;
    let hostile_files = summaries
        .iter()
        .filter(|s| s.risk.max_crit >= CRIT_HOSTILE)
        .count() as f32;
    let suspicious_files = summaries
        .iter()
        .filter(|s| s.risk.max_crit == CRIT_SUSPICIOUS)
        .count() as f32;
    let notable_files = summaries
        .iter()
        .filter(|s| s.risk.max_crit == CRIT_NOTABLE)
        .count() as f32;
    w.set("agg:file_hostile_fraction", hostile_files / n_files);
    w.set("agg:file_suspicious_fraction", suspicious_files / n_files);
    w.set("agg:file_notable_fraction", notable_files / n_files);
    w.set("agg:file_hostile_count_log", hostile_files.ln_1p());
    w.set("agg:file_suspicious_count_log", suspicious_files.ln_1p());
    w.set("agg:file_notable_count_log", notable_files.ln_1p());
    w.set("agg:hostile_depth_weight", 0.0);
}

/// ATT&CK / MBC code counts and the objective-path co-occurrence counts.
pub(super) fn write_code_features(
    summary: &FindingSummary,
    attack_techniques: &[String],
    mbc_behaviors: &[String],
    w: &mut FeatureWriter<'_>,
) {
    w.set("agg:attack_technique_count", attack_techniques.len() as f32);
    // An ATT&CK technique ID is ASCII ("T1059.001"), but `atk` is whatever the
    // finding carried and nothing validates it, so a multi-byte codepoint can
    // sit inside the first four bytes. `len() >= 4` counts bytes and says
    // nothing about where characters begin: "Tü─…" is T(0) ü(1..3) ─(3..6), so
    // a byte-4 slice lands mid-character and panics, taking the whole analysis
    // task with it. `get` yields None on a non-boundary index instead.
    let tactic_prefixes: HashSet<&str> = attack_techniques
        .iter()
        .filter_map(|t| t.get(..4))
        .filter(|prefix| prefix.starts_with('T'))
        .collect();
    w.set("agg:attack_tactic_count", tactic_prefixes.len() as f32);
    w.set("agg:mbc_behavior_count", mbc_behaviors.len() as f32);

    // ATT&CK / MBC co-occurrence aggregates: log1p of the count of unordered
    // pairs / triples among the distinct technique and behavior codes seen.
    // Mirrors `agg:attack_bigram_count` and friends in the collimator extractor
    // (features.py around line 2197).  Combinations are computed analytically
    // (n choose k) — equivalent to enumerating but cheaper for large finding
    // counts.  saturating_sub guards the n=0 / n=1 cases where the product
    // would otherwise underflow on usize.
    let n_atk = attack_techniques.len();
    let atk_bi = (n_atk * n_atk.saturating_sub(1)) / 2;
    let atk_tri = (n_atk * n_atk.saturating_sub(1) * n_atk.saturating_sub(2)) / 6;
    w.set("agg:attack_bigram_count", (atk_bi as f32).ln_1p());
    w.set("agg:attack_trigram_count", (atk_tri as f32).ln_1p());
    let n_mbc = mbc_behaviors.len();
    let mbc_bi = (n_mbc * n_mbc.saturating_sub(1)) / 2;
    w.set("agg:mbc_bigram_count", (mbc_bi as f32).ln_1p());
    let has_objectives = summary
        .sample_paths
        .keys()
        .any(|p| p.starts_with("objectives/"));
    w.set(
        "agg:has_attack_and_objective",
        f32::from(!attack_techniques.is_empty() && has_objectives),
    );

    // Objective-path co-occurrence aggregates.  Mirrors collimator's
    // `agg:objective_bigram_count` / `agg:objective_trigram_count` (features.py
    // around line 2155).  Counts unordered pairs and triples among the distinct
    // `objectives/*` and `well-known/*` sample paths.  The trigram inner loop
    // is capped at 20 per (i, j) pair to bound work on samples with many
    // objective paths — the same cap collimator uses, so the values match.
    let n_obj = summary
        .sample_paths
        .keys()
        .filter(|p| p.starts_with("objectives/") || p.starts_with("well-known/"))
        .count();
    let n_obj_bi = (n_obj * n_obj.saturating_sub(1)) / 2;
    let mut n_obj_tri: usize = 0;
    for i in 0..n_obj {
        for j in (i + 1)..n_obj {
            let k_end = (j + 20).min(n_obj);
            n_obj_tri += k_end.saturating_sub(j + 1);
        }
    }
    w.set("agg:objective_bigram_count", (n_obj_bi as f32).ln_1p());
    w.set("agg:objective_trigram_count", (n_obj_tri as f32).ln_1p());
}

pub(super) fn tier_prefix(crit: u32) -> &'static str {
    match crit {
        CRIT_HOSTILE => "h",
        CRIT_SUSPICIOUS => "s",
        _ => "n",
    }
}

/// Categories whose notable-or-above paths feed the `crit:` n-grams.
pub(super) const CRIT_CATEGORIES: &[&str] = &[
    "objectives",
    "well-known",
    "supply-chain",
    "anti-analysis",
    "anti-static",
    "command-and-control",
    "evasion",
    "execution",
    "exfiltration",
];

/// Sorted `tier:category[/sub]` tokens: each notable-or-above sample path in
/// a [`CRIT_CATEGORIES`] category, cut to two levels, at its highest tier.
pub(super) fn crit_category_tokens(summary: &FindingSummary) -> Vec<String> {
    let mut max_crit: HashMap<String, u32> = HashMap::new();
    for (path, &mo) in &summary.sample_paths {
        if mo < CRIT_NOTABLE {
            continue;
        }
        let parts: Vec<&str> = path.split('/').collect();
        if !CRIT_CATEGORIES.contains(&parts[0]) {
            continue;
        }
        let key = if parts.len() >= 2 {
            format!("{}/{}", parts[0], parts[1])
        } else {
            parts[0].to_string()
        };
        let e = max_crit.entry(key).or_insert(0);
        *e = (*e).max(mo);
    }
    let mut tokens: Vec<String> = max_crit
        .iter()
        .map(|(k, &c)| format!("{}:{k}", tier_prefix(c)))
        .collect();
    tokens.sort();
    tokens
}

/// Sorted `tier:path` tokens from prefixes at the model's trait depth,
/// shared by the tiered bigram and trigram feature builders.
pub(super) fn tiered_path_tokens(summary: &FindingSummary) -> Vec<String> {
    tiered_path_tokens_with_options(summary, 3, &HashMap::new())
}

pub(super) fn tiered_path_tokens_with_options(
    summary: &FindingSummary,
    minimum: u32,
    branches: &HashMap<String, u32>,
) -> Vec<String> {
    let mut tokens: Vec<String> = summary
        .sample_paths
        .iter()
        .filter_map(|(path, &crit)| {
            let branch = path.split('/').next().unwrap_or("");
            let floor = branches.get(branch).copied().unwrap_or(minimum);
            (crit > 0 && crit >= floor).then(|| format!("{}:{path}", tier_prefix(crit)))
        })
        .collect();
    tokens.sort();
    tokens
}

/// Keep real finding evidence, with deterministic lexical ties. The score for
/// each path was accumulated as max(severity * confidence), never a product
/// of unrelated maxima. Match collimator's v20 token budgets exactly.
pub(super) fn ranked_tiered_path_tokens(
    summary: &FindingSummary,
    minimum: u32,
    branches: &HashMap<String, u32>,
    order: usize,
) -> Vec<String> {
    let mut tokens = tiered_path_tokens_with_options(summary, minimum, branches);
    let limit = match order {
        2 => 512,
        3 => 128,
        _ => 64,
    };
    if tokens.len() > limit {
        // Each token is `<tier>:<path>`; rank by the path's score.
        let score = |token: &str| {
            let path = token.split_once(':').map_or(token, |(_, path)| path);
            summary.path_scores.get(path).copied().unwrap_or(0.0)
        };
        tokens.sort_unstable_by(|a, b| {
            let (score_a, score_b) = (score(a), score(b));
            score_b.total_cmp(&score_a).then_with(|| a.cmp(b))
        });
        tokens.truncate(limit);
        tokens.sort();
    }
    tokens
}

/// `crit:` unigrams over the category tokens, then their n-grams.
pub(super) fn write_crit_ngrams(tokens: &[String], w: &mut FeatureWriter<'_>) {
    let mut key = String::new();
    for t in tokens {
        key.clear();
        let _ = write!(key, "crit:{t}");
        w.set_token(&key, 1.0);
    }
    write_ngrams(w, tokens, "critbi:", "crittri:", "crit");
}

/// Set the pair (`<bi>a + b`) and triple (`<tri>a + b + c`) tokens over
/// sorted, distinct `tokens`. Pairs are O(n²) and triples O(n³), so a
/// bloated report skips both past 512 tokens and triples past 128.
pub(super) fn write_ngrams(
    w: &mut FeatureWriter<'_>,
    tokens: &[String],
    bi: &str,
    tri: &str,
    what: &str,
) {
    if tokens.len() > 512 {
        tracing::warn!(
            tokens = tokens.len(),
            "too many unique {what} tokens; skipping n-gram generation"
        );
        return;
    }
    let mut key = String::new();
    for (i, t1) in tokens.iter().enumerate() {
        for (j, t2) in tokens.iter().enumerate().skip(i + 1) {
            key.clear();
            let _ = write!(key, "{bi}{t1} + {t2}");
            w.set_token(&key, 1.0);
            if tokens.len() <= 128 {
                for t3 in &tokens[j + 1..] {
                    key.clear();
                    let _ = write!(key, "{tri}{t1} + {t2} + {t3}");
                    w.set_token(&key, 1.0);
                }
            }
        }
    }
}

pub(super) fn topk_file_risk_features_from_summaries(summaries: &[FileSummary]) -> [f32; 8] {
    // Descending, so `min_by` keeps the first of equals, as a stable sort and
    // `take(1)` would.
    let top_susp = summaries.iter().map(|s| &s.risk).min_by(|a, b| {
        b.suspicious_ratio
            .total_cmp(&a.suspicious_ratio)
            .then(b.suspicious_findings.cmp(&a.suspicious_findings))
            .then(b.hostile_ratio.total_cmp(&a.hostile_ratio))
            .then(b.hostile_findings.cmp(&a.hostile_findings))
    });
    let top_host = summaries.iter().map(|s| &s.risk).min_by(|a, b| {
        b.hostile_ratio
            .total_cmp(&a.hostile_ratio)
            .then(b.hostile_findings.cmp(&a.hostile_findings))
            .then(b.suspicious_ratio.total_cmp(&a.suspicious_ratio))
            .then(b.suspicious_findings.cmp(&a.suspicious_findings))
    });

    let (Some(top_susp), Some(top_host)) = (top_susp, top_host) else {
        return [0.0; 8];
    };

    [
        top_susp.suspicious_ratio,
        top_host.hostile_ratio,
        (top_susp.suspicious_findings as f32 + 1.0).ln(),
        (top_host.hostile_findings as f32 + 1.0).ln(),
        top_susp.suspicious_density,
        top_host.hostile_density,
        top_susp.suspicious_category_breadth as f32,
        top_host.hostile_category_breadth as f32,
    ]
}
