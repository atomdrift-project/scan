//! One spec resolved for extraction: name lookups and metric slots built
//! once, reused for every report.

#![expect(
    clippy::cast_possible_truncation,
    reason = "feature values are narrowed to the model's f32 input dtype"
)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use super::report::{FileSummary, FindingSummary, ParsedReport, RawNeeds, ranked_ngram_paths};
use super::spec::{FeatureSpec, KEY_METRICS, KeywordSource};
use super::structure::{
    MetricMap, derived_metric, write_derived_metric_features, write_external_summary_features,
    write_format_hint_features, write_intent_gap_features, write_logic_gap_features,
    write_negative_space_features, write_silent_packer_signal, write_structural_extensions,
    write_structural_features, write_suspicious_ngram_counts, write_symbol_features,
};
use super::write::{
    ranked_tiered_path_tokens, write_aggregate_features, write_crit_ngrams, write_ngrams,
    write_optional_finding_features,
};
use super::{CRIT_BASELINE, EXTENDED_KEYWORD_ABI, FeatureWriter, METRIC_TRANSFORM_ABI};

/// One `metrics:` feature resolved to its slot: the value at
/// `metrics[group][field]`, optionally `ln(|v| + 1)`-scaled.
#[derive(Debug)]
pub(super) struct MetricSlot {
    pub(super) slot: usize,
    pub(super) group: String,
    pub(super) field: String,
    pub(super) log: bool,
    pub(super) getter: Option<String>,
    pub(super) correlation: Option<[[String; 2]; 2]>,
}

/// Pre-built lookup tables for fast repeated extraction against a spec.
#[derive(Debug)]
pub struct ExtractContext {
    /// A description-keyword model refuses a finding with no description.
    pub(super) strict_keyword_descriptions: bool,
    pub(super) keyword_slots: HashMap<KeywordSource, HashMap<String, usize>>,
    /// Archive structure is measured relative to the sample, not absolutely.
    pub(super) relative_structure: bool,
    pub(super) explicit_metrics: bool,
    pub(super) metric_reference_year: u32,
    pub(super) trait_path_depth: usize,
    pub(super) tiered_min_crit: [u32; 3],
    pub(super) tiered_branch_min_crit: HashMap<String, u32>,
    pub(super) presence_lookup: HashMap<String, usize>,
    /// Global feature name → index lookup for name-written features.
    pub(super) absolute_lookup: HashMap<String, usize>,
    /// Group 5: the base [`KEY_METRICS`] then the spec's extended metric
    /// vocab, each resolved to its slot (absent ones dropped).
    pub(super) metric_slots: Vec<MetricSlot>,
    /// Group 12: `ghost:` vocab paths that have a slot in this spec.
    pub(super) ghost_slots: Vec<(String, usize)>,
    pub(super) total_features: usize,
    /// Optional families this spec carries; their inputs are only gathered
    /// (and their writers only run) when a route needs them.
    pub(super) kv: bool,
    pub(super) kv_shape: bool,
    pub(super) kv_split: bool,
    /// Any of the `symbol:`, `symbol_bi:` and `symbol_tri:` families.
    pub(super) symbols: bool,
    pub(super) symbol_bigrams: bool,
    pub(super) symbol_trigrams: bool,
    pub(super) suspicious_ngram_counts: bool,
    pub(super) optional_finding_features: bool,
    pub(super) tiered_slots: Vec<(usize, Vec<String>)>,

    // Optimized bigram/trigram lookups
    pub(super) path_to_id: HashMap<String, u32>,
    pub(super) bigram_id_lookup: HashMap<(u32, u32), usize>,
    pub(super) trigram_id_lookup: HashMap<(u32, u32, u32), usize>,

    // Per-family map from vocab index -> the real `feature_names` slot for that
    // member, resolved by NAME in `new()`. `None` at a position means that vocab
    // entry has no slot in this spec (pruned) and is skipped. Indexed by the same
    // vocab index the runtime lookups (`presence_lookup`, `bigram_id_lookup`,
    // `trigram_id_lookup`) produce, so a family that is a SUBSET of its (possibly
    // shared) vocab still writes each present member to its true slot — no
    // contiguous `base + idx` block required. See `new()`.
    pub(super) present_slots: Vec<Option<usize>>,
    pub(super) maxcrit_slots: Vec<Option<usize>>,
    pub(super) bigram_slots: Vec<Option<usize>>,
    pub(super) trigram_slots: Vec<Option<usize>>,
    pub(super) unsigned_bigram_slots: Vec<Option<usize>>,
}

impl ExtractContext {
    /// Build lookup tables from a feature specification.
    #[must_use]
    pub fn new(spec: &FeatureSpec) -> Self {
        let presence_lookup: HashMap<String, usize> = spec
            .presence_vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let (path_to_id, bigram_id_lookup, trigram_id_lookup) = ngram_id_lookups(spec);
        let absolute_lookup: HashMap<String, usize> = spec
            .feature_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        // Resolve each offset-written family (presence, max-crit, bigram, trigram,
        // unsigned-bigram) to the real `feature_names` slot of every vocab member,
        // by NAME, rather than assuming a contiguous `base + idx` run. A family
        // pruned from the spec maps every entry to `None` and is skipped; a
        // family that keeps only a SUBSET of its (often shared) vocab maps each
        // kept member to its true slot, so writes never land on the wrong
        // feature.
        let family_slots = |prefix: &str, vocab: &[String]| -> Vec<Option<usize>> {
            vocab
                .iter()
                .map(|v| absolute_lookup.get(&format!("{prefix}{v}")).copied())
                .collect()
        };
        let present_slots = family_slots("present:", &spec.presence_vocab);
        let maxcrit_slots = family_slots("maxcrit:", &spec.presence_vocab);
        let bigram_slots = family_slots("bigrams:", &spec.bigram_vocab);
        let trigram_slots = family_slots("trigram:", &spec.trigram_vocab);
        let unsigned_bigram_slots = family_slots("unsigned_bigram:", &spec.bigram_vocab);

        let has_prefix = |prefix: &str| spec.feature_names.iter().any(|n| n.starts_with(prefix));
        let has_name = |name: &str| absolute_lookup.contains_key(name);
        Self {
            strict_keyword_descriptions: spec.abi_version < EXTENDED_KEYWORD_ABI,
            keyword_slots: keyword_slots(spec, &absolute_lookup),
            trait_path_depth: spec.trait_path_depth,
            relative_structure: spec.feature_semantics_abi() >= METRIC_TRANSFORM_ABI,
            explicit_metrics: spec.metric_transforms.is_some(),
            metric_reference_year: spec.metric_reference_year,
            tiered_min_crit: spec.tiered_min_crit,
            tiered_branch_min_crit: spec.tiered_branch_min_crit.clone(),
            metric_slots: metric_slots(&absolute_lookup, spec),
            ghost_slots: spec
                .ghost_vocab
                .iter()
                .filter_map(|path| {
                    let slot = *absolute_lookup.get(&format!("ghost:{path}"))?;
                    Some((path.clone(), slot))
                })
                .collect(),
            total_features: spec.total_features,
            kv: has_prefix("kv:"),
            kv_shape: spec
                .kv_vocab
                .iter()
                .any(|v| v.split('=').next().is_some_and(|key| key.contains(':'))),
            kv_split: spec.kv_vocab.iter().any(|v| v.contains("=part:")),
            symbols: has_prefix("symbol:") || has_prefix("symbol_bi:") || has_prefix("symbol_tri:"),
            symbol_bigrams: has_prefix("symbol_bi:"),
            symbol_trigrams: has_prefix("symbol_tri:"),
            tiered_slots: tiered_slots(spec),
            optional_finding_features: has_prefix("mbc:")
                || has_prefix("agg:docobf_")
                || [
                    "agg:confidence_mean",
                    "agg:confidence_std",
                    "agg:confidence_skew",
                    "agg:confidence_kurtosis",
                    "agg:trait_id_lexical_distance",
                ]
                .iter()
                .any(|name| has_name(name)),
            suspicious_ngram_counts: has_name("agg:suspicious_bigram_count")
                || has_name("agg:suspicious_trigram_count"),
            presence_lookup,
            absolute_lookup,
            path_to_id,
            bigram_id_lookup,
            trigram_id_lookup,
            present_slots,
            maxcrit_slots,
            bigram_slots,
            trigram_slots,
            unsigned_bigram_slots,
        }
    }

    /// Extract this route's feature vector from a cleave compact report.
    #[must_use]
    pub fn extract(&self, report: &cleave::types::CompactReport) -> Vec<f32> {
        let parsed = ParsedReport::from_compact_report(report, self.raw_needs(), None);
        self.extract_from_parsed(&parsed)
    }

    /// This route's active raw-subtree needs (which optional families it emits).
    pub(crate) fn raw_needs(&self) -> RawNeeds {
        RawNeeds {
            kv: self.kv,
            kv_shape: self.kv_shape,
            kv_split: self.kv_split,
            symbol: self.symbols,
            rule_keywords: self.reads(KeywordSource::RuleId),
            description_keywords: self.reads(KeywordSource::Description)
                || self.reads(KeywordSource::IdAndDescription),
            full_rule_keywords: self.reads(KeywordSource::FullId)
                || self.reads(KeywordSource::IdAndDescription),
            context_keywords: self.reads(KeywordSource::Context),
        }
    }

    /// Whether this route has a slot for any word of `source`.
    pub(super) fn reads(&self, source: KeywordSource) -> bool {
        self.keyword_slots.contains_key(&source)
    }

    /// Allocate and fill this route's feature vector from a shared [`ParsedReport`].
    pub(crate) fn extract_from_parsed(&self, parsed: &ParsedReport) -> Vec<f32> {
        if parsed.trait_path_depth != self.trait_path_depth {
            let adjusted = parsed.at_trait_path_depth(self.trait_path_depth);
            return self.extract_from_parsed(&adjusted);
        }
        let mut vec = vec![0.0f32; self.total_features];
        self.write_features(parsed, &mut vec);
        vec
    }

    /// Emit this route's features into `vec` from an already-parsed report.
    /// The parse is route-independent (see [`ParsedReport`]); only the writes
    /// below depend on this context's vocab/slots. Families are written in
    /// layout order.
    fn write_features(&self, parsed: &ParsedReport, vec: &mut [f32]) {
        let summaries = parsed.summaries.as_slice();
        let combined = &parsed.combined;
        let w = &mut FeatureWriter {
            vec,
            lookup: &self.absolute_lookup,
        };

        self.write_keyword_features(parsed, w);
        self.write_path_features(combined, parsed.sample_score, w); // G1, G2
        write_aggregate_features(parsed, w); // G3
        if self.optional_finding_features {
            write_optional_finding_features(parsed, w);
        }
        write_crit_ngrams(&parsed.crit_tokens, w);
        write_ngrams(w, &parsed.attack_codes, "atkbi:", "atktri:", "ATT&CK");
        write_ngrams(w, &parsed.mbc_codes, "mbcbi:", "mbctri:", "MBC");
        write_external_summary_features(combined, w); // G4
        self.write_metric_features(&parsed.merged_metrics, summaries, w); // G5
        // G6 (filetype one-hots) is never written: models are trained blind
        // to the file type.
        write_format_hint_features(summaries, w); // G6b
        write_structural_features(w, summaries, combined.filtered_finding_count); // G7
        self.write_formula_features(parsed, w); // G8–G10, G12–G14
        self.write_bigram_features(summaries, w, &self.bigram_slots); // G11
        for (index, order) in [2, 3, 4].into_iter().enumerate() {
            if !self
                .tiered_slots
                .iter()
                .any(|(_, parts)| parts.len() == order)
            {
                continue;
            }
            let tokens = ranked_tiered_path_tokens(
                combined,
                self.tiered_min_crit[index],
                &self.tiered_branch_min_crit,
                order,
            );
            for (slot, parts) in &self.tiered_slots {
                if parts.len() == order
                    && parts.iter().all(|part| tokens.binary_search(part).is_ok())
                {
                    w.set_slot(*slot, 1.0);
                }
            }
        }
        write_structural_extensions(summaries, combined, self.relative_structure, w); // G15
        self.write_trigram_features(summaries, w); // G16
        write_logic_gap_features(combined, summaries, w); // G19
        // G20: signature synergy — the bigram block again, for unsigned samples.
        if combined.sample_paths.contains_key("metadata/unsigned") {
            self.write_bigram_features(summaries, w, &self.unsigned_bigram_slots);
        }
        write_intent_gap_features(combined, w); // G22
        write_negative_space_features(combined, summaries, w); // G23

        // Families only some route specs carry. G24 (textenc) has no input in
        // the compact report and always extracts as zero.
        if self.suspicious_ngram_counts {
            write_suspicious_ngram_counts(combined, w);
        }
        if !self.explicit_metrics {
            write_derived_metric_features(&parsed.merged_metrics, w);
        }
        write_silent_packer_signal(summaries, combined.filtered_finding_count, w);
        if self.kv {
            for s in summaries {
                for token in &s.kv_tokens {
                    w.set_token(token, 1.0);
                }
            }
        }
        if self.symbols {
            write_symbol_features(summaries, w, self.symbol_bigrams, self.symbol_trigrams);
        }
    }

    /// G1 presence and G2 max-criticality, per sample path, weighted by the
    /// sample's risk score and the path's best confidence.
    fn write_path_features(
        &self,
        summary: &FindingSummary,
        sample_score: i64,
        w: &mut FeatureWriter<'_>,
    ) {
        let score_weight: f64 = if sample_score > 0 {
            (sample_score as f64).ln_1p()
        } else {
            1.0
        };
        for (path, &max_ord) in &summary.sample_paths {
            if max_ord >= CRIT_BASELINE
                && let Some(&idx) = self.presence_lookup.get(path.as_str())
                && let Some(&Some(slot)) = self.present_slots.get(idx)
            {
                let conf = summary.path_confidences.get(path).copied().unwrap_or(1.0);
                w.set_slot(slot, (score_weight * conf) as f32);
            }
        }
        for (path, &max_ord) in &summary.sample_paths {
            if let Some(&idx) = self.presence_lookup.get(path.as_str())
                && let Some(&Some(slot)) = self.maxcrit_slots.get(idx)
            {
                let conf = summary.path_confidences.get(path).copied().unwrap_or(1.0);
                w.set_slot(slot, (f64::from(max_ord) * score_weight * conf) as f32);
            }
        }
    }

    /// G5: the base [`KEY_METRICS`] then the spec's extended metric vocab.
    fn write_metric_features(
        &self,
        metrics: &MetricMap,
        summaries: &[FileSummary],
        w: &mut FeatureWriter<'_>,
    ) {
        for m in &self.metric_slots {
            let value = if let Some([[ga, ka], [gb, kb]]) = &m.correlation {
                let get = |g: &String, k: &String| {
                    metrics
                        .get(g)
                        .and_then(|fields| fields.get(k))
                        .copied()
                        .unwrap_or(0.0)
                };
                get(ga, ka) * get(gb, kb)
            } else if let Some(getter) = &m.getter {
                derived_metric(getter, metrics, summaries, self.metric_reference_year)
            } else {
                metrics
                    .get(&m.group)
                    .and_then(|g| g.get(&m.field))
                    .copied()
                    .unwrap_or(0.0)
            };
            // A spec without explicit transforms was trained on the f32 value,
            // logged after narrowing.
            let output = if self.explicit_metrics {
                (if m.log { value.abs().ln_1p() } else { value }) as f32
            } else {
                let value = value as f32;
                if m.log {
                    (value.abs() + 1.0).ln()
                } else {
                    value
                }
            };
            w.set_slot(m.slot, output);
        }
    }

    /// G8 elements, G9 formula, G10 score, G12 ghost, G13 skeleton and G14
    /// rare elements: the families read off the primary file's formula and
    /// risk score.
    fn write_formula_features(&self, parsed: &ParsedReport, w: &mut FeatureWriter<'_>) {
        let combined = &parsed.combined;
        let formula_str = parsed.formula_str.as_str();
        let elements_str = parsed.elements_str.as_str();
        let sample_score = parsed.sample_score;
        let mut key = String::new();

        // G8: Elements
        if !elements_str.is_empty() {
            for el in elements_str.split(',') {
                w.set_prefixed(&mut key, "elements:", el.trim(), 1.0);
            }
        }

        // G9: Formula
        let skeleton_str: String = formula_str.chars().filter(|c| c.is_alphabetic()).collect();
        let unique_skel_chars: HashSet<char> = skeleton_str.chars().collect();
        w.set("formula:skeleton_len", skeleton_str.chars().count() as f32);
        w.set("formula:unique_elements", unique_skel_chars.len() as f32);
        if combined.filtered_finding_count > 0 {
            w.set(
                "formula:complexity_ratio",
                formula_str.chars().count() as f32 / combined.filtered_finding_count as f32,
            );
        }

        // G10: Score
        let total_size_bytes: f64 = parsed.summaries.iter().map(|s| s.size_bytes).sum();
        w.set("score:hopper_score", sample_score as f32);
        w.set(
            "score:density",
            if total_size_bytes > 0.0 {
                sample_score as f32 / (total_size_bytes as f32).ln_1p()
            } else {
                0.0
            },
        );
        for s in &parsed.summaries {
            key.clear();
            let _ = write!(key, "inter:{}*score", s.file_type);
            w.set_token(&key, sample_score as f32);
        }

        // G12: Ghost — an expected path the sample never reached at baseline.
        for (ghost_path, slot) in &self.ghost_slots {
            let missing = combined
                .sample_paths
                .get(ghost_path)
                .is_none_or(|&max_ord| max_ord < CRIT_BASELINE);
            if missing {
                w.set_slot(*slot, 1.0);
            }
        }

        // G13: Skeleton
        if !skeleton_str.is_empty() {
            w.set_prefixed(&mut key, "skeleton:", &skeleton_str, 1.0);
        }

        // G14: Rare Elements, weighted by the mean finding confidence.
        if !elements_str.is_empty() {
            let weight: f32 = if combined.finding_confidences.is_empty() {
                1.0
            } else {
                let sum: f64 = combined.finding_confidences.iter().sum();
                (sum / combined.finding_confidences.len() as f64) as f32
            };
            for el in elements_str.split(',') {
                w.set_prefixed(&mut key, "rare:", el.trim(), weight);
            }
        }
    }

    /// Path-pair features: a vocab bigram is set when both of its 3-level
    /// paths fired in the same file. `slots` picks the family (`bigrams:` or
    /// `unsigned_bigram:`), which share one vocab.
    pub(super) fn write_bigram_features(
        &self,
        summaries: &[FileSummary],
        w: &mut FeatureWriter<'_>,
        slots: &[Option<usize>],
    ) {
        for s in summaries {
            let ids = self.sorted_path_ids(s, 512);
            let n = ids.len();
            let mut mark = |idx: usize| {
                if let Some(&Some(slot)) = slots.get(idx) {
                    w.set_slot(slot, 1.0);
                }
            };
            // Probe whichever side is smaller: a file usually fires a handful
            // of paths, while the vocab holds thousands of pairs, and this
            // runs once per archive member.
            if n * (n + 1) / 2 <= self.bigram_id_lookup.len() {
                for (i, &a) in ids.iter().enumerate() {
                    for &b in &ids[i..] {
                        if let Some(&idx) = self.bigram_id_lookup.get(&(a, b)) {
                            mark(idx);
                        }
                    }
                }
            } else {
                for (&(a, b), &idx) in &self.bigram_id_lookup {
                    if [a, b].iter().all(|id| ids.binary_search(id).is_ok()) {
                        mark(idx);
                    }
                }
            }
        }
    }

    /// The vocab ids of the file's strongest `rank_limit` n-gram base paths,
    /// ascending — the order the bigram and trigram lookup keys are built in.
    /// The limit bounds the pair and triple probes.
    fn sorted_path_ids(&self, s: &FileSummary, rank_limit: usize) -> Vec<u32> {
        let mut ids: Vec<u32> = ranked_ngram_paths(&s.raw_findings, rank_limit)
            .iter()
            .filter_map(|p| self.path_to_id.get(p).copied())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Path-triple features, as [`Self::write_bigram_features`].
    pub(super) fn write_trigram_features(
        &self,
        summaries: &[FileSummary],
        w: &mut FeatureWriter<'_>,
    ) {
        for s in summaries {
            let ids = self.sorted_path_ids(s, 128);
            let n = ids.len();
            let mut mark = |idx: usize| {
                if let Some(&Some(slot)) = self.trigram_slots.get(idx) {
                    w.set_slot(slot, 1.0);
                }
            };
            // The smaller side again, as for bigrams.
            if n * (n + 1) * (n + 2) / 6 <= self.trigram_id_lookup.len() {
                for (i, &a) in ids.iter().enumerate() {
                    for (j, &b) in ids.iter().enumerate().skip(i) {
                        for &c in &ids[j..] {
                            if let Some(&idx) = self.trigram_id_lookup.get(&(a, b, c)) {
                                mark(idx);
                            }
                        }
                    }
                }
            } else {
                for (&(a, b, c), &idx) in &self.trigram_id_lookup {
                    if [a, b, c].iter().all(|id| ids.binary_search(id).is_ok()) {
                        mark(idx);
                    }
                }
            }
        }
    }
}

/// Interned ids of the n-gram base paths, and the bigram and trigram vocabs
/// keyed by their sorted id tuples, so a file's pairs are probed by integer.
#[expect(
    clippy::type_complexity,
    reason = "the three tables move straight into ExtractContext's fields"
)]
fn ngram_id_lookups(
    spec: &FeatureSpec,
) -> (
    HashMap<String, u32>,
    HashMap<(u32, u32), usize>,
    HashMap<(u32, u32, u32), usize>,
) {
    let mut path_to_id: HashMap<String, u32> = HashMap::new();
    let mut id = |path: &str| {
        let next = u32::try_from(path_to_id.len()).unwrap_or(u32::MAX);
        *path_to_id.entry(path.to_owned()).or_insert(next)
    };
    let mut bigrams = HashMap::new();
    for (i, bigram) in spec.bigram_vocab.iter().enumerate() {
        if let [a, b] = bigram.split(" + ").collect::<Vec<_>>()[..] {
            let (a, b) = (id(a), id(b));
            bigrams.insert((a.min(b), a.max(b)), i);
        }
    }
    let mut trigrams = HashMap::new();
    for (i, trigram) in spec.trigram_vocab.iter().enumerate() {
        if let [a, b, c] = trigram.split(" + ").collect::<Vec<_>>()[..] {
            let mut ids = [id(a), id(b), id(c)];
            ids.sort_unstable();
            trigrams.insert((ids[0], ids[1], ids[2]), i);
        }
    }
    (path_to_id, bigrams, trigrams)
}

/// Each keyword family's words that have a slot in this spec; a family with
/// none is left out.
fn keyword_slots(
    spec: &FeatureSpec,
    lookup: &HashMap<String, usize>,
) -> HashMap<KeywordSource, HashMap<String, usize>> {
    spec.keyword_vocabs
        .iter()
        .filter_map(|(&source, words)| {
            let slots: HashMap<String, usize> = words
                .iter()
                .filter_map(|word| {
                    lookup
                        .get(&format!("{}:{word}", source.name()))
                        .map(|&slot| (word.clone(), slot))
                })
                .collect();
            (!slots.is_empty()).then_some((source, slots))
        })
        .collect()
}

/// The `tierbi:`/`tiertri:`/`tierquad:` features: each slot with the
/// `tier:path` tokens that must all be present to set it.
fn tiered_slots(spec: &FeatureSpec) -> Vec<(usize, Vec<String>)> {
    spec.feature_names
        .iter()
        .enumerate()
        .filter_map(|(slot, name)| {
            let (family, value) = name.split_once(':')?;
            ["tierbi", "tiertri", "tierquad"]
                .contains(&family)
                .then(|| (slot, value.split(" + ").map(str::to_owned).collect()))
        })
        .collect()
}

/// Resolve the `metrics:` family to slots: every [`KEY_METRICS`] entry, then
/// each extended-vocab key (`<group>_<field>`) that is not a base metric.
/// Extended keys are log-scaled when the field names a count or size.
fn metric_slots(lookup: &HashMap<String, usize>, spec: &FeatureSpec) -> Vec<MetricSlot> {
    if let Some(transforms) = &spec.metric_transforms {
        return lookup
            .iter()
            .filter_map(|(name, &slot)| {
                let suffix = name.strip_prefix("metrics:")?;
                let (group, field) = suffix.split_once('_')?;
                Some(MetricSlot {
                    slot,
                    group: group.into(),
                    field: field.into(),
                    log: transforms.get(suffix).is_some_and(|t| t == "log1p_abs"),
                    getter: spec
                        .metric_getters
                        .iter()
                        .any(|g| g == suffix)
                        .then(|| suffix.to_owned()),
                    correlation: spec.metric_correlations.get(suffix).cloned(),
                })
            })
            .collect();
    }
    let metric_vocab = &spec.metric_vocab;
    let base = KEY_METRICS.iter().map(|&(group, field, log)| {
        (
            format!("metrics:{group}_{field}"),
            group.to_string(),
            field.to_string(),
            log,
        )
    });
    let base_keys: HashSet<String> = KEY_METRICS
        .iter()
        .map(|&(group, field, _)| format!("{group}_{field}"))
        .collect();
    let extended = metric_vocab.iter().filter_map(|key| {
        let (group, field) = key.split_once('_')?;
        if base_keys.contains(key.as_str()) {
            return None;
        }
        let log = ["count", "size", "total", "bytes", "length"]
            .iter()
            .any(|word| field.contains(word));
        Some((
            format!("metrics:{key}"),
            group.to_string(),
            field.to_string(),
            log,
        ))
    });
    base.chain(extended)
        .filter_map(|(name, group, field, log)| {
            Some(MetricSlot {
                slot: *lookup.get(&name)?,
                group,
                field,
                log,
                getter: None,
                correlation: None,
            })
        })
        .collect()
}
