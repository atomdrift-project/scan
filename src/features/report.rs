//! A compact report parsed once into the per-file and per-finding summaries
//! the writers read.

#![expect(
    clippy::cast_possible_truncation,
    reason = "feature values are narrowed to the model's f32 input dtype"
)]

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// The finding fields retained for cross-file aggregation: ID dedup and
/// ATT&CK/MBC code rollups. Keeping just these avoids cloning the whole
/// finding JSON subtree for every file.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct RawFinding {
    pub(super) id: String,
    pub(super) conf: f64,
    pub(super) crit: u32,
    pub(super) atk: Option<String>,
    pub(super) mbc: Option<String>,
}

/// Select base paths before filtering against a model vocabulary, as in training.
pub(super) fn ranked_ngram_paths(findings: &[RawFinding], limit: usize) -> Vec<String> {
    let mut scores: HashMap<&str, f64> = HashMap::new();
    for finding in findings {
        if finding.id.is_empty()
            || finding.conf.is_nan()
            || finding.conf < MIN_CONFIDENCE
            || finding.crit < CRIT_NOTABLE
        {
            continue;
        }
        let path = trait_path(&finding.id);
        let score = if finding.conf.is_finite() {
            f64::from(finding.crit) * f64::from(finding.conf.min(1.0) as f32)
        } else {
            0.0
        };
        scores
            .entry(path)
            .and_modify(|v| *v = v.max(score))
            .or_insert(score);
    }
    let mut paths: Vec<_> = scores.keys().copied().collect();
    if paths.len() > limit {
        paths.sort_unstable_by(|a, b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(b)));
        paths.truncate(limit);
    }
    paths.sort_unstable();
    paths.into_iter().map(str::to_owned).collect()
}

/// Pre-calculated data for a single file entry in a report.
///
/// Compact placement metadata supplies archive structure. Timestamps and text
/// encodings remain unavailable in the retained-facts contract.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct FileSummary {
    pub(super) id: Option<u32>,
    pub(super) parent: Option<u32>,
    pub(super) depth: u32,
    pub(super) entry_section_nontext: bool,
    pub(super) path: String,
    pub(super) file_type: String,
    pub(super) size_bytes: f64,
    pub(super) overall_entropy: f64,
    pub(super) metrics: HashMap<String, HashMap<String, f64>>,
    pub(super) findings: FindingSummary,
    pub(super) risk: FileRiskStats,
    pub(super) unique_3level_paths: Vec<String>,
    /// Import tokens (`name` and `lib!name`) for the logic-gap features.
    pub(super) imports: HashSet<String>,
    /// Full `kv:` feature names from the metrics block. Empty unless a route
    /// reads the `kv:` family.
    pub(super) kv_tokens: Vec<String>,
    /// Normalized import/export/function/AST symbols, sorted and unique.
    /// Empty unless a route reads the `symbol:` family.
    pub(super) symbols: Vec<String>,
    /// Findings reduced to the fields used for cross-file aggregation.
    pub(super) raw_findings: Vec<RawFinding>,
}

/// Which optional feature families are active for this route, so
/// [`FileSummary::from_compact`] only gathers the inputs they read.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RawNeeds {
    pub(super) kv: bool,
    pub(super) kv_shape: bool,
    pub(super) kv_split: bool,
    pub(super) symbol: bool,
}

impl RawNeeds {
    /// All optional inputs any specialist feature family may read.
    pub(crate) const fn all() -> Self {
        Self {
            kv: true,
            kv_shape: true,
            kv_split: true,
            symbol: true,
        }
    }

    /// The needs of any of two consumers — used to parse a report once for the
    /// general pass and every route, gathering an input if *any* of them reads it.
    /// `RawNeeds::default()` (no families) is the identity for folding these.
    pub(crate) const fn union(self, other: Self) -> Self {
        Self {
            kv: self.kv || other.kv,
            kv_shape: self.kv_shape || other.kv_shape,
            kv_split: self.kv_split || other.kv_split,
            symbol: self.symbol || other.symbol,
        }
    }
}

/// A report parsed into the route-independent summaries the feature writers read.
///
/// Building this is the expensive part of extraction (per-file `FileSummary`
/// construction, cross-file finding/metric aggregation, token sets). It depends
/// only on the report and the union of active families, never on a specific
/// route's vocab — so the general pass and every ML route share one
/// `ParsedReport` instead of each re-summarizing the same report.
#[derive(Debug)]
pub(crate) struct ParsedReport {
    pub(super) trait_path_depth: usize,
    pub(super) depth_variants: std::sync::Mutex<HashMap<usize, std::sync::Arc<ParsedReport>>>,
    pub(super) summaries: Vec<FileSummary>,
    pub(super) combined: FindingSummary,
    pub(super) merged_metrics: MetricMap,
    pub(super) formula_str: String,
    pub(super) elements_str: String,
    pub(super) sample_score: i64,
    /// Sorted `tier:category[/sub]` tokens for the `crit:` n-grams.
    pub(super) crit_tokens: Vec<String>,
    /// Sorted `tier:path` tokens for the `tierbi:`/`tiertri:` families.
    pub(super) tiered_tokens: Vec<String>,
    /// Distinct ATT&CK technique and MBC behavior codes, sorted.
    pub(super) attack_codes: Vec<String>,
    pub(super) mbc_codes: Vec<String>,
}

impl ParsedReport {
    /// Reuse decoded facts while rebuilding depth-dependent summaries for a
    /// route. The shared default-depth parse remains untouched for other routes.
    pub(super) fn at_trait_path_depth(&self, depth: usize) -> std::sync::Arc<Self> {
        let mut variants = self
            .depth_variants
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::sync::Arc::clone(
            variants
                .entry(depth)
                .or_insert_with(|| std::sync::Arc::new(self.rebuild_at_trait_path_depth(depth))),
        )
    }

    pub(super) fn rebuild_at_trait_path_depth(&self, depth: usize) -> Self {
        let mut summaries = self.summaries.clone();
        for file in &mut summaries {
            let views: Vec<_> = file
                .raw_findings
                .iter()
                .map(|f| FindingView {
                    id: &f.id,
                    conf: f.conf,
                    crit: f.crit,
                })
                .collect();
            file.findings = summarize_findings_at_depth(&views, depth);
            file.risk.suspicious_category_breadth = file.findings.suspicious_category_breadth;
            file.risk.hostile_category_breadth = file.findings.hostile_category_breadth;
        }
        let mut adjusted =
            Self::from_summaries(summaries, Some((&self.formula_str, self.sample_score)));
        adjusted.trait_path_depth = depth;
        adjusted
    }

    /// Featurize a typed compact report.
    ///
    /// `keep` restricts the pass to the sample's own files by sha — used when
    /// fetched payloads have been grafted in and must not dilute the sample's
    /// own aggregate. `None` featurizes every file. The primary file is the
    /// first entry at depth 0, resolved within whatever subset survives `keep`.
    pub(crate) fn from_compact_report(
        report: &cleave::types::CompactReport,
        needs: RawNeeds,
        keep: Option<&HashSet<String>>,
    ) -> Self {
        let files: Vec<&cleave::types::CompactFile> = match keep {
            Some(shas) => report
                .files
                .iter()
                .filter(|f| shas.contains(&f.sha))
                .collect(),
            None => report.files.iter().collect(),
        };
        Self::from_compact_files(&files, needs)
    }

    /// Featurize a single compact file entry (an archive member) on its own.
    pub(crate) fn from_compact_file(file: &cleave::types::CompactFile, needs: RawNeeds) -> Self {
        // Placement depth belongs to the enclosing archive. When grading this
        // member alone, its own metadata must match its training database row.
        Self::from_summaries(
            vec![FileSummary::from_compact(file, needs)],
            Some((file.formula.as_deref().unwrap_or(""), file.risk)),
        )
    }

    pub(super) fn from_compact_files(
        files: &[&cleave::types::CompactFile],
        needs: RawNeeds,
    ) -> Self {
        // Small warm-cache reports are cheaper to summarize serially than to fan
        // out into rayon jobs.
        let file_summaries: Vec<FileSummary> = if files.len() < 8 {
            files
                .iter()
                .map(|&f| FileSummary::from_compact(f, needs))
                .collect()
        } else {
            files
                .par_iter()
                .map(|&f| FileSummary::from_compact(f, needs))
                .collect()
        };
        let primary = files
            .iter()
            .copied()
            .find(|f| f.depth == 0)
            .map(|f| (f.formula.as_deref().unwrap_or(""), f.risk));
        Self::from_summaries(file_summaries, primary)
    }

    pub(super) fn from_summaries(
        file_summaries: Vec<FileSummary>,
        primary: Option<(&str, i64)>,
    ) -> Self {
        let summaries = if file_summaries.is_empty() {
            vec![FileSummary::default()]
        } else {
            file_summaries
        };
        let (formula_str, elements_str, sample_score) = primary.map_or_else(
            || (String::new(), String::new(), 0),
            |(f, risk)| {
                (
                    f.to_owned(),
                    f.chars()
                        .filter(|c| !('\u{2080}'..='\u{2089}').contains(c))
                        .collect(),
                    risk,
                )
            },
        );

        let combined = summarize_report_summaries(&summaries);
        let merged_metrics = merge_metric_summaries(&summaries);
        let crit_tokens = crit_category_tokens(&combined);
        let tiered_tokens = tiered_path_tokens(&combined);
        let mut attack_codes: HashSet<&str> = HashSet::new();
        let mut mbc_codes: HashSet<&str> = HashSet::new();
        for finding in summaries.iter().flat_map(|s| &s.raw_findings) {
            if let Some(a) = &finding.atk {
                attack_codes.insert(a);
            }
            if let Some(m) = &finding.mbc {
                mbc_codes.insert(m);
            }
        }
        let sorted = |codes: HashSet<&str>| {
            let mut codes: Vec<String> = codes.into_iter().map(str::to_string).collect();
            codes.sort();
            codes
        };
        let attack_codes = sorted(attack_codes);
        let mbc_codes = sorted(mbc_codes);
        Self {
            trait_path_depth: default_trait_path_depth(),
            depth_variants: std::sync::Mutex::default(),
            summaries,
            combined,
            merged_metrics,
            formula_str,
            elements_str,
            sample_score,
            crit_tokens,
            tiered_tokens,
            attack_codes,
            mbc_codes,
        }
    }
}

impl FileSummary {
    /// Summarize one compact file entry.
    pub(super) fn from_compact(file: &cleave::types::CompactFile, needs: RawNeeds) -> Self {
        let raw_findings: Vec<RawFinding> = file
            .findings
            .iter()
            .map(|f| RawFinding {
                id: f.id.clone(),
                conf: f64::from(f.confidence),
                crit: u32::from(f.criticality),
                atk: f.attack.clone(),
                mbc: f.mbc.clone(),
            })
            .collect();
        let facts = &file.facts;
        let finding_views: Vec<FindingView<'_>> = raw_findings
            .iter()
            .map(|f| FindingView {
                id: &f.id,
                conf: f.conf,
                crit: f.crit,
            })
            .collect();
        let findings = summarize_findings(&finding_views);

        let size_bytes = file.size as f64;
        let size_kb = (size_bytes / 1024.0).max(1.0) as f32;
        let denom = findings.filtered_finding_count.max(1) as f32;
        let max_crit = findings.sample_paths.values().copied().max().unwrap_or(0);

        let risk = FileRiskStats {
            suspicious_ratio: findings.suspicious_finding_count as f32 / denom,
            hostile_ratio: findings.hostile_finding_count as f32 / denom,
            suspicious_findings: findings.suspicious_finding_count,
            hostile_findings: findings.hostile_finding_count,
            suspicious_density: findings.suspicious_finding_count as f32 / size_kb,
            hostile_density: findings.hostile_finding_count as f32 / size_kb,
            suspicious_category_breadth: findings.suspicious_category_breadth,
            hostile_category_breadth: findings.hostile_category_breadth,
            max_crit,
        };

        let mut unique_3level_paths: Vec<String> = finding_views
            .iter()
            .filter(|f| f.conf >= MIN_CONFIDENCE && f.crit >= CRIT_NOTABLE)
            .map(|f| trait_path(f.id).to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        unique_3level_paths.sort();

        let metrics_value = facts
            .metrics
            .as_ref()
            .map(|m| &m.0)
            .filter(|v| v.is_object());
        let metrics = numeric_metrics(metrics_value);
        let overall_entropy = metrics
            .get("binary")
            .and_then(|m| m.get("overall_entropy"))
            .copied()
            .unwrap_or(0.0);

        let imports: HashSet<String> = facts
            .imports
            .iter()
            .flat_map(|imp| import_tokens_from_parts(&imp.library, &imp.name))
            .collect();

        Self {
            id: Some(file.id),
            parent: file.parent,
            depth: file.depth,
            entry_section_nontext: metrics_value
                .and_then(|v| v.get("pe"))
                .and_then(|v| v.get("entry_section"))
                .and_then(serde_json::Value::as_str)
                .is_some_and(|v| !v.is_empty() && v != ".text"),
            path: file.path.clone(),
            file_type: file.file_type.clone(),
            size_bytes,
            overall_entropy,
            metrics,
            findings,
            risk,
            unique_3level_paths,
            imports,
            kv_tokens: if needs.kv {
                kv_tokens_with_options(metrics_value, needs.kv_shape, needs.kv_split)
            } else {
                Vec::new()
            },
            symbols: if needs.symbol {
                file_symbols(facts)
            } else {
                Vec::new()
            },
            raw_findings,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(super) struct FindingSummary {
    pub(super) sample_paths: HashMap<String, u32>,
    pub(super) path_confidences: HashMap<String, f64>,
    pub(super) path_scores: HashMap<String, f64>,
    pub(super) finding_confidences: Vec<f64>,
    pub(super) filtered_finding_count: u32,
    pub(super) notable_finding_count: u32,
    pub(super) suspicious_finding_count: u32,
    pub(super) hostile_finding_count: u32,
    pub(super) unique_notable_ids: usize,
    pub(super) unique_suspicious_ids: usize,
    pub(super) unique_hostile_ids: usize,
    pub(super) suspicious_category_breadth: usize,
    pub(super) hostile_category_breadth: usize,
    pub(super) third_party_max_crit: u32,
    pub(super) third_party_count: u32,
    pub(super) well_known_max_crit: u32,
    pub(super) well_known_hostile: u32,
    pub(super) well_known_suspicious: u32,
    pub(super) has_yara: bool,
}

/// Borrow numeric/categorical facts under old and new metric spellings.
/// Missing aliases are filled; explicit values and malformed groups win.
pub(super) fn metric_fields(
    source: Option<&serde_json::Value>,
) -> impl Iterator<Item = (&str, &str, &serde_json::Value)> {
    let groups = source.and_then(serde_json::Value::as_object);
    let original = groups
        .into_iter()
        .flat_map(|groups| groups.iter())
        .flat_map(|(group, fields)| {
            fields.as_object().into_iter().flat_map(move |fields| {
                fields
                    .iter()
                    .map(move |(key, value)| (group.as_str(), key.as_str(), value))
            })
        });
    let aliases = METRIC_EQUIVALENTS.iter().flat_map(move |equivalents| {
        let value = equivalents
            .iter()
            .find_map(|&(group, key)| groups?.get(group)?.as_object()?.get(key));
        equivalents.iter().filter_map(move |&(group, key)| {
            let groups = groups?;
            if let Some(fields) = groups.get(group)
                && fields.as_object()?.contains_key(key)
            {
                return None;
            }
            Some((group, key, value?))
        })
    });
    original.chain(aliases)
}

/// Keep numbers and booleans (`true` → 1.0), dropping strings/nulls.
pub(super) fn numeric_metrics(source: Option<&serde_json::Value>) -> MetricMap {
    let mut result = MetricMap::new();
    for (group, key, value) in metric_fields(source) {
        if let Some(number) = value
            .as_f64()
            .or_else(|| value.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
        {
            result
                .entry(group.to_owned())
                .or_default()
                .insert(key.to_owned(), number);
        }
    }
    result
}

/// The tokens a `(library, name)` import pair contributes: the bare name and
/// the `lib!name` form, each subject to collimator's per-token `len >= 2`
/// gate.
pub(super) fn import_tokens_from_parts(lib: &str, name: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(2);
    if name.len() >= 2 {
        out.push(name.to_string());
    }
    let combined = if !lib.is_empty() && !name.is_empty() {
        format!("{lib}!{name}")
    } else if name.is_empty() {
        lib.to_string()
    } else {
        name.to_string()
    };
    if combined.len() >= 2 && !out.contains(&combined) {
        out.push(combined);
    }
    out
}

/// The whole of a finding that scoring reads: its id, confidence and
/// criticality ordinal.
#[derive(Debug, Clone, Copy)]
pub(super) struct FindingView<'a> {
    pub(super) id: &'a str,
    pub(super) conf: f64,
    pub(super) crit: u32,
}

pub(super) fn summarize_findings(findings: &[FindingView<'_>]) -> FindingSummary {
    summarize_findings_at_depth(findings, default_trait_path_depth())
}

pub(super) fn summarize_findings_at_depth(
    findings: &[FindingView<'_>],
    depth: usize,
) -> FindingSummary {
    let mut summary = FindingSummary::default();
    let mut notable_ids: HashSet<&str> = HashSet::new();
    let mut suspicious_ids: HashSet<&str> = HashSet::new();
    let mut hostile_ids: HashSet<&str> = HashSet::new();

    for finding in findings {
        let fid = finding.id;
        if fid.is_empty() {
            continue;
        }
        let conf = finding.conf;
        if conf < MIN_CONFIDENCE {
            continue;
        }

        summary.filtered_finding_count += 1;
        summary.finding_confidences.push(conf);
        let crit_ord = finding.crit;

        if crit_ord >= CRIT_NOTABLE {
            summary.notable_finding_count += 1;
            notable_ids.insert(fid);
        }
        if crit_ord >= CRIT_SUSPICIOUS {
            summary.suspicious_finding_count += 1;
            suspicious_ids.insert(fid);
        }
        if crit_ord >= CRIT_HOSTILE {
            summary.hostile_finding_count += 1;
            hostile_ids.insert(fid);
        }

        let top = fid.split('/').next().unwrap_or("");
        match top {
            "third_party" => {
                summary.third_party_count += 1;
                summary.third_party_max_crit = summary.third_party_max_crit.max(crit_ord);
                if fid.starts_with("third_party/yara") {
                    summary.has_yara = true;
                }
            }
            "well-known" => {
                summary.well_known_max_crit = summary.well_known_max_crit.max(crit_ord);
                if crit_ord >= CRIT_HOSTILE {
                    summary.well_known_hostile += 1;
                } else if crit_ord >= CRIT_SUSPICIOUS {
                    summary.well_known_suspicious += 1;
                }
            }
            _ => {}
        }

        // `entry(path.to_owned())` would allocate a fresh `String` on every
        // call regardless of whether the key already exists. Most findings
        // share path prefixes, so hitting an existing entry is the common
        // case — `get_mut` + conditional `insert` pays for the allocation
        // only on first insert.
        let score = if conf.is_finite() {
            f64::from(crit_ord) * f64::from(conf.clamp(0.0, 1.0) as f32)
        } else {
            0.0
        };
        for path in finding_paths(fid, depth) {
            match summary.path_scores.get_mut(path) {
                Some(v) => *v = v.max(score),
                None => {
                    summary.path_scores.insert(path.to_owned(), score);
                }
            }
            match summary.sample_paths.get_mut(path) {
                Some(v) => *v = (*v).max(crit_ord),
                None => {
                    summary.sample_paths.insert(path.to_owned(), crit_ord);
                }
            }
            match summary.path_confidences.get_mut(path) {
                Some(v) => *v = v.max(conf),
                None => {
                    summary.path_confidences.insert(path.to_owned(), conf);
                }
            }
        }
    }

    summary.unique_notable_ids = notable_ids.len();
    summary.unique_suspicious_ids = suspicious_ids.len();
    summary.unique_hostile_ids = hostile_ids.len();
    (
        summary.suspicious_category_breadth,
        summary.hostile_category_breadth,
    ) = category_breadth(&summary.sample_paths);
    summary
}

/// Distinct top-level categories reached at suspicious and at hostile
/// criticality.
pub(super) fn category_breadth(sample_paths: &HashMap<String, u32>) -> (usize, usize) {
    let mut susp_cats: HashSet<&str> = HashSet::new();
    let mut host_cats: HashSet<&str> = HashSet::new();
    for (path, &max_ord) in sample_paths {
        if max_ord >= CRIT_SUSPICIOUS {
            susp_cats.insert(path.split('/').next().unwrap_or(""));
        }
        if max_ord >= CRIT_HOSTILE {
            host_cats.insert(path.split('/').next().unwrap_or(""));
        }
    }
    (susp_cats.len(), host_cats.len())
}

pub(super) fn summarize_report_summaries(summaries: &[FileSummary]) -> FindingSummary {
    let mut combined = FindingSummary::default();

    // Deduplicate unique IDs across all files (matching Python's second pass
    // in _summarize_report_files which iterates all findings again). The
    // finding IDs borrow from `summaries` for the lifetime of this function,
    // so the sets hold `&str` — matching `summarize_findings`'s shape — and
    // avoid one `String` allocation per qualifying finding per criticality
    // tier.
    let mut notable_ids: HashSet<&str> = HashSet::new();
    let mut suspicious_ids: HashSet<&str> = HashSet::new();
    let mut hostile_ids: HashSet<&str> = HashSet::new();

    for s in summaries {
        let fs = &s.findings;
        combined.filtered_finding_count += fs.filtered_finding_count;
        combined.notable_finding_count += fs.notable_finding_count;
        combined.suspicious_finding_count += fs.suspicious_finding_count;
        combined.hostile_finding_count += fs.hostile_finding_count;
        combined.third_party_count += fs.third_party_count;
        combined.well_known_hostile += fs.well_known_hostile;
        combined.well_known_suspicious += fs.well_known_suspicious;
        combined.third_party_max_crit = combined.third_party_max_crit.max(fs.third_party_max_crit);
        combined.well_known_max_crit = combined.well_known_max_crit.max(fs.well_known_max_crit);
        combined.has_yara |= fs.has_yara;

        // Same motivation as `summarize_findings`: skip the `String` clone
        // when the path already aggregates into `combined`.
        for (path, max_ord) in &fs.sample_paths {
            match combined.sample_paths.get_mut(path) {
                Some(v) => *v = (*v).max(*max_ord),
                None => {
                    combined.sample_paths.insert(path.clone(), *max_ord);
                }
            }
        }
        for (path, &conf) in &fs.path_confidences {
            match combined.path_confidences.get_mut(path) {
                Some(v) => *v = v.max(conf),
                None => {
                    combined.path_confidences.insert(path.clone(), conf);
                }
            }
        }
        for (path, &score) in &fs.path_scores {
            match combined.path_scores.get_mut(path) {
                Some(v) => *v = v.max(score),
                None => {
                    combined.path_scores.insert(path.clone(), score);
                }
            }
        }
        combined
            .finding_confidences
            .extend(fs.finding_confidences.iter().copied());

        // Re-scan raw findings to deduplicate unique IDs across files.
        for finding in &s.raw_findings {
            let fid = finding.id.as_str();
            if fid.is_empty() {
                continue;
            }
            if finding.conf < MIN_CONFIDENCE {
                continue;
            }
            let crit = finding.crit;
            if crit >= CRIT_NOTABLE {
                notable_ids.insert(fid);
            }
            if crit >= CRIT_SUSPICIOUS {
                suspicious_ids.insert(fid);
            }
            if crit >= CRIT_HOSTILE {
                hostile_ids.insert(fid);
            }
        }
    }

    combined.unique_notable_ids = notable_ids.len();
    combined.unique_suspicious_ids = suspicious_ids.len();
    combined.unique_hostile_ids = hostile_ids.len();
    (
        combined.suspicious_category_breadth,
        combined.hostile_category_breadth,
    ) = category_breadth(&combined.sample_paths);
    combined
}
