//! Writers for the format, structure, gap, token and metric features.

#![expect(
    clippy::cast_possible_truncation,
    reason = "feature values are narrowed to the model's f32 input dtype"
)]

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// The format group of a cleave-reported file type; `None` for one filefacts
/// does not identify.
pub(super) fn format_group_for_type(file_type: &str) -> Option<filefacts::Group> {
    let group = filefacts::FileType::from_label(&file_type.trim().to_ascii_lowercase())?.group();
    (group != filefacts::Group::Unknown).then_some(group)
}

/// G6b: portable format-group hints derived only from cleave file types.
pub(super) fn write_format_hint_features(summaries: &[FileSummary], w: &mut FeatureWriter<'_>) {
    let total_files = summaries.len().max(1) as f32;
    let mut known_files = 0usize;
    let mut present_groups = HashSet::new();

    // Each file's format group, resolved once — the loop below is O(groups × files).
    let file_groups: Vec<Option<filefacts::Group>> = summaries
        .iter()
        .map(|s| format_group_for_type(&s.file_type))
        .collect();

    for (&group, names) in FORMAT_GROUPS.iter().zip(FORMAT_FEATURE_NAMES.iter()) {
        let mut group_count = 0usize;
        let mut suspicious_count = 0usize;
        let mut hostile_count = 0usize;

        for (s, file_group) in summaries.iter().zip(&file_groups) {
            if *file_group == Some(group) {
                group_count += 1;
                present_groups.insert(group);
                if s.findings.suspicious_finding_count > 0 {
                    suspicious_count += 1;
                }
                if s.findings.hostile_finding_count > 0 {
                    hostile_count += 1;
                }
            }
        }

        known_files += group_count;
        let group_denom = group_count.max(1) as f32;
        let [
            present,
            file_fraction,
            inner_fraction,
            suspicious_fraction,
            hostile_fraction,
        ] = names;
        w.set(present, f32::from(group_count > 0));
        w.set(file_fraction, group_count as f32 / total_files);
        // No compact file has a parent link, so no file is "inner".
        w.set(inner_fraction, 0.0);
        w.set(suspicious_fraction, suspicious_count as f32 / group_denom);
        w.set(hostile_fraction, hostile_count as f32 / group_denom);
    }

    w.set(
        "format:group_count_log",
        (present_groups.len() as f32).ln_1p(),
    );
    w.set(
        "format:mixed_script_binary",
        f32::from(
            present_groups.contains(&filefacts::Group::Scripts)
                && present_groups.contains(&filefacts::Group::Native),
        ),
    );
    w.set(
        "format:mixed_archive_script",
        f32::from(
            present_groups.contains(&filefacts::Group::Archives)
                && present_groups.contains(&filefacts::Group::Scripts),
        ),
    );
    w.set(
        "format:mixed_archive_binary",
        f32::from(
            present_groups.contains(&filefacts::Group::Archives)
                && present_groups.contains(&filefacts::Group::Native),
        ),
    );
    w.set(
        "format:unknown_file_fraction",
        (summaries.len().saturating_sub(known_files)) as f32 / total_files,
    );
}

/// Resolve each parent edge once; malformed cycles have no depth evidence.
pub(super) fn relative_depths(files: &[FileSummary]) -> Vec<u32> {
    let by_id: HashMap<_, _> = files
        .iter()
        .enumerate()
        .filter_map(|(i, f)| Some((f.id?, i)))
        .collect();
    let mut depths = vec![None; files.len()];
    for start in 0..files.len() {
        let mut chain = Vec::new();
        let mut seen = HashSet::new();
        let mut i = start;
        while depths[i].is_none() {
            if !seen.insert(i) {
                for &(j, _) in &chain {
                    depths[j] = Some(0);
                }
                break;
            }
            if files[i].depth > 0 {
                depths[i] = Some(files[i].depth);
                break;
            }
            let parent = files[i]
                .parent
                .and_then(|p| by_id.get(&p).copied())
                .filter(|&p| p != i);
            let Some(parent) = parent else {
                depths[i] = Some(0);
                break;
            };
            chain.push((i, parent));
            i = parent;
        }
        for &(j, parent) in chain.iter().rev() {
            if depths[j].is_none() {
                depths[j] = Some(depths[parent].unwrap_or(0u32).saturating_add(1));
            }
        }
    }
    let base = depths.iter().flatten().copied().min().unwrap_or(0);
    depths
        .into_iter()
        .map(|d| d.unwrap_or(0).saturating_sub(base))
        .collect()
}

/// G15: structure relative to the graded artifact (legacy bundles keep zeros).
pub(super) fn write_structural_extensions(
    summaries: &[FileSummary],
    combined: &FindingSummary,
    relative: bool,
    w: &mut FeatureWriter<'_>,
) {
    let binary_like = ["pe", "elf", "macho"];
    let source_types = ["javascript", "python", "typescript", "ruby", "php"];
    let text_exts = ["txt", "md", "json", "png", "jpg"];
    let code_types = ["javascript", "python", "pe", "elf", "macho"];

    let mut entropies = Vec::new();
    let mut code_entropies = Vec::new();
    let mut max_entropy = 0.0_f64;
    let mut hostile_files = 0;
    let mut total_loc: u64 = 0;
    let mut extension_mismatches = 0;
    let mut has_source_files = relative
        && summaries
            .iter()
            .any(|s| source_types.contains(&s.file_type.as_str()));
    let mut has_foreign_binaries = false;

    for s in summaries {
        if source_types.contains(&s.file_type.as_str()) {
            has_source_files = true;
        }
        if binary_like.contains(&s.file_type.as_str()) && has_source_files {
            has_foreign_binaries = true;
        }

        let lines = s
            .metrics
            .get("text")
            .and_then(|t| t.get("total_lines"))
            .copied()
            .unwrap_or(0.0);
        #[expect(clippy::cast_sign_loss, reason = "a line count is never negative")]
        let lines = lines as u64;
        total_loc += lines;

        if let Some((_, ext)) = s.path.rsplit_once('.')
            && binary_like.contains(&s.file_type.as_str())
            && text_exts.contains(&ext.to_ascii_lowercase().as_str())
        {
            extension_mismatches += 1;
        }

        if s.overall_entropy > 0.0 {
            entropies.push(s.overall_entropy);
        }
        max_entropy = max_entropy.max(s.overall_entropy);

        if s.findings.hostile_finding_count > 0 {
            hostile_files += 1;
        }

        if s.overall_entropy > 0.0 && code_types.contains(&s.file_type.as_str()) {
            code_entropies.push(s.overall_entropy);
        }
    }

    w.set(
        "struct:packaged_capability",
        (combined.sample_paths.len() as f64 * max_entropy) as f32,
    );
    let depths = if relative {
        relative_depths(summaries)
    } else {
        vec![0; summaries.len()]
    };
    w.set(
        "struct:max_nesting_depth_log",
        f64::from(*depths.iter().max().unwrap_or(&0)).ln_1p() as f32,
    );
    w.set(
        "struct:inner_file_ratio",
        depths.iter().filter(|&&d| d > 0).count() as f32 / summaries.len().max(1) as f32,
    );
    if relative {
        w.set(
            "agg:hostile_depth_weight",
            summaries
                .iter()
                .zip(&depths)
                .map(|(s, &d)| f64::from(s.findings.hostile_finding_count) * f64::from(d))
                .sum::<f64>() as f32,
        );
    }
    let mean_entropy = if entropies.is_empty() {
        0.0
    } else {
        entropies.iter().sum::<f64>() / entropies.len() as f64
    };
    if entropies.len() > 1 {
        let mean = mean_entropy;
        let var =
            entropies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / entropies.len() as f64;
        w.set("struct:entropy_std_dev", var.sqrt() as f32);
        let mx = entropies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        w.set("struct:entropy_max_diff", (mx - mean) as f32);
    }
    w.set(
        "struct:air_gap_signal",
        f32::from(
            hostile_files > 0
                && !summaries
                    .iter()
                    .zip(&depths)
                    .any(|(s, &d)| d > 0 && s.findings.hostile_finding_count > 0),
        ),
    );
    if !code_entropies.is_empty() {
        let max_code_ent = code_entropies
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        w.set(
            "struct:code_entropy_spike",
            (max_code_ent - mean_entropy) as f32,
        );
    }
    w.set(
        "struct:foreign_binary_signal",
        f32::from(has_foreign_binaries),
    );
    w.set(
        "struct:extension_mismatch_signal",
        extension_mismatches as f32,
    );
    if total_loc > 0 {
        w.set(
            "struct:hostile_finding_density",
            (hostile_files as f32 * 1000.0) / total_loc as f32,
        );
    }
}

/// G19: a capability imported but never exercised (no notable behavior).
pub(super) fn write_logic_gap_features(
    summary: &FindingSummary,
    summaries: &[FileSummary],
    w: &mut FeatureWriter<'_>,
) {
    let logic_gaps: &[(&str, &[&str], &[&str])] = &[
        (
            "crypto",
            &[
                "cryptography",
                "Crypto",
                "hashlib",
                "CryptAcquireContext",
                "BCryptOpenAlgorithmProvider",
            ],
            &["micro-behaviors/crypto", "metadata/encoded-payload"],
        ),
        (
            "network",
            &[
                "socket", "urllib", "requests", "http", "curl", "wininet", "winhttp",
            ],
            &["micro-behaviors/network", "objectives/command-and-control"],
        ),
        (
            "process",
            &[
                "subprocess",
                "os.spawn",
                "os.system",
                "CreateProcess",
                "ShellExecute",
                "posix_spawn",
            ],
            &["micro-behaviors/process/create", "objectives/execution"],
        ),
    ];

    let mut all_imports: HashSet<&str> = HashSet::new();
    for s in summaries {
        for imp in &s.imports {
            all_imports.insert(imp.as_str());
        }
    }

    for (target_cat, name) in LOGIC_GAP_CATEGORIES.iter().zip(GAP_FEATURE_NAMES.iter()) {
        if let Some((_, imports_set, traits_set)) =
            logic_gaps.iter().find(|(c, _, _)| c == target_cat)
        {
            let has_import = imports_set.iter().any(|imp| all_imports.contains(*imp));
            let has_behavior = summary.sample_paths.iter().any(|(path, &max_ord)| {
                max_ord >= CRIT_NOTABLE && traits_set.iter().any(|t| path.starts_with(t))
            });
            if has_import && !has_behavior {
                w.set(name, 1.0);
            }
        }
    }
}

/// G22: a risky behavior with no documentation explaining it.
pub(super) fn write_intent_gap_features(summary: &FindingSummary, w: &mut FeatureWriter<'_>) {
    let intent_signal = summary
        .sample_paths
        .contains_key("metadata/package/documentation")
        || summary.sample_paths.contains_key("metadata/package/help");
    let risky: &[(&str, &[&str])] = &[
        (
            "network",
            &["objectives/network", "micro-behaviors/network"],
        ),
        (
            "filesystem",
            &["objectives/persistence", "micro-behaviors/filesystem"],
        ),
        (
            "execution",
            &["objectives/execution", "micro-behaviors/process/create"],
        ),
        ("crypto", &["objectives/crypto", "micro-behaviors/crypto"]),
    ];
    for (target_cat, name) in INTENT_GAP_CATEGORIES
        .iter()
        .zip(INTENT_GAP_FEATURE_NAMES.iter())
    {
        let traits = risky
            .iter()
            .find_map(|&(c, t)| (c == *target_cat).then_some(t))
            .unwrap_or_default();
        let has_behavior = summary.sample_paths.iter().any(|(path, &max_ord)| {
            max_ord >= CRIT_SUSPICIOUS && traits.iter().any(|t| path.starts_with(t))
        });
        if has_behavior && !intent_signal {
            w.set(name, 1.0);
        }
    }
}

/// G23: negative space — a file type present without a trait it should have.
pub(super) fn write_negative_space_features(
    summary: &FindingSummary,
    summaries: &[FileSummary],
    w: &mut FeatureWriter<'_>,
) {
    let mut present_types: HashSet<&str> = HashSet::new();
    for s in summaries {
        if !s.file_type.is_empty() {
            present_types.insert(s.file_type.as_str());
        }
    }
    let expected = EXPECTED_GHOSTS
        .iter()
        .flat_map(|&(ftype, traits)| traits.iter().map(move |t| (ftype, *t)));
    for ((ftype, trait_path), name) in expected.zip(MISSING_FEATURE_NAMES.iter()) {
        if present_types.contains(ftype) && !summary.sample_paths.contains_key(trait_path) {
            w.set(name, 1.0);
        }
    }
}

pub(super) fn write_external_summary_features(summary: &FindingSummary, w: &mut FeatureWriter<'_>) {
    w.set(
        "ext:third_party_max_crit",
        summary.third_party_max_crit as f32,
    );
    w.set(
        "ext:third_party_count",
        (summary.third_party_count as f32 + 1.0).ln(),
    );
    w.set(
        "ext:well_known_max_crit",
        summary.well_known_max_crit as f32,
    );
    w.set(
        "ext:well_known_hostile_count",
        summary.well_known_hostile as f32,
    );
    w.set(
        "ext:well_known_suspicious_count",
        summary.well_known_suspicious as f32,
    );
    w.set("ext:has_yara_match", f32::from(summary.has_yara));
}

pub(super) type MetricMap = HashMap<String, HashMap<String, f64>>;

pub(super) fn merge_metric_summaries(summaries: &[FileSummary]) -> MetricMap {
    let mut merged: MetricMap = HashMap::new();
    for s in summaries {
        for (group, fields) in &s.metrics {
            let group_map = merged.entry(group.clone()).or_default();
            for (fname, &val) in fields {
                let e = group_map.entry(fname.clone()).or_insert(f64::NEG_INFINITY);
                *e = f64::max(*e, val);
            }
        }
    }
    merged
}

/// G7: structure.
pub(super) fn write_structural_features(
    w: &mut FeatureWriter<'_>,
    summaries: &[FileSummary],
    filtered_finding_count: u32,
) {
    let binary_like = ["pe", "elf", "macho"];
    let mut any_tiny_binary = false;
    let mut max_entropy = 0.0_f64;
    let mut suspicious_files = 0;
    let mut hostile_files = 0;

    for s in summaries {
        if binary_like.contains(&s.file_type.as_str()) && s.size_bytes < 20_000.0 {
            any_tiny_binary = true;
        }
        max_entropy = max_entropy.max(s.overall_entropy);
        if s.findings.suspicious_finding_count > 0 {
            suspicious_files += 1;
        }
        if s.findings.hostile_finding_count > 0 {
            hostile_files += 1;
        }
    }

    w.set("struct:tiny_executable", f32::from(any_tiny_binary));
    // Collimator sets this when every file with an imports key imports
    // nothing; the compact schema has no such key, so it is always 0.
    w.set("struct:no_imports", 0.0);
    w.set(
        "struct:zero_findings",
        f32::from(filtered_finding_count == 0),
    );
    w.set(
        "struct:finding_count_log",
        (filtered_finding_count as f32 + 1.0).ln(),
    );
    let file_count = summaries.len() as f32;
    w.set("struct:file_count_log", (file_count + 1.0).ln());
    w.set(
        "struct:inner_file_count_log",
        ((file_count - 1.0).max(0.0) + 1.0).ln(),
    );
    w.set(
        "struct:stealth_potential",
        f32::from(filtered_finding_count < 5 && max_entropy > 6.5),
    );
    let denom = summaries.len().max(1) as f32;
    w.set(
        "struct:suspicious_file_fraction",
        suspicious_files as f32 / denom,
    );
    w.set("struct:hostile_file_fraction", hostile_files as f32 / denom);
    w.set(
        "struct:suspicious_file_count_log",
        (suspicious_files as f32).ln_1p(),
    );
    w.set(
        "struct:hostile_file_count_log",
        (hostile_files as f32).ln_1p(),
    );
}

pub(super) fn finding_paths(finding_id: &str, depth: usize) -> FindingPaths<'_> {
    let base = trait_path(finding_id);
    FindingPaths {
        base,
        slashes: base.match_indices('/'),
        remaining: depth,
        finished: false,
    }
}

pub(super) struct FindingPaths<'a> {
    pub(super) base: &'a str,
    pub(super) slashes: std::str::MatchIndices<'a, char>,
    pub(super) remaining: usize,
    pub(super) finished: bool,
}

impl<'a> Iterator for FindingPaths<'a> {
    type Item = &'a str;
    #[expect(
        clippy::string_slice,
        reason = "ASCII slash offsets are UTF-8 boundaries"
    )]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 || self.finished {
            return None;
        }
        self.remaining -= 1;
        let end = self.slashes.next().map_or_else(
            || {
                self.finished = true;
                self.base.len()
            },
            |(i, _)| i,
        );
        Some(&self.base[..end])
    }
}

// ============================================================================
// kv: / symbol: / textenc: / derived metric helpers — ported from
// collimator/src/collimator/features.py. Each helper mirrors the Python
// behavior exactly so the feature vectors stay bit-identical with the
// training-time extraction.
// ============================================================================

/// Normalize a value to a bounded vocab token (collapse whitespace, truncate).
/// Mirrors `_normalize_vocab_token` in collimator.
pub(super) fn normalize_vocab_token(value: &str, max_len: usize) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    // Collapse runs of whitespace to single spaces ("a   b\nc" → "a b c").
    let mut out = String::with_capacity(trimmed.len());
    let mut prev_ws = false;
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            if !prev_ws && !out.is_empty() {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    if out.chars().count() > max_len {
        out = out.chars().take(max_len).collect();
    }
    out
}

pub(super) fn count_bucket(n: usize) -> &'static str {
    match n {
        0 => "0",
        1 => "1",
        2..=3 => "2_3",
        4..=7 => "4_7",
        8..=15 => "8_15",
        16..=31 => "16_31",
        32..=63 => "32_63",
        64..=127 => "64_127",
        _ => "128_plus",
    }
}

pub(super) fn number_bucket(n: f64) -> String {
    if !n.is_finite() {
        return "nonfinite".into();
    }
    if n == 0.0 {
        return "zero".into();
    }
    let magnitude = n.abs();
    let bucket = if magnitude < 1.0 {
        "lt1"
    } else if magnitude < 10.0 {
        "1_9"
    } else if magnitude < 100.0 {
        "10_99"
    } else if magnitude < 1000.0 {
        "100_999"
    } else if magnitude < 10000.0 {
        "1k_9k"
    } else if magnitude < 1000000.0 {
        "10k_999k"
    } else {
        "1m_plus"
    };
    format!("{}_{bucket}", if n < 0.0 { "neg" } else { "pos" })
}

/// Emit only bounded categorical/shape tokens; the model selects its slots.
pub(super) fn kv_tokens_with_options(
    metrics: Option<&serde_json::Value>,
    shape: bool,
    split: bool,
) -> Vec<String> {
    use serde_json::Value;
    let mut tokens = Vec::new();
    for (group, key, value) in metric_fields(metrics) {
        let base = format!("kv:{group}.{key}");
        if shape {
            tokens.push(format!("{base}:exists"));
        }
        match value {
            Value::Bool(b) => {
                tokens.push(format!("{base}={b}"));
                if shape {
                    tokens.push(format!("{base}:bool={b}"));
                }
            }
            Value::String(s) => {
                let val = normalize_vocab_token(s, 80);
                if !val.is_empty() {
                    tokens.push(format!("{base}={val}"));
                }
                if shape {
                    tokens.push(format!("{base}:strlen={}", count_bucket(s.chars().count())));
                    tokens.push(format!(
                        "{base}:{}",
                        if s.is_empty() { "empty" } else { "nonempty" }
                    ));
                }
                if split {
                    for part in s.split(|c: char| c.is_whitespace() || ",;:/\\|".contains(c)) {
                        let part = normalize_vocab_token(part, 64);
                        if !part.is_empty() {
                            tokens.push(format!("{base}=part:{part}"));
                        }
                    }
                }
            }
            Value::Number(n) if shape => {
                if let Some(n) = n.as_f64() {
                    tokens.push(format!("{base}:number={}", number_bucket(n)));
                }
            }
            Value::Array(items) if shape => {
                tokens.push(format!("{base}:len={}", count_bucket(items.len())));
                if items.is_empty() {
                    tokens.push(format!("{base}:empty"));
                }
            }
            Value::Object(items) if shape => {
                tokens.push(format!("{base}:len={}", count_bucket(items.len())));
                if items.is_empty() {
                    tokens.push(format!("{base}:empty"));
                }
            }
            Value::Null if shape => tokens.push(format!("{base}:empty")),
            _ => {}
        }
    }
    tokens
}

#[cfg(test)]
pub(super) fn kv_tokens(metrics: Option<&serde_json::Value>) -> Vec<String> {
    kv_tokens_with_options(metrics, false, false)
}

/// A file's normalized symbols, sorted and unique. Mirrors `_file_symbols`
/// in collimator: an import contributes its bare name and its `lib!name`
/// form (or whichever half it has); exports, function names, AST call
/// targets and member chains contribute their name. Tokens shorter than two
/// characters are dropped.
pub(super) fn file_symbols(facts: &cleave::types::CompactFacts) -> Vec<String> {
    let mut out = HashSet::new();
    let mut insert = |raw: &str| {
        let token = normalize_vocab_token(raw, 96);
        if token.chars().count() >= 2 {
            out.insert(token);
        }
    };
    for imp in &facts.imports {
        let (lib, name) = (imp.library.as_str(), imp.name.as_str());
        insert(name);
        match (lib.is_empty(), name.is_empty()) {
            (false, false) => insert(&format!("{lib}!{name}")),
            (false, true) => insert(lib),
            (true, _) => {}
        }
    }
    for name in facts
        .exports
        .iter()
        .map(|e| e.name.as_str())
        .chain(facts.functions.iter().map(|f| f.name.as_str()))
        .chain(facts.targets.iter().map(String::as_str))
        .chain(facts.members.iter().map(String::as_str))
    {
        insert(name);
    }
    let mut symbols: Vec<String> = out.into_iter().collect();
    symbols.sort();
    symbols
}

/// Per-file caps from collimator (`_SYMBOL_BIGRAM_CAP`/`_SYMBOL_TRIGRAM_CAP`).
pub(super) const SYMBOL_BIGRAM_CAP: usize = 64;
pub(super) const SYMBOL_TRIGRAM_CAP: usize = 24;

/// Emit symbol:*, symbol_bi:*, symbol_tri:* for a set of summaries.
pub(super) fn write_symbol_features(
    summaries: &[FileSummary],
    w: &mut FeatureWriter<'_>,
    emit_bigrams: bool,
    emit_trigrams: bool,
) {
    let mut key = String::new();
    for s in summaries {
        let sorted = s.symbols.as_slice();
        for sym in sorted {
            key.clear();
            let _ = write!(key, "symbol:{sym}");
            w.set_token(&key, 1.0);
        }
        if emit_bigrams {
            let cap = sorted.len().min(SYMBOL_BIGRAM_CAP);
            for i in 0..cap {
                for j in (i + 1)..cap {
                    key.clear();
                    let _ = write!(key, "symbol_bi:{}||{}", sorted[i], sorted[j]);
                    w.set_token(&key, 1.0);
                }
            }
        }
        if emit_trigrams {
            let cap = sorted.len().min(SYMBOL_TRIGRAM_CAP);
            for i in 0..cap {
                for j in (i + 1)..cap {
                    for k in (j + 1)..cap {
                        key.clear();
                        let _ = write!(
                            key,
                            "symbol_tri:{}||{}||{}",
                            sorted[i], sorted[j], sorted[k]
                        );
                        w.set_token(&key, 1.0);
                    }
                }
            }
        }
    }
}

pub(super) fn known_metric_getter(name: &str) -> bool {
    KEY_METRICS
        .iter()
        .any(|(g, f, _)| name == format!("{g}_{f}"))
        || matches!(
            name,
            "pe_is_dotnet"
                | "pe_linker_major_version"
                | "pe_subsystem"
                | "pe_checksum_missing"
                | "pe_entry_section_nontext"
                | "pe_year_distance"
                | "pe_year_pre_2000"
                | "pe_year_future"
                | "text_mixed_indent"
                | "binary_has_overlay"
                | "binary_nonstandard_section_name_count"
                | "derived_string_per_function"
                | "derived_imports_per_dependency"
                | "derived_wide_string_ratio"
                | "derived_imports_per_kb"
                | "derived_sections_per_kb"
                | "derived_strings_per_kb"
                | "text_lines_over_200"
                | "text_lines_over_500"
                | "text_lines_over_1000"
                | "text_lines_in_200_499"
                | "text_lines_in_500_999"
        )
}

pub(super) fn derived_metric(
    name: &str,
    metrics: &MetricMap,
    files: &[FileSummary],
    year: u32,
) -> f64 {
    let get = |group: &str, field: &str| {
        metrics
            .get(group)
            .and_then(|g| g.get(field))
            .copied()
            .unwrap_or(0.0)
    };
    let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
    match name {
        "pe_year_distance" => {
            if get("pe", "timestamp_year") > 0.0 {
                (f64::from(year) - get("pe", "timestamp_year")).abs()
            } else {
                0.0
            }
        }
        "pe_entry_section_nontext" => f64::from(files.iter().any(|f| f.entry_section_nontext)),
        "pe_year_pre_2000" => {
            f64::from(get("pe", "timestamp_year") > 0.0 && get("pe", "timestamp_year") < 2000.0)
        }
        "pe_year_future" => f64::from(get("pe", "timestamp_year") > 2030.0),
        "derived_string_per_function" => ratio(
            get("binary", "string_count"),
            get("binary", "function_count"),
        ),
        "derived_imports_per_dependency" => ratio(
            get("binary", "import_count"),
            get("binary", "dependency_count"),
        ),
        "derived_wide_string_ratio" => ratio(
            get("binary", "wide_string_count"),
            get("binary", "string_count"),
        ),
        "derived_imports_per_kb" => ratio(
            get("binary", "import_count"),
            get("binary", "file_size") / 1024.0,
        ),
        "derived_sections_per_kb" => ratio(
            get("binary", "section_count"),
            get("binary", "file_size") / 1024.0,
        ),
        "derived_strings_per_kb" => ratio(
            get("binary", "string_count"),
            get("binary", "file_size") / 1024.0,
        ),
        "text_lines_in_200_499" => {
            (get("text", "lines_over_200") - get("text", "lines_over_500")).max(0.0)
        }
        "text_lines_in_500_999" => {
            (get("text", "lines_over_500") - get("text", "lines_over_1000")).max(0.0)
        }
        _ => {
            let (g, f) = name.split_once('_').unwrap_or(("", ""));
            let value = get(g, f);
            if matches!(
                name,
                "pe_is_dotnet" | "pe_checksum_missing" | "text_mixed_indent" | "binary_has_overlay"
            ) {
                f64::from(value != 0.0)
            } else {
                value
            }
        }
    }
}

/// Cross-metric derived ratios. Mirrors collimator `_BATCH1_RATIOS`.
pub(super) fn write_derived_metric_features(merged_metrics: &MetricMap, w: &mut FeatureWriter<'_>) {
    let get = |group: &str, field: &str| -> f64 {
        merged_metrics
            .get(group)
            .and_then(|g| g.get(field))
            .copied()
            .unwrap_or(0.0)
    };
    let ratio = |num: f64, denom: f64| -> f64 { if denom == 0.0 { 0.0 } else { num / denom } };
    let string_count = get("binary", "string_count");
    let function_count = get("binary", "function_count");
    let import_count = get("binary", "import_count");
    let dependency_count = get("binary", "dependency_count");
    let wide_string_count = get("binary", "wide_string_count");
    w.set(
        "metrics:derived_string_per_function",
        ratio(string_count, function_count) as f32,
    );
    w.set(
        "metrics:derived_imports_per_dependency",
        ratio(import_count, dependency_count) as f32,
    );
    w.set(
        "metrics:derived_wide_string_ratio",
        ratio(wide_string_count, string_count) as f32,
    );
}

/// `struct:silent_packer_signal` — large file size / few findings → packer.
/// Mirrors collimator's gated computation (Exp 43).
pub(super) fn write_silent_packer_signal(
    summaries: &[FileSummary],
    filtered_finding_count: u32,
    w: &mut FeatureWriter<'_>,
) {
    let total_size: f64 = summaries.iter().map(|s| s.size_bytes).sum();
    let size_mb = total_size / (1024.0 * 1024.0);
    let denom = f64::from(filtered_finding_count + 1);
    w.set(
        "struct:silent_packer_signal",
        (size_mb.ln_1p() / denom) as f32,
    );
}

/// Aggregate-level suspicious n-gram co-occurrence counts. Mirrors
/// collimator's `include_suspicious_trigrams` branch.
pub(super) fn write_suspicious_ngram_counts(combined: &FindingSummary, w: &mut FeatureWriter<'_>) {
    let n = combined
        .sample_paths
        .values()
        .filter(|&&mo| mo >= 4)
        .count();
    let mut n_bi: u64 = 0;
    let mut n_tri: u64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            n_bi += 1;
            // Python caps the third-element span to 20 to bound O(n^3).
            let k_end = n.min(j + 20);
            n_tri += (k_end.saturating_sub(j + 1)) as u64;
        }
    }
    w.set("agg:suspicious_bigram_count", (n_bi as f64).ln_1p() as f32);
    w.set(
        "agg:suspicious_trigram_count",
        (n_tri as f64).ln_1p() as f32,
    );
}
