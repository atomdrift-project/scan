//! The fixed feature names this extractor writes, and the metric renames
//! that keep older specs readable.

#[allow(
    clippy::wildcard_imports,
    reason = "one module split across files: each part shares the parent's namespace"
)]
use super::*;

/// Group 6b's format groups: filefacts' groups in label order, the order
/// collimator lays them out in. `unknown` is left out; a file with no
/// identified type counts toward `format:unknown_file_fraction` instead.
pub(super) static FORMAT_GROUPS: LazyLock<Vec<filefacts::Group>> = LazyLock::new(|| {
    let mut groups: Vec<_> = filefacts::Group::ALL
        .iter()
        .copied()
        .filter(|&group| group != filefacts::Group::Unknown)
        .collect();
    groups.sort_by_key(|group| group.label());
    groups
});

/// Expected ghosts (v16 group 23).
pub(super) const EXPECTED_GHOSTS: &[(&str, &[&str])] = &[
    (
        "elf",
        &[
            "metadata/binary/layout",
            "metadata/binary/metrics",
            "metadata/binary/symbols",
            "metadata/binary/linking",
        ],
    ),
    (
        "javascript",
        &[
            // Names no trait, so this feature is 1 for every JavaScript file.
            // Deployed specs carry it by this name; replace it only at a
            // retrain, in lockstep with collimator.
            "micro-behaviors/javascript/async",
            "metadata/package/versioning",
        ],
    ),
    (
        "pe",
        &[
            "metadata/binary/layout",
            "metadata/binary/metrics",
            "metadata/binary/resource",
            "metadata/binary/symbols",
            "metadata/binary/linking",
        ],
    ),
];

/// Group 3: report-level aggregates.
pub(super) const AGG_FEATURES: &[&str] = &[
    "agg:max_crit",
    "agg:category_breadth",
    "agg:path_breadth_any",
    "agg:total_active_paths",
    "agg:suspicious_concentration",
    "agg:hostile_concentration",
    "agg:escalation_rate",
    "agg:notable_only_fraction",
    "agg:notable_findings_log",
    "agg:suspicious_findings_log",
    "agg:hostile_findings_log",
    "agg:notable_finding_ratio",
    "agg:suspicious_finding_ratio",
    "agg:hostile_finding_ratio",
    "agg:unique_suspicious_ids_log",
    "agg:unique_hostile_ids_log",
    // The `top1` block summarizes the single riskiest file; see
    // `topk_file_risk_features_from_summaries`.
    "agg:top1_file_suspicious_ratio_sum",
    "agg:top1_file_hostile_ratio_sum",
    "agg:top1_file_suspicious_findings_log",
    "agg:top1_file_hostile_findings_log",
    "agg:suspicious_category_breadth",
    "agg:hostile_category_breadth",
    "agg:suspicious_category_density",
    "agg:hostile_category_density",
    "agg:suspicious_findings_per_kb",
    "agg:hostile_findings_per_kb",
    "agg:suspicious_categories_per_kb",
    "agg:hostile_categories_per_kb",
    "agg:top1_file_suspicious_density_sum",
    "agg:top1_file_hostile_density_sum",
    "agg:top1_file_suspicious_category_breadth_sum",
    "agg:top1_file_hostile_category_breadth_sum",
    // Size-invariant crit-tier severity fractions (collimator's
    // include_severity_fractions group).
    "agg:confidence_mean",
    "agg:confidence_std",
    "agg:confidence_skew",
    "agg:confidence_kurtosis",
    "agg:trait_id_lexical_distance",
    "agg:docobf_obfuscation_count",
    "agg:docobf_eval_count",
    "agg:docobf_lure_count",
    "agg:docobf_total_count",
    "agg:docobf_has_any",
    "agg:crit3_finding_fraction",
    "agg:crit4_finding_fraction",
    "agg:hostile_finding_fraction",
    "agg:severe_to_mundane_ratio",
    "agg:crit4_present",
    "agg:hostile_escalation_rate",
    "agg:hostile_share_of_suspicious",
    "agg:suspicious_finding_escalation_rate",
    "agg:hostile_finding_escalation_rate",
    "agg:hostile_share_of_suspicious_findings",
    "agg:hostile_weighted_density",
    "agg:top1_file_hostile_weighted_density_sum",
    "agg:suspicious_id_repeat_ratio",
    "agg:hostile_id_repeat_ratio",
    "agg:suspicious_category_repeat_ratio",
    "agg:hostile_category_repeat_ratio",
    "agg:file_hostile_fraction",
    "agg:file_suspicious_fraction",
    "agg:file_notable_fraction",
    "agg:file_hostile_count_log",
    "agg:file_suspicious_count_log",
    "agg:file_notable_count_log",
    "agg:hostile_depth_weight",
    "agg:suspicious_2level_breadth",
    "agg:hostile_2level_breadth",
    "agg:objectives_breadth",
    // From here to `static_signed_file_fraction`, collimator computes these
    // and scan does not: they are declared so a spec carrying them is not
    // reported as degraded, and extract as zero.
    "agg:kill_chain_span",
    "agg:objective_micro_ratio",
    "agg:avg_finding_depth",
    "agg:objective_hostile_density",
    "agg:static_file_bytes_log",
    "agg:static_import_count_log",
    "agg:static_export_count_log",
    "agg:static_dependency_count_log",
    "agg:static_string_count_log",
    "agg:static_wide_string_ratio",
    "agg:static_max_string_length_log",
    "agg:static_string_entropy_max",
    "agg:static_text_lines_log",
    "agg:static_function_count_log",
    "agg:static_code_bytes_log",
    "agg:static_code_to_data_ratio_max",
    "agg:static_wx_units_log",
    "agg:static_writable_unit_ratio",
    "agg:static_executable_unit_ratio",
    "agg:static_nonstandard_unit_names_log",
    "agg:static_largest_unit_ratio_max",
    "agg:static_resource_ratio_max",
    "agg:static_signed_file_fraction",
    "agg:attack_technique_count",
    "agg:attack_tactic_count",
    "agg:mbc_behavior_count",
    "agg:has_attack_and_objective",
    // ATT&CK / MBC co-occurrence aggregates (log1p of unordered combinations
    // among distinct technique / behavior codes seen in raw_findings).
    "agg:attack_bigram_count",
    "agg:attack_trigram_count",
    "agg:mbc_bigram_count",
    // Objective path co-occurrence aggregates (log1p of unordered combinations
    // among distinct `objectives/*` and `well-known/*` paths seen in
    // sample_paths). Trigram is bounded with a per-pair cap of 20 inner
    // elements to avoid O(n^3) explosion on samples with many objectives.
    "agg:objective_bigram_count",
    "agg:objective_trigram_count",
];

/// Group 4: external-signal summary.
pub(super) const EXT_FEATURES: &[&str] = &[
    "ext:third_party_max_crit",
    "ext:third_party_count",
    "ext:well_known_max_crit",
    "ext:well_known_hostile_count",
    "ext:well_known_suspicious_count",
    "ext:has_yara_match",
];

/// Group 6b: portable format-group hints, after the per-group block.
pub(super) const FORMAT_SUMMARY_FEATURES: &[&str] = &[
    "format:group_count_log",
    "format:mixed_script_binary",
    "format:mixed_archive_script",
    "format:mixed_archive_binary",
    "format:unknown_file_fraction",
];

/// Group 7: structure.
pub(super) const STRUCT_FEATURES: &[&str] = &[
    "struct:tiny_executable",
    "struct:no_imports",
    "struct:zero_findings",
    "struct:finding_count_log",
    "struct:file_count_log",
    "struct:inner_file_count_log",
    "struct:stealth_potential",
    "struct:suspicious_file_fraction",
    "struct:hostile_file_fraction",
    "struct:suspicious_file_count_log",
    "struct:hostile_file_count_log",
];

/// Group 9: formula.
pub(super) const FORMULA_FEATURES: &[&str] = &[
    "formula:skeleton_len",
    "formula:unique_elements",
    "formula:complexity_ratio",
];

/// Group 10: score (the `inter:` family follows it).
pub(super) const SCORE_FEATURES: &[&str] = &["score:hopper_score", "score:density"];

/// Group 15: structural extensions.
pub(super) const STRUCT_EXTENSION_FEATURES: &[&str] = &[
    "struct:packaged_capability",
    "struct:mtime_range_hours",
    "struct:mtime_std_dev_hours",
    "struct:max_nesting_depth_log",
    "struct:inner_file_ratio",
    "struct:entropy_std_dev",
    "struct:entropy_max_diff",
    "struct:air_gap_signal",
    "struct:anachronistic_injection",
    "struct:code_entropy_spike",
    "struct:foreign_binary_signal",
    "struct:extension_mismatch_signal",
    "struct:hostile_finding_density",
];

/// Features only some route specs carry (e.g. filetypes/makefile): suspicious
/// co-occurrence counts, cross-metric ratios, and the silent-packer signal.
pub(super) const TAIL_FEATURES: &[&str] = &[
    "agg:suspicious_bigram_count",
    "agg:suspicious_trigram_count",
    "metrics:derived_string_per_function",
    "metrics:derived_imports_per_dependency",
    "metrics:derived_wide_string_ratio",
    "struct:silent_packer_signal",
];

/// Group 24: textenc — collimator's `_apply_text_encoding_features` ratios
/// over a file's strings. The compact report carries no strings, so these
/// always extract as zero; they are declared so a spec carrying them is not
/// reported as degraded.
pub(super) const TEXTENC_FEATURES: &[&str] = &[
    "textenc:string_count_log",
    "textenc:avg_len_log",
    "textenc:max_len_log",
    "textenc:base64ish_ratio",
    "textenc:hexish_ratio",
    "textenc:urlish_ratio",
    "textenc:pathish_ratio",
    "textenc:unicode_escape_ratio",
    "textenc:wide_ratio",
    "textenc:high_entropy_ratio",
    "textenc:long_token_ratio",
    "textenc:short_junk_ratio",
];

/// Group 5: `metrics:<group>_<field>` for every [`KEY_METRICS`] entry.
pub(super) fn metric_base_feature_names() -> impl Iterator<Item = String> {
    KEY_METRICS
        .iter()
        .map(|&(group, field, _)| format!("metrics:{group}_{field}"))
}

/// Group 6b: `format:<group>` and its four fractions, per [`FORMAT_GROUPS`] entry.
pub(super) static FORMAT_FEATURE_NAMES: LazyLock<Vec<[String; 5]>> = LazyLock::new(|| {
    FORMAT_GROUPS
        .iter()
        .map(|group| {
            let group = group.label();
            [
                format!("format:{group}"),
                format!("format:{group}_file_fraction"),
                format!("format:{group}_inner_fraction"),
                format!("format:{group}_suspicious_fraction"),
                format!("format:{group}_hostile_fraction"),
            ]
        })
        .collect()
});

/// Group 19: `gap:<category>`, per [`LOGIC_GAP_CATEGORIES`] entry.
pub(super) static GAP_FEATURE_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    LOGIC_GAP_CATEGORIES
        .iter()
        .map(|cat| format!("gap:{cat}"))
        .collect()
});

/// Group 22: `intent_gap:<category>`, per [`INTENT_GAP_CATEGORIES`] entry.
pub(super) static INTENT_GAP_FEATURE_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    INTENT_GAP_CATEGORIES
        .iter()
        .map(|cat| format!("intent_gap:{cat}"))
        .collect()
});

/// Group 23: `missing:<type>*<trait>`, in [`EXPECTED_GHOSTS`] order.
pub(super) static MISSING_FEATURE_NAMES: LazyLock<Vec<String>> = LazyLock::new(|| {
    EXPECTED_GHOSTS
        .iter()
        .flat_map(|&(ftype, traits)| traits.iter().map(move |t| format!("missing:{ftype}*{t}")))
        .collect()
});

/// Every fixed-name feature: the layout minus its vocabulary-driven families.
/// The single list [`FeatureWriter::set`] checks names against.
pub(super) static FIXED_FEATURE_NAMES: LazyLock<HashSet<String>> = LazyLock::new(|| {
    [
        AGG_FEATURES,
        EXT_FEATURES,
        FORMAT_SUMMARY_FEATURES,
        STRUCT_FEATURES,
        FORMULA_FEATURES,
        SCORE_FEATURES,
        STRUCT_EXTENSION_FEATURES,
        TAIL_FEATURES,
        TEXTENC_FEATURES,
    ]
    .concat()
    .into_iter()
    .map(str::to_string)
    .chain(metric_base_feature_names())
    .chain(FORMAT_FEATURE_NAMES.iter().flatten().cloned())
    .chain(GAP_FEATURE_NAMES.iter().cloned())
    .chain(INTENT_GAP_FEATURE_NAMES.iter().cloned())
    .chain(MISSING_FEATURE_NAMES.iter().cloned())
    .collect()
});

// Compatibility aliases for metrics cleave has renamed since August 2026. Keep
// in sync with collimator.features._METRIC_RENAMES: existing names (even
// null/zero) win. Keep every entry: hopper keeps reports from every generation.
pub(super) const METRIC_RENAMES: &[(&str, &str, &str, &str)] = &[
    ("archive", "total_compressed", "archive", "compressed_size"),
    (
        "archive",
        "total_uncompressed",
        "archive",
        "uncompressed_size",
    ),
    ("ast", "array_literal_max_length", "ast", "max_array_length"),
    (
        "ast",
        "numeric_array_max_length",
        "ast",
        "max_numeric_array",
    ),
    (
        "ast",
        "numeric_sequence_max_length",
        "ast",
        "max_numeric_sequence",
    ),
    (
        "ast",
        "string_concat_chain_max_length",
        "ast",
        "max_concat_chain",
    ),
    (
        "binary",
        "embedded_archive_count",
        "binary",
        "embedded_archives",
    ),
    (
        "binary",
        "embedded_binary_count",
        "binary",
        "embedded_binaries",
    ),
    ("binary", "embedded_file_count", "binary", "embedded_files"),
    ("binary", "overall_entropy", "file", "entropy"),
    (
        "binary",
        "total_basic_blocks",
        "binary",
        "basic_block_count",
    ),
    ("comments", "base64_in_comments", "comments", "base64"),
    ("comments", "code_in_comments", "comments", "code"),
    ("comments", "empty_comments", "comments", "empty"),
    (
        "comments",
        "high_entropy_comments",
        "comments",
        "high_entropy",
    ),
    ("comments", "total", "comments", "count"),
    ("elf", "gnu_stack_section_absent", "elf", "no_gnu_stack"),
    (
        "elf",
        "stripped_but_symtab_present",
        "elf",
        "stripped_with_symtab",
    ),
    ("file", "size_bytes", "file", "size"),
    (
        "functions",
        "code_in_functions_ratio",
        "functions",
        "code_ratio",
    ),
    ("functions", "density_per_100_lines", "functions", "density"),
    ("functions", "nested_functions", "functions", "nested"),
    ("identifiers", "total", "identifiers", "count"),
    ("identifiers", "unique_count", "identifiers", "unique"),
    ("imports", "aliased_imports", "imports", "aliased"),
    ("imports", "dynamic_imports", "imports", "dynamic"),
    ("imports", "relative_imports", "imports", "relative"),
    ("imports", "unique_modules", "imports", "count"),
    ("imports", "wildcard_imports", "imports", "wildcard"),
    ("pe", "headers_size_bytes", "pe", "headers_size"),
    ("strings", "hex_strings", "strings", "hex"),
    ("strings", "shell_command_strings", "strings", "shell"),
    ("strings", "sql_strings", "strings", "sql"),
    ("strings", "total", "strings", "count"),
    ("strings", "total_bytes", "strings", "bytes"),
    (
        "strings",
        "unicode_heavy_strings",
        "strings",
        "unicode_heavy",
    ),
    ("strings", "url_encoded_strings", "strings", "url_encoded"),
    ("strings", "very_long_strings", "strings", "very_long"),
    ("text", "high_byte_ratio", "text", "non_ascii_ratio"),
    (
        "text",
        "max_inline_whitespace_run",
        "text",
        "max_whitespace_run",
    ),
    ("text", "most_common_char_codepoint", "text", "top_char"),
    ("text", "most_common_char_is_null", "text", "top_char_null"),
    ("text", "most_common_ratio", "text", "top_char_ratio"),
    ("text", "total_lines", "text", "lines"),
    ("ast", "member_depth_max", "ast", "max_member_depth"),
    (
        "binary",
        "high_entropy_strings",
        "binary",
        "high_entropy_string_count",
    ),
    (
        "binary",
        "normalized_string_count",
        "text",
        "normalized_string_count",
    ),
    ("comments", "url_in_comments", "comments", "url_count"),
    ("elf", "build_id_present", "elf", "has_build_id"),
    ("elf", "class_bits", "elf", "bits"),
    ("elf", "debuglink_present", "elf", "has_debuglink"),
    ("elf", "entry_point", "elf", "entry"),
    ("elf", "gnu_hash_present", "elf", "has_gnu_hash"),
    ("elf", "hidden_symbols", "elf", "hidden_symbol_count"),
    (
        "elf",
        "load_segment_max_p_filesz",
        "elf",
        "load_segment_max_file_size",
    ),
    (
        "elf",
        "load_segment_max_p_memsz",
        "elf",
        "load_segment_max_memory_size",
    ),
    ("pe", "certificate_table_size", "pe", "cert_table_size"),
    ("pe", "delay_load_imports", "pe", "delay_import_count"),
    ("pe", "manifest_present", "pe", "has_manifest"),
    ("pe", "size_of_headers", "pe", "headers_size"),
    ("pe", "size_of_image", "pe", "image_size"),
    ("pe", "tls_callbacks", "pe", "tls_callback_count"),
    ("pe", "version_info_present", "pe", "has_version_info"),
    ("text", "max_ws_run", "text", "max_whitespace_run"),
    ("apk", "declared_size", "apk", "installed_size"),
    ("apk", "depend_count", "apk", "dependency_count"),
    ("archive", "prefix_bytes", "archive", "leading_bytes"),
    (
        "archive",
        "timing.mtime_dominant_fraction",
        "archive",
        "timing.mtime_dominant_ratio",
    ),
    ("ast", "max_array_len", "ast", "max_array_length"),
    ("ast", "max_numeric_seq", "ast", "max_numeric_sequence"),
    ("binary", "basic_blocks", "binary", "basic_block_count"),
    ("binary", "huge_func_count", "binary", "huge_function_count"),
    ("binary", "leaf_func_count", "binary", "leaf_function_count"),
    ("binary", "tiny_func_count", "binary", "tiny_function_count"),
    (
        "class",
        "constant_pool_size",
        "class",
        "constant_pool_count",
    ),
    ("deb", "depends_count", "deb", "dependency_count"),
    ("deb", "installed_size", "deb", "installed_size_kib"),
    (
        "dmg",
        "total_uncompressed_bytes",
        "dmg",
        "uncompressed_size",
    ),
    (
        "elf",
        "compressed_sections_count",
        "elf",
        "compressed_section_count",
    ),
    (
        "elf",
        "has_direct_loader_dep",
        "elf",
        "has_direct_loader_dependency",
    ),
    (
        "iso",
        "blank_identifier_fields",
        "iso",
        "blank_identifier_count",
    ),
    ("iso", "dir_count", "iso", "directory_count"),
    ("iso", "udf.dir_count", "iso", "udf.directory_count"),
    ("jpeg", "appended_bytes", "jpeg", "trailing_bytes"),
    (
        "lnk",
        "args_leading_spaces",
        "lnk",
        "arguments_leading_spaces",
    ),
    ("lnk", "args_leading_tabs", "lnk", "arguments_leading_tabs"),
    (
        "lnk",
        "args_max_whitespace_run",
        "lnk",
        "arguments_max_whitespace_run",
    ),
    (
        "lnk",
        "args_whitespace_total",
        "lnk",
        "arguments_whitespace_count",
    ),
    (
        "office",
        "vba.mean_identifier_length",
        "office",
        "vba.avg_identifier_length",
    ),
    ("office", "vba_project_size", "office", "vba.project_size"),
    (
        "pdf",
        "decoded_form_value_max_len",
        "pdf",
        "decoded_form_value_max_length",
    ),
    ("pdf", "leading_bytes_before_header", "pdf", "leading_bytes"),
    ("pdf", "trailing_bytes_after_eof", "pdf", "trailing_bytes"),
    (
        "pe",
        "clr.strong_name_sig_size",
        "pe",
        "clr.strong_name_signature_size",
    ),
    ("pe", "recovered_exports", "pe", "recovered_export_count"),
    (
        "pe",
        "recovered_functions",
        "pe",
        "recovered_function_count",
    ),
    ("pe", "recovered_imports", "pe", "recovered_import_count"),
    ("pe", "recovered_sections", "pe", "recovered_section_count"),
    (
        "pe",
        "reloc_overhang_bytes",
        "pe",
        "relocation_overhang_bytes",
    ),
    (
        "pe",
        "reloc_overhang_ratio",
        "pe",
        "relocation_overhang_ratio",
    ),
    ("png", "chunks_after_iend", "png", "trailing_chunk_count"),
    ("rar", "encrypted_header", "rar", "headers_encrypted"),
    (
        "registry",
        "first_published_at",
        "registry",
        "first_published_unix",
    ),
    ("registry", "maintainers", "registry", "maintainer_count"),
    (
        "registry",
        "previous_published_at",
        "registry",
        "previous_published_unix",
    ),
    ("registry", "published_at", "registry", "published_unix"),
    ("rtf", "group_depth_max", "rtf", "max_group_depth"),
    ("scpt", "calls", "scpt", "call_count"),
    ("scpt", "decoded", "scpt", "decoded_count"),
    ("scpt", "events", "scpt", "event_count"),
    ("scpt", "handlers", "scpt", "handler_count"),
    ("sections", "entropy_max", "sections", "max_entropy"),
    ("sections", "entropy_mean", "sections", "avg_entropy"),
    ("source", "ast_walk_panic", "source", "ast_walk_panicked"),
    ("source", "extract_panic", "source", "extract_panicked"),
];

pub(super) type MetricAliasKey = (&'static str, &'static str);

// Same equivalence classes/order as collimator: canonical spelling first,
// then historical names in the table's order. Never overwrite explicit data.
pub(super) static METRIC_EQUIVALENTS: LazyLock<Vec<Vec<MetricAliasKey>>> = LazyLock::new(|| {
    let renames: HashMap<MetricAliasKey, MetricAliasKey> = METRIC_RENAMES
        .iter()
        .map(|&(og, ok, ng, nk)| ((og, ok), (ng, nk)))
        .collect();
    let mut groups: Vec<Vec<MetricAliasKey>> = Vec::new();
    let mut positions = HashMap::new();
    for &(og, ok, _, _) in METRIC_RENAMES {
        let old = (og, ok);
        let mut canonical = old;
        let mut seen = HashSet::new();
        while let Some(&next) = renames.get(&canonical) {
            assert!(seen.insert(canonical), "cyclic metric rename");
            canonical = next;
        }
        let index = *positions.entry(canonical).or_insert_with(|| {
            groups.push(vec![canonical]);
            groups.len() - 1
        });
        if !groups[index].contains(&old) {
            groups[index].push(old);
        }
    }
    groups
});
