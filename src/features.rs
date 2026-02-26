//! Feature extraction from cleave analysis reports.
//!
//! Extracts features for ML classification matching the Python training pipeline:
//! - Capability presence vectors (~500)
//! - Criticality counts (7)
//! - Import one-hot encoding (~2,000)
//! - String/trait features (8)
//! - Text metrics (8)
//! - Source code metrics (11)
//! - Binary properties (5)
//! - Structural features (7)
//! - Byte histogram (256)

use cleave::types::{AnalysisReport, Criticality, TraitKind};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Total feature count (approximate)
const TOTAL_FEATURES: usize = 10000;

/// A feature vector extracted from a cleave report
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Raw feature values
    pub values: Vec<f32>,
    /// Feature names (for explainability)
    pub names: Vec<String>,
}

impl FeatureVector {
    /// Create a new feature vector with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            names: Vec::with_capacity(capacity),
        }
    }

    /// Add a feature
    pub fn push(&mut self, name: &str, value: f32) {
        self.names.push(name.to_string());
        self.values.push(value);
    }

    /// Get feature values as slice
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Get number of features
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the feature vector is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Sanitize a feature name for XGBoost compatibility (no [, ], <, >)
fn sanitize_feature_name(name: &str) -> String {
    name.replace(['[', ']', '<', '>'], "_")
}

/// Feature extractor configuration
pub struct FeatureExtractor {
    /// Known capability IDs (for one-hot encoding)
    known_capabilities: Vec<String>,
    /// Known import symbols (for one-hot encoding, pre-sanitized)
    known_imports: Vec<String>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor with default configuration
    pub fn new() -> Self {
        // TODO: Load from embedded config or file
        Self {
            known_capabilities: Self::default_capabilities(),
            known_imports: Self::default_imports(),
        }
    }

    /// Load feature extractor from default vocabulary files
    /// Searches: LITMUS_MODEL_PATH env var, ~/.litmus/models/, ./models/
    pub fn load_default() -> anyhow::Result<Self> {
        let caps_path = Self::find_vocab_file("capabilities.json")?;
        let imports_path = Self::find_vocab_file("imports.json")?;
        Self::from_config(caps_path.to_str().unwrap(), imports_path.to_str().unwrap())
    }

    /// Find a vocabulary file in standard locations
    fn find_vocab_file(filename: &str) -> anyhow::Result<std::path::PathBuf> {
        // Check LITMUS_MODEL_PATH directory
        if let Ok(path) = std::env::var("LITMUS_MODEL_PATH") {
            let model_dir = std::path::Path::new(&path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            let file = model_dir.join(filename);
            if file.exists() {
                return Ok(file);
            }
        }

        // Check ~/.litmus/models/
        if let Some(home) = dirs::home_dir() {
            let file = home.join(".litmus").join("models").join(filename);
            if file.exists() {
                return Ok(file);
            }
        }

        // Check ./models/
        let file = std::path::PathBuf::from("models").join(filename);
        if file.exists() {
            return Ok(file);
        }

        anyhow::bail!("Could not find vocabulary file: {}", filename)
    }

    /// Load feature extractor from configuration files
    pub fn from_config(capabilities_path: &str, imports_path: &str) -> anyhow::Result<Self> {
        let capabilities: Vec<String> =
            serde_json::from_str(&std::fs::read_to_string(capabilities_path)?)?;
        let imports: Vec<String> =
            serde_json::from_str(&std::fs::read_to_string(imports_path)?)?;

        // Sanitize imports for XGBoost compatibility
        let imports = imports.into_iter().map(|s| sanitize_feature_name(&s)).collect();

        Ok(Self {
            known_capabilities: capabilities,
            known_imports: imports,
        })
    }

    /// Extract features from a cleave analysis report
    ///
    /// The file_path is used for byte histogram extraction (reads first 64KB).
    /// Order of features matches Python extract_features.py exactly.
    pub fn extract_with_path(&self, report: &AnalysisReport, file_path: Option<&Path>) -> FeatureVector {
        let mut features = FeatureVector::with_capacity(TOTAL_FEATURES);

        // Order matches Python extract_features.py exactly:
        // 1. Capability presence vectors (1621 features)
        self.extract_capability_features(report, &mut features);

        // 2. Criticality counts (18 features - expanded with binary indicators)
        self.extract_criticality_features(report, &mut features);

        // 2b. Hierarchical trait features (~200 features - replaces inefficient cap_* one-hot)
        self.extract_hierarchical_traits(report, &mut features);

        // 3. Import one-hot encoding + count (3901 features)
        self.extract_import_features(report, &mut features);

        // 4. String/trait features (8 features)
        self.extract_string_features(report, &mut features);

        // 5. Text metrics (11 features)
        self.extract_text_metric_features(report, &mut features);

        // 6. Identifier metrics (15 features)
        self.extract_identifier_features(report, &mut features);

        // 7. String metrics from cleave (9 features)
        self.extract_cleave_string_metrics(report, &mut features);

        // 8. Comment metrics (3 features)
        self.extract_comment_features(report, &mut features);

        // 9. Function metrics (14 features)
        self.extract_function_features(report, &mut features);

        // 10. Language-specific metrics (2 features)
        self.extract_language_features(report, &mut features);

        // 11. Legacy source code metrics (11 features)
        self.extract_scm_features(report, &mut features);

        // 12. Suspicious function name count (1 feature)
        self.extract_suspicious_func_features(report, &mut features);

        // 13. Supply chain indicators (22 features)
        self.extract_supply_chain_features(file_path, &mut features);

        // 14. Binary property features (5 features)
        self.extract_binary_features(report, &mut features);

        // 15. Structural features (7 features)
        self.extract_structural_features(report, &mut features);

        // 16. File type one-hot encoding (9 features)
        self.extract_file_type_features(report, &mut features);

        // 17. Byte histogram + anomaly features (260 features)
        self.extract_byte_histogram(report, file_path, &mut features);

        // 18. Binary metrics from radare2 (35 features)
        self.extract_binary_metrics(report, &mut features);

        // 19. Composite scores (15 features)
        self.extract_composite_scores(report, &mut features);

        // 20. Capability confidence aggregates (8 features)
        self.extract_capability_confidence(report, &mut features);

        // 21. Supply chain attack patterns - xz-style detection (18 features)
        self.extract_xz_style_features(report, &mut features);

        // 22. Capability category ratios (14 features)
        self.extract_category_ratios(report, &mut features);

        // 23. Archive contents features (12 features)
        self.extract_archive_features(report, &mut features);

        features
    }

    /// Extract ratio of findings in each top-level capability category (23 features)
    /// Categories from cleave/traits/
    fn extract_category_ratios(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        let total = report.findings.len().max(1) as f32;

        // cleave trait categories (from traits/ dir + YARA rules)
        let categories = [
            "anti-analysis",
            "anti-static",
            "c2",
            "collect",
            "comm",
            "cred",
            "crypto",
            "cryptominer", // from YARA
            "data",
            "discovery",
            "evasion",
            "exec",
            "exfil",
            "feat",
            "fs",
            "hw",
            "impact",
            "known-tools",
            "lateral",
            "mem",
            "net",
            "os",
            "persist",
            "privesc",
            "process",
        ];

        for cat in categories {
            let prefix = format!("{}/", cat);
            let count = report.findings.iter()
                .filter(|f| f.id.starts_with(&prefix))
                .count();
            let name = format!("cat_{}_ratio", cat.replace('-', "_"));
            features.push(&name, count as f32 / total);
        }
    }

    /// Extract features from archive contents (12 features)
    /// Counts of file types within archives, nesting depth, etc.
    fn extract_archive_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        let contents = &report.archive_contents;

        // Total file count
        features.push("archive_file_count", contents.len() as f32);

        // Count files by type
        let mut shell_count = 0;
        let mut python_count = 0;
        let mut javascript_count = 0;
        let mut elf_count = 0;
        let mut pe_count = 0;
        let mut java_count = 0;
        let mut archive_count = 0; // Nested archives
        let mut other_count = 0;
        let mut max_depth = 0;

        for entry in contents {
            // Count nesting depth (number of ! separators)
            let depth = entry.path.matches('!').count();
            max_depth = max_depth.max(depth);

            // Count by file type
            match entry.file_type.as_str() {
                "shell" => shell_count += 1,
                "python" => python_count += 1,
                "javascript" => javascript_count += 1,
                "elf" => elf_count += 1,
                "pe" => pe_count += 1,
                "javaclass" | "java" => java_count += 1,
                "archive" => archive_count += 1,
                _ => other_count += 1,
            }
        }

        features.push("archive_shell_count", shell_count as f32);
        features.push("archive_python_count", python_count as f32);
        features.push("archive_javascript_count", javascript_count as f32);
        features.push("archive_elf_count", elf_count as f32);
        features.push("archive_pe_count", pe_count as f32);
        features.push("archive_java_count", java_count as f32);
        features.push("archive_nested_count", archive_count as f32);
        features.push("archive_other_count", other_count as f32);
        features.push("archive_max_depth", max_depth as f32);

        // Has nested archives (binary)
        features.push("archive_has_nested", if archive_count > 0 { 1.0 } else { 0.0 });
    }

    /// Extract capability presence as binary features
    fn extract_capability_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        let present: HashSet<&str> = report.findings.iter().map(|f| f.id.as_str()).collect();

        for cap_id in &self.known_capabilities {
            let value = if present.contains(cap_id.as_str()) {
                1.0
            } else {
                0.0
            };
            features.push(&format!("cap_{}", cap_id.replace('/', "_")), value);
        }
    }

    /// Extract hierarchical trait features with criticality weighting
    ///
    /// This replaces the inefficient 1621 one-hot cap_* features with hierarchical aggregations.
    /// Captures patterns at multiple abstraction levels (e.g., stealer/* vs stealer/browser/chrome).
    fn extract_hierarchical_traits(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        use std::collections::HashMap;

        // Aggregate counts and weighted scores by hierarchy level
        let mut level1_counts: HashMap<String, (u32, u32, u32)> = HashMap::new(); // (hostile, suspicious, notable)
        let mut level2_counts: HashMap<String, (u32, u32, u32)> = HashMap::new();
        let mut level3_counts: HashMap<String, (u32, u32, u32)> = HashMap::new();

        let mut level1_conf_sum: HashMap<String, f32> = HashMap::new();
        let mut level2_conf_sum: HashMap<String, f32> = HashMap::new();

        for finding in &report.findings {
            let parts: Vec<&str> = finding.id.split('/').collect();
            if parts.is_empty() {
                continue;
            }

            // Count by criticality
            let crit_tuple = match finding.crit {
                Criticality::Hostile => (1, 0, 0),
                Criticality::Suspicious => (0, 1, 0),
                Criticality::Notable => (0, 0, 1),
                _ => (0, 0, 0),
            };

            // Level 1: Top category (e.g., "stealer", "c2", "exfil")
            if !parts.is_empty() {
                let level1 = parts[0].to_string();
                let entry = level1_counts.entry(level1.clone()).or_insert((0, 0, 0));
                entry.0 += crit_tuple.0;
                entry.1 += crit_tuple.1;
                entry.2 += crit_tuple.2;
                *level1_conf_sum.entry(level1).or_insert(0.0) += finding.conf;
            }

            // Level 2: Subcategory (e.g., "stealer/browser", "c2/backdoor")
            if parts.len() >= 2 {
                let level2 = format!("{}/{}", parts[0], parts[1]);
                let entry = level2_counts.entry(level2.clone()).or_insert((0, 0, 0));
                entry.0 += crit_tuple.0;
                entry.1 += crit_tuple.1;
                entry.2 += crit_tuple.2;
                *level2_conf_sum.entry(level2).or_insert(0.0) += finding.conf;
            }

            // Level 3: Specific capability (e.g., "stealer/browser/chrome")
            if parts.len() >= 3 {
                let level3 = format!("{}/{}/{}", parts[0], parts[1], parts[2]);
                let entry = level3_counts.entry(level3).or_insert((0, 0, 0));
                entry.0 += crit_tuple.0;
                entry.1 += crit_tuple.1;
                entry.2 += crit_tuple.2;
            }
        }

        // Top-level categories (critical for pattern detection)
        let top_categories = [
            "stealer", "c2", "exfil", "ransomware", "backdoor",
            "lateral", "exec", "fs", "comm", "crypto",
            "anti-analysis", "anti-static", "evasion", "persist",
            "discovery", "cred", "impact", "collect", "data",
        ];

        for cat in &top_categories {
            let (hostile, suspicious, notable) = level1_counts.get(*cat).unwrap_or(&(0, 0, 0));
            let conf_sum = level1_conf_sum.get(*cat).unwrap_or(&0.0);

            // Binary presence
            features.push(&format!("hier1_{}_present", cat), if hostile + suspicious + notable > 0 { 1.0 } else { 0.0 });

            // Counts by criticality
            features.push(&format!("hier1_{}_hostile", cat), *hostile as f32);
            features.push(&format!("hier1_{}_suspicious", cat), *suspicious as f32);

            // Weighted danger score (hostile worth 10x, suspicious 2x)
            let danger_score = (*hostile as f32 * 10.0) + (*suspicious as f32 * 2.0) + (*notable as f32 * 0.1);
            features.push(&format!("hier1_{}_danger", cat), danger_score);

            // Confidence-weighted score
            let conf_weighted = danger_score * (*conf_sum / (hostile + suspicious + notable).max(1) as f32);
            features.push(&format!("hier1_{}_conf_weighted", cat), conf_weighted);
        }

        // High-risk subcategories (level 2) - most discriminative
        let high_risk_subcats = [
            "stealer/browser", "stealer/credential", "stealer/wallet",
            "c2/backdoor", "c2/infrastructure", "c2/beacon",
            "exfil/http", "exfil/data", "exfil/channel",
            "lateral/supply-chain", "lateral/exploit",
            "exec/dropper", "exec/command",
            "crypto/ransomware", "impact/ransomware",
            "anti-analysis/vm", "anti-analysis/debugger",
            "evasion/rootkit", "evasion/obfuscation",
            "persist/autostart", "persist/hook",
        ];

        for subcat in &high_risk_subcats {
            let (hostile, suspicious, notable) = level2_counts.get(*subcat).unwrap_or(&(0, 0, 0));

            let safe_name = subcat.replace('/', "_");
            features.push(&format!("hier2_{}_present", safe_name), if hostile + suspicious + notable > 0 { 1.0 } else { 0.0 });
            features.push(&format!("hier2_{}_hostile", safe_name), *hostile as f32);

            let danger_score = (*hostile as f32 * 10.0) + (*suspicious as f32 * 2.0);
            features.push(&format!("hier2_{}_danger", safe_name), danger_score);
        }

        // Interaction patterns (subtle supply chain indicators)
        let has_supply_chain_lateral = level2_counts.get("lateral/supply-chain").is_some_and(|(h, s, _)| h + s > 0);
        let has_exec_dropper = level2_counts.get("exec/dropper").is_some_and(|(h, s, _)| h + s > 0);
        let has_obfuscation = level2_counts.get("anti-static/obfuscation").is_some_and(|(_, s, _)| *s > 2);
        let has_evasion = level1_counts.get("evasion").is_some_and(|(h, s, _)| h + s > 0);

        // xz-style: supply chain + obfuscation + evasion
        features.push("pattern_supply_chain_attack",
            if has_supply_chain_lateral && (has_obfuscation || has_evasion) { 1.0 } else { 0.0 });

        // Dropper + C2 = staged attack
        features.push("pattern_staged_attack",
            if has_exec_dropper && level1_counts.contains_key("c2") { 1.0 } else { 0.0 });

        // Stealer + exfil + obfuscation = sophisticated stealer
        features.push("pattern_sophisticated_stealer",
            if level1_counts.contains_key("stealer") &&
               level1_counts.contains_key("exfil") &&
               has_obfuscation { 1.0 } else { 0.0 });
    }

    /// Extract criticality distribution
    fn extract_criticality_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        let mut hostile = 0;
        let mut suspicious = 0;
        let mut notable = 0;
        let mut baseline = 0;

        for finding in &report.findings {
            match finding.crit {
                Criticality::Hostile => hostile += 1,
                Criticality::Suspicious => suspicious += 1,
                Criticality::Notable => notable += 1,
                Criticality::Baseline | Criticality::Component => baseline += 1,
                Criticality::Filtered => {}
            }
        }

        features.push("crit_hostile_count", hostile as f32);
        features.push("crit_suspicious_count", suspicious as f32);
        features.push("crit_notable_count", notable as f32);
        features.push("crit_baseline_count", baseline as f32);
        features.push("crit_total_findings", report.findings.len() as f32);

        // Ratios (normalized by total findings)
        let total = report.findings.len().max(1) as f32;
        features.push("crit_hostile_ratio", hostile as f32 / total);
        features.push("crit_suspicious_ratio", suspicious as f32 / total);
        features.push("crit_notable_ratio", notable as f32 / total);
        features.push("crit_baseline_ratio", baseline as f32 / total);

        // Danger ratio: (hostile + suspicious) / (notable + baseline + 1)
        // Higher = more dangerous findings relative to benign ones
        let dangerous = (hostile + suspicious) as f32;
        let benign = (notable + baseline).max(1) as f32;
        features.push("crit_danger_ratio", dangerous / benign);

        // Hostile+suspicious combined ratio
        features.push("crit_dangerous_ratio", dangerous / total);

        // CRITICAL: Binary indicators for strong signal (can't be overwhelmed by byte patterns)
        // These have high discriminative power even with low variance in training data
        features.push("has_hostile_trait", if hostile > 0 { 1.0 } else { 0.0 });
        features.push("has_multiple_hostile", if hostile >= 2 { 1.0 } else { 0.0 });
        features.push("has_many_suspicious", if suspicious >= 3 { 1.0 } else { 0.0 });
        features.push("has_very_many_suspicious", if suspicious >= 10 { 1.0 } else { 0.0 });

        // Composite danger flags combining different signals
        let is_dangerous = hostile > 0 || suspicious >= 5;
        features.push("is_dangerous_composite", if is_dangerous { 1.0 } else { 0.0 });

        // Weighted danger score (hostile worth 10x suspicious)
        let weighted_danger_score = (hostile as f32 * 10.0) + (suspicious as f32 * 1.0);
        features.push("weighted_danger_score", weighted_danger_score);

        // Per-size density features (findings per KB)
        // Helps distinguish large benign bundles from small dense malware
        let size_kb = (report.target.size_bytes as f32 / 1024.0).max(0.001); // Avoid divide-by-zero
        features.push("hostile_per_kb", hostile as f32 / size_kb);
        features.push("suspicious_per_kb", suspicious as f32 / size_kb);
        features.push("danger_per_kb", weighted_danger_score / size_kb);
        features.push("findings_per_kb", report.findings.len() as f32 / size_kb);

        // Check for specific high-risk capability categories
        let has_stealer = report.findings.iter()
            .any(|f| f.id.contains("stealer") && f.crit == Criticality::Hostile);
        let has_ransomware = report.findings.iter()
            .any(|f| (f.id.contains("ransomware") || f.id.contains("encrypt")) && f.crit == Criticality::Hostile);
        let has_backdoor = report.findings.iter()
            .any(|f| (f.id.contains("backdoor") || f.id.contains("c2")) && f.crit == Criticality::Hostile);

        features.push("has_hostile_stealer", if has_stealer { 1.0 } else { 0.0 });
        features.push("has_hostile_ransomware", if has_ransomware { 1.0 } else { 0.0 });
        features.push("has_hostile_backdoor", if has_backdoor { 1.0 } else { 0.0 });
    }

    /// Extract import features as one-hot encoding
    fn extract_import_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        // Sanitize import names for consistent matching
        let present: HashSet<String> = report
            .imports
            .iter()
            .map(|i| sanitize_feature_name(&i.symbol))
            .collect();

        for import in &self.known_imports {
            let value = if present.contains(import) { 1.0 } else { 0.0 };
            features.push(&format!("import_{}", import), value);
        }

        // Import count features
        features.push("imports_count", report.imports.len() as f32);
    }

    /// Extract string-based features
    fn extract_string_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        let strings = &report.strings;

        // Basic counts
        features.push("string_total_count", strings.len() as f32);

        // Length statistics
        if !strings.is_empty() {
            let mut max_length = 0usize;
            let mut total_length = 0usize;

            for s in strings {
                let len = s.value.len();
                total_length += len;
                if len > max_length {
                    max_length = len;
                }
            }

            features.push("string_max_length", max_length as f32);
            features.push("string_avg_length", total_length as f32 / strings.len() as f32);
        } else {
            features.push("string_max_length", 0.0);
            features.push("string_avg_length", 0.0);
        }

        // Trait-based counts (matches Python extract_features.py)
        let url_count = report
            .traits
            .iter()
            .filter(|t| matches!(t.kind, TraitKind::Url))
            .count();
        let ip_count = report
            .traits
            .iter()
            .filter(|t| matches!(t.kind, TraitKind::Ip))
            .count();
        let domain_count = report
            .traits
            .iter()
            .filter(|t| matches!(t.kind, TraitKind::Domain))
            .count();
        let path_count = report
            .traits
            .iter()
            .filter(|t| matches!(t.kind, TraitKind::Path))
            .count();
        let base64_count = report
            .traits
            .iter()
            .filter(|t| matches!(t.kind, TraitKind::Base64))
            .count();

        features.push("string_url_count", url_count as f32);
        features.push("string_ip_count", ip_count as f32);
        features.push("string_domain_count", domain_count as f32);
        features.push("string_path_count", path_count as f32);
        features.push("string_base64_count", base64_count as f32);
    }

    /// Extract text metric features (11 features)
    fn extract_text_metric_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        if let Some(ref metrics) = report.metrics {
            if let Some(ref text) = metrics.text {
                features.push("metric_char_entropy", text.char_entropy);
                features.push("metric_avg_line_length", text.avg_line_length);
                features.push("metric_max_line_length", text.max_line_length as f32);
                features.push("metric_non_ascii_ratio", text.non_ascii_ratio);
                features.push("metric_hex_escape_count", text.hex_escape_count as f32);
                features.push("metric_unicode_escape_count", text.unicode_escape_count as f32);
                features.push("metric_total_lines", text.total_lines as f32);
                features.push("metric_whitespace_ratio", text.whitespace_ratio);
                features.push("metric_digit_ratio", text.digit_ratio);
                features.push("metric_repeated_char_sequences", text.repeated_char_sequences as f32);
                features.push("metric_trailing_whitespace_lines", text.trailing_whitespace_lines as f32);
            } else {
                Self::push_zeros(features, &[
                    "metric_char_entropy", "metric_avg_line_length", "metric_max_line_length",
                    "metric_non_ascii_ratio", "metric_hex_escape_count", "metric_unicode_escape_count",
                    "metric_total_lines", "metric_whitespace_ratio", "metric_digit_ratio",
                    "metric_repeated_char_sequences", "metric_trailing_whitespace_lines",
                ]);
            }
        } else {
            Self::push_zeros(features, &[
                "metric_char_entropy", "metric_avg_line_length", "metric_max_line_length",
                "metric_non_ascii_ratio", "metric_hex_escape_count", "metric_unicode_escape_count",
                "metric_total_lines", "metric_whitespace_ratio", "metric_digit_ratio",
                "metric_repeated_char_sequences", "metric_trailing_whitespace_lines",
            ]);
        }
    }

    /// Extract identifier metrics (15 features) - obfuscation detection
    fn extract_identifier_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        if let Some(ref metrics) = report.metrics {
            if let Some(ref idents) = metrics.identifiers {
                features.push("ident_total", idents.total as f32);
                features.push("ident_unique", idents.unique_count as f32);
                features.push("ident_reuse_ratio", idents.reuse_ratio);
                features.push("ident_avg_length", idents.avg_length);
                features.push("ident_single_char_count", idents.single_char_count as f32);
                features.push("ident_single_char_ratio", idents.single_char_ratio);
                features.push("ident_avg_entropy", idents.avg_entropy);
                features.push("ident_high_entropy_count", idents.high_entropy_count as f32);
                features.push("ident_high_entropy_ratio", idents.high_entropy_ratio);
                features.push("ident_all_uppercase_ratio", idents.all_uppercase_ratio);
                features.push("ident_underscore_prefix_count", idents.underscore_prefix_count as f32);
                features.push("ident_double_underscore_count", idents.double_underscore_count as f32);
                features.push("ident_sequential_names", idents.sequential_names as f32);
                features.push("ident_has_digit_ratio", idents.has_digit_ratio);
                features.push("ident_numeric_suffix_count", idents.numeric_suffix_count as f32);
            } else {
                Self::push_zeros(features, &[
                    "ident_total", "ident_unique", "ident_reuse_ratio", "ident_avg_length",
                    "ident_single_char_count", "ident_single_char_ratio", "ident_avg_entropy",
                    "ident_high_entropy_count", "ident_high_entropy_ratio", "ident_all_uppercase_ratio",
                    "ident_underscore_prefix_count", "ident_double_underscore_count",
                    "ident_sequential_names", "ident_has_digit_ratio", "ident_numeric_suffix_count",
                ]);
            }
        } else {
            Self::push_zeros(features, &[
                "ident_total", "ident_unique", "ident_reuse_ratio", "ident_avg_length",
                "ident_single_char_count", "ident_single_char_ratio", "ident_avg_entropy",
                "ident_high_entropy_count", "ident_high_entropy_ratio", "ident_all_uppercase_ratio",
                "ident_underscore_prefix_count", "ident_double_underscore_count",
                "ident_sequential_names", "ident_has_digit_ratio", "ident_numeric_suffix_count",
            ]);
        }
    }

    /// Extract string metrics from cleave metrics.strings (9 features)
    fn extract_cleave_string_metrics(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        if let Some(ref metrics) = report.metrics {
            if let Some(ref str_metrics) = metrics.strings {
                features.push("str_total", str_metrics.total as f32);
                features.push("str_total_bytes", str_metrics.total_bytes as f32);
                features.push("str_avg_length", str_metrics.avg_length);
                features.push("str_max_length", str_metrics.max_length as f32);
                features.push("str_avg_entropy", str_metrics.avg_entropy);
                features.push("str_entropy_stddev", str_metrics.entropy_stddev);
                features.push("str_shell_command_strings", str_metrics.shell_command_strings as f32);
                features.push("str_base64_candidates", str_metrics.base64_candidates as f32);
                features.push("str_path_count", str_metrics.path_count as f32);
            } else {
                Self::push_zeros(features, &[
                    "str_total", "str_total_bytes", "str_avg_length", "str_max_length",
                    "str_avg_entropy", "str_entropy_stddev", "str_shell_command_strings",
                    "str_base64_candidates", "str_path_count",
                ]);
            }
        } else {
            Self::push_zeros(features, &[
                "str_total", "str_total_bytes", "str_avg_length", "str_max_length",
                "str_avg_entropy", "str_entropy_stddev", "str_shell_command_strings",
                "str_base64_candidates", "str_path_count",
            ]);
        }
    }

    /// Extract comment metrics (3 features)
    fn extract_comment_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        if let Some(ref metrics) = report.metrics {
            if let Some(ref comments) = metrics.comments {
                features.push("comment_total", comments.total as f32);
                features.push("comment_lines", comments.lines as f32);
                features.push("comment_to_code_ratio", comments.to_code_ratio);
            } else {
                Self::push_zeros(features, &["comment_total", "comment_lines", "comment_to_code_ratio"]);
            }
        } else {
            Self::push_zeros(features, &["comment_total", "comment_lines", "comment_to_code_ratio"]);
        }
    }

    /// Extract function metrics (14 features)
    fn extract_function_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        if let Some(ref metrics) = report.metrics {
            if let Some(ref funcs) = metrics.functions {
                features.push("func_total", funcs.total as f32);
                features.push("func_anonymous", funcs.anonymous as f32);
                features.push("func_async_count", funcs.async_count as f32);
                features.push("func_avg_length_lines", funcs.avg_length_lines);
                features.push("func_max_length_lines", funcs.max_length_lines as f32);
                features.push("func_one_liners", funcs.one_liners as f32);
                features.push("func_avg_params", funcs.avg_params);
                features.push("func_no_params_count", funcs.no_params_count as f32);
                features.push("func_single_char_params", funcs.single_char_params as f32);
                features.push("func_max_nesting_depth", funcs.max_nesting_depth as f32);
                features.push("func_avg_nesting_depth", funcs.avg_nesting_depth);
                features.push("func_nested_functions", funcs.nested_functions as f32);
                features.push("func_density_per_100_lines", funcs.density_per_100_lines);
                features.push("func_code_in_functions_ratio", funcs.code_in_functions_ratio);
            } else {
                Self::push_zeros(features, &[
                    "func_total", "func_anonymous", "func_async_count", "func_avg_length_lines",
                    "func_max_length_lines", "func_one_liners", "func_avg_params", "func_no_params_count",
                    "func_single_char_params", "func_max_nesting_depth", "func_avg_nesting_depth",
                    "func_nested_functions", "func_density_per_100_lines", "func_code_in_functions_ratio",
                ]);
            }
        } else {
            Self::push_zeros(features, &[
                "func_total", "func_anonymous", "func_async_count", "func_avg_length_lines",
                "func_max_length_lines", "func_one_liners", "func_avg_params", "func_no_params_count",
                "func_single_char_params", "func_max_nesting_depth", "func_avg_nesting_depth",
                "func_nested_functions", "func_density_per_100_lines", "func_code_in_functions_ratio",
            ]);
        }
    }

    /// Extract language-specific metrics (2 features)
    fn extract_language_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        let mut py_try_except = 0.0;
        let mut js_arrow = 0.0;

        if let Some(ref metrics) = report.metrics {
            if let Some(ref py) = metrics.python {
                py_try_except = py.try_except_count as f32;
            }
            if let Some(ref js) = metrics.javascript {
                js_arrow = js.arrow_function_count as f32;
            }
        }

        features.push("lang_python_try_except_count", py_try_except);
        features.push("lang_js_arrow_function_count", js_arrow);
    }

    /// Extract legacy source code metrics (11 features)
    fn extract_scm_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        if let Some(ref scm) = report.source_code_metrics {
            features.push("scm_function_count", scm.function_count as f32);
            features.push("scm_dynamic_exec_count", scm.dynamic_execution_count as f32);
            features.push("scm_obfuscation_indicators", scm.obfuscation_indicators as f32);
            features.push("scm_high_entropy_strings", scm.high_entropy_strings as f32);
            features.push("scm_total_lines", scm.total_lines as f32);
            features.push("scm_code_lines", scm.code_lines as f32);
            features.push("scm_comment_lines", scm.comment_lines as f32);
            features.push("scm_avg_function_size", scm.avg_function_size);
            features.push("scm_max_function_size", scm.max_function_size as f32);
            features.push("scm_string_count", scm.string_count as f32);
            features.push("scm_avg_string_entropy", scm.avg_string_entropy);
        } else {
            Self::push_zeros(features, &[
                "scm_function_count", "scm_dynamic_exec_count", "scm_obfuscation_indicators",
                "scm_high_entropy_strings", "scm_total_lines", "scm_code_lines", "scm_comment_lines",
                "scm_avg_function_size", "scm_max_function_size", "scm_string_count", "scm_avg_string_entropy",
            ]);
        }
    }

    /// Extract suspicious function name count (1 feature)
    fn extract_suspicious_func_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        const SUSPICIOUS_KEYWORDS: &[&str] = &[
            "inject", "shell", "payload", "execute", "exec", "download", "upload",
            "encrypt", "decrypt", "exfil", "backdoor", "keylog", "hook", "dump",
            "steal", "capture", "reverse", "bind", "connect", "socket", "spawn",
            "persist", "hide", "obfuscate", "encode", "decode", "base64", "eval",
            "credential", "password", "token", "cookie", "browser", "chrome", "firefox",
        ];

        let mut count = 0;
        for func in &report.functions {
            let name_lower = func.name.to_lowercase();
            for kw in SUSPICIOUS_KEYWORDS {
                if name_lower.contains(kw) {
                    count += 1;
                    break;
                }
            }
        }
        features.push("suspicious_func_name_count", count as f32);
    }

    /// Helper to push multiple zero features
    fn push_zeros(features: &mut FeatureVector, names: &[&str]) {
        for name in names {
            features.push(name, 0.0);
        }
    }

    /// Extract binary property features (5 features, matches Python)
    fn extract_binary_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        if let Some(ref bp) = report.binary_properties {
            let sec = &bp.security;
            features.push("binary_has_canary", if sec.canary { 1.0 } else { 0.0 });
            features.push("binary_has_nx", if sec.nx { 1.0 } else { 0.0 });
            features.push("binary_has_pic", if sec.pic { 1.0 } else { 0.0 });
            features.push("binary_is_stripped", if sec.stripped { 1.0 } else { 0.0 });
            features.push("binary_uses_crypto", if sec.uses_crypto { 1.0 } else { 0.0 });
        } else {
            // Fill with zeros if no binary properties
            features.push("binary_has_canary", 0.0);
            features.push("binary_has_nx", 0.0);
            features.push("binary_has_pic", 0.0);
            features.push("binary_is_stripped", 0.0);
            features.push("binary_uses_crypto", 0.0);
        }
    }

    /// Extract structural features (7 features, matches Python)
    fn extract_structural_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        features.push("struct_feature_count", report.structure.len() as f32);
        features.push("struct_section_count", report.sections.len() as f32);
        features.push("struct_function_count", report.functions.len() as f32);
        features.push("struct_syscall_count", report.syscalls.len() as f32);
        features.push("struct_path_count", report.paths.len() as f32);
        features.push("struct_env_var_count", report.env_vars.len() as f32);

        // File size (log-normalized using ln(1+x), matches Python np.log1p)
        let size_log = (report.target.size_bytes as f64 + 1.0).ln() as f32;
        features.push("struct_file_size_log", size_log);
    }

    /// Extract supply chain attack indicators (22 features) - matches Python exactly
    fn extract_supply_chain_features(&self, file_path: Option<&Path>, features: &mut FeatureVector) {
        let content = if let Some(path) = file_path {
            std::fs::read_to_string(path).unwrap_or_default()
        } else {
            String::new()
        };
        let content_lower = content.to_lowercase();

        // 1. Sensitive paths (SSH, cloud, package managers)
        const SENSITIVE_PATHS: &[&str] = &[
            ".ssh/", "id_rsa", "id_ed25519", "known_hosts", "authorized_keys",
            ".aws/credentials", ".aws/config", ".azure/", ".config/gcloud",
            ".npmrc", ".pypirc", ".gem/credentials", ".docker/config",
            ".kube/config", ".terraform", ".vault-token", ".gnupg/", ".pgp",
            "private_key", "privkey",
        ];
        let sensitive_count = SENSITIVE_PATHS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_sensitive_path_count", sensitive_count as f32);

        // 2. Browser credential paths
        const BROWSER_PATHS: &[&str] = &[
            "chrome", "chromium", "firefox", "mozilla", "safari", "edge", "opera", "brave",
            "local state", "login data", "cookies", "web data",
            "application support/google", "appdata/local/google",
            "appdata/roaming/mozilla", ".config/google-chrome",
        ];
        let browser_count = BROWSER_PATHS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_browser_path_count", browser_count as f32);

        // 3. Exfiltration endpoints
        const EXFIL_ENDPOINTS: &[&str] = &[
            "discord.com/api/webhook", "discord.gg", "discordapp.com",
            "telegram.org", "api.telegram", "t.me/",
            "pastebin.com", "hastebin", "ghostbin", "paste.ee", "dpaste",
            "ngrok.io", "serveo.net", "localhost.run",
            "transfer.sh", "file.io", "0x0.st",
        ];
        let exfil_count = EXFIL_ENDPOINTS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_exfil_endpoint_count", exfil_count as f32);

        // 4-5. Hardcoded public IPs
        let (public_ip_count, unique_public_ip_count) = Self::count_public_ips(&content);
        features.push("sc_public_ip_count", public_ip_count as f32);
        features.push("sc_unique_public_ip_count", unique_public_ip_count as f32);

        // 6. Environment variable access
        const ENV_ACCESS: &[&str] = &[
            "os.environ", "os.getenv", "process.env", "env[",
            "getenv(", "environ.get", "environment.",
        ];
        let env_count = ENV_ACCESS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_env_access_count", env_count as f32);

        // 7. Sensitive environment variables
        const SENSITIVE_ENVS: &[&str] = &[
            "api_key", "apikey", "api-key", "secret", "token", "password", "passwd",
            "aws_access", "aws_secret", "github_token", "npm_token", "pypi_token",
            "discord_token", "slack_token", "telegram_token",
            "private_key", "encryption_key", "signing_key",
            "database_url", "db_password", "redis_url", "mongodb_uri",
        ];
        let sensitive_env_count = SENSITIVE_ENVS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_sensitive_env_var_count", sensitive_env_count as f32);

        // 8. Crypto wallet patterns (capped at 10)
        let crypto_count = Self::count_crypto_wallets(&content).min(10);
        features.push("sc_crypto_wallet_count", crypto_count as f32);

        // 9. Install hook indicators
        const INSTALL_HOOKS: &[&str] = &[
            "cmdclass", "install_requires", "setup(", "setuptools",
            "postinstall", "preinstall", "prepare", "build:",
            "__import__", "exec(", "eval(", "compile(",
            "subprocess.run", "subprocess.popen", "subprocess.call",
            "os.system(", "os.popen(",
        ];
        let install_count = INSTALL_HOOKS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_install_hook_count", install_count as f32);

        // 10. Data exfiltration methods
        const EXFIL_METHODS: &[&str] = &[
            "requests.post", "requests.put", "urllib.request",
            "http.client", "httplib", "aiohttp",
            "socket.connect", "socket.send",
            "ftplib", "smtplib", "paramiko",
            "base64.b64encode", "codecs.encode", "binascii",
        ];
        let exfil_method_count = EXFIL_METHODS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_exfil_method_count", exfil_method_count as f32);

        // 11. Obfuscation indicators (count occurrences, capped at 100)
        const OBFUS_PATTERNS: &[&str] = &[
            "\\x", "\\u00", "chr(", "ord(",
            "lambda", "map(", "reduce(",
            "getattr(", "setattr(", "__dict__",
            "globals()", "locals()", "vars(",
            "importlib", "__import__", "exec(",
        ];
        let obfus_count: usize = OBFUS_PATTERNS.iter()
            .map(|p| content_lower.matches(p).count())
            .sum();
        features.push("sc_obfuscation_count", obfus_count.min(100) as f32);

        // 12. Persistence mechanisms
        const PERSIST_PATTERNS: &[&str] = &[
            "crontab", "cron.d", "/etc/rc", "systemd", "launchd", "plist",
            "registry", "hkey_", "run\\", "runonce",
            "startup", "autostart", ".bashrc", ".zshrc", ".profile",
            "scheduled task", "at job",
        ];
        let persist_count = PERSIST_PATTERNS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_persistence_count", persist_count as f32);

        // 13. Anti-analysis / evasion
        const EVASION_PATTERNS: &[&str] = &[
            "virtualbox", "vmware", "qemu", "sandbox", "malware",
            "wireshark", "fiddler", "burp", "debugger", "ida",
            "isdebuggerpresent", "ptrace", "strace",
            "time.sleep", "sleep(", "timeout",
        ];
        let evasion_count = EVASION_PATTERNS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_evasion_count", evasion_count as f32);

        // 14. Data collection (supply-chain exfil)
        const DATA_COLLECTION: &[&str] = &[
            "gethostname", "getfqdn", "platform.node", "socket.gethostname",
            "platform.system", "platform.release", "platform.machine",
            "getpass.getuser", "os.getlogin", "pwd.getpwuid",
            "uuid.getnode", "get_mac", "netifaces",
            "psutil", "wmi", "subprocess.check_output",
        ];
        let data_count = DATA_COLLECTION.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_data_collection_count", data_count as f32);

        // 15. Suspicious imports
        const SUS_IMPORTS: &[&str] = &[
            "colorama", "requests", "urllib3",
            "__builtins__", "__code__", "__globals__",
            "marshal.loads", "types.functiontype", "types.codetype",
        ];
        let sus_import_count = SUS_IMPORTS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_suspicious_import_count", sus_import_count as f32);

        // 16. Steganography / hidden data patterns
        const STEGO_PATTERNS: &[&str] = &[
            "from_bytes", "to_bytes", "struct.unpack", "struct.pack",
            "zlib.decompress", "gzip.decompress", "lzma.decompress", "bz2.decompress",
            "pil", "image.open", "png", "jpg", "jpeg", "bmp",
        ];
        let stego_count = STEGO_PATTERNS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_stego_count", stego_count as f32);

        // 17. Code injection patterns
        const INJECT_PATTERNS: &[&str] = &[
            "ctypes", "windll", "cdll", "cfunctype",
            "mmap", "prot_exec", "page_execute",
            "virtualloc", "writeprocessmemory", "createremotethread",
            "dlopen", "dlsym", "ld_preload",
        ];
        let inject_count = INJECT_PATTERNS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_injection_count", inject_count as f32);

        // 18. Keylogging / input capture
        const KEYLOG_PATTERNS: &[&str] = &[
            "pynput", "keyboard", "getasynckeystate", "setwindowshookex",
            "pyautogui", "pyscreenshot", "mss", "screenshot",
            "clipboard", "pyperclip", "win32clipboard",
        ];
        let keylog_count = KEYLOG_PATTERNS.iter().filter(|p| content_lower.contains(*p)).count();
        features.push("sc_keylog_count", keylog_count as f32);

        // 19-20. URL counts
        let url_count = content_lower.matches("http://").count() + content_lower.matches("https://").count();
        features.push("sc_url_count", url_count as f32);

        let raw_ip_url_count = Self::count_ip_urls(&content_lower);
        features.push("sc_raw_ip_url_count", raw_ip_url_count as f32);

        // 21-22. Base64 encoded content
        let (b64_count, long_b64_count) = Self::count_base64_strings(&content);
        features.push("sc_base64_string_count", b64_count as f32);
        features.push("sc_long_base64_count", long_b64_count as f32);
    }

    /// Count public (non-private) IP addresses in content
    fn count_public_ips(content: &str) -> (usize, usize) {
        let mut ips = Vec::new();
        // Simple IP pattern matching
        for part in content.split(|c: char| !c.is_ascii_digit() && c != '.') {
            let octets: Vec<&str> = part.split('.').collect();
            if octets.len() == 4 {
                let valid = octets.iter().all(|o| {
                    o.parse::<u8>().is_ok()
                });
                if valid {
                    // Check if it's a public IP
                    if let Ok(first) = octets[0].parse::<u8>() {
                        if let Ok(second) = octets[1].parse::<u8>() {
                            let is_private = first == 127 || first == 10 || first == 0
                                || (first == 192 && second == 168)
                                || (first == 172 && (16..=31).contains(&second));
                            if !is_private && part != "255.255.255.255" {
                                ips.push(part.to_string());
                            }
                        }
                    }
                }
            }
        }
        let total = ips.len();
        let unique: std::collections::HashSet<_> = ips.into_iter().collect();
        (total, unique.len())
    }

    /// Count crypto wallet addresses
    fn count_crypto_wallets(content: &str) -> usize {
        let mut count = 0;
        // Bitcoin-like addresses (start with 1 or 3, 26-35 chars)
        for word in content.split_whitespace() {
            if (word.starts_with('1') || word.starts_with('3'))
                && word.len() >= 26
                && word.len() <= 35
                && word.chars().all(|c| c.is_alphanumeric())
            {
                count += 1;
            }
            // Ethereum addresses (0x + 40 hex chars)
            if word.starts_with("0x")
                && word.len() == 42
                && word[2..].chars().all(|c| c.is_ascii_hexdigit())
            {
                count += 1;
            }
        }
        count
    }

    /// Count URLs with raw IP addresses
    fn count_ip_urls(content: &str) -> usize {
        let mut count = 0;
        for part in content.split_whitespace() {
            let after_proto = part
                .strip_prefix("https://")
                .or_else(|| part.strip_prefix("http://"));

            if let Some(host) = after_proto {
                // Check if starts with digit (potential IP)
                if host.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Count base64-like strings (40+ chars of base64 alphabet)
    fn count_base64_strings(content: &str) -> (usize, usize) {
        let mut total = 0;
        let mut long_count = 0;

        for word in content.split(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == '`') {
            if word.len() >= 40 {
                let is_b64 = word.chars().all(|c| {
                    c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='
                });
                if is_b64 {
                    total += 1;
                    if word.len() > 100 {
                        long_count += 1;
                    }
                }
            }
        }
        (total, long_count)
    }

    /// Extract file type one-hot encoding (9 features)
    fn extract_file_type_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        const FILE_TYPES: &[&str] = &[
            "python_script", "javascript", "shell_script", "elf",
            "macho", "pe", "go", "java_class", "unknown",
        ];

        let file_type = &report.target.file_type;
        let mut normalized = Self::normalize_file_type(file_type);

        // If this is an archive and we have contents, use the most common file type from contents
        if !report.archive_contents.is_empty() && normalized == "unknown" {
            let mut type_counts = std::collections::HashMap::new();
            for entry in &report.archive_contents {
                let entry_type = Self::normalize_file_type(&entry.file_type);
                if entry_type != "unknown" {
                    *type_counts.entry(entry_type).or_insert(0) += 1;
                }
            }
            // Use the most common file type from archive contents
            if let Some((most_common, _)) = type_counts.iter().max_by_key(|(_, count)| *count) {
                normalized = most_common;
            }
        }

        for ft in FILE_TYPES {
            let value = if *ft == normalized { 1.0 } else { 0.0 };
            features.push(&format!("ftype_{}", ft), value);
        }
    }

    /// Normalize file type string to known types
    fn normalize_file_type(raw: &str) -> &'static str {
        match raw.to_lowercase().as_str() {
            "python" | "python_script" => "python_script",
            "js" | "javascript" => "javascript",
            "bash" | "sh" | "zsh" | "shell_script" | "shell" => "shell_script",
            "elf" | "elf64" | "elf32" => "elf",
            "macho" | "macho64" | "macho32" => "macho",
            "pe" | "exe" | "dll" => "pe",
            "go" | "golang" => "go",
            "java_class" | "class" => "java_class",
            _ => "unknown",
        }
    }

    /// Extract byte histogram + anomaly features (260 features)
    fn extract_byte_histogram(&self, report: &AnalysisReport, file_path: Option<&Path>, features: &mut FeatureVector) {
        const BYTES_TO_READ: usize = 65536; // 64KB, matches Python

        let mut byte_counts = [0u32; 256];
        let mut total_bytes = 0u32;

        if let Some(path) = file_path {
            if let Ok(mut file) = File::open(path) {
                let mut buffer = vec![0u8; BYTES_TO_READ];
                if let Ok(n) = file.read(&mut buffer) {
                    for &byte in &buffer[..n] {
                        byte_counts[byte as usize] += 1;
                    }
                    total_bytes = n as u32;
                }
            }
        }

        // Normalize and add features
        let total = total_bytes.max(1) as f32;
        let mut frequencies = [0.0f32; 256];
        for i in 0..256 {
            frequencies[i] = byte_counts[i] as f32 / total;
            features.push(&format!("byte_{:02x}", i), frequencies[i]);
        }

        // Byte anomaly features (4 features)
        // Get baseline for file type (use uniform if not available)
        let baseline = Self::get_byte_baseline(Self::normalize_file_type(&report.target.file_type));

        // Calculate anomaly metrics
        let mut anomaly_sum = 0.0f32;
        let mut anomaly_max = 0.0f32;
        let mut overrep_count = 0u32;
        let mut underrep_count = 0u32;

        for i in 0..256 {
            let diff = (frequencies[i] - baseline[i]).abs();
            anomaly_sum += diff;
            if diff > anomaly_max {
                anomaly_max = diff;
            }
            // Count bytes significantly over/under represented
            if frequencies[i] > baseline[i] * 2.0 && frequencies[i] > 0.01 {
                overrep_count += 1;
            }
            if frequencies[i] < baseline[i] * 0.5 && baseline[i] > 0.01 {
                underrep_count += 1;
            }
        }

        features.push("byte_anomaly_mean", anomaly_sum / 256.0);
        features.push("byte_anomaly_max", anomaly_max);
        features.push("byte_overrep_count", overrep_count as f32);
        features.push("byte_underrep_count", underrep_count as f32);
    }

    /// Get baseline byte distribution for a file type (uniform for now)
    fn get_byte_baseline(_file_type: &str) -> [f32; 256] {
        // For simplicity, use uniform distribution
        // In production, this could load precomputed baselines per file type
        [1.0 / 256.0; 256]
    }

    /// Default known capabilities (subset - full list loaded from config)
    fn default_capabilities() -> Vec<String> {
        vec![
            // Execution
            "exec/eval/code",
            "exec/command/shell",
            "exec/command/subprocess",
            "exec/dylib/load",
            "exec/download/execute",
            // Credentials
            "credential/browser/cookies",
            "credential/browser/passwords",
            "credential/environment/api_keys",
            "credential/keychain",
            // Network
            "net/http/client",
            "net/http/post",
            "net/socket/connect",
            "net/socket/raw",
            // C2
            "c2/hardcoded_ip",
            "c2/dns_tunnel",
            "c2/beacon",
            // Anti-analysis
            "anti-analysis/vm_detect",
            "anti-analysis/debugger_detect",
            "anti-analysis/sandbox_detect",
            // Exfiltration
            "exfil/data/encode",
            "exfil/data/compress",
            "exfil/screenshot",
            // Persistence
            "persist/launchd",
            "persist/cron",
            "persist/startup",
            // Filesystem
            "fs/read/sensitive",
            "fs/write/system",
            "fs/delete",
            // Crypto
            "crypto/encrypt",
            "crypto/decrypt",
            "crypto/base64/encode",
            "crypto/base64/decode",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    /// Extract binary metrics from radare2 analysis (35 features)
    fn extract_binary_metrics(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        let binary_opt = report.metrics.as_ref().and_then(|m| m.binary.as_ref());
        if let Some(binary) = binary_opt {
            // Entropy features (5)
            features.push("bm_overall_entropy", binary.overall_entropy);
            features.push("bm_code_entropy", binary.code_entropy);
            features.push("bm_data_entropy", binary.data_entropy);
            features.push("bm_entropy_variance", binary.entropy_variance);
            features.push("bm_high_entropy_regions", binary.high_entropy_regions as f32);

            // Section features (6)
            features.push("bm_section_count", binary.section_count as f32);
            features.push("bm_executable_sections", binary.executable_sections as f32);
            features.push("bm_writable_sections", binary.writable_sections as f32);
            features.push("bm_wx_sections", binary.wx_sections as f32); // W+X is suspicious
            features.push("bm_section_name_entropy", binary.section_name_entropy);
            features.push("bm_largest_section_ratio", binary.largest_section_ratio);

            // Import/Export features (3)
            features.push("bm_import_count", binary.import_count as f32);
            features.push("bm_export_count", binary.export_count as f32);
            features.push("bm_import_entropy", binary.import_entropy);

            // String features (4)
            features.push("bm_string_count", binary.string_count as f32);
            features.push("bm_avg_string_entropy", binary.avg_string_entropy);
            features.push("bm_high_entropy_strings", binary.high_entropy_strings as f32);
            features.push("bm_strings_in_code", binary.strings_in_code as f32);

            // Function features (6)
            features.push("bm_function_count", binary.function_count as f32);
            features.push("bm_avg_function_size", binary.avg_function_size);
            features.push("bm_tiny_functions", binary.tiny_functions as f32);
            features.push("bm_huge_functions", binary.huge_functions as f32);
            features.push("bm_indirect_calls", binary.indirect_calls as f32);
            features.push("bm_indirect_jumps", binary.indirect_jumps as f32);

            // Complexity features (5)
            features.push("bm_avg_complexity", binary.avg_complexity);
            features.push("bm_max_complexity", binary.max_complexity as f32);
            features.push("bm_high_complexity_funcs", binary.high_complexity_functions as f32);
            features.push("bm_total_basic_blocks", binary.total_basic_blocks as f32);
            features.push("bm_avg_basic_blocks", binary.avg_basic_blocks);

            // Control flow features (4)
            features.push("bm_linear_functions", binary.linear_functions as f32);
            features.push("bm_noreturn_functions", binary.noreturn_functions as f32);
            features.push("bm_leaf_functions", binary.leaf_functions as f32);
            features.push("bm_recursive_functions", binary.recursive_functions as f32);

            // Overlay features (2)
            features.push("bm_has_overlay", if binary.has_overlay { 1.0 } else { 0.0 });
            features.push("bm_overlay_entropy", binary.overlay_entropy);
        } else {
            // Fill with zeros for non-binary files
            let names = [
                "bm_overall_entropy", "bm_code_entropy", "bm_data_entropy",
                "bm_entropy_variance", "bm_high_entropy_regions",
                "bm_section_count", "bm_executable_sections", "bm_writable_sections",
                "bm_wx_sections", "bm_section_name_entropy", "bm_largest_section_ratio",
                "bm_import_count", "bm_export_count", "bm_import_entropy",
                "bm_string_count", "bm_avg_string_entropy", "bm_high_entropy_strings",
                "bm_strings_in_code",
                "bm_function_count", "bm_avg_function_size", "bm_tiny_functions",
                "bm_huge_functions", "bm_indirect_calls", "bm_indirect_jumps",
                "bm_avg_complexity", "bm_max_complexity", "bm_high_complexity_funcs",
                "bm_total_basic_blocks", "bm_avg_basic_blocks",
                "bm_linear_functions", "bm_noreturn_functions", "bm_leaf_functions",
                "bm_recursive_functions",
                "bm_has_overlay", "bm_overlay_entropy",
            ];
            for name in names {
                features.push(name, 0.0);
            }
        }
    }

    /// Extract composite scores (15 features)
    fn extract_composite_scores(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        // Obfuscation score (6 features)
        let obf_opt = report.metrics.as_ref().and_then(|m| m.obfuscation.as_ref());
        if let Some(obf) = obf_opt {
            features.push("cs_obfuscation_score", obf.score);
            features.push("cs_obfuscation_conf", obf.conf);
            features.push("cs_obf_naming", obf.naming_score);
            features.push("cs_obf_string", obf.string_score);
            features.push("cs_obf_structure", obf.structure_score);
            features.push("cs_obf_encoding", obf.encoding_score);
        } else {
            features.push("cs_obfuscation_score", 0.0);
            features.push("cs_obfuscation_conf", 0.0);
            features.push("cs_obf_naming", 0.0);
            features.push("cs_obf_string", 0.0);
            features.push("cs_obf_structure", 0.0);
            features.push("cs_obf_encoding", 0.0);
        }

        // Packing score (5 features)
        let pack_opt = report.metrics.as_ref().and_then(|m| m.packing.as_ref());
        if let Some(pack) = pack_opt {
            features.push("cs_packing_score", pack.score);
            features.push("cs_packing_conf", pack.conf);
            features.push("cs_pack_entropy", pack.entropy_score);
            features.push("cs_pack_import", pack.import_score);
            features.push("cs_pack_section", pack.section_score);
        } else {
            features.push("cs_packing_score", 0.0);
            features.push("cs_packing_conf", 0.0);
            features.push("cs_pack_entropy", 0.0);
            features.push("cs_pack_import", 0.0);
            features.push("cs_pack_section", 0.0);
        }

        // Supply chain score (4 features)
        let sc_opt = report.metrics.as_ref().and_then(|m| m.supply_chain.as_ref());
        if let Some(sc) = sc_opt {
            features.push("cs_supply_chain_score", sc.score);
            features.push("cs_sc_install_script", sc.install_script_score);
            features.push("cs_sc_dependency", sc.dependency_score);
            features.push("cs_sc_typosquat", sc.typosquat_score);
        } else {
            features.push("cs_supply_chain_score", 0.0);
            features.push("cs_sc_install_script", 0.0);
            features.push("cs_sc_dependency", 0.0);
            features.push("cs_sc_typosquat", 0.0);
        }
    }

    /// Extract capability confidence aggregates (8 features)
    fn extract_capability_confidence(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        let mut total_conf = 0.0f32;
        let mut max_conf = 0.0f32;
        let mut hostile_conf = 0.0f32;
        let mut suspicious_conf = 0.0f32;
        let mut high_conf_count = 0u32; // conf > 0.8
        let mut evidence_count = 0u32;

        for finding in &report.findings {
            total_conf += finding.conf;
            if finding.conf > max_conf {
                max_conf = finding.conf;
            }
            if finding.conf > 0.8 {
                high_conf_count += 1;
            }
            evidence_count += finding.evidence.len() as u32;

            // Sum confidence by criticality
            match finding.crit {
                cleave::Criticality::Hostile => hostile_conf += finding.conf,
                cleave::Criticality::Suspicious => suspicious_conf += finding.conf,
                _ => {}
            }
        }

        let finding_count = report.findings.len().max(1) as f32;

        features.push("conf_mean", total_conf / finding_count);
        features.push("conf_max", max_conf);
        features.push("conf_hostile_sum", hostile_conf);
        features.push("conf_suspicious_sum", suspicious_conf);
        features.push("conf_high_count", high_conf_count as f32);
        features.push("conf_evidence_total", evidence_count as f32);
        features.push("conf_evidence_mean", evidence_count as f32 / finding_count);
        features.push("conf_total", total_conf);
    }

    /// Extract xz-style supply chain attack indicators (18 features)
    ///
    /// These features are designed to detect sophisticated supply chain attacks like
    /// the xz backdoor (CVE-2024-3094) which exhibited:
    /// - Massive function size anomalies (avg jumped from 681 to 2360 bytes)
    /// - Injection of huge functions into otherwise normal libraries
    /// - Abused IFUNC resolvers and constructors
    /// - Unusual section names in object files
    fn extract_xz_style_features(&self, report: &AnalysisReport, features: &mut FeatureVector) {
        let binary_opt = report.metrics.as_ref().and_then(|m| m.binary.as_ref());

        // --- Function size distribution anomalies (6 features) ---
        if let Some(binary) = binary_opt {
            // Function size variance - high variance indicates injected outliers
            // Normal libraries have relatively uniform function sizes
            let func_count = binary.function_count.max(1) as f32;
            let avg_size = binary.avg_function_size;

            // Ratio of huge functions to total - xz had 4 huge funcs among many normal ones
            let huge_ratio = binary.huge_functions as f32 / func_count;
            features.push("xz_huge_func_ratio", huge_ratio);

            // Huge to tiny ratio - backdoors inject huge funcs without tiny stubs
            let tiny_plus_one = (binary.tiny_functions + 1) as f32;
            features.push("xz_huge_tiny_ratio", binary.huge_functions as f32 / tiny_plus_one);

            // Avg function size anomaly - normal .so files have avg ~100-500 bytes
            // xz backdoor pushed this to 2360, a 3-4x increase
            let size_anomaly = if avg_size > 1000.0 { 1.0 } else { avg_size / 1000.0 };
            features.push("xz_func_size_anomaly", size_anomaly);

            // Max complexity vs avg - huge gap indicates injected complex function
            let complexity_gap = if binary.avg_complexity > 0.0 {
                binary.max_complexity as f32 / binary.avg_complexity
            } else {
                0.0
            };
            features.push("xz_complexity_gap", complexity_gap.min(100.0));

            // High complexity + huge function combo (strong indicator)
            let complex_and_huge = if binary.max_complexity > 50 && binary.huge_functions > 0 {
                1.0
            } else {
                0.0
            };
            features.push("xz_complex_huge_combo", complex_and_huge);

            // Indirect calls/jumps ratio - backdoors use these for control flow obfuscation
            let indirect_ratio = (binary.indirect_calls + binary.indirect_jumps) as f32 / func_count;
            features.push("xz_indirect_ratio", indirect_ratio);
        } else {
            Self::push_zeros(features, &[
                "xz_huge_func_ratio", "xz_huge_tiny_ratio", "xz_func_size_anomaly",
                "xz_complexity_gap", "xz_complex_huge_combo", "xz_indirect_ratio",
            ]);
        }

        // --- Section name anomalies (4 features) ---
        // xz backdoor's .o file had 242 sections with garbled names like "lzma_block_buffer_encoda"
        let section_count = report.sections.len();
        let excessive_sections = if section_count > 50 { 1.0 } else { 0.0 };
        features.push("xz_excessive_sections", excessive_sections);

        // Section name entropy/anomaly detection
        let mut truncated_names = 0u32;
        let mut garbled_names = 0u32;
        for section in &report.sections {
            let name = &section.name;
            // Truncated names often end abruptly (like "lzma_block_buffer_encoda" instead of "encode")
            if name.len() > 10 && !name.ends_with('_') && !name.ends_with('t') {
                // Check if name looks truncated (ends in middle of word)
                let last_char = name.chars().last().unwrap_or('_');
                if last_char.is_alphabetic() && !"etsnd".contains(last_char) {
                    truncated_names += 1;
                }
            }
            // Check for unusual section name patterns
            if !name.starts_with('.') && !name.starts_with("__") && name.len() > 5 {
                // Legitimate sections typically start with . or __
                // or are short standard names
                garbled_names += 1;
            }
        }
        features.push("xz_truncated_section_names", truncated_names as f32);
        features.push("xz_garbled_section_names", garbled_names as f32);
        features.push("xz_section_count_log", (section_count as f32 + 1.0).ln());

        // --- IFUNC/Constructor abuse (3 features) ---
        // The xz backdoor hijacked symbol resolution via IFUNC
        let mut has_ifunc = false;
        let mut has_init_array = false;
        let mut has_fini_array = false;

        for section in &report.sections {
            let name_lower = section.name.to_lowercase();
            if name_lower.contains("ifunc") || name_lower.contains("gnu.hash") {
                has_ifunc = true;
            }
            if name_lower.contains("init_array") || name_lower.contains(".init") {
                has_init_array = true;
            }
            if name_lower.contains("fini_array") || name_lower.contains(".fini") {
                has_fini_array = true;
            }
        }

        // Also check functions for IFUNC-related patterns
        for func in &report.functions {
            let name_lower = func.name.to_lowercase();
            if name_lower.contains("ifunc") || name_lower.contains("resolver") {
                has_ifunc = true;
            }
        }

        features.push("xz_has_ifunc", if has_ifunc { 1.0 } else { 0.0 });
        features.push("xz_has_init_array", if has_init_array { 1.0 } else { 0.0 });
        features.push("xz_has_fini_array", if has_fini_array { 1.0 } else { 0.0 });

        // --- Specific hostile pattern detection (3 features) ---
        // Weight known supply chain attack patterns heavily
        let mut xz_style_attack = 0.0f32;
        let mut supply_chain_patterns = 0u32;
        let mut backdoor_conf_sum = 0.0f32;

        for finding in &report.findings {
            let id_lower = finding.id.to_lowercase();

            // Direct xz-style attack detection
            if id_lower.contains("xz") || id_lower.contains("supply-chain") || id_lower.contains("supplychain") {
                xz_style_attack = 1.0;
                backdoor_conf_sum += finding.conf;
            }

            // Other supply chain indicators
            if id_lower.contains("backdoor")
                || id_lower.contains("trojan")
                || id_lower.contains("implant")
                || id_lower.contains("hook")
                || id_lower.contains("hijack")
            {
                supply_chain_patterns += 1;
                backdoor_conf_sum += finding.conf;
            }
        }

        features.push("xz_attack_pattern", xz_style_attack);
        features.push("xz_supply_chain_patterns", supply_chain_patterns as f32);
        features.push("xz_backdoor_conf_sum", backdoor_conf_sum);

        // --- Shared library anomaly (2 features) ---
        // Shared libraries (.so, .dylib) with constructors + huge functions = suspicious
        let is_shared_lib = report.target.file_type.to_lowercase().contains("shared")
            || report.target.path.ends_with(".so")
            || report.target.path.contains(".so.")
            || report.target.path.ends_with(".dylib");

        let shared_lib_with_init = if is_shared_lib && has_init_array { 1.0 } else { 0.0 };
        features.push("xz_shared_lib_init", shared_lib_with_init);

        // Composite: shared lib + huge functions + init/fini = very suspicious
        let shared_lib_suspicious = if is_shared_lib {
            let binary_opt = report.metrics.as_ref().and_then(|m| m.binary.as_ref());
            if let Some(binary) = binary_opt {
                if binary.huge_functions > 0 && (has_init_array || has_fini_array) {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        features.push("xz_shared_lib_suspicious", shared_lib_suspicious);
    }

    /// Default known imports (subset - full list loaded from config)
    fn default_imports() -> Vec<String> {
        vec![
            // Dangerous C functions
            "system",
            "popen",
            "exec",
            "execve",
            "fork",
            "ptrace",
            "dlopen",
            "dlsym",
            // Network
            "socket",
            "connect",
            "send",
            "recv",
            "bind",
            "listen",
            "accept",
            // File operations
            "open",
            "read",
            "write",
            "unlink",
            "chmod",
            "chown",
            // Memory
            "mmap",
            "mprotect",
            "malloc",
            "free",
            // Crypto
            "EVP_EncryptInit",
            "EVP_DecryptInit",
            "AES_encrypt",
            // Python
            "eval",
            "exec",
            "compile",
            "__import__",
            "subprocess",
            "os.system",
            "socket.socket",
            // JavaScript
            "eval",
            "Function",
            "child_process",
            "fs.writeFile",
            "http.request",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vector_creation() {
        let mut fv = FeatureVector::with_capacity(100);
        fv.push("test_feature", 1.0);
        assert_eq!(fv.len(), 1);
        assert_eq!(fv.values[0], 1.0);
        assert_eq!(fv.names[0], "test_feature");
    }

    #[test]
    fn test_sanitize_feature_name() {
        // Brackets and angle brackets should be replaced with underscores
        assert_eq!(
            sanitize_feature_name("import_common_prefix[:-1].rfind"),
            "import_common_prefix_:-1_.rfind"
        );
        assert_eq!(
            sanitize_feature_name("std::__1::basic_string<char>"),
            "std::__1::basic_string_char_"
        );
        // Already clean names should pass through unchanged
        assert_eq!(sanitize_feature_name("simple_name"), "simple_name");
    }
}
