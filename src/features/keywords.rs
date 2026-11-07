//! Model-local words from rule IDs, descriptions and decoded context windows.
use std::collections::HashSet;

use anyhow::Result;

use super::context::ExtractContext;
use super::names::legacy_finding_id;
use super::report::{ParsedReport, RawNeeds};
use super::spec::KeywordSource;
use super::{FeatureWriter, MIN_CONFIDENCE};

/// Add the keywords of `text` to `into`: lowercased runs of ASCII letters and
/// digits, at least two long and holding a letter.
///
/// Lowercasing comes first and is Unicode-aware, matching the Python reference
/// (`str.lower()`): `K` (U+212A) becomes `k`, so it extends an ASCII word,
/// and `İ` becomes `i` plus a combining dot, which ends one. ASCII text skips
/// that copy, and a word already in the set costs no allocation.
fn add_words(text: &str, into: &mut HashSet<String>) {
    let lowered;
    let text = if text.is_ascii() {
        text
    } else {
        lowered = text.to_lowercase();
        &lowered
    };
    for word in text.split(|c: char| !c.is_ascii_alphanumeric()) {
        if word.len() < 2 || !word.bytes().any(|c| c.is_ascii_alphabetic()) {
            continue;
        }
        if word.bytes().any(|c| c.is_ascii_uppercase()) {
            into.insert(word.to_ascii_lowercase());
        } else if !into.contains(word) {
            into.insert(word.to_owned());
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(super) struct FileKeywords {
    ids: HashSet<String>,
    descriptions: HashSet<String>,
    full_ids: HashSet<String>,
    context: HashSet<String>,
    /// The first finding whose description is blank, which a description
    /// model trained before ABI 23 cannot score.
    missing_description: Option<String>,
}

pub(super) fn file_keywords(file: &cleave::types::CompactFile, needs: RawNeeds) -> FileKeywords {
    let mut result = FileKeywords::default();
    if needs.context_keywords {
        // The typed reader has already decoded byte text and legacy Z85.
        for window in &file.context {
            add_words(&String::from_utf8_lossy(&window.data), &mut result.context);
        }
    }
    if !needs.rule_keywords && !needs.description_keywords && !needs.full_rule_keywords {
        return result;
    }
    // Match the f32 representation used by compact findings at the threshold.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the gate compares in the compact report's f32"
    )]
    const COMPACT_MIN_CONFIDENCE: f32 = MIN_CONFIDENCE as f32;
    for finding in &file.findings {
        if finding.id.is_empty()
            || finding.confidence.is_nan()
            || finding.confidence < COMPACT_MIN_CONFIDENCE
        {
            continue;
        }
        let id = legacy_finding_id(&finding.id);
        if needs.rule_keywords {
            let relative = id.split_once("::").map_or_else(
                || id.rsplit('/').next().unwrap_or(""),
                |(_, relative)| relative,
            );
            add_words(relative, &mut result.ids);
        }
        if needs.full_rule_keywords {
            add_words(&id, &mut result.full_ids);
        }
        if needs.description_keywords {
            // Blank as Python's `str.isspace` has it, which training used:
            // Unicode whitespace plus the separators U+001C..=U+001F.
            if finding
                .description
                .chars()
                .all(|c| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
            {
                if result.missing_description.is_none() {
                    result.missing_description = Some(finding.id.clone());
                }
            } else {
                add_words(&finding.description, &mut result.descriptions);
            }
        }
    }
    result
}

impl ExtractContext {
    /// Checked extraction, including the input contract of description models.
    ///
    /// # Errors
    /// An ABI 22 description model refuses a finding with no description;
    /// from ABI 23 absent text reads as empty and the other features remain.
    pub fn try_extract(&self, report: &cleave::types::CompactReport) -> Result<Vec<f32>> {
        let parsed = ParsedReport::from_compact_report(report, self.raw_needs(), None);
        self.validate_keyword_inputs(&parsed)?;
        Ok(self.extract_from_parsed(&parsed))
    }

    pub(crate) fn validate_keyword_inputs(&self, parsed: &ParsedReport) -> Result<()> {
        if self.strict_keyword_descriptions && self.reads(KeywordSource::Description) {
            for summary in &parsed.summaries {
                if let Some(id) = &summary.keywords.missing_description {
                    anyhow::bail!("description keywords require descriptions; missing for {id}");
                }
            }
        }
        Ok(())
    }

    pub(super) fn write_keyword_features(
        &self,
        parsed: &ParsedReport,
        writer: &mut FeatureWriter<'_>,
    ) {
        for (source, slots) in &self.keyword_slots {
            for summary in &parsed.summaries {
                let keywords = &summary.keywords;
                let sets: &[&HashSet<String>] = match source {
                    KeywordSource::RuleId => &[&keywords.ids],
                    KeywordSource::Description => &[&keywords.descriptions],
                    KeywordSource::FullId => &[&keywords.full_ids],
                    KeywordSource::IdAndDescription => {
                        &[&keywords.full_ids, &keywords.descriptions]
                    }
                    KeywordSource::Context => &[&keywords.context],
                };
                for words in sets {
                    for word in *words {
                        if let Some(&slot) = slots.get(word) {
                            writer.set_slot(slot, 1.0);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::spec::FeatureSpec;
    use std::collections::HashMap;

    #[test]
    fn words_lowercase_like_python_before_splitting() {
        let mut got = HashSet::new();
        add_words("Hello WORLD hello x1 42 a ab", &mut got);
        // U+212A KELVIN SIGN lowercases to ASCII `k`; U+0130 lowercases to
        // `i` plus a combining dot, which splits the word.
        add_words("\u{212A}ey \u{130}stanbul", &mut got);
        let mut got: Vec<_> = got.into_iter().collect();
        got.sort();
        assert_eq!(got, ["ab", "hello", "key", "stanbul", "world", "x1"]);
    }

    #[test]
    fn keywords_match_python_fixture_and_preserve_base_abi() -> Result<()> {
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../testdata/features/keyword_parity.json"))?;
        for case in cases {
            let spec: FeatureSpec = serde_json::from_value(case["spec"].clone())?;
            spec.validate()?;
            assert!(spec.degraded_feature_names().is_empty());
            let context = ExtractContext::new(&spec);
            assert!(!context.relative_structure);
            let report = serde_json::from_value(case["report"].clone())?;
            let result = context.try_extract(&report);
            if case["error"].as_bool() == Some(true) {
                assert!(result.is_err(), "{}", case["name"]);
            } else {
                let expected: Vec<f32> = serde_json::from_value(case["expected"].clone())?;
                assert_eq!(result?, expected, "{}", case["name"]);
            }
        }
        Ok(())
    }

    #[test]
    fn confidence_boundary_agrees_between_base_and_keyword_features() -> Result<()> {
        let spec: FeatureSpec = serde_json::from_value(serde_json::json!({
            "version":23, "abi_version":23, "base_feature_abi":20, "trait_path_depth":3,
            "presence_vocab":["metadata"], "keyword_vocabs":{"descword":["boundary"]},
            "feature_names":["present:metadata", "agg:notable_findings_log", "descword:boundary"],
            "total_features":3, "standardized":false
        }))?;
        spec.validate()?;
        let context = ExtractContext::new(&spec);
        for (confidence, included) in [(0.65_f32, true), (0.65_f32.next_down(), false)] {
            let report = serde_json::from_value(serde_json::json!({"files":[{
                "id":0,"path":"sample","type":"pe","sha":"fixture","size":10,
                "traits":[{"id":"metadata/test::boundary", "crit":3,
                           "conf":confidence,"desc":"boundary"}]
            }]}))?;
            let expected = if included {
                vec![0.65, 2.0_f32.ln(), 1.0]
            } else {
                vec![0.0; 3]
            };
            assert_eq!(context.try_extract(&report)?, expected);
        }
        Ok(())
    }

    #[test]
    fn every_keyword_source_combination_matches_python() -> Result<()> {
        let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
            "../../testdata/features/keyword_combinations_parity.json"
        ))?;
        assert_eq!(cases.len(), 31 * 9);
        for case in cases {
            let spec: FeatureSpec = serde_json::from_value(case["spec"].clone())?;
            spec.validate()?;
            assert!(spec.degraded_feature_names().is_empty());
            let context = ExtractContext::new(&spec);
            let report = serde_json::from_value(case["report"].clone())?;
            let expected: Vec<f32> = serde_json::from_value(case["expected"].clone())?;
            assert_eq!(context.try_extract(&report)?, expected, "{}", case["name"]);
            let parsed = ParsedReport::from_compact_report(&report, RawNeeds::all(), None);
            context.validate_keyword_inputs(&parsed)?;
            assert_eq!(
                context.extract_from_parsed(&parsed),
                expected,
                "{}",
                case["name"]
            );
        }
        Ok(())
    }

    #[test]
    fn shared_parse_validates_descriptions_only_for_models_that_use_them() -> Result<()> {
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../testdata/features/keyword_parity.json"))?;
        let case = cases
            .iter()
            .find(|case| case["name"] == "missing-description")
            .ok_or_else(|| anyhow::anyhow!("missing fixture"))?;
        let mut spec: FeatureSpec = serde_json::from_value(case["spec"].clone())?;
        let description = ExtractContext::new(&spec);
        spec.feature_names
            .retain(|name| !name.starts_with("descword:"));
        spec.total_features = spec.feature_names.len();
        let ids = ExtractContext::new(&spec);
        let report = serde_json::from_value(case["report"].clone())?;
        let parsed = ParsedReport::from_compact_report(
            &report,
            ids.raw_needs().union(description.raw_needs()),
            None,
        );
        ids.validate_keyword_inputs(&parsed)?;
        assert!(description.validate_keyword_inputs(&parsed).is_err());
        assert_eq!(ids.extract_from_parsed(&parsed), ids.try_extract(&report)?);
        Ok(())
    }

    #[test]
    fn extended_sources_match_python_with_both_context_encodings() -> Result<()> {
        let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
            "../../testdata/features/extended_keyword_parity.json"
        ))?;
        for case in cases {
            let spec: FeatureSpec = serde_json::from_value(case["spec"].clone())?;
            spec.validate()?;
            assert!(spec.degraded_feature_names().is_empty());
            let context = ExtractContext::new(&spec);
            let report = serde_json::from_value(case["report"].clone())?;
            let expected: Vec<f32> = serde_json::from_value(case["expected"].clone())?;
            assert_eq!(context.try_extract(&report)?, expected, "{}", case["name"]);
            let parsed = ParsedReport::from_compact_report(&report, RawNeeds::all(), None);
            context.validate_keyword_inputs(&parsed)?;
            assert_eq!(
                context.extract_from_parsed(&parsed),
                expected,
                "{}",
                case["name"]
            );
            let mut legacy = spec.clone();
            legacy.version = 22;
            legacy.abi_version = 22;
            assert!(legacy.validate().is_err(), "new sources require ABI 23");
        }
        Ok(())
    }

    #[test]
    fn keyword_abi_requires_explicit_compatible_base() -> Result<()> {
        for (abi, base, transforms, valid) in [
            (22, None, false, false),
            (22, Some(22), false, false),
            (22, Some(20), false, true),
            (22, Some(21), false, false),
            (22, Some(21), true, true),
            (20, Some(20), false, false),
        ] {
            let spec: FeatureSpec = serde_json::from_value(serde_json::json!({
                "version":abi,"abi_version":abi,"base_feature_abi":base,
                "trait_path_depth":3,"presence_vocab":["metadata"],"feature_names":["present:metadata"],
                "total_features":1,"standardized":false,
                "metric_transforms":if transforms {Some(HashMap::<String,String>::new())} else {None}
            }))?;
            assert_eq!(spec.validate().is_ok(), valid);
        }
        Ok(())
    }
}
