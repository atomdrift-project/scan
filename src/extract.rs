//! Fast parallel feature extraction for training.
//!
//! Extracts features from samples using Rust (much faster than Python).
//! Outputs JSON format compatible with training pipeline.
//!
//! Crash recovery: Creates a .checkpoint file that tracks processed samples.
//! If the pipeline crashes (e.g., tree-sitter assertion), re-run the same
//! command and it will resume from where it left off, skipping already-processed samples.

use crate::features::FeatureExtractor;
use anyhow::{Context, Result};
use cleave::{CapabilityMapper, Criticality};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Run feature extraction on a batch of samples
pub fn run(
    input: &Path,
    output: &Path,
    label: u8,
    workers: usize,
    limit: Option<usize>,
    require_suspicious: bool,
    extension: Option<&str>,
) -> Result<()> {
    let start = Instant::now();

    // Collect sample paths (filters out "clean" files if require_suspicious)
    // Don't apply limit here if filtering - we'll apply it to the output instead
    let input_limit = if require_suspicious { None } else { limit };
    let samples = collect_samples(input, input_limit, require_suspicious, extension)?;
    let total = samples.len();

    if total == 0 {
        anyhow::bail!("No samples found in {:?}", input);
    }

    // Load checkpoint for crash recovery (returns both paths and samples)
    let checkpoint = load_checkpoint(output)?;
    let resume_count = checkpoint.samples.len();
    if resume_count > 0 {
        eprintln!("Resuming: {} samples already processed", resume_count);
    }

    let ext_info = extension.map(|e| format!(" [.{} only]", e)).unwrap_or_default();
    if require_suspicious {
        eprintln!(
            "Extracting features from {} samples using {} workers{} (filtering for suspicious traits)...",
            total, workers, ext_info
        );
    } else {
        eprintln!(
            "Extracting features from {} samples using {} workers{}...",
            total, workers, ext_info
        );
    }

    // Configure thread pool with larger stack size for complex analysis (e.g., deeply nested archives)
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .stack_size(16 * 1024 * 1024) // 16MB stack per thread (handles deep archive nesting)
        .build_global()
        .ok(); // Ignore if already initialized

    // Load feature extractor (vocabulary)
    let extractor = FeatureExtractor::load_default().unwrap_or_else(|e| {
        eprintln!("Warning: Could not load vocabulary: {}. Using defaults.", e);
        FeatureExtractor::new()
    });

    // Load cleave capability mapper once (shared across all threads)
    let capability_mapper = Arc::new(CapabilityMapper::new());

    // Progress counters
    let processed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let filtered = AtomicUsize::new(0);
    let accepted = AtomicUsize::new(0);

    // Filter out already processed samples for crash recovery
    let samples: Vec<_> = samples
        .into_iter()
        .filter(|path| {
            !checkpoint
                .processed_paths
                .contains(path.to_str().unwrap_or_default())
        })
        .collect();

    let remaining = samples.len();
    if resume_count > 0 {
        eprintln!("  {} new samples to process ({} already done)", remaining, resume_count);
    }

    // Process samples in parallel
    let mut results: Vec<_> = samples
        .par_iter()
        .filter_map(|path| {
            // Early exit if we've collected enough samples (with buffer for in-flight work)
            if let Some(lim) = limit {
                if accepted.load(Ordering::Relaxed) >= lim + workers * 2 {
                    return None;
                }
            }

            // Log current file for crash debugging (written before analysis starts)
            if std::env::var("LITMUS_DEBUG").is_ok() {
                eprintln!("  [DEBUG] Processing: {:?}", path);
            } else if is_archive_file(path) {
                // Always log before analyzing archives (in case of deep nesting crashes)
                eprintln!("  [ARCHIVE] Analyzing: {:?}", path);
            }

            // Catch panics from cleave analysis (e.g., tree-sitter crashes, stack overflows)
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                extract_sample(&extractor, &capability_mapper, path, label)
            }));

            let n = processed.fetch_add(1, Ordering::Relaxed) + 1;

            // Convert panic to error, then handle as normal
            let result = result.unwrap_or_else(|panic| {
                let panic_msg = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                Err(anyhow::anyhow!("Analysis panicked: {}", panic_msg))
            });

            match result {
                Ok(sample) => {
                    // Filter samples without suspicious/hostile traits if required
                    if require_suspicious && !sample.has_suspicious_traits {
                        filtered.fetch_add(1, Ordering::Relaxed);

                        // Update progress
                        if n.is_multiple_of(50) || n == remaining {
                            let acc = accepted.load(Ordering::Relaxed);
                            let filt = filtered.load(Ordering::Relaxed);
                            let fail = failed.load(Ordering::Relaxed);
                            let pct = (n * 100) / remaining;
                            eprint!("\r  [{:3}%] {}/{} processed | {} accepted, {} filtered, {} failed    ",
                                pct, n, remaining, acc, filt, fail);
                        }
                        return None;
                    }

                    let acc = accepted.fetch_add(1, Ordering::Relaxed) + 1;

                    // Checkpoint progress every 10 samples for crash recovery
                    if acc.is_multiple_of(10) {
                        if let Err(e) = append_checkpoint(output, &sample) {
                            eprintln!("  [WARN] Failed to write checkpoint: {}", e);
                        }
                    }

                    // Update progress
                    if n.is_multiple_of(50) || n == remaining {
                        let filt = filtered.load(Ordering::Relaxed);
                        let fail = failed.load(Ordering::Relaxed);
                        let pct = (n * 100) / remaining;
                        eprint!("\r  [{:3}%] {}/{} processed | {} accepted, {} filtered, {} failed    ",
                            pct, n, remaining, acc, filt, fail);
                    }

                    Some(sample)
                }
                Err(e) => {
                    // Silently skip unsupported file types
                    let err_str = format!("{:#}", e);
                    if !err_str.contains("Unsupported file type") {
                        let fail = failed.fetch_add(1, Ordering::Relaxed) + 1;

                        // Print warning without breaking progress line
                        let acc = accepted.load(Ordering::Relaxed);
                        let filt = filtered.load(Ordering::Relaxed);
                        let pct = (n * 100) / remaining;
                        eprint!("\r  [{:3}%] {}/{} processed | {} accepted, {} filtered, {} failed    \n",
                            pct, n, remaining, acc, filt, fail);
                        eprintln!("  [WARN] Failed: {:?}: {:#}", path, e);
                    } else {
                        // Update progress for skipped unsupported types
                        if n.is_multiple_of(50) || n == remaining {
                            let acc = accepted.load(Ordering::Relaxed);
                            let filt = filtered.load(Ordering::Relaxed);
                            let fail = failed.load(Ordering::Relaxed);
                            let pct = (n * 100) / remaining;
                            eprint!("\r  [{:3}%] {}/{} processed | {} accepted, {} filtered, {} failed    ",
                                pct, n, remaining, acc, filt, fail);
                        }
                    }
                    None
                }
            }
        })
        .collect();

    eprintln!(); // Newline after progress

    let filtered_count = filtered.load(Ordering::Relaxed);
    if filtered_count > 0 {
        eprintln!("  Filtered {} samples without suspicious traits", filtered_count);
    }

    // Merge with previously processed samples from checkpoint
    if !checkpoint.samples.is_empty() {
        let mut previous_results = checkpoint.samples;

        // Get feature names from new results, or use existing if only resuming
        let feature_names = results.first()
            .map(|s| s.feature_names.clone())
            .or_else(|| previous_results.first().map(|s| s.feature_names.clone()))
            .ok_or_else(|| anyhow::anyhow!("No samples successfully processed"))?;

        // Update feature names in checkpoint samples (may be empty from old checkpoint)
        for sample in &mut previous_results {
            if sample.feature_names.is_empty() {
                sample.feature_names = feature_names.clone();
            }
        }

        // Merge
        previous_results.extend(results);
        results = previous_results;
    }

    // Shuffle and apply limit to output (after filtering) when require_suspicious is set
    if let Some(lim) = limit {
        if results.len() > lim {
            eprintln!("  Randomly sampling {} from {} filtered results", lim, results.len());
            results.shuffle(&mut rand::rng());
            results.truncate(lim);
        }
    }

    let success = results.len();
    let fail_count = failed.load(Ordering::Relaxed);

    // Get feature names from first result
    let feature_names = results
        .first()
        .map(|s| s.feature_names.clone())
        .ok_or_else(|| anyhow::anyhow!("No samples successfully processed"))?;

    // Write output (newline to clear any cleave progress messages)
    eprintln!("\nWriting {} samples to {:?}...", success, output);
    write_output(output, &results, &feature_names)?;

    let elapsed = start.elapsed();
    let filter_count = filtered.load(Ordering::Relaxed);

    if filter_count > 0 {
        eprintln!(
            "Done! {} accepted, {} filtered, {} failed in {:.1}s ({:.1} samples/sec)",
            success,
            filter_count,
            fail_count,
            elapsed.as_secs_f64(),
            success as f64 / elapsed.as_secs_f64()
        );
    } else {
        eprintln!(
            "Done! {} succeeded, {} failed in {:.1}s ({:.1} samples/sec)",
            success,
            fail_count,
            elapsed.as_secs_f64(),
            success as f64 / elapsed.as_secs_f64()
        );
    }

    Ok(())
}

/// A processed sample with extracted features.
///
/// Contains both the raw feature vector and metadata needed for training/inference.
#[derive(Debug, Clone)]
struct ProcessedSample {
    /// Path to the original sample file
    path: String,
    /// Classification label (0=benign, 1=malicious)
    label: u8,
    /// Feature vector values
    features: Vec<f32>,
    /// Feature names (for explainability)
    feature_names: Vec<String>,
    /// Whether sample has suspicious/hostile traits
    has_suspicious_traits: bool,
}

/// Extract features from a single sample
fn extract_sample(
    extractor: &FeatureExtractor,
    _capability_mapper: &Arc<CapabilityMapper>,
    path: &Path,
    label: u8,
) -> Result<ProcessedSample> {
    // DEBUG: Before cleave
    if std::env::var("LITMUS_DEBUG_EXTRACT").is_ok() {
        eprintln!("[EXTRACT DEBUG] Starting analysis of: {:?}", path);
    }

    // Run cleave analysis
    // NOTE: Using analyze_file instead of analyze_file_with_mapper due to bug
    // where shared mapper returns empty findings in parallel execution
    let options = cleave::AnalysisOptions {
        disable_yara: false,
        disable_radare2: false, // Enable radare2 for binary metrics
        disable_upx: false,
        ..Default::default()
    };

    let report = cleave::analyze_file(path, &options)
        .context(format!("cleave failed for {:?}", path))?;


    // DEBUG: Check what cleave returned
    if std::env::var("LITMUS_DEBUG_EXTRACT").is_ok() {
        for (i, f) in report.findings.iter().take(3).enumerate() {
            eprintln!("[EXTRACT DEBUG]   {}: {} ({:?})", i, f.id, f.crit);
        }
    }

    // Check if sample has any suspicious or hostile findings
    let has_suspicious_traits = report
        .findings
        .iter()
        .any(|f| f.crit >= Criticality::Suspicious);

    // Extract features
    let fv = extractor.extract_with_path(&report, Some(path));


    Ok(ProcessedSample {
        path: path.display().to_string(),
        label,
        features: fv.values,
        feature_names: fv.names,
        has_suspicious_traits,
    })
}

/// Check if a path contains known archive file extensions.
fn is_archive_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_lowercase().as_str(),
                "zip" | "tar" | "gz" | "7z" | "tgz"
            )
        })
        .unwrap_or(false)
        || path
            .to_str()
            .map(|s| s.ends_with(".tar.gz"))
            .unwrap_or(false)
}

/// Check if a path should be filtered out (contains "clean" in the name or in .git/).
fn should_filter_path(path: &Path) -> bool {
    let path_str = path.to_string_lossy().to_lowercase();
    path_str.contains("clean") || path_str.contains("/.git/")
}

/// Check if a path matches the required extension.
///
/// Returns true if no extension filter is specified, or if the path's extension
/// matches the filter (case-insensitive).
fn matches_extension(path: &Path, extension: Option<&str>) -> bool {
    extension.is_none_or(|ext| {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case(ext))
    })
}

/// Collect sample paths from input (directory, file list, or stdin with "-")
fn collect_samples(
    input: &Path,
    limit: Option<usize>,
    filter_clean: bool,
    extension: Option<&str>,
) -> Result<Vec<PathBuf>> {
    let mut samples = Vec::new();

    if input.as_os_str() == "-" {
        // Read paths from stdin
        let stdin_handle = std::io::stdin();
        let reader = BufReader::new(stdin_handle.lock());
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                let path = PathBuf::from(trimmed);
                if path.is_file()
                    && !(filter_clean && should_filter_path(&path))
                    && matches_extension(&path, extension)
                {
                    samples.push(path);
                }
            }
        }
    } else if input.is_dir() {
        // Recursively collect files from directory
        collect_from_dir(input, &mut samples, filter_clean, extension)?;
    } else if input.is_file() {
        // Read file list (one path per line)
        let file = File::open(input)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            let path = PathBuf::from(line.trim());
            if path.is_file()
                && !(filter_clean && should_filter_path(&path))
                && matches_extension(&path, extension)
            {
                samples.push(path);
            }
        }
    } else {
        anyhow::bail!("Input path does not exist: {:?}", input);
    }

    // Shuffle and apply limit for random sampling
    if let Some(lim) = limit {
        samples.shuffle(&mut rand::rng());
        samples.truncate(lim);
    }

    Ok(samples)
}

/// Recursively collect files from a directory
fn collect_from_dir(
    dir: &Path,
    samples: &mut Vec<PathBuf>,
    filter_clean: bool,
    extension: Option<&str>,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip .git directories entirely
        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name == ".git" {
                    continue;
                }
            }
            collect_from_dir(&path, samples, filter_clean, extension)?;
        } else if path.is_file() {
            // Skip hidden files and common non-sample files
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with('.') && !name.ends_with(".json") && !name.ends_with(".txt") {
                // Skip files with "clean" in the path if filtering
                if filter_clean && should_filter_path(&path) {
                    continue;
                }
                // Skip files that don't match the extension filter
                if !matches_extension(&path, extension) {
                    continue;
                }
                samples.push(path);
            }
        }
    }
    Ok(())
}

/// Write output in JSON format compatible with training
fn write_output(path: &Path, samples: &[ProcessedSample], feature_names: &[String]) -> Result<()> {
    let mut file = File::create(path)?;

    // Write as JSON object with metadata
    writeln!(file, "{{")?;
    writeln!(file, "  \"feature_names\": {},", serde_json::to_string(feature_names)?)?;
    writeln!(file, "  \"feature_count\": {},", feature_names.len())?;
    writeln!(file, "  \"sample_count\": {},", samples.len())?;
    writeln!(file, "  \"samples\": [")?;

    let last_idx = samples.len().saturating_sub(1);
    for (i, sample) in samples.iter().enumerate() {
        let comma = if i < last_idx { "," } else { "" };
        writeln!(
            file,
            "    {{\"path\": {}, \"label\": {}, \"features\": {}}}{}",
            serde_json::to_string(&sample.path)?,
            sample.label,
            serde_json::to_string(&sample.features)?,
            comma
        )?;
    }

    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;

    // Clean up checkpoint file after successful write
    let checkpoint_path = checkpoint_path(path);
    if checkpoint_path.exists() {
        let _ = fs::remove_file(&checkpoint_path); // Ignore errors
    }

    Ok(())
}

/// Get checkpoint file path for an output file.
///
/// The checkpoint file has the same name as the output file but with a
/// `.checkpoint` extension (e.g., `malware.json` → `malware.checkpoint`).
fn checkpoint_path(output: &Path) -> PathBuf {
    output.with_extension("checkpoint")
}

/// Checkpoint data loaded from disk for crash recovery.
///
/// Contains both a fast lookup set of processed paths and the full sample data
/// to enable resuming from where the pipeline left off.
#[derive(Debug, Default)]
struct Checkpoint {
    /// Set of already-processed sample paths for fast lookup
    processed_paths: HashSet<String>,
    /// Previously processed samples with features
    samples: Vec<ProcessedSample>,
}

/// Load checkpoint in a single pass (returns both path set and samples)
fn load_checkpoint(output: &Path) -> Result<Checkpoint> {
    let checkpoint_file = checkpoint_path(output);
    if !checkpoint_file.exists() {
        return Ok(Checkpoint::default());
    }

    let file = File::open(&checkpoint_file)?;
    let reader = BufReader::new(file);

    let (processed_paths, samples) = reader
        .lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let line = line.ok()?;

            // First line might be metadata (old format), skip if it parses as CheckpointMeta
            if i == 0 && serde_json::from_str::<CheckpointMeta>(&line).is_ok() {
                return None;
            }

            serde_json::from_str::<ProcessedSampleJson>(&line).ok()
        })
        .fold(
            (HashSet::new(), Vec::new()),
            |(mut paths, mut samples), json_sample| {
                paths.insert(json_sample.path.clone());
                samples.push(ProcessedSample {
                    path: json_sample.path,
                    label: json_sample.label,
                    features: json_sample.features,
                    feature_names: json_sample.feature_names.unwrap_or_default(),
                    has_suspicious_traits: true,
                });
                (paths, samples)
            },
        );

    Ok(Checkpoint {
        processed_paths,
        samples,
    })
}

/// Append a sample to the checkpoint file (one JSON object per line).
///
/// This enables crash recovery by incrementally saving processed samples.
/// Each line is a complete JSON object that can be independently parsed.
fn append_checkpoint(output: &Path, sample: &ProcessedSample) -> Result<()> {
    use std::fs::OpenOptions;

    let checkpoint = checkpoint_path(output);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(checkpoint)?;

    let json_sample = ProcessedSampleJson {
        path: sample.path.clone(),
        label: sample.label,
        features: sample.features.clone(),
        feature_names: Some(sample.feature_names.clone()),
    };

    writeln!(file, "{}", serde_json::to_string(&json_sample)?)?;
    Ok(())
}

/// JSON representation of a processed sample for checkpoint serialization.
///
/// Stored as one JSON object per line in the checkpoint file.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ProcessedSampleJson {
    path: String,
    label: u8,
    features: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    feature_names: Option<Vec<String>>,
}

/// Checkpoint metadata for backward compatibility with old checkpoint format.
///
/// Old checkpoints had feature names on the first line; new format includes
/// feature names with each sample. This struct is used only for detection
/// (to skip the metadata line), not to read the actual data.
#[derive(Debug, serde::Deserialize)]
struct CheckpointMeta {
    #[allow(dead_code)]
    feature_names: Vec<String>,
}
