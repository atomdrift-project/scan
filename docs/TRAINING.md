# litmus ML Model Training Guide

This guide walks you through training the litmus malware detection model from scratch.

## Overview

litmus uses XGBoost machine learning models to classify files as benign or malicious based on static features extracted from binaries, scripts, and archives. The training pipeline consists of four main stages:

1. **Data Collection** - Gather benign and malicious samples
2. **Feature Extraction** - Extract static features from samples using litmus
3. **Model Training** - Train XGBoost classifier(s)
4. **Evaluation** - Test model performance and validate results

## Prerequisites

- Rust toolchain (for building litmus CLI)
- Python 3.8+ with virtualenv
- Sufficient disk space (datasets can be large)
- cleave tool (for capability detection) - symlinked to `traits/`

## Quick Start

If you've already run feature extraction, here's what to do next:

```bash
# You are here: features extracted
# training/data/benign.json and training/data/malware.json exist

# Step 1: Set up Python environment
make venv

# Step 2: Train a single unified model
make train

# Step 3: Evaluate the model
make evaluate

# Step 4: Test on held-out samples (optional)
make test-eval
```

## Stage 1: Data Collection

### Benign Samples

Use the Go-based collector to download benign samples from trusted package repositories:

```bash
# Build collector
make collector

# Download from all sources (default: 50 packages per source)
make data MAX_PACKAGES=100 WORKERS=8

# Or download from specific sources
make data-pypi MAX_PACKAGES=500    # PyPI Python packages
make data-npm MAX_PACKAGES=500     # NPM JavaScript packages
make data-cargo MAX_PACKAGES=100   # Rust crates
make data-go MAX_PACKAGES=100      # Go modules
make data-debian MAX_PACKAGES=100  # Debian packages
make data-wolfi MAX_PACKAGES=100   # Wolfi packages
make data-bsd                      # BSD core sets
```

Benign samples are stored in `$HOME/data/known-good` by default (configurable with `DATA_DIR`).

### Malicious Samples

Collect malware samples from:
- [Backstabbers Knife Collection](https://github.com/backstabbers-knife-collection/backdoored-pypi-packages) (PyPI malware)
- [MalwareBazaar](https://bazaar.abuse.ch/)
- [VirusShare](https://virusshare.com/)
- Your own malware repository

Place malicious samples in `$HOME/data/malware` by default (configurable with `MALWARE_DIR`).

**Important:** Ensure you have sufficient samples per class:
- Minimum: 100 samples per class
- Recommended: 1,000+ samples per class
- Optimal: 10,000+ samples per class

## Stage 2: Feature Extraction

**You are here!** You've already run feature extraction.

### What You Ran

```bash
make features-batched \
    LIMIT=50000 \
    WORKERS=16 \
    BATCH_SIZE=2048 \
    DATA_DIR=$HOME/data/known-good \
    MALWARE_DIR=$HOME/data/malware \
    FEATURES_DIR=training/data
```

This command:
1. Found all files in `DATA_DIR` and `MALWARE_DIR`
2. Randomly sampled up to 50,000 files from each directory
3. Extracted static features using 16 parallel workers
4. Processed files in batches of 2,048 (crash-resilient)
5. Created two JSON files:
   - `training/data/benign.json` - Features from benign samples (label=0)
   - `training/data/malware.json` - Features from malicious samples (label=1)

### Feature Types

litmus extracts several categories of features:

- **File type features** - PE, ELF, Python, JavaScript, shell, etc.
- **Imports/Exports** - API calls, library imports, system calls
- **Capabilities** - Behaviors detected by cleave (network, crypto, process, etc.)
- **Strings** - URLs, IP addresses, suspicious patterns
- **Entropy** - Randomness metrics, compression ratios
- **Structural** - Section sizes, header characteristics
- **Archive contents** - For tarballs, wheels, gems, etc.

### Batch Extraction (What You Used)

The `features-batched` target is **crash-resilient** and **resumable**:
- Processes files in small batches (default: 1024 files)
- Skips already-processed batches on restart
- Saves batch files to `training/data/batches/`
- Merges all batches into final JSON files

If extraction crashes, just re-run the same command - it will resume from where it left off.

### Feature Extraction Options

```bash
# Extract features with limits
make features LIMIT=10000          # Max 10k samples per class
make features WORKERS=32           # Use 32 CPU cores

# Filter by file extension
make features-batched EXTENSION=py # Only Python files
make features-batched EXTENSION=js # Only JavaScript files
make features-batched EXTENSION=exe # Only Windows PE files

# Require suspicious traits for malware (filter false positives)
# This is done automatically for malware in features-batched
```

### Verify Extracted Features

Check what you extracted:

```bash
# Install jq for JSON parsing
brew install jq  # macOS

# Check benign samples
jq '.samples | length' training/data/benign.json
jq '.feature_names | length' training/data/benign.json

# Check malware samples
jq '.samples | length' training/data/malware.json

# View sample paths
jq '.samples[0:5] | .[].path' training/data/benign.json
```

## Stage 3: Model Training

Now that you have extracted features, you're ready to train.

### Option A: Single Unified Model (Recommended for Starting)

Train one model that works for all file types:

```bash
# Set up Python environment (first time only)
make venv

# Train the model
make train
```

This creates:
- `models/litmus_v1.json` - XGBoost model in JSON format
- `models/feature_names.json` - Feature name mapping
- `models/evaluation.json` - Training metrics (ROC AUC, F1, etc.)
- `models/capabilities.json` - List of capability features
- `models/imports.json` - List of import features

**Training Process:**
1. Loads benign and malware features from JSON
2. Aligns feature sets (handles missing features)
3. Performs 5-fold stratified cross-validation
4. Handles class imbalance with `scale_pos_weight`
5. Trains final model on all data with early stopping
6. Computes SHAP feature importance (top contributing features)
7. Finds optimal decision threshold (maximizes F1 score)

**Expected Output:**

```
Loading training/data/benign.json...
  Loaded 45823 samples, 6023 features
Loading training/data/malware.json...
  Loaded 48912 samples, 6023 features

Combined dataset: 94735 samples
  Benign:    45823
  Malicious: 48912

Training with 5-fold cross-validation...
  Class distribution: 45823 benign, 48912 malware
  Class weight ratio: 0.94

  Fold    AUC       F1     Prec   Recall
  ----------------------------------------------
  1      0.9945   0.9812   0.9823   0.9801
  2      0.9938   0.9798   0.9815   0.9781
  3      0.9951   0.9825   0.9831   0.9819
  4      0.9943   0.9807   0.9822   0.9792
  5      0.9947   0.9815   0.9828   0.9802
  ----------------------------------------------
  Mean   0.9945   0.9811
  Std    0.0005   0.0010

Cross-Validation Results (unbiased estimate):
  Accuracy:  0.9811
  Precision: 0.9824
  Recall:    0.9799
  F1 Score:  0.9811
  ROC AUC:   0.9945
  Avg Prec:  0.9943

Optimal threshold (max F1): 0.587

Top 20 Most Important Features (SHAP):
  1.  cap_execution/shell-execution           0.2134
  2.  cap_network/http-post                   0.1876
  3.  cap_obfuscation/base64                  0.1654
  4.  import_eval                             0.1432
  5.  import_exec                             0.1298
  ...

Model saved to models/litmus_v1.json
```

### Option B: Per-Filetype Models (Advanced)

Train separate models for each file type to eliminate file-type bias:

```bash
# Train per-filetype models (requires 50+ samples per type by default)
make train-per-filetype

# Or with stricter requirements
make train-per-filetype MIN_SAMPLES=100
```

This creates:
- `models/python.json` - Model for Python files
- `models/javascript.json` - Model for JavaScript files
- `models/pe.json` - Model for Windows PE files
- `models/elf.json` - Model for Linux ELF files
- `models/shell.json` - Model for shell scripts
- `models/unknown.json` - Model for unknown file types
- `models/model_registry.json` - Metadata about all models

**When to use per-filetype models:**
- You have 500+ malware samples per file type
- You want to eliminate file-type bias (e.g., "Python files are benign")
- You need specialized detection for specific languages

**Hybrid Classifier:**
litmus automatically uses per-filetype models when they have sufficient training data (500+ malware samples), falling back to the single model for rare file types.

### Training Parameters

Edit parameters in `training/train_from_json.py`:

```python
params = {
    "objective": "binary:logistic",
    "max_depth": 6,              # Tree depth (4-10)
    "eta": 0.1,                  # Learning rate (0.01-0.3)
    "subsample": 0.8,            # Row sampling (0.5-1.0)
    "colsample_bytree": 0.8,     # Column sampling (0.5-1.0)
    "min_child_weight": 1,       # Minimum samples per leaf
    "scale_pos_weight": ratio,   # Class imbalance handling
}
```

## Stage 4: Evaluation

### Cross-Validation Results

The training process automatically performs 5-fold cross-validation and reports unbiased performance estimates. Look for:

- **ROC AUC** - Area under ROC curve (0.99+ is excellent)
- **F1 Score** - Harmonic mean of precision and recall (0.95+ is excellent)
- **Precision** - True positives / predicted positives (minimize false alarms)
- **Recall** - True positives / actual positives (minimize false negatives)

### Evaluate on Training Data

```bash
# Re-evaluate the model (requires features.npz)
make evaluate
```

This uses the NPZ features file created during training. To create it:

```bash
# Generate NPZ features during training
make train FEATURES_NPZ=training/data/features.npz
```

### Evaluate on Held-Out Test Set

Use completely separate test samples that were **never used for training**:

```bash
# Evaluate on test set (should be in different directory)
make test-eval TEST_DIR=/path/to/test/samples
```

This:
1. Extracts features from test samples using litmus
2. Runs predictions using the trained model
3. Computes metrics on truly unseen data
4. Saves report to `evaluation/test_report.json`

**Important:** Test samples must be from a different source than training data to avoid data leakage.

### Predict on Individual Files

```bash
# Classify a single file
./target/release/litmus scan /path/to/suspicious.py

# Get detailed scores
./target/release/litmus scan --verbose /path/to/sample.exe

# Explain prediction with SHAP values
make explain FILE=/path/to/sample.py
```

## Understanding Model Performance

### Confusion Matrix

```
                 Predicted
                 Benign  Malware
  Actual Benign   9145     32     (TN=9145, FP=32)
  Actual Malware    48   9698     (FN=48, TP=9698)
```

- **True Negatives (TN):** Benign files correctly identified as benign
- **False Positives (FP):** Benign files incorrectly flagged as malicious (minimize these!)
- **False Negatives (FN):** Malware files incorrectly marked as benign (minimize these!)
- **True Positives (TP):** Malware correctly detected as malicious

### Feature Importance (SHAP)

SHAP (SHapley Additive exPlanations) values show which features contribute most to predictions:

```
Top features for malware detection:
  1. cap_execution/shell-execution   - Executes shell commands
  2. cap_network/http-post           - Sends HTTP POST requests
  3. cap_obfuscation/base64          - Uses base64 encoding
  4. import_eval                     - Calls eval() (code execution)
  5. import_exec                     - Calls exec() (command execution)
```

Features with low SHAP importance can be removed to reduce model size.

### Optimal Threshold

The default threshold is 0.5, but the training process finds an optimal threshold that maximizes F1 score:

```python
# In Rust: src/main.rs or src/hybrid_model.rs
const THRESHOLD: f32 = 0.587;  // Adjust this based on training output
```

Higher threshold = fewer false positives, more false negatives
Lower threshold = more false positives, fewer false negatives

## Common Issues and Solutions

### 1. Class Imbalance

**Problem:** 10,000 benign samples, 100 malware samples
**Solution:** Use `scale_pos_weight` (automatically computed during training)

```bash
# Downsample benign samples during extraction
make features-batched LIMIT=1000  # Limits both classes to 1000 samples
```

### 2. Poor Performance on Specific File Types

**Problem:** JavaScript malware detection is poor (F1 < 0.80)
**Solution:** Train per-filetype models or collect more JavaScript malware

```bash
# Train per-filetype models
make train-per-filetype MIN_SAMPLES=50
```

### 3. High False Positive Rate

**Problem:** Too many benign files flagged as malicious
**Causes:**
- Insufficient benign training data
- Training data contamination (.git directories, test files)
- Threshold too low

**Solutions:**

```bash
# Collect more diverse benign samples
make data-pypi MAX_PACKAGES=1000
make data-npm MAX_PACKAGES=1000
make data-debian MAX_PACKAGES=500

# Re-extract features (filtering is built-in)
make features-batched LIMIT=10000

# Increase threshold (edit Rust code)
const THRESHOLD: f32 = 0.65;  // Increase from 0.5
```

### 4. Extraction Crashes or Hangs

**Problem:** Feature extraction crashes on large files or archives
**Solution:** Use batch extraction (you already did this!)

```bash
# Resume from where it crashed
make features-batched BATCH_SIZE=1024

# Debug specific batch
cat training/data/batches/benign_batch_0042.files | \
  ./target/release/litmus extract - -o /tmp/debug.json -l 0 -w 1
```

### 5. SHAP Analysis Fails

**Problem:** "SHAP not installed" during training
**Solution:**

```bash
# Install SHAP in virtual environment
training/.venv/bin/pip install shap

# Re-run training
make train
```

### 6. Model Size Too Large

**Problem:** Model file is very large (>100 MB)
**Solutions:**
- Remove low-importance features (SHAP analysis identifies them)
- Reduce tree depth: `max_depth=4` instead of `max_depth=6`
- Reduce number of trees: `num_boost_round=200` instead of `num_boost_round=500`

## Advanced Topics

### Hyperparameter Tuning

Use grid search or random search to find optimal parameters:

```python
# In train_from_json.py, try different values:
param_grid = {
    'max_depth': [4, 6, 8],
    'eta': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}
```

### Feature Engineering

Add custom features in `src/features.rs`:

```rust
pub fn extract_features(file: &Path, traits: &Traits) -> Features {
    // Add your custom features here
    features.suspicious_patterns = detect_patterns(file);
    features
}
```

### Ensemble Models

Combine multiple models for better accuracy:

```bash
# Train multiple models with different seeds
make train SEED=42
make train SEED=123
make train SEED=456

# Average predictions (implement in Rust)
```

### Model Versioning

Keep track of model versions:

```bash
# Tag models with version and date
cp models/litmus_v1.json models/litmus_v1_2024-01-27.json

# Update model in Rust
const MODEL_PATH: &str = "models/litmus_v1_2024-01-27.json";
```

### Continuous Training

Set up automated retraining with new samples:

```bash
#!/bin/bash
# retrain.sh - Daily retraining script

# Collect new benign samples
make data MAX_PACKAGES=10

# Extract features (append to existing)
make features-batched LIMIT=0  # No limit, use all

# Retrain
make train

# Evaluate
make test-eval

# Deploy if performance improves
if [ "$(check_metrics)" == "improved" ]; then
    cp models/litmus_v1.json models/litmus_production.json
fi
```

## Complete Pipeline Example

From scratch to production model:

```bash
# 1. Clone and build
git clone https://github.com/yourusername/litmus.git
cd litmus
make build

# 2. Collect benign samples (takes 30-60 minutes)
make data MAX_PACKAGES=500 WORKERS=16

# 3. Add malware samples (manual step)
cp -r /path/to/malware/* ~/data/malware/

# 4. Extract features (takes 2-4 hours for 50k samples)
make features-batched \
    LIMIT=25000 \
    WORKERS=16 \
    BATCH_SIZE=2048

# 5. Train model (takes 10-30 minutes)
make train

# 6. Evaluate on held-out test set
make test-eval TEST_DIR=/path/to/test/samples

# 7. Deploy
cargo build --release
./target/release/litmus scan /path/to/suspicious/file
```

## Performance Benchmarks

Typical performance on well-balanced dataset:

| Metric | Single Model | Per-Filetype (500+ samples) |
|--------|--------------|------------------------------|
| ROC AUC | 0.994 | 0.996 |
| F1 Score | 0.981 | 0.985 |
| Precision | 0.982 | 0.987 |
| Recall | 0.980 | 0.983 |
| False Positive Rate | 0.35% | 0.25% |
| Training Time | 15 min | 45 min |

## Next Steps

After training your model:

1. **Test thoroughly** - Evaluate on diverse samples
2. **Tune threshold** - Adjust based on your tolerance for FP/FN
3. **Monitor in production** - Track false positives and update model
4. **Collect feedback** - Use misclassifications to improve training data
5. **Retrain periodically** - Add new malware families and benign patterns

## Troubleshooting

### Training Fails with "Not enough samples"

Minimum 10 samples required. Collect more data:

```bash
make data MAX_PACKAGES=100
```

### "Feature mismatch" error during prediction

Features extracted during scanning don't match training features. Rebuild model:

```bash
make clean
make build
make features-batched
make train
```

### Model performs poorly on real-world samples

Common causes:
- Training/test data from same source (data leakage)
- Insufficient diversity in training data
- Malware samples outdated (model hasn't seen new techniques)

Solution: Expand and diversify training data.

## Resources

- XGBoost Documentation: https://xgboost.readthedocs.io/
- SHAP Documentation: https://shap.readthedocs.io/
- cleave Tool: https://github.com/yourusername/cleave
- Malware Datasets: See `docs/DATASETS.md`

## Getting Help

If you encounter issues:

1. Check `make help` for all available commands
2. Enable verbose mode: `litmus_VERBOSE=1 ./target/release/litmus scan file`
3. Review training logs in `training/data/`
4. Open an issue on GitHub with training output

## Summary

You've completed feature extraction. Next steps:

```bash
# Train the model
make train

# Evaluate it
make evaluate

# Test on real samples
./target/release/litmus scan /path/to/file
```

Good luck with your malware detection model!
