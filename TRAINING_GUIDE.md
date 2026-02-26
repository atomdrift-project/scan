# Training Guide for 500+ Malware Samples Per File Type

## Prerequisites

You need at least **500 malware samples per file type** for per-filetype models to be effective.

### Current Status (as of last training)
```
Python:      32 malware samples (need 468 more)
JavaScript:  22 malware samples (need 478 more)
Shell:       36 malware samples (need 464 more)
Go:          28 malware samples (need 472 more)
PE:         359 malware samples (need 141 more)
Unknown:    226 malware samples (need 274 more)
ELF:         41 malware samples (need 459 more)
```

## Step 1: Collect More Malware Samples

### PyPI Malware (Python)
```bash
# The Backstabbers Knife Collection has many PyPI packages
ls -d /Users/t/data/malware/datasets/Backstabbers-Knife-Collection/samples/pypi/*/ | wc -l
# Should show 1000+ malicious packages available

# Add them to your malware dataset
cp -r /Users/t/data/malware/datasets/Backstabbers-Knife-Collection/samples/pypi/* \
      /Users/t/data/malware/pypi/
```

### NPM Malware (JavaScript)
Check for npm malware packages in your datasets or collect from:
- Backstabbers Knife Collection npm section
- Other malware repositories

### Shell Script Malware
```bash
# Find existing shell malware
find /Users/t/data/malware -name "*.sh" -o -name "*.bash"
```

## Step 2: Re-extract Features

Once you have 500+ samples per type:

```bash
# Clean old training data
rm -rf training/data/batches training/data/benign.json training/data/malware.json

# Extract features with larger limit (if needed)
make features-batched LIMIT=10000 WORKERS=8

# This will:
# - Skip .git/ directories (no false positives)
# - Detect file types from archive contents
# - Filter out samples without suspicious traits
# - Create training/data/benign.json and training/data/malware.json
```

## Step 3: Train Per-Filetype Models

```bash
# Train per-filetype models (min 50 samples by default)
make train-per-filetype

# For stricter requirements (min 100 samples):
make train-per-filetype MIN_SAMPLES=100

# This creates:
# - models/python.json (if Python has 50+ samples)
# - models/javascript.json (if JavaScript has 50+ samples)
# - models/pe.json (if PE has 50+ samples)
# - models/unknown.json (if unknown has 50+ samples)
# - models/model_registry.json (metadata)
```

## Step 4: Verify Hybrid Classifier Activation

```bash
# Check which models are "well-trained" (500+ malware samples)
litmus_VERBOSE=1 ./target/release/litmus scan /path/to/sample.py 2>&1 | grep -A 10 "Hybrid Classifier"

# Expected output when you have 500+ samples:
# Hybrid Classifier Statistics:
#   Per-filetype registry: available
#   Well-trained models: 3
#
#   File types with per-type models (500+ malware samples):
#     python: 650 malware samples
#     javascript: 520 malware samples
#     pe: 800 malware samples
#
#   Fallback: single model (for rare types)
```

## Step 5: Train Single Model Too

Even with per-filetype models, train a single model as fallback:

```bash
# Train single unified model
make train

# This creates:
# - models/litmus_v1.json (fallback for rare file types)
# - models/feature_names.json
# - models/evaluation.json
```

## How the Hybrid Classifier Works

The hybrid classifier automatically:

1. **Detects file type** from path and archive contents
2. **Checks if per-filetype model is well-trained** (500+ malware samples)
3. **Routes intelligently**:
   - If well-trained per-type model exists → use it (no file type bias)
   - Otherwise → use single model (better than poorly-trained per-type)

### Example Routing Logic

```
classify("malware.py", features):
  - Detect type: "python"
  - Check registry: python model has 650 malware samples (>500) ✓
  - Use: models/python.json (eliminates Python file type bias)

classify("malware.sh", features):
  - Detect type: "shell"
  - Check registry: shell model has 45 malware samples (<500) ✗
  - Use: models/litmus_v1.json (single model fallback)
```

## Expected Performance Improvements

### With 500+ Samples Per Type

**Python malware detection:**
```
Before (32 samples):  F1 = 0.421 (poor)
After (500+ samples): F1 = 0.95+  (excellent, estimated)
```

**JavaScript malware detection:**
```
Before (22 samples):  Model not trained (too few samples)
After (500+ samples): F1 = 0.95+  (excellent, estimated)
```

**Overall detection:**
```
Current (single model):        F1 = 0.982 (excellent, but has biases)
Future (hybrid with per-type): F1 = 0.985+ (excellent, no biases)
```

## Monitoring Training Quality

### Check Per-Filetype Model Stats

```bash
# View evaluation results for each model
cat models/python_evaluation.json | jq '{roc_auc, f1, n_samples, n_malware}'

# Expected output:
# {
#   "roc_auc": 0.985,
#   "f1": 0.952,
#   "n_samples": 3500,
#   "n_malware": 650
# }
```

### Run Comparison Analysis

```bash
# Compare single vs per-filetype performance
training/.venv/bin/python scripts/compare-models.py

# Look for:
# - Per-type models should have F1 > 0.90 if well-trained
# - Cross-filetype patterns should be identified
# - File type bias should be minimal
```

## Troubleshooting

### "Well-trained models: 0" even after training

Check if models actually have 500+ malware samples:

```bash
# Inspect model registry
jq '.models | to_entries[] | {type: .key, malware: .value.n_malware}' models/model_registry.json

# If n_malware < 500, you need more samples
```

### Per-filetype model performance is poor

Possible causes:
1. **Class imbalance**: Too many benign vs malware
   - Solution: Collect more malware or downsample benign
2. **Label noise**: False positives in training data
   - Solution: Use `--require-suspicious` flag during extraction
3. **Insufficient variety**: All malware is similar
   - Solution: Collect diverse malware families

### Archive file types not detected correctly

Test cleave analysis:

```bash
# Analyze an archive
./target/release/litmus extract <(echo "/path/to/malware.tar.gz") \
  -o /tmp/test.json -l 1 -w 1

# Check detected file type
jq '.samples[0] | .path, (.features[17])' /tmp/test.json

# feature[17] should be ftype_python_script if it's a Python package
```

## Advanced: Training Parameters

### Adjust Minimum Samples Threshold

Edit `src/hybrid_model.rs`:

```rust
/// Minimum malware samples required
const MIN_MALWARE_SAMPLES: usize = 500;  // Adjust this

// Lower = more aggressive (uses per-type models sooner)
// Higher = more conservative (waits for more data)
```

Recommended thresholds:
- **500**: Balanced (default)
- **300**: Aggressive (for limited datasets)
- **1000**: Conservative (for large datasets with high quality requirements)

### Custom Model Registry

Create `models/model_registry.json` manually:

```json
{
  "models": {
    "python": {
      "model_file": "python.json",
      "evaluation_file": "python_evaluation.json",
      "n_samples": 3500,
      "n_benign": 2850,
      "n_malware": 650,
      "roc_auc": 0.985,
      "f1": 0.952,
      "optimal_threshold": 0.65
    }
  },
  "feature_count": 6023,
  "default_model": "litmus_v1.json"
}
```

## Summary Checklist

Before retraining with 500+ samples:

- [ ] Collected 500+ malware samples per file type
- [ ] Verified samples are properly labeled (suspicious/hostile traits)
- [ ] Cleaned out .git/ directories and false positives
- [ ] Extracted features with `make features-batched`
- [ ] Trained per-filetype models with `make train-per-filetype`
- [ ] Trained single model fallback with `make train`
- [ ] Verified hybrid classifier activation with `litmus_VERBOSE=1`
- [ ] Tested on known malware samples
- [ ] Compared performance with `scripts/compare-models.py`

Once complete, the hybrid classifier will automatically use the best model for each file type!
