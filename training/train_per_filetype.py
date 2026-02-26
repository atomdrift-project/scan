#!/usr/bin/env python3
"""
Train separate XGBoost models per file type.

This addresses the problem where file type features dominate predictions
(e.g., "Python files are benign" because training only included JS malware).

Usage:
    python train_per_filetype.py --benign benign.json --malware malware.json --output-dir models/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Import the improved training functions
from train_from_json import (
    load_features,
    align_features,
    train_model_cv,
    compute_shap_importance,
    export_vocabulary,
    sanitize_feature_name,
)


def extract_filetype_from_path(path: str) -> str:
    """Extract file type from sample path (hacky but works)."""
    path_lower = path.lower()

    # Check for common extensions
    if path_lower.endswith('.py'):
        return 'python'
    elif path_lower.endswith('.js'):
        return 'javascript'
    elif path_lower.endswith(('.exe', '.dll', '.sys')):
        return 'pe'
    elif path_lower.endswith('.elf') or '/bin/' in path_lower or '/sbin/' in path_lower:
        return 'elf'
    elif path_lower.endswith(('.sh', '.bash')):
        return 'shell'
    elif path_lower.endswith('.rb'):
        return 'ruby'
    elif path_lower.endswith('.go'):
        return 'go'
    elif path_lower.endswith('.rs'):
        return 'rust'
    elif path_lower.endswith('.jar'):
        return 'java'
    elif path_lower.endswith('.apk'):
        return 'android'

    # Default to 'unknown'
    return 'unknown'


def group_by_filetype(X, y, paths, feature_names):
    """Group samples by file type."""
    groups = defaultdict(lambda: {'X': [], 'y': [], 'paths': []})

    for i in range(len(X)):
        ftype = extract_filetype_from_path(paths[i])
        groups[ftype]['X'].append(X[i])
        groups[ftype]['y'].append(y[i])
        groups[ftype]['paths'].append(paths[i])

    # Convert lists to arrays
    for ftype in groups:
        groups[ftype]['X'] = np.array(groups[ftype]['X'], dtype=np.float32)
        groups[ftype]['y'] = np.array(groups[ftype]['y'], dtype=np.int32)

    return groups


def main():
    parser = argparse.ArgumentParser(description="Train per-filetype XGBoost models")
    parser.add_argument("--benign", type=Path, required=True, help="Benign features JSON")
    parser.add_argument("--malware", type=Path, required=True, help="Malware features JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for models")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum samples per filetype")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis")
    args = parser.parse_args()

    # Load features
    X_benign, y_benign, feature_names_benign, paths_benign = load_features(args.benign)
    X_malware, y_malware, feature_names_malware, paths_malware = load_features(args.malware)

    # Align feature names if needed
    if feature_names_benign != feature_names_malware:
        print("\nAligning feature sets...")
        unified_names = sorted(set(feature_names_benign) | set(feature_names_malware))
        print(f"  Benign:  {len(feature_names_benign)} -> {len(unified_names)} features")
        print(f"  Malware: {len(feature_names_malware)} -> {len(unified_names)} features")
        X_benign = align_features(X_benign, feature_names_benign, unified_names)
        X_malware = align_features(X_malware, feature_names_malware, unified_names)
        feature_names = unified_names
    else:
        feature_names = feature_names_benign

    # Combine datasets
    X = np.vstack([X_benign, X_malware])
    y = np.hstack([y_benign, y_malware])
    paths = paths_benign + paths_malware

    print(f"\nCombined dataset: {len(X)} samples")
    print(f"  Benign:    {len(X_benign)}")
    print(f"  Malicious: {len(X_malware)}")

    # Group by file type
    print("\nGrouping by file type...")
    groups = group_by_filetype(X, y, paths, feature_names)

    # Print distribution
    print(f"\nFile type distribution:")
    print(f"  {'Type':<15} {'Total':>8} {'Benign':>8} {'Malware':>8}")
    print(f"  {'-'*43}")

    filetypes_to_train = []
    for ftype, data in sorted(groups.items(), key=lambda x: -len(x[1]['X'])):
        n_total = len(data['X'])
        n_benign = np.sum(data['y'] == 0)
        n_malware = np.sum(data['y'] == 1)

        meets_threshold = n_total >= args.min_samples and n_benign >= 5 and n_malware >= 5
        status = "✓" if meets_threshold else "✗"

        print(f"  {status} {ftype:<15} {n_total:>8} {n_benign:>8} {n_malware:>8}")

        if meets_threshold:
            filetypes_to_train.append(ftype)

    if not filetypes_to_train:
        print(f"\nERROR: No file types have enough samples (min {args.min_samples})")
        sys.exit(1)

    print(f"\nWill train models for {len(filetypes_to_train)} file types")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train models for each file type
    model_metadata = {}

    for ftype in filetypes_to_train:
        print(f"\n{'='*60}")
        print(f"Training model for: {ftype}")
        print(f"{'='*60}")

        data = groups[ftype]
        X_ftype = data['X']
        y_ftype = data['y']

        # Train model
        model, eval_results = train_model_cv(
            X_ftype, y_ftype, feature_names, n_folds=args.folds
        )

        # SHAP analysis
        if not args.no_shap:
            shap_results = compute_shap_importance(model, X_ftype, feature_names)
            eval_results["shap"] = shap_results

        # Save model
        model_path = args.output_dir / f"{ftype}.json"
        model.save_model(str(model_path))
        print(f"\nModel saved to {model_path}")

        # Save evaluation results
        eval_path = args.output_dir / f"{ftype}_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        # Store metadata
        model_metadata[ftype] = {
            "model_file": f"{ftype}.json",
            "evaluation_file": f"{ftype}_evaluation.json",
            "n_samples": len(X_ftype),
            "n_benign": int(np.sum(y_ftype == 0)),
            "n_malware": int(np.sum(y_ftype == 1)),
            "roc_auc": eval_results["roc_auc"],
            "f1": eval_results["f1"],
            "optimal_threshold": eval_results["optimal_threshold"],
        }

    # Save global feature names (shared across all models)
    feature_names_path = args.output_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f)
    print(f"\nFeature names saved to {feature_names_path}")

    # Save model registry
    registry_path = args.output_dir / "model_registry.json"
    registry = {
        "models": model_metadata,
        "feature_count": len(feature_names),
        "default_model": "litmus_v1.json",  # Fallback for unknown types
    }
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Model registry saved to {registry_path}")

    # Export vocabulary for Rust
    export_vocabulary(feature_names, args.output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("PER-FILETYPE TRAINING SUMMARY")
    print("=" * 60)
    print(f"Trained models for {len(filetypes_to_train)} file types:")
    for ftype in filetypes_to_train:
        meta = model_metadata[ftype]
        print(f"  {ftype:12s}: {meta['n_samples']:4d} samples, "
              f"AUC={meta['roc_auc']:.3f}, F1={meta['f1']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
