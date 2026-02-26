#!/usr/bin/env python3
"""Merge batch JSON files into a single output file."""

import json
import sys
from pathlib import Path


def merge_batches(batch_prefix: str, output_path: str) -> None:
    """Merge all batch files matching prefix into single output.

    Handles batches with different feature sets by aligning to a unified
    feature list (union of all batch features).
    """
    batch_dir = Path(batch_prefix).parent
    batch_name = Path(batch_prefix).name

    # Find all batch files
    batch_files = sorted(batch_dir.glob(f"{batch_name}_batch_*.json"))

    if not batch_files:
        print(f"  Error: No batch files found matching {batch_prefix}_batch_*.json")
        sys.exit(1)

    print(f"  Found {len(batch_files)} batch files to merge")

    # First pass: collect all feature names and detect inconsistencies
    all_feature_names: set[str] = set()
    batch_data: list[dict] = []
    feature_counts: dict[str, int] = {}

    for batch_file in batch_files:
        try:
            with open(batch_file) as f:
                data = json.load(f)
            batch_data.append((data, batch_file))
            feature_names = data.get("feature_names", [])
            all_feature_names.update(feature_names)
            feature_counts[batch_file.name] = len(feature_names)
        except (json.JSONDecodeError, IOError) as e:
            print(f"    Warning: Failed to load {batch_file}: {e}")
            continue

    # Check for inconsistencies
    unique_counts = set(feature_counts.values())
    if len(unique_counts) > 1:
        print(f"  WARNING: Batches have different feature counts!")
        for name, count in sorted(feature_counts.items()):
            print(f"    {name}: {count} features")
        print("  Aligning to unified feature set...")

    # Create unified sorted feature list
    unified_features = sorted(all_feature_names)
    unified_index = {name: i for i, name in enumerate(unified_features)}
    feature_count = len(unified_features)

    print(f"  Unified feature count: {feature_count}")

    # Second pass: align samples to unified feature list
    all_samples = []

    for data, batch_file in batch_data:
        batch_features = data.get("feature_names", [])
        samples = data.get("samples", [])

        # Check if this batch needs alignment
        needs_alignment = batch_features != unified_features

        if needs_alignment:
            # Build mapping from batch index to unified index
            batch_to_unified = [unified_index.get(name, -1) for name in batch_features]

            for sample in samples:
                old_features = sample["features"]
                # Validate sample has expected feature count for this batch
                if len(old_features) != len(batch_features):
                    path = sample.get("path", "unknown")
                    print(f"      WARNING: Sample {path} has {len(old_features)} features, expected {len(batch_features)}")
                # Initialize with zeros
                new_features = [0.0] * feature_count
                # Map old features to new positions
                for old_idx, unified_idx in enumerate(batch_to_unified):
                    if unified_idx >= 0 and old_idx < len(old_features):
                        new_features[unified_idx] = old_features[old_idx]
                sample["features"] = new_features
        else:
            # Validate samples even when no alignment needed
            for sample in samples:
                if len(sample["features"]) != feature_count:
                    path = sample.get("path", "unknown")
                    print(f"      WARNING: Sample {path} has {len(sample['features'])} features, expected {feature_count}")

        all_samples.extend(samples)
        print(f"    Loaded {len(samples)} samples from {batch_file.name}")

    if not all_samples:
        print("  Error: No samples loaded from any batch file")
        sys.exit(1)

    # Write merged output
    output = {
        "feature_names": unified_features,
        "feature_count": feature_count,
        "sample_count": len(all_samples),
        "samples": all_samples,
    }

    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"  Merged {len(all_samples)} samples into {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: merge-batches.py <batch_prefix> <output_file>")
        print("  Example: merge-batches.py training/data/batches/benign training/data/benign.json")
        sys.exit(1)

    merge_batches(sys.argv[1], sys.argv[2])
