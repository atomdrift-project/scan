#!/usr/bin/env python3
"""
Evaluate trained model against a held-out test dataset.

This script scans all files in a test directory and reports detection metrics.
The test dataset should NOT be used for training - only for final evaluation.

Usage:
    python evaluate_test.py --test-dir ../test_samples_NOT_TRAINING --model ../models/litmus_v1.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict


def scan_file(litmus_bin: Path, file_path: Path, threshold: float | None = None) -> dict | None:
    """Scan a single file using litmus CLI."""
    cmd = [str(litmus_bin), "scan", "--format", "json", str(file_path)]
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Failed to scan {file_path}: {e}", file=sys.stderr)
    return None


def find_test_files(test_dir: Path) -> list[tuple[Path, str]]:
    """Find all test files and their category (subdirectory name)."""
    files = []
    for category_dir in test_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        for file_path in category_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                # Skip non-code files
                if file_path.suffix.lower() in {".txt", ".md", ".json", ".yaml", ".yml", ".toml"}:
                    continue
                files.append((file_path, category_dir.name))
    return files


def main():
    parser = argparse.ArgumentParser(description="Evaluate model against test dataset")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("../test_samples_NOT_TRAINING"),
        help="Path to test samples directory (NOT used for training)",
    )
    parser.add_argument(
        "--litmus",
        type=Path,
        default=Path("./target/release/litmus"),
        help="Path to litmus binary",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom classification threshold",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON report path",
    )
    parser.add_argument(
        "--expected-label",
        type=int,
        default=1,
        choices=[0, 1],
        help="Expected label for all test samples (0=benign, 1=malware)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-file results",
    )
    args = parser.parse_args()

    # Validate paths
    if not args.test_dir.exists():
        print(f"Error: Test directory not found: {args.test_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.litmus.exists():
        print(f"Error: litmus binary not found: {args.litmus}", file=sys.stderr)
        print("Run 'cargo build --release' first", file=sys.stderr)
        sys.exit(1)

    # Find test files
    print(f"Scanning test directory: {args.test_dir}")
    test_files = find_test_files(args.test_dir)
    print(f"Found {len(test_files)} test files")

    if not test_files:
        print("No test files found!", file=sys.stderr)
        sys.exit(1)

    # Track results
    results = {
        "total": 0,
        "scanned": 0,
        "hostile": 0,
        "suspicious": 0,
        "benign": 0,
        "failed": 0,
        "by_category": defaultdict(lambda: {"total": 0, "hostile": 0, "suspicious": 0, "benign": 0, "failed": 0}),
        "misclassified": [],
        "probabilities": [],
    }

    expected_malware = args.expected_label == 1

    # Scan each file
    for i, (file_path, category) in enumerate(test_files):
        results["total"] += 1
        results["by_category"][category]["total"] += 1

        if args.verbose:
            print(f"[{i+1}/{len(test_files)}] Scanning: {file_path}")

        scan_result = scan_file(args.litmus, file_path, args.threshold)

        if scan_result is None:
            results["failed"] += 1
            results["by_category"][category]["failed"] += 1
            continue

        results["scanned"] += 1
        classification = scan_result.get("classification", "").lower()
        probability = scan_result.get("probability", 0.0)
        results["probabilities"].append(probability)

        if classification == "hostile":
            results["hostile"] += 1
            results["by_category"][category]["hostile"] += 1
        elif classification == "suspicious":
            results["suspicious"] += 1
            results["by_category"][category]["suspicious"] += 1
        else:
            results["benign"] += 1
            results["by_category"][category]["benign"] += 1

        # Track misclassifications
        is_detected = classification in ("hostile", "suspicious")
        if expected_malware and not is_detected:
            results["misclassified"].append({
                "path": str(file_path),
                "category": category,
                "classification": classification,
                "probability": probability,
            })
        elif not expected_malware and is_detected:
            results["misclassified"].append({
                "path": str(file_path),
                "category": category,
                "classification": classification,
                "probability": probability,
            })

        if args.verbose:
            status = "OK" if (expected_malware == is_detected) else "MISS"
            print(f"  {status}: {classification} (p={probability:.3f})")

    # Calculate metrics
    if results["scanned"] > 0:
        if expected_malware:
            # True positives = hostile + suspicious, False negatives = benign
            true_positives = results["hostile"] + results["suspicious"]
            false_negatives = results["benign"]
            detection_rate = true_positives / results["scanned"]
            miss_rate = false_negatives / results["scanned"]
        else:
            # True negatives = benign, False positives = hostile + suspicious
            true_negatives = results["benign"]
            false_positives = results["hostile"] + results["suspicious"]
            detection_rate = true_negatives / results["scanned"]
            miss_rate = false_positives / results["scanned"]

        avg_probability = sum(results["probabilities"]) / len(results["probabilities"])
    else:
        detection_rate = 0.0
        miss_rate = 0.0
        avg_probability = 0.0

    # Print summary
    print("\n" + "=" * 70)
    print("TEST EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Test directory: {args.test_dir}")
    print(f"Expected label: {'MALWARE' if expected_malware else 'BENIGN'}")
    print(f"\nFiles scanned: {results['scanned']}/{results['total']}")
    print(f"  - Hostile:    {results['hostile']}")
    print(f"  - Suspicious: {results['suspicious']}")
    print(f"  - Benign:     {results['benign']}")
    print(f"  - Failed:     {results['failed']}")

    print(f"\nDetection rate: {detection_rate:.2%}")
    print(f"Miss rate:      {miss_rate:.2%}")
    print(f"Avg probability: {avg_probability:.3f}")

    # Per-category breakdown
    print("\n" + "-" * 70)
    print("BY CATEGORY")
    print("-" * 70)
    for category in sorted(results["by_category"].keys()):
        cat_data = results["by_category"][category]
        cat_scanned = cat_data["total"] - cat_data["failed"]
        if cat_scanned > 0:
            cat_detected = cat_data["hostile"] + cat_data["suspicious"]
            cat_rate = cat_detected / cat_scanned
            print(f"  {category:20s}: {cat_detected:3d}/{cat_scanned:3d} detected ({cat_rate:.0%})")

    # Show misclassifications
    if results["misclassified"]:
        print("\n" + "-" * 70)
        print(f"MISCLASSIFIED ({len(results['misclassified'])} files)")
        print("-" * 70)
        for item in results["misclassified"][:20]:
            print(f"  [{item['category']}] {item['path']}")
            print(f"    -> {item['classification']} (p={item['probability']:.3f})")
        if len(results["misclassified"]) > 20:
            print(f"  ... and {len(results['misclassified']) - 20} more")

    # Save report
    if args.output:
        report = {
            "test_dir": str(args.test_dir),
            "expected_label": "malware" if expected_malware else "benign",
            "total_files": results["total"],
            "scanned_files": results["scanned"],
            "failed_files": results["failed"],
            "classifications": {
                "hostile": results["hostile"],
                "suspicious": results["suspicious"],
                "benign": results["benign"],
            },
            "metrics": {
                "detection_rate": detection_rate,
                "miss_rate": miss_rate,
                "avg_probability": avg_probability,
            },
            "by_category": dict(results["by_category"]),
            "misclassified": results["misclassified"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {args.output}")

    # Exit with error if detection rate is below threshold
    if expected_malware and detection_rate < 0.90:
        print(f"\nWARNING: Detection rate {detection_rate:.2%} is below 90% threshold!")
        sys.exit(1)
    elif not expected_malware and detection_rate < 0.90:
        print(f"\nWARNING: True negative rate {detection_rate:.2%} is below 90% threshold!")
        sys.exit(1)

    print("\nTest evaluation PASSED")


if __name__ == "__main__":
    main()
