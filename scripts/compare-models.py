#!/usr/bin/env python3
"""
Compare single-model vs per-filetype model approaches.

This script analyzes:
1. Feature importance across file types
2. Cross-filetype patterns (trait categories)
3. Performance metrics for each approach
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_evaluation(path: Path) -> dict:
    """Load evaluation JSON."""
    with open(path) as f:
        return json.load(f)


def analyze_feature_importance(eval_data: dict, prefix: str = "") -> dict:
    """Analyze feature importance patterns."""
    if "shap" not in eval_data:
        return {}

    shap_data = eval_data["shap"]
    top_features = shap_data.get("top_features", [])

    # Group by feature category
    categories = defaultdict(list)
    for feat in top_features:
        name = feat["name"]
        importance = feat.get("importance", feat.get("mean_abs_shap", 0))

        # Extract category (e.g., "cap_comms_", "ftype_", "import_")
        if name.startswith("cap_"):
            category = name.split("_")[1] if "_" in name[4:] else "cap_other"
        elif name.startswith("ftype_"):
            category = "ftype"
        elif name.startswith("import_"):
            category = "import"
        elif name.startswith("metric_"):
            category = "metric"
        else:
            category = "other"

        categories[category].append({
            "name": name,
            "importance": importance
        })

    return categories


def compare_single_vs_multi(single_eval: dict, multi_evals: dict):
    """Compare single model vs per-filetype models."""
    print("=" * 80)
    print("SINGLE MODEL vs PER-FILETYPE MODELS COMPARISON")
    print("=" * 80)

    # Single model performance
    print("\n### Single Model ###")
    print(f"  ROC AUC:           {single_eval.get('roc_auc', 0):.4f}")
    print(f"  F1 Score:          {single_eval.get('f1', 0):.4f}")
    print(f"  Optimal Threshold: {single_eval.get('optimal_threshold', 0):.4f}")

    # Feature importance categories
    single_cats = analyze_feature_importance(single_eval)
    print(f"\n  Top Feature Categories:")
    for cat, feats in sorted(single_cats.items(), key=lambda x: -sum(f['importance'] for f in x[1]))[:5]:
        total_imp = sum(f['importance'] for f in feats)
        print(f"    {cat:15s}: {len(feats):2d} features, total importance {total_imp:.3f}")

    # Per-filetype models
    print("\n### Per-Filetype Models ###")
    print(f"  {'Type':<15} {'AUC':>7} {'F1':>7} {'Samples':>8} {'Top Category':<20}")
    print("  " + "-" * 70)

    for ftype, eval_data in sorted(multi_evals.items()):
        auc = eval_data.get('roc_auc', 0)
        f1 = eval_data.get('f1', 0)
        n_samples = eval_data.get('n_samples', 0)

        # Find top category
        cats = analyze_feature_importance(eval_data)
        if cats:
            top_cat = max(cats.items(), key=lambda x: sum(f['importance'] for f in x[1]))[0]
        else:
            top_cat = "N/A"

        print(f"  {ftype:<15} {auc:>7.4f} {f1:>7.4f} {n_samples:>8} {top_cat:<20}")

    # Cross-filetype patterns
    print("\n### Cross-Filetype Pattern Analysis ###")

    # For single model, show which capability categories are most important
    if single_cats:
        print("\nSingle model - Top capability categories:")
        cap_cats = {k: v for k, v in single_cats.items() if k not in ['ftype', 'import', 'metric', 'other']}
        for cat, feats in sorted(cap_cats.items(), key=lambda x: -sum(f['importance'] for f in x[1]))[:10]:
            total_imp = sum(f['importance'] for f in feats)
            print(f"  {cat:20s}: {total_imp:6.3f} (n={len(feats)})")

    # For per-filetype models, show which categories appear across multiple types
    print("\nPer-filetype models - Categories appearing across types:")
    category_by_type = defaultdict(lambda: defaultdict(float))
    for ftype, eval_data in multi_evals.items():
        cats = analyze_feature_importance(eval_data)
        for cat, feats in cats.items():
            if cat not in ['ftype', 'import', 'metric', 'other']:
                total_imp = sum(f['importance'] for f in feats)
                category_by_type[cat][ftype] = total_imp

    # Show categories that appear in 3+ file types
    cross_type_cats = {cat: types for cat, types in category_by_type.items() if len(types) >= 3}
    if cross_type_cats:
        for cat in sorted(cross_type_cats.keys()):
            types = category_by_type[cat]
            avg_imp = np.mean(list(types.values()))
            type_list = ", ".join(f"{t}({v:.2f})" for t, v in sorted(types.items(), key=lambda x: -x[1])[:3])
            print(f"  {cat:20s}: avg_imp={avg_imp:.3f}, appears in {len(types)} types")
            print(f"    Top types: {type_list}")
    else:
        print("  (No categories found across 3+ file types)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Average performance
    multi_aucs = [e.get('roc_auc', 0) for e in multi_evals.values()]
    multi_f1s = [e.get('f1', 0) for e in multi_evals.values()]

    print(f"Single model AUC:      {single_eval.get('roc_auc', 0):.4f}")
    print(f"Per-type avg AUC:      {np.mean(multi_aucs):.4f} (std={np.std(multi_aucs):.4f})")
    print(f"Single model F1:       {single_eval.get('f1', 0):.4f}")
    print(f"Per-type avg F1:       {np.mean(multi_f1s):.4f} (std={np.std(multi_f1s):.4f})")

    # Cross-type patterns
    n_cross_type = len(cross_type_cats) if cross_type_cats else 0
    print(f"\nCross-filetype categories: {n_cross_type}")
    print(f"File types with models:    {len(multi_evals)}")


def main():
    parser = argparse.ArgumentParser(description="Compare single vs per-filetype models")
    parser.add_argument("--single", type=Path, default=Path("models/divine_v1_evaluation.json"),
                        help="Single model evaluation JSON")
    parser.add_argument("--multi-dir", type=Path, default=Path("models"),
                        help="Directory containing per-filetype evaluation JSONs")
    args = parser.parse_args()

    # Load single model evaluation
    if not args.single.exists():
        print(f"ERROR: Single model evaluation not found: {args.single}")
        return 1

    single_eval = load_evaluation(args.single)

    # Load per-filetype evaluations
    multi_evals = {}
    for eval_file in args.multi_dir.glob("*_evaluation.json"):
        if eval_file.name == "divine_v1_evaluation.json":
            continue  # Skip single model

        ftype = eval_file.stem.replace("_evaluation", "")
        multi_evals[ftype] = load_evaluation(eval_file)

    if not multi_evals:
        print(f"ERROR: No per-filetype evaluations found in {args.multi_dir}")
        return 1

    # Run comparison
    compare_single_vs_multi(single_eval, multi_evals)

    return 0


if __name__ == "__main__":
    exit(main())
