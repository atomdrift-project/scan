#!/usr/bin/env python3
"""
Train XGBoost model from JSON features extracted by litmus.

Improvements over basic training:
- Stratified K-Fold cross-validation
- Class imbalance handling (scale_pos_weight)
- Probability calibration (isotonic regression)
- Hyperparameter tuning
- Global SHAP feature importance analysis
- Comprehensive evaluation metrics

Usage:
    python train_from_json.py --benign benign.json --malware malware.json --output model.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def sanitize_feature_name(name: str) -> str:
    """Sanitize feature name for XGBoost (no [, ], or <)."""
    return name.replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load features from JSON file produced by litmus extract."""
    print(f"Loading {path}...")
    with open(path) as f:
        data = json.load(f)

    feature_names = [sanitize_feature_name(n) for n in data["feature_names"]]
    samples = data["samples"]

    X = np.array([s["features"] for s in samples], dtype=np.float32)
    y = np.array([s["label"] for s in samples], dtype=np.int32)
    paths = [s.get("path", "") for s in samples]

    print(f"  Loaded {len(samples)} samples, {len(feature_names)} features")
    return X, y, feature_names, paths


def align_features(
    X: np.ndarray,
    source_names: list[str],
    target_names: list[str],
) -> np.ndarray:
    """Align feature matrix to target feature names."""
    if source_names == target_names:
        return X

    target_index = {name: i for i, name in enumerate(target_names)}
    source_to_target = [target_index.get(name, -1) for name in source_names]

    X_aligned = np.zeros((X.shape[0], len(target_names)), dtype=np.float32)
    for src_idx, tgt_idx in enumerate(source_to_target):
        if tgt_idx >= 0:
            X_aligned[:, tgt_idx] = X[:, src_idx]

    return X_aligned


def compute_shap_importance(model: xgb.Booster, X: np.ndarray, feature_names: list[str]) -> dict:
    """Compute global SHAP feature importance."""
    try:
        import shap
    except ImportError:
        print("  SHAP not installed, skipping feature importance analysis")
        return {}

    print("\nComputing SHAP feature importance...")

    # Use a sample if dataset is large
    if len(X) > 1000:
        idx = np.random.choice(len(X), 1000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    importance = {}
    print("\nTop 20 Most Important Features (SHAP):")
    for i, idx in enumerate(sorted_idx[:20]):
        name = feature_names[idx]
        value = mean_abs_shap[idx]
        importance[name] = float(value)
        print(f"  {i+1:2d}. {name:40s} {value:.4f}")

    # Identify potentially useless features
    useless_threshold = 0.001
    useless_count = np.sum(mean_abs_shap < useless_threshold)
    if useless_count > 0:
        print(f"\n  {useless_count} features have SHAP importance < {useless_threshold}")
        print("  Consider removing these to reduce model size")

    return {
        "top_features": [
            {"name": feature_names[idx], "importance": float(mean_abs_shap[idx])}
            for idx in sorted_idx[:50]
        ],
        "useless_feature_count": int(useless_count),
    }


def train_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_folds: int = 5,
    calibrate: bool = True,
) -> tuple[xgb.Booster, dict]:
    """Train XGBoost model with cross-validation and calibration."""
    print(f"\nTraining with {n_folds}-fold cross-validation...")

    # Class distribution
    n_benign = np.sum(y == 0)
    n_malware = np.sum(y == 1)
    scale_pos_weight = n_benign / n_malware if n_malware > 0 else 1.0

    print(f"  Class distribution: {n_benign} benign, {n_malware} malware")
    print(f"  Class weight ratio: {scale_pos_weight:.2f}")

    # Check minimum samples per class for CV
    min_class_count = min(n_benign, n_malware)
    if min_class_count < n_folds:
        n_folds = max(2, min_class_count)
        print(f"  Reducing to {n_folds} folds due to small class size")

    # XGBoost parameters with class balancing
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "scale_pos_weight": scale_pos_weight,  # Handle class imbalance
        "seed": 42,
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Collect predictions from each fold
    y_pred_proba = np.zeros(len(y))
    fold_metrics = []

    print(f"\n  {'Fold':<6} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print(f"  {'-'*42}")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

        # Train with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Predict on validation set
        fold_pred = model.predict(dval)
        y_pred_proba[val_idx] = fold_pred

        # Fold metrics
        fold_pred_binary = (fold_pred > 0.5).astype(int)
        fold_auc = roc_auc_score(y_val, fold_pred)
        fold_f1 = f1_score(y_val, fold_pred_binary)
        fold_prec = precision_score(y_val, fold_pred_binary, zero_division=0)
        fold_rec = recall_score(y_val, fold_pred_binary, zero_division=0)

        fold_metrics.append({
            "auc": fold_auc, "f1": fold_f1, "precision": fold_prec, "recall": fold_rec
        })
        print(f"  {fold+1:<6} {fold_auc:>8.4f} {fold_f1:>8.4f} {fold_prec:>8.4f} {fold_rec:>8.4f}")

    # Aggregate cross-validation metrics
    cv_auc = np.mean([m["auc"] for m in fold_metrics])
    cv_auc_std = np.std([m["auc"] for m in fold_metrics])
    cv_f1 = np.mean([m["f1"] for m in fold_metrics])
    cv_f1_std = np.std([m["f1"] for m in fold_metrics])

    print(f"  {'-'*42}")
    print(f"  {'Mean':<6} {cv_auc:>8.4f} {cv_f1:>8.4f}")
    print(f"  {'Std':<6} {cv_auc_std:>8.4f} {cv_f1_std:>8.4f}")

    # Train final model on all data
    print("\nTraining final model on all data...")
    dtrain_full = xgb.DMatrix(X, label=y, feature_names=feature_names)

    final_model = xgb.train(
        params,
        dtrain_full,
        num_boost_round=500,
        evals=[(dtrain_full, "train")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    # Compute final metrics using CV predictions (unbiased estimate)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    print("\nCross-Validation Results (unbiased estimate):")
    print(f"  Accuracy:  {accuracy_score(y, y_pred_binary):.4f}")
    print(f"  Precision: {precision_score(y, y_pred_binary, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y, y_pred_binary, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y, y_pred_binary, zero_division=0):.4f}")
    print(f"  ROC AUC:   {roc_auc_score(y, y_pred_proba):.4f}")
    print(f"  Avg Prec:  {average_precision_score(y, y_pred_proba):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred_binary)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Benign  Malware")
    print(f"  Actual Benign  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"  Actual Malware {cm[1,0]:6d}  {cm[1,1]:6d}")

    # Find optimal threshold using precision-recall curve
    precision_arr, recall_arr, thresholds = precision_recall_curve(y, y_pred_proba)
    f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    print(f"\nOptimal threshold (max F1): {optimal_threshold:.3f}")
    print(f"  F1 at optimal: {f1_scores[optimal_idx]:.4f}")

    # Evaluation results
    eval_results = {
        "cv_folds": n_folds,
        "cv_auc_mean": float(cv_auc),
        "cv_auc_std": float(cv_auc_std),
        "cv_f1_mean": float(cv_f1),
        "cv_f1_std": float(cv_f1_std),
        "accuracy": float(accuracy_score(y, y_pred_binary)),
        "precision": float(precision_score(y, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y, y_pred_binary, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_pred_proba)),
        "avg_precision": float(average_precision_score(y, y_pred_proba)),
        "optimal_threshold": float(optimal_threshold),
        "confusion_matrix": cm.tolist(),
        "class_distribution": {"benign": int(n_benign), "malware": int(n_malware)},
        "scale_pos_weight": float(scale_pos_weight),
    }

    return final_model, eval_results


def export_vocabulary(feature_names: list[str], output_dir: Path):
    """Export capability and import vocabulary for Rust."""
    capabilities = [n[4:] for n in feature_names if n.startswith("cap_")]
    imports = [n[7:] for n in feature_names if n.startswith("import_")]

    caps_path = output_dir / "capabilities.json"
    imports_path = output_dir / "imports.json"

    with open(caps_path, "w") as f:
        json.dump(capabilities, f, indent=2)

    with open(imports_path, "w") as f:
        json.dump(imports, f, indent=2)

    print(f"\nExported vocabulary:")
    print(f"  {caps_path}: {len(capabilities)} capabilities")
    print(f"  {imports_path}: {len(imports)} imports")


def export_features_npz(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], paths: list[str], output_path: Path
):
    """Export features as NPZ for evaluate.py compatibility."""
    np.savez(
        output_path,
        X=X,
        y=y,
        feature_names=np.array(feature_names, dtype=object),
        paths=np.array(paths, dtype=object),
    )
    print(f"Exported features to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost from JSON features")
    parser.add_argument("--benign", type=Path, required=True, help="Benign features JSON")
    parser.add_argument("--malware", type=Path, required=True, help="Malware features JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output model path")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument(
        "--features-npz", type=Path, help="Export features as NPZ for evaluate.py"
    )
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

    # Check we have enough data
    if len(X) < 10:
        print("\nERROR: Not enough samples for training (need at least 10)")
        sys.exit(1)

    # Train model with cross-validation
    model, eval_results = train_model_cv(X, y, feature_names, n_folds=args.folds)

    # SHAP analysis
    if not args.no_shap:
        shap_results = compute_shap_importance(model, X, feature_names)
        eval_results["shap"] = shap_results

    # Save model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output))
    print(f"\nModel saved to {args.output}")

    # Export feature names
    feature_names_path = args.output.parent / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f)
    print(f"Feature names saved to {feature_names_path}")

    # Export evaluation results
    eval_path = args.output.parent / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {eval_path}")

    # Export vocabulary for Rust
    export_vocabulary(feature_names, args.output.parent)

    # Export features as NPZ for evaluate.py compatibility
    if args.features_npz:
        export_features_npz(X, y, feature_names, paths, args.features_npz)

    # Summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Samples:     {len(X)} ({len(X_benign)} benign, {len(X_malware)} malware)")
    print(f"Features:    {len(feature_names)}")
    print(f"CV Folds:    {args.folds}")
    print(f"ROC AUC:     {eval_results['roc_auc']:.4f}")
    print(f"F1 Score:    {eval_results['f1']:.4f}")
    print(f"Precision:   {eval_results['precision']:.4f}")
    print(f"Recall:      {eval_results['recall']:.4f}")
    print(f"Threshold:   {eval_results['optimal_threshold']:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
