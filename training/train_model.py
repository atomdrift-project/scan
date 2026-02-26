#!/usr/bin/env python3
"""
Train XGBoost binary classifier and export to ONNX.

This script:
1. Loads features from extract_features.py output
2. Trains an XGBoost classifier
3. Evaluates on held-out test set
4. Exports to ONNX format for Rust inference

Usage:
    python train_model.py --input features.npz --output ../models/litmus_v1.onnx
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Try to import ONNX conversion tools
try:
    import onnx
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnxmltools not installed. ONNX export will be skipped.")


def load_features(path: Path) -> tuple:
    """Load features from numpy file."""
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = list(data["feature_names"])
    paths = list(data.get("paths", []))
    return X, y, feature_names, paths


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> xgb.XGBClassifier:
    """Train XGBoost classifier."""
    # Calculate class weights for imbalanced data
    n_benign = sum(y_train == 0)
    n_hostile = sum(y_train == 1)
    scale_pos_weight = n_benign / max(n_hostile, 1)

    print(f"Training samples: {len(y_train)} ({n_benign} benign, {n_hostile} hostile)")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",  # Fast for large feature sets
        random_state=42,
        n_jobs=-1,
    )

    # Train with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )

    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
):
    """Evaluate model and print metrics."""
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Basic metrics
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Hostile"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}")
    print(f"  FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}")

    # Feature importance
    print("\nTop 20 Most Important Features:")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:20]
    for i, idx in enumerate(indices):
        print(f"  {i + 1:2d}. {feature_names[idx]:40s} {importance[idx]:.4f}")

    # Find optimal thresholds
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    # Precision-recall analysis
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob)

    # Find threshold for 95% recall (catch most malware)
    for i, (p, r, t) in enumerate(zip(precisions, recalls, thresholds_pr)):
        if r < 0.95:
            print(f"\nAt 95% recall:")
            print(f"  Threshold: {thresholds_pr[max(0, i - 1)]:.3f}")
            print(f"  Precision: {precisions[max(0, i - 1)]:.3f}")
            break

    # Find threshold for 95% precision (minimize false positives)
    for i, (p, r, t) in enumerate(zip(precisions, recalls, thresholds_pr)):
        if p >= 0.95:
            print(f"\nAt 95% precision:")
            print(f"  Threshold: {t:.3f}")
            print(f"  Recall: {r:.3f}")
            break

    # Save evaluation results
    eval_results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
        "top_features": [
            {"name": feature_names[idx], "importance": float(importance[idx])}
            for idx in indices
        ],
    }
    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved evaluation results to {eval_path}")


def export_onnx(
    model: xgb.XGBClassifier,
    n_features: int,
    output_path: Path,
):
    """Export model to ONNX format."""
    if not HAS_ONNX:
        print("Skipping ONNX export (onnxmltools not installed)")
        return

    print(f"\nExporting to ONNX: {output_path}")

    # Convert to ONNX
    initial_types = [("features", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_types)

    # Save ONNX model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, str(output_path))
    print(f"Saved ONNX model to {output_path}")


def export_xgboost_native(model: xgb.XGBClassifier, output_path: Path):
    """Export model in XGBoost native format (for SHAP)."""
    json_path = output_path.with_suffix(".json")
    model.save_model(json_path)
    print(f"Saved XGBoost model to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost classifier")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("features.npz"),
        help="Input features file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../models/litmus_v1.onnx"),
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Load features
    print(f"Loading features from {args.input}")
    X, y, feature_names, paths = load_features(args.input)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Train model
    print("\nTraining XGBoost classifier...")
    model = train_xgboost(X_train, y_train, X_test, y_test)

    # Evaluate
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluate_model(model, X_test, y_test, feature_names, output_dir)

    # Export models
    export_onnx(model, X.shape[1], args.output)
    export_xgboost_native(model, args.output)

    # Copy feature names to models directory
    feature_names_src = args.input.with_suffix(".json")
    if feature_names_src.exists():
        feature_names_dst = args.output.with_name("feature_names.json")
        shutil.copy(feature_names_src, feature_names_dst)
        print(f"Copied feature names to {feature_names_dst}")

    print("\nDone!")


if __name__ == "__main__":
    main()
