#!/usr/bin/env python3
"""
Detailed model evaluation with visualizations.

This script provides:
- ROC curves
- Precision-recall curves
- Confusion matrix heatmap
- Feature importance analysis
- Threshold optimization recommendations

Usage:
    python evaluate.py --model ../models/litmus_v1.json --features features.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_val_score

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_model(model_path: Path) -> xgb.XGBClassifier:
    """Load XGBoost model."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def load_features(features_path: Path) -> tuple:
    """Load features from numpy file."""
    data = np.load(features_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = list(data["feature_names"])
    paths = list(data.get("paths", []))
    return X, y, feature_names, paths


def find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Find optimal thresholds for different use cases."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    results = {}

    # Find threshold for maximum F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    results["max_f1"] = {
        "threshold": float(thresholds[best_f1_idx]) if best_f1_idx < len(thresholds) else 0.5,
        "precision": float(precisions[best_f1_idx]),
        "recall": float(recalls[best_f1_idx]),
        "f1": float(f1_scores[best_f1_idx]),
    }

    # Find threshold for 95% recall (security-focused)
    for i, r in enumerate(recalls):
        if r < 0.95:
            idx = max(0, i - 1)
            results["high_recall"] = {
                "threshold": float(thresholds[idx]) if idx < len(thresholds) else 0.5,
                "precision": float(precisions[idx]),
                "recall": float(recalls[idx]),
                "description": "Catches 95% of malware (more false positives)",
            }
            break

    # Find threshold for 95% precision (low false positive)
    for i, p in enumerate(precisions):
        if p >= 0.95:
            results["high_precision"] = {
                "threshold": float(thresholds[i]) if i < len(thresholds) else 0.5,
                "precision": float(precisions[i]),
                "recall": float(recalls[i]),
                "description": "95% of alerts are real malware (may miss some)",
            }
            break

    return results


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path):
    """Plot ROC curve."""
    if not HAS_MATPLOTLIB:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve to {output_path}")


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path):
    """Plot precision-recall curve."""
    if not HAS_MATPLOTLIB:
        return

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved precision-recall curve to {output_path}")


def plot_feature_importance(
    model: xgb.XGBClassifier, feature_names: list[str], output_path: Path, top_n: int = 30
):
    """Plot feature importance."""
    if not HAS_MATPLOTLIB:
        return

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.barh(
        range(top_n),
        importance[indices][::-1],
        align="center",
    )
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to {output_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """Plot confusion matrix."""
    if not HAS_MATPLOTLIB:
        return

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Benign", "Hostile"])
    plt.yticks([0, 1], ["Benign", "Hostile"])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def analyze_errors(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    paths: list[str],
    threshold: float = 0.5,
) -> dict:
    """Analyze false positives and false negatives."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    errors = {
        "false_positives": [],  # Benign classified as hostile
        "false_negatives": [],  # Hostile classified as benign
    }

    for i, (true, pred, prob, path) in enumerate(zip(y, y_pred, y_prob, paths)):
        if true == 0 and pred == 1:  # False positive
            errors["false_positives"].append({
                "path": path,
                "probability": float(prob),
            })
        elif true == 1 and pred == 0:  # False negative
            errors["false_negatives"].append({
                "path": path,
                "probability": float(prob),
            })

    # Sort by probability (most confident errors first)
    errors["false_positives"].sort(key=lambda x: x["probability"], reverse=True)
    errors["false_negatives"].sort(key=lambda x: x["probability"])

    return errors


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("../models/litmus_v1.json"),
        help="Path to XGBoost model",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("features.npz"),
        help="Path to features file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./evaluation"),
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for error analysis",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    print(f"Loading model from {args.model}")
    model = load_model(args.model)

    print(f"Loading features from {args.features}")
    X, y, feature_names, paths = load_features(args.features)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

    # Get predictions
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob > args.threshold).astype(int)

    # Print summary metrics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nThreshold: {args.threshold}")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall: {recall_score(y, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y, y_prob):.4f}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Benign", "Hostile"]))

    # Find optimal thresholds
    print("\n" + "=" * 60)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 60)
    thresholds = find_optimal_thresholds(y, y_prob)
    for name, info in thresholds.items():
        print(f"\n{name}:")
        for k, v in info.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Generate plots
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        plot_roc_curve(y, y_prob, args.output_dir / "roc_curve.png")
        plot_precision_recall(y, y_prob, args.output_dir / "precision_recall.png")
        plot_feature_importance(model, feature_names, args.output_dir / "feature_importance.png")
        plot_confusion_matrix(y, y_pred, args.output_dir / "confusion_matrix.png")
    else:
        print("\nMatplotlib not installed, skipping plots")

    # Error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    errors = analyze_errors(model, X, y, paths, args.threshold)

    print(f"\nFalse Positives (benign flagged as hostile): {len(errors['false_positives'])}")
    for fp in errors["false_positives"][:5]:
        print(f"  {fp['path']} (p={fp['probability']:.3f})")

    print(f"\nFalse Negatives (hostile missed): {len(errors['false_negatives'])}")
    for fn in errors["false_negatives"][:5]:
        print(f"  {fn['path']} (p={fn['probability']:.3f})")

    # Save detailed report
    report = {
        "threshold": args.threshold,
        "metrics": {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred)),
            "f1": float(f1_score(y, y_pred)),
            "auc_roc": float(roc_auc_score(y, y_prob)),
        },
        "recommended_thresholds": thresholds,
        "errors": {
            "false_positives": errors["false_positives"][:20],
            "false_negatives": errors["false_negatives"][:20],
        },
    }

    report_path = args.output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved detailed report to {report_path}")


if __name__ == "__main__":
    main()
