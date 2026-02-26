#!/usr/bin/env python3
"""
Compute SHAP values for a single prediction.

This script is called by the Rust CLI when --explain is passed.
It loads the XGBoost model and computes SHAP values for the given features.

Usage:
    python explain.py '<features_json>' <model_path> <feature_names_path>

Output:
    JSON with base_value and top contributing features
"""

import json
import sys
from pathlib import Path

import numpy as np
import shap
import xgboost as xgb


def explain(features_json: str, model_path: str, feature_names_path: str) -> dict:
    """Compute SHAP values and return top contributors."""
    # Parse features
    features = np.array(json.loads(features_json), dtype=np.float32).reshape(1, -1)

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Load feature names
    with open(feature_names_path) as f:
        feature_names = json.load(f)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(features)

    # For binary classification, shap_values is 1D array
    if isinstance(shap_values, list):
        # Multi-output format (older SHAP versions)
        shap_values = shap_values[1]  # Class 1 (hostile)

    shap_values = shap_values.flatten()

    # Get base value (expected value)
    if isinstance(explainer.expected_value, np.ndarray):
        base_value = float(explainer.expected_value[1])
    else:
        base_value = float(explainer.expected_value)

    # Combine features, values, and SHAP contributions
    contributions = []
    for i, (name, val, shap_val) in enumerate(
        zip(feature_names, features[0], shap_values)
    ):
        contributions.append({
            "name": name,
            "value": float(val),
            "shap": float(shap_val),
            "index": i,
        })

    # Sort by absolute SHAP value
    contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

    # Return top 20 contributors
    return {
        "base_value": base_value,
        "prediction": float(model.predict_proba(features)[0, 1]),
        "top_features": contributions[:20],
    }


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python explain.py '<features_json>' <model_path> <feature_names_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    features_json = sys.argv[1]
    model_path = sys.argv[2]
    feature_names_path = sys.argv[3]

    # Validate paths
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(feature_names_path).exists():
        print(f"Error: Feature names not found: {feature_names_path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = explain(features_json, model_path, feature_names_path)
        print(json.dumps(result))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
