#!/usr/bin/env python3
"""
Predict classification for a single file using the trained litmus model.

Usage:
    python predict.py /bin/ls
    python predict.py --json /bin/ls
    python predict.py --explain /bin/ls    # Show SHAP explanations
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

from extract_features import run_cleave, extract_features, expand_capability_hierarchy

# Thresholds for classification
HOSTILE_THRESHOLD = 0.80
SUSPICIOUS_THRESHOLD = 0.40


def load_model_and_vocab(model_dir: Path):
    """Load trained model and vocabulary from model directory."""
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "litmus_v1.json")

    # Load vocabulary from JSON files
    with open(model_dir / "capabilities.json") as f:
        capabilities = json.load(f)
    with open(model_dir / "imports.json") as f:
        imports = json.load(f)
    with open(model_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    return model, capabilities, imports, feature_names


def predict_file(file_path: Path, model, capabilities, imports, feature_names, return_features: bool = False):
    """Run prediction on a single file."""
    # Run cleave
    report, exit_code, err = run_cleave(file_path)
    if report is None:
        result = {
            "error": f"cleave failed: {err}",
            "exit_code": exit_code,
        }
        return (result, None) if return_features else result

    # We need to set the global variables for extract_features
    import extract_features as ef
    ef.KNOWN_CAPABILITIES = capabilities
    ef.KNOWN_IMPORTS = imports
    ef.FEATURE_NAMES = feature_names

    # Extract features
    features = extract_features(report, file_path)

    # Predict
    prob = model.predict_proba(features.reshape(1, -1))[0]
    p_hostile = prob[1]

    # Classify
    if p_hostile > HOSTILE_THRESHOLD:
        classification = "HOSTILE"
    elif p_hostile > SUSPICIOUS_THRESHOLD:
        classification = "SUSPICIOUS"
    else:
        classification = "BENIGN"

    # Get findings summary
    findings = report.get("findings", [])
    hostile_findings = [f for f in findings if f.get("crit", f.get("criticality")) == "hostile"]
    suspicious_findings = [f for f in findings if f.get("crit", f.get("criticality")) == "suspicious"]

    # Extract supply-chain indicators from features
    sc_indicators = []
    sc_feature_map = {
        "sc_sensitive_path_count": "Sensitive file paths (SSH, AWS, etc.)",
        "sc_browser_path_count": "Browser credential paths",
        "sc_exfil_endpoint_count": "Exfiltration endpoints (Discord, Telegram)",
        "sc_public_ip_count": "Hardcoded public IPs",
        "sc_env_access_count": "Environment variable access",
        "sc_sensitive_env_var_count": "Sensitive env vars (API_KEY, TOKEN)",
        "sc_crypto_wallet_count": "Cryptocurrency wallet addresses",
        "sc_install_hook_count": "Install hooks (setup.py, exec)",
        "sc_exfil_method_count": "Data exfiltration methods",
        "sc_obfuscation_count": "Code obfuscation patterns",
        "sc_persistence_count": "Persistence mechanisms",
        "sc_evasion_count": "Anti-analysis / evasion",
        "sc_data_collection_count": "System data collection",
        "sc_keylog_count": "Keylogging / screen capture",
        "sc_raw_ip_url_count": "URLs with raw IPs",
        "sc_base64_string_count": "Base64 encoded strings",
        "sc_long_base64_count": "Large encoded payloads",
    }

    for feat_name, description in sc_feature_map.items():
        if feat_name in feature_names:
            idx = feature_names.index(feat_name)
            value = features[idx]
            if value > 0:
                sc_indicators.append({
                    "indicator": description,
                    "count": int(value),
                })

    result = {
        "file": str(file_path),
        "classification": classification,
        "probability": float(p_hostile),
        "findings": {
            "total": len(findings),
            "hostile": len(hostile_findings),
            "suspicious": len(suspicious_findings),
        },
        "top_hostile": [
            {"id": f.get("id"), "description": f.get("description", "")[:80]}
            for f in hostile_findings[:5]
        ],
        "top_suspicious": [
            {"id": f.get("id"), "description": f.get("description", "")[:80]}
            for f in suspicious_findings[:5]
        ],
        "supply_chain_indicators": sc_indicators,
    }

    if return_features:
        return result, features
    return result


def compute_shap_explanation(model, features, feature_names, top_n: int = 50):
    """Compute SHAP values for a prediction."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features.reshape(1, -1))

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (hostile) for binary
    shap_values = shap_values.flatten()

    # Get base value
    if isinstance(explainer.expected_value, np.ndarray):
        base_value = float(explainer.expected_value[1])
    else:
        base_value = float(explainer.expected_value)

    # Combine and sort by absolute SHAP value
    contributions = []
    for name, val, shap_val in zip(feature_names, features, shap_values):
        contributions.append({
            "name": name,
            "value": float(val),
            "shap": float(shap_val),
        })

    contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "base_value": base_value,
        "contributions": contributions[:top_n],
    }


def main():
    parser = argparse.ArgumentParser(description="Predict file classification")
    parser.add_argument("file", type=Path, help="File to classify")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--explain", action="store_true", help="Show SHAP explanations for the prediction")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Directory containing model files",
    )
    parser.add_argument(
        "--cleave-dir",
        type=Path,
        default=None,
        help="cleave source directory (contains traits/). If not set, uses CLEAVE_DIR env var.",
    )
    args = parser.parse_args()

    # Set CLEAVE_DIR for extract_features module
    import os
    import extract_features as ef
    if args.cleave_dir:
        ef.CLEAVE_DIR = args.cleave_dir
    elif os.environ.get("CLEAVE_DIR"):
        ef.CLEAVE_DIR = Path(os.environ["CLEAVE_DIR"])

    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # Load model
    model, capabilities, imports, feature_names = load_model_and_vocab(args.model_dir)

    # Predict (with features if explaining)
    if args.explain:
        result, features = predict_file(args.file, model, capabilities, imports, feature_names, return_features=True)
    else:
        result = predict_file(args.file, model, capabilities, imports, feature_names)

    # Compute SHAP if requested
    shap_result = None
    if args.explain and features is not None:
        shap_result = compute_shap_explanation(model, features, feature_names)
        result["shap"] = shap_result

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)

        # Terminal output
        classification = result["classification"]
        prob = result["probability"]

        # Color codes
        if classification == "HOSTILE":
            color = "\033[91m"  # Red
        elif classification == "SUSPICIOUS":
            color = "\033[93m"  # Yellow
        else:
            color = "\033[92m"  # Green
        reset = "\033[0m"
        dim = "\033[2m"
        bold = "\033[1m"

        print(f"\n{bold}litmus Classification{reset}")
        print("=" * 50)
        print(f"File: {result['file']}")
        print(f"Classification: {color}{classification}{reset}")
        print(f"Hostile Probability: {prob:.4f}")
        print(f"\nFindings: {result['findings']['total']} total")
        print(f"  Hostile: {result['findings']['hostile']}")
        print(f"  Suspicious: {result['findings']['suspicious']}")

        if result["top_hostile"]:
            print(f"\nTop Hostile Capabilities:")
            for f in result["top_hostile"]:
                print(f"  - {f['id']}")

        if result["top_suspicious"]:
            print(f"\nTop Suspicious Capabilities:")
            for f in result["top_suspicious"]:
                print(f"  - {f['id']}")

        if result.get("supply_chain_indicators"):
            print(f"\n{color}Supply-Chain Indicators:{reset}")
            for ind in result["supply_chain_indicators"][:10]:
                print(f"  - {ind['indicator']}: {ind['count']}")

        # SHAP explanations
        if shap_result:
            print(f"\n{bold}SHAP Explanation{reset}")
            print("-" * 50)
            print(f"Base value (prior): {shap_result['base_value']:.4f}")
            print(f"\nTop Contributing Features:")
            print(f"{'Feature':<40} {'Value':>10} {'SHAP':>12}")
            print("-" * 64)
            for c in shap_result["contributions"]:
                name = c["name"][:38]
                val = c["value"]
                shap_val = c["shap"]

                # Color SHAP value by direction
                if shap_val > 0.01:
                    shap_color = "\033[91m"  # Red (toward hostile)
                    arrow = "+"
                elif shap_val < -0.01:
                    shap_color = "\033[92m"  # Green (toward benign)
                    arrow = ""
                else:
                    shap_color = dim
                    arrow = " "

                # Format value based on type
                if val == 0:
                    val_str = "absent"
                elif val == 1 and name.startswith(("cap_", "import_")):
                    val_str = "present"
                else:
                    val_str = f"{val:.2f}"

                print(f"{name:<40} {val_str:>10} {shap_color}{arrow}{shap_val:>+.4f}{reset}")


if __name__ == "__main__":
    main()
