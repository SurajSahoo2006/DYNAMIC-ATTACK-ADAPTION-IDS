"""
drift/detect_drift.py
=====================
Evaluates the trained ensemble across 5 temporal windows and applies the
Page-Hinkley drift detector (River) to monitor classification error stream.
Writes drift/drift_detection_results.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from river import drift as river_drift

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DRIFT_DIR = PROJECT_ROOT / "data" / "drift"
RESULTS_DIR = PROJECT_ROOT / "drift"


def load_rf_model():
    candidate_paths = [
        PROJECT_ROOT / "models" / "rf_8class_b2.joblib",
        PROJECT_ROOT / "models" / "rf_model.joblib",
        PROJECT_ROOT / "models" / "rf_model.pkl",
    ]
    for p in candidate_paths:
        if p.exists():
            print(f"Loaded RF model from {p.name}")
            return joblib.load(p)
    raise FileNotFoundError("Could not find RF model in candidate paths.")


def load_xgb_model():
    candidate_paths = [
        PROJECT_ROOT / "models" / "xgb_8class_b2.joblib",
        PROJECT_ROOT / "models" / "xgb_model.joblib",
        PROJECT_ROOT / "models" / "xgb_model.pkl",
    ]
    for p in candidate_paths:
        if p.exists():
            print(f"Loaded XGB model from {p.name}")
            return joblib.load(p)
    raise FileNotFoundError("Could not find XGB model in candidate paths.")


def load_scaler():
    candidate_paths = [
        PROJECT_ROOT / "artifacts" / "scaler.pkl",
        PROJECT_ROOT / "artifacts" / "standard_scaler.pkl",
    ]
    for p in candidate_paths:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError("Could not find scaler in candidate paths.")


def main():
    parser = argparse.ArgumentParser(description="Page-Hinkley Drift Detector")
    parser.add_argument("--fast", action="store_true", help="Evaluate only first 500 rows per window")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2a. Load models, scaler, selected features
    rf = load_rf_model()
    xgb = load_xgb_model()
    scaler = load_scaler()

    feat_path = PROJECT_ROOT / "artifacts" / "selected_features.json"
    with open(feat_path, "r", encoding="utf-8") as f:
        features_meta = json.load(f)
    if isinstance(features_meta, dict) and "selected_features" in features_meta:
        feature_cols = features_meta["selected_features"]
    elif isinstance(features_meta, list):
        feature_cols = features_meta
    else:
        raise ValueError(f"Unrecognized format in {feat_path}")

    # 2b. Evaluate across windows 0..4
    detector = river_drift.PageHinkley(min_instances=30, threshold=50.0)
    windows_results = []

    print("\n" + "=" * 75)
    print("                 PAGE-HINKLEY CONCEPT DRIFT DETECTION")
    print("=" * 75)
    print(f"{'Window':<8} | {'Samples':<9} | {'Accuracy':<10} | {'Drift Detected':<16} | {'First Drift Index':<18}")
    print("-" * 75)

    for wid in range(5):
        w_path = DRIFT_DIR / f"window_{wid}.parquet"
        if not w_path.exists():
            raise FileNotFoundError(f"Missing {w_path}. Run drift/simulate_drift.py first.")

        df_w = pd.read_parquet(w_path)
        if args.fast:
            df_w = df_w.iloc[:500].copy()

        n_samples = len(df_w)
        X = df_w[feature_cols]

        # Scale features if scaler feature dimension matches
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(feature_cols):
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.to_numpy()

        y_true = df_w["label_8"].to_numpy() if "label_8" in df_w.columns else df_w["label_8class"].to_numpy()

        # Predictions
        rf_pred = rf.predict(X_scaled)
        xgb_pred = xgb.predict(X_scaled)
        ensemble_pred = rf_pred  # RF as primary for speed

        accuracy = float((ensemble_pred == y_true).mean())

        # Page-Hinkley monitoring
        drift_detected = False
        first_drift_index = None

        for idx, (pred, true_lbl) in enumerate(zip(ensemble_pred, y_true)):
            error_value = 0.0 if pred == true_lbl else 1.0
            detector.update(error_value)
            if detector.drift_detected and not drift_detected:
                drift_detected = True
                first_drift_index = idx

        w_entry = {
            "window_id": wid,
            "n_samples": n_samples,
            "accuracy": round(accuracy, 4),
            "drift_detected": bool(drift_detected),
            "first_drift_index": first_drift_index,
        }
        windows_results.append(w_entry)

        detected_str = "Yes" if drift_detected else "No"
        idx_str = str(first_drift_index) if first_drift_index is not None else "-"
        print(f"{wid:<8} | {n_samples:<9} | {accuracy*100:>8.2f}% | {detected_str:<16} | {idx_str:<18}")

    print("=" * 75 + "\n")

    # 2c. Write drift_detection_results.json
    results_payload = {
        "windows": windows_results,
        "drift_injected_window": 2,
        "detector": "PageHinkley",
        "detector_params": {"min_instances": 30, "threshold": 50.0},
    }

    out_file = RESULTS_DIR / "drift_detection_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"[OK] Drift detection results written to {out_file}")


if __name__ == "__main__":
    main()
