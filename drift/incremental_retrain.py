"""
drift/incremental_retrain.py
============================
Performs incremental adaptation / retraining on accumulated windows [0, 1, 2]
and evaluates against the unseen next window 3.
Writes drift/retrain_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DRIFT_DIR = PROJECT_ROOT / "data" / "drift"
RESULTS_DIR = PROJECT_ROOT / "drift"


def load_original_rf_model():
    candidate_paths = [
        PROJECT_ROOT / "models" / "rf_8class_b2.joblib",
        PROJECT_ROOT / "models" / "rf_model.joblib",
        PROJECT_ROOT / "models" / "rf_model.pkl",
    ]
    for p in candidate_paths:
        if p.exists():
            print(f"Loaded original RF model from {p.name}")
            return joblib.load(p)
    raise FileNotFoundError("Could not find RF model in candidate paths.")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 3a. Load trained RF model and feature list
    original_rf = load_original_rf_model()

    feat_path = PROJECT_ROOT / "artifacts" / "selected_features.json"
    with open(feat_path, "r", encoding="utf-8") as f:
        features_meta = json.load(f)
    if isinstance(features_meta, dict) and "selected_features" in features_meta:
        feature_cols = features_meta["selected_features"]
    elif isinstance(features_meta, list):
        feature_cols = features_meta
    else:
        raise ValueError(f"Unrecognized format in {feat_path}")

    # 3b. Define RETRAIN TRIGGER from drift_detection_results.json
    det_results_path = RESULTS_DIR / "drift_detection_results.json"
    w2_drift_detected = False
    if det_results_path.exists():
        with open(det_results_path, "r", encoding="utf-8") as f:
            det_data = json.load(f)
        for w in det_data.get("windows", []):
            if w.get("window_id") == 2 and w.get("drift_detected"):
                w2_drift_detected = True
                break

    retrain_triggered_by = "drift" if w2_drift_detected else "scheduled"
    print(f"Retrain Trigger: {retrain_triggered_by} (drift detected in window 2: {w2_drift_detected})")

    # 3c. Load windows 0, 1, 2 and retrain
    print("Loading windows 0, 1, 2 for retraining ...")
    w0 = pd.read_parquet(DRIFT_DIR / "window_0.parquet")
    w1 = pd.read_parquet(DRIFT_DIR / "window_1.parquet")
    w2 = pd.read_parquet(DRIFT_DIR / "window_2.parquet")

    df_retrain = pd.concat([w0, w1, w2], ignore_index=True)
    X_retrain = df_retrain[feature_cols].to_numpy()
    y_retrain = df_retrain["label_8"].to_numpy() if "label_8" in df_retrain.columns else df_retrain["label_8class"].to_numpy()

    print(f"Fitting new RandomForestClassifier (50 trees, max_depth=15, warm_start=True) on {len(df_retrain)} samples ...")
    new_rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        warm_start=True,
        class_weight="balanced",
        n_jobs=-1,
    )
    new_rf.fit(X_retrain, y_retrain)
    print("Retraining complete.")

    # 3d. Evaluate BOTH models on unseen window 3
    print("Evaluating on unseen window 3 ...")
    w3 = pd.read_parquet(DRIFT_DIR / "window_3.parquet")
    X_eval = w3[feature_cols].to_numpy()
    y_eval = w3["label_8"].to_numpy() if "label_8" in w3.columns else w3["label_8class"].to_numpy()

    orig_preds = original_rf.predict(X_eval)
    retrained_preds = new_rf.predict(X_eval)

    original_acc = float((orig_preds == y_eval).mean())
    retrained_acc = float((retrained_preds == y_eval).mean())
    delta = retrained_acc - original_acc

    # 3e. Write drift/retrain_results.json
    results_payload = {
        "retrain_triggered_by": retrain_triggered_by,
        "retrain_windows_used": [0, 1, 2],
        "eval_window": 3,
        "original_rf_accuracy_window3": round(original_acc, 4),
        "retrained_rf_accuracy_window3": round(retrained_acc, 4),
        "accuracy_delta": round(float(delta), 6),
        "n_retrain_samples": int(len(df_retrain)),
    }

    out_file = RESULTS_DIR / "retrain_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[OK] Retrain results saved to {out_file}\n")

    # 3f. Print interpretation
    delta_pct = delta * 100.0
    sign = "+" if delta_pct >= 0 else ""
    interpretation = (
        "Retrained model shows improvement after drift"
        if delta > 0.0
        else "No improvement — drift may not affect this window"
    )

    print("=" * 65)
    print("            INCREMENTAL RETRAINING EVALUATION")
    print("=" * 65)
    print(f"[RETRAIN] Original RF accuracy on window 3 : {original_acc * 100:.2f}%")
    print(f"[RETRAIN] Retrained RF accuracy on window 3: {retrained_acc * 100:.2f}%")
    print(f"[RETRAIN] Delta: {sign}{delta_pct:.2f}%")
    print(f"[RETRAIN] Interpretation: \"{interpretation}\"")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
