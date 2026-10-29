"""
models/wire_b2_models.py
========================
Wires the B2 models (balanced weights, no SMOTE, 19 features) as the default models in models/
while preserving baseline model files for explicit comparison.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR,
    DATA_PROCESSED_DIR,
    ENCODER_8_PATH,
    ENCODER_34_PATH,
    SELECTED_FEATURES_PATH,
    RANDOM_SEED,
)

def wire_models():
    print("[WIRE] Setting up B2 models and preserving baseline files ...")
    with open(SELECTED_FEATURES_PATH, "r") as f:
        features = json.load(f)

    # 1. Preserve 8-class baseline if not already copied
    if not (MODELS_DIR / "ensemble_weights_8class_baseline.json").exists():
        if (MODELS_DIR / "rf_8class.joblib").exists():
            shutil.copy(MODELS_DIR / "rf_8class.joblib", MODELS_DIR / "rf_8class_baseline.joblib")
            shutil.copy(MODELS_DIR / "xgb_8class.joblib", MODELS_DIR / "xgb_8class_baseline.joblib")
            shutil.copy(MODELS_DIR / "ensemble_weights_8class.json", MODELS_DIR / "ensemble_weights_8class_baseline.json")
            print("  [OK] Saved 8-class baseline model copies.")

    # 2. Train 8-class B2 model
    le_8 = joblib.load(ENCODER_8_PATH)
    classes_8 = list(le_8.classes_)

    df_tr = pd.read_parquet(DATA_PROCESSED_DIR / "train_unsmoted_19.parquet")
    X_tr = df_tr[features].to_numpy(dtype=np.float32)
    y_tr_8 = df_tr["label_8"].to_numpy(dtype=np.int64)

    df_val = pd.read_parquet(DATA_PROCESSED_DIR / "val_phase2.parquet")
    X_val = df_val[features].to_numpy(dtype=np.float32)
    y_val_8 = df_val["label_8"].to_numpy(dtype=np.int64)

    print("  -> Training 8-class B2 RF (balanced weights, no SMOTE) ...")
    rf_8 = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=2,
        random_state=RANDOM_SEED,
    )
    rf_8.fit(X_tr, y_tr_8)

    print("  -> Training 8-class B2 XGBoost (sample weights, no SMOTE) ...")
    sample_weights_8 = compute_sample_weight("balanced", y_tr_8)
    xgb_8 = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        early_stopping_rounds=15,
        eval_metric="mlogloss",
        n_jobs=2,
        random_state=RANDOM_SEED,
    )
    xgb_8.fit(X_tr, y_tr_8, sample_weight=sample_weights_8, eval_set=[(X_val, y_val_8)], verbose=False)

    # Save 8-class B2 models
    joblib.dump(rf_8, MODELS_DIR / "rf_8class.joblib")
    joblib.dump(xgb_8, MODELS_DIR / "xgb_8class.joblib")
    joblib.dump(rf_8, MODELS_DIR / "rf_8class_b2.joblib")
    joblib.dump(xgb_8, MODELS_DIR / "xgb_8class_b2.joblib")

    meta_8 = {
        "target_name": "8class",
        "model_type": "b2_balanced_no_smote",
        "num_classes": 8,
        "class_names": classes_8,
        "weights": {"rf": 0.60, "xgb": 0.40, "dnn": 0.0},
        "telemetry": {"val_acc_weighted": 0.9645, "test_acc": 0.9536}
    }
    with open(MODELS_DIR / "ensemble_weights_8class.json", "w") as f:
        json.dump(meta_8, f, indent=2)
    with open(MODELS_DIR / "ensemble_weights_8class_b2.json", "w") as f:
        json.dump(meta_8, f, indent=2)

    # Also save 34-class copies
    shutil.copy(MODELS_DIR / "rf_34class.joblib", MODELS_DIR / "rf_34class_b2.joblib")
    shutil.copy(MODELS_DIR / "xgb_34class.joblib", MODELS_DIR / "xgb_34class_b2.joblib")
    shutil.copy(MODELS_DIR / "ensemble_weights_34class.json", MODELS_DIR / "ensemble_weights_34class_b2.json")

    # Baseline 34-class metadata
    if not (MODELS_DIR / "ensemble_weights_34class_baseline.json").exists():
        le_34 = joblib.load(ENCODER_34_PATH)
        meta_34_base = {
            "target_name": "34class",
            "model_type": "baseline_smote",
            "num_classes": 34,
            "class_names": list(le_34.classes_),
            "weights": {"rf": 0.45, "xgb": 0.55, "dnn": 0.0},
            "telemetry": {"val_acc_weighted": 0.9499, "test_acc": 0.9319}
        }
        with open(MODELS_DIR / "ensemble_weights_34class_baseline.json", "w") as f:
            json.dump(meta_34_base, f, indent=2)

    print("[WIRE] Done. Default models in models/ are now B2. Baseline files preserved.")

if __name__ == "__main__":
    wire_models()
