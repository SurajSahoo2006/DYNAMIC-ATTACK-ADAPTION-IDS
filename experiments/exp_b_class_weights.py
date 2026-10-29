"""
experiments/exp_b_class_weights.py
==================================
Experiment B: Class Weighting & Sample Weighting (with and without SMOTE)
Tests:
  - B1: RF class_weight='balanced' + XGBoost sample_weight WITH SMOTE (19 features)
  - B2: RF class_weight='balanced' + XGBoost sample_weight WITHOUT SMOTE (19 features)
"""

from __future__ import annotations

import json
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
    TRAIN_PARQUET,
    DATA_PROCESSED_DIR,
    VAL_PARQUET,
    TEST_PARQUET,
    ENCODER_34_PATH,
    SELECTED_FEATURES_PATH,
    RANDOM_SEED,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_N_JOBS,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_EARLY_STOPPING,
    XGB_N_JOBS,
)
from experiments.eval_utils import evaluate_34class, print_experiment_result


def train_weighted_models(
    train_parquet_path: Path,
    use_smote_label: str,
    eval_test: bool = True,
):
    le_34 = joblib.load(ENCODER_34_PATH)
    classes = list(le_34.classes_)
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_19 = json.load(f)

    # Load data
    df_train = pd.read_parquet(train_parquet_path)
    X_train = df_train[selected_19].to_numpy(dtype=np.float32)
    y_train = df_train["label_34"].to_numpy(dtype=np.int64)

    df_val = pd.read_parquet(VAL_PARQUET)
    X_val = df_val[selected_19].to_numpy(dtype=np.float32)
    y_val = df_val["label_34"].to_numpy(dtype=np.int64)

    # 1. Random Forest with balanced class weights
    print(f"\n[EXP B: {use_smote_label}] Fitting RF (class_weight='balanced') ...")
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_SEED,
    )
    rf.fit(X_train, y_train)

    # 2. XGBoost with sample weights
    print(f"[EXP B: {use_smote_label}] Fitting XGBoost with balanced sample weights ...")
    sample_weights = compute_sample_weight("balanced", y_train)
    xgb = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        tree_method="hist",
        early_stopping_rounds=XGB_EARLY_STOPPING,
        eval_metric="mlogloss",
        n_jobs=XGB_N_JOBS,
        random_state=RANDOM_SEED,
    )
    xgb.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 3. Probability prediction and weight tuning on validation
    P_val_rf = rf.predict_proba(X_val)
    # Ensure shape (N, 34)
    if P_val_rf.shape[1] < 34:
        full_p = np.zeros((len(X_val), 34), dtype=np.float32)
        for i, c in enumerate(rf.classes_):
            full_p[:, c] = P_val_rf[:, i]
        P_val_rf = full_p

    P_val_xgb = xgb.predict_proba(X_val)

    # Tune ensemble weight w on validation
    best_w = 0.5
    best_acc = 0.0
    for w in np.linspace(0.0, 1.0, 21):
        P_comb = w * P_val_rf + (1 - w) * P_val_xgb
        preds = np.argmax(P_comb, axis=1)
        acc = np.mean(preds == y_val)
        if acc > best_acc:
            best_acc = acc
            best_w = w

    print(f"Optimal validation weights: RF={best_w:.2f}, XGB={1-best_w:.2f} (Val Acc: {best_acc*100:.2f}%)")

    P_val_ens = best_w * P_val_rf + (1 - best_w) * P_val_xgb
    preds_val = np.argmax(P_val_ens, axis=1)
    res_val = evaluate_34class(y_val, preds_val, classes, split_name="Validation")

    res_test = None
    if eval_test:
        df_test = pd.read_parquet(TEST_PARQUET)
        X_test = df_test[selected_19].to_numpy(dtype=np.float32)
        y_test = df_test["label_34"].to_numpy(dtype=np.int64)

        P_test_rf = rf.predict_proba(X_test)
        if P_test_rf.shape[1] < 34:
            full_p = np.zeros((len(X_test), 34), dtype=np.float32)
            for i, c in enumerate(rf.classes_):
                full_p[:, c] = P_test_rf[:, i]
            P_test_rf = full_p
        P_test_xgb = xgb.predict_proba(X_test)
        P_test_ens = best_w * P_test_rf + (1 - best_w) * P_test_xgb
        preds_test = np.argmax(P_test_ens, axis=1)
        res_test = evaluate_34class(y_test, preds_test, classes, split_name="Test")

    exp_title = f"B: Balanced Class Weights ({use_smote_label}, 19 Features)"
    print_experiment_result(exp_title, res_val, res_test)
    return {"val": res_val, "test": res_test, "rf": rf, "xgb": xgb, "w": best_w}


def run_exp_b(eval_test: bool = True):
    res_b1 = train_weighted_models(
        TRAIN_PARQUET,
        use_smote_label="WITH SMOTE",
        eval_test=eval_test,
    )
    p_unsmote = DATA_PROCESSED_DIR / "train_unsmoted_19.parquet"
    res_b2 = train_weighted_models(
        p_unsmote,
        use_smote_label="WITHOUT SMOTE",
        eval_test=eval_test,
    )
    return {"B1_smote": res_b1, "B2_no_smote": res_b2}


if __name__ == "__main__":
    run_exp_b(eval_test=True)
