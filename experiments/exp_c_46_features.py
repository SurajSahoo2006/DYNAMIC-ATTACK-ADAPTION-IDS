"""
experiments/exp_c_46_features.py
================================
Experiment C: All 46 Features (Tree models only)
Tests RF + XGBoost on all 46 raw features instead of 19.
Evaluates accuracy, macro F1, rare class metrics, and inference latency trade-offs.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROCESSED_DIR,
    ENCODER_34_PATH,
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
from ingest.loader import FEATURE_COLS
from experiments.eval_utils import evaluate_34class, print_experiment_result


def run_exp_c(eval_test: bool = True):
    le_34 = joblib.load(ENCODER_34_PATH)
    classes = list(le_34.classes_)

    p_train = DATA_PROCESSED_DIR / "train_smote_46.parquet"
    p_val   = DATA_PROCESSED_DIR / "val_46.parquet"
    p_test  = DATA_PROCESSED_DIR / "test_46.parquet"

    print("\n[EXP C] Loading 46-feature datasets ...")
    df_train = pd.read_parquet(p_train)
    X_train = df_train[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = df_train["label_34"].to_numpy(dtype=np.int64)

    df_val = pd.read_parquet(p_val)
    X_val = df_val[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = df_val["label_34"].to_numpy(dtype=np.int64)

    # 1. Random Forest (46 features)
    print("[EXP C] Training RF (46 features) ...")
    t0_rf = time.time()
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_SEED,
    )
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0_rf
    print(f"       RF train time: {rf_train_time:.2f} s")

    # 2. XGBoost (46 features)
    print("[EXP C] Training XGBoost (46 features) ...")
    t0_xgb = time.time()
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
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_train_time = time.time() - t0_xgb
    print(f"       XGB train time: {xgb_train_time:.2f} s")

    # 3. Validation evaluation & weight tuning
    P_val_rf = rf.predict_proba(X_val)
    if P_val_rf.shape[1] < 34:
        full_p = np.zeros((len(X_val), 34), dtype=np.float32)
        for i, c in enumerate(rf.classes_):
            full_p[:, c] = P_val_rf[:, i]
        P_val_rf = full_p
    P_val_xgb = xgb.predict_proba(X_val)

    best_w = 0.5
    best_acc = 0.0
    for w in np.linspace(0.0, 1.0, 21):
        P_comb = w * P_val_rf + (1 - w) * P_val_xgb
        preds = np.argmax(P_comb, axis=1)
        acc = np.mean(preds == y_val)
        if acc > best_acc:
            best_acc = acc
            best_w = w

    print(f"Optimal validation weights (46 features): RF={best_w:.2f}, XGB={1-best_w:.2f} (Val Acc: {best_acc*100:.2f}%)")

    P_val_ens = best_w * P_val_rf + (1 - best_w) * P_val_xgb
    preds_val = np.argmax(P_val_ens, axis=1)
    res_val = evaluate_34class(y_val, preds_val, classes, split_name="Validation")

    res_test = None
    inf_latency_ms = 0.0
    if eval_test:
        df_test = pd.read_parquet(p_test)
        X_test = df_test[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_test = df_test["label_34"].to_numpy(dtype=np.int64)

        t0_inf = time.time()
        P_test_rf = rf.predict_proba(X_test)
        if P_test_rf.shape[1] < 34:
            full_p = np.zeros((len(X_test), 34), dtype=np.float32)
            for i, c in enumerate(rf.classes_):
                full_p[:, c] = P_test_rf[:, i]
            P_test_rf = full_p
        P_test_xgb = xgb.predict_proba(X_test)
        P_test_ens = best_w * P_test_rf + (1 - best_w) * P_test_xgb
        inf_latency_ms = ((time.time() - t0_inf) / len(X_test)) * 1000.0

        preds_test = np.argmax(P_test_ens, axis=1)
        res_test = evaluate_34class(y_test, preds_test, classes, split_name="Test")
        res_test["inf_latency_ms"] = round(inf_latency_ms, 4)

    print_experiment_result("C: All 46 Features (Tree models: RF + XGBoost)", res_val, res_test)
    if res_test is not None:
        print(f"Test Inference Latency: {inf_latency_ms:.4f} ms/sample")
    return {"val": res_val, "test": res_test, "rf": rf, "xgb": xgb, "w": best_w}


if __name__ == "__main__":
    run_exp_c(eval_test=True)
