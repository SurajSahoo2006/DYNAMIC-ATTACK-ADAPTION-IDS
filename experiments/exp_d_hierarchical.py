"""
experiments/exp_d_hierarchical.py
=================================
Experiment D: Hierarchical Classification Architecture
Stage 1: Predicts the 8-class family (Benign, BruteForce, DDoS, DoS, Mirai, Recon, Spoofing, Web-based)
Stage 2: Specialist models trained ONLY on their respective family sub-types:
  - Web Specialist: Backdoor_Malware, BrowserHijacking, CommandInjection, SqlInjection, Uploading_Attack, XSS
  - Recon Specialist: Recon-HostDiscovery, Recon-OSScan, Recon-PingSweep, Recon-PortScan, VulnerabilityScan
  - DDoS Specialist: 12 sub-types
  - DoS Specialist: 4 sub-types
  - Mirai Specialist: 3 sub-types
  - Spoofing Specialist: DNS_Spoofing, MITM-ArpSpoofing
  - BruteForce & Benign: Direct 1-to-1 mapping
Reports end-to-end 34-class metrics on Validation and Test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_PARQUET,
    VAL_PARQUET,
    TEST_PARQUET,
    ENCODER_34_PATH,
    ENCODER_8_PATH,
    SELECTED_FEATURES_PATH,
    MODELS_DIR,
    RANDOM_SEED,
)
from models.ensemble import EnsembleClassifier
from experiments.eval_utils import evaluate_34class, print_experiment_result


class HierarchicalIDS:
    def __init__(self, features: list[str]):
        self.features = features
        self.stage1_model = EnsembleClassifier.load(MODELS_DIR, target_name="8class")
        self.specialists: Dict[int, Any] = {}
        self.label_mapping: Dict[int, list[int]] = {}

    def fit_specialists(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        print("\n[HIERARCHICAL] Training Stage-2 Specialist Models ...")
        # 8-class IDs:
        # 0: Benign (1 class: 1)
        # 1: BruteForce (1 class: 17)
        # 2: DDoS (12 classes)
        # 3: DoS (4 classes)
        # 4: Mirai (3 classes)
        # 5: Recon (5 classes)
        # 6: Spoofing (2 classes)
        # 7: Web-based (6 classes)

        multi_sub_families = [2, 3, 4, 5, 6, 7]

        for fam_id in multi_sub_families:
            sub_tr = df_train[df_train["label_8"] == fam_id]
            sub_val = df_val[df_val["label_8"] == fam_id]

            unique_classes = sorted(sub_tr["label_34"].unique().tolist())
            self.label_mapping[fam_id] = unique_classes

            # Local re-indexing (0 to K-1)
            cls_to_local = {c: i for i, c in enumerate(unique_classes)}
            y_tr_local = np.array([cls_to_local[c] for c in sub_tr["label_34"]], dtype=np.int64)
            y_val_local = np.array([cls_to_local[c] for c in sub_val["label_34"] if c in cls_to_local], dtype=np.int64)

            X_tr_sub = sub_tr[self.features].to_numpy(dtype=np.float32)
            # Filter val rows belonging to this family
            sub_val_valid = sub_val[sub_val["label_34"].isin(unique_classes)]
            X_val_sub = sub_val_valid[self.features].to_numpy(dtype=np.float32)

            fam_names = {2: "DDoS", 3: "DoS", 4: "Mirai", 5: "Recon", 6: "Spoofing", 7: "Web-based"}
            print(f"  -> Fitting Specialist for '{fam_names[fam_id]}' ({len(unique_classes)} classes, {len(X_tr_sub):,} train rows) ...")

            # Specialist: RF with balanced class weights + XGBoost
            rf = RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=2,
                random_state=RANDOM_SEED,
            )
            rf.fit(X_tr_sub, y_tr_local)

            xgb = XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                n_jobs=2,
                random_state=RANDOM_SEED,
            )
            xgb.fit(X_tr_sub, y_tr_local)

            self.specialists[fam_id] = {
                "rf": rf,
                "xgb": xgb,
                "classes": unique_classes,
            }

        print("[HIERARCHICAL] All Stage-2 Specialists trained successfully.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features]
        # Stage 1: Predict 8-class family
        preds_8, _ = self.stage1_model.predict(X, mode="weighted")

        final_preds_34 = np.zeros(len(df), dtype=np.int64)
        X_np = X.to_numpy(dtype=np.float32)

        for pred_fam in range(8):
            mask = (preds_8 == pred_fam)
            if not np.any(mask):
                continue
            if pred_fam == 0:    # Benign -> BenignTraffic (1)
                final_preds_34[mask] = 1
            elif pred_fam == 1:  # BruteForce -> DictionaryBruteForce (17)
                final_preds_34[mask] = 17
            else:
                spec = self.specialists[pred_fam]
                X_fam = X_np[mask]
                p_rf = spec["rf"].predict_proba(X_fam)
                p_xgb = spec["xgb"].predict_proba(X_fam)
                p_ens = 0.5 * p_rf + 0.5 * p_xgb
                local_preds = np.argmax(p_ens, axis=1)
                classes_arr = np.array(spec["classes"], dtype=np.int64)
                final_preds_34[mask] = classes_arr[local_preds]

        return final_preds_34


def run_exp_d(eval_test: bool = True):
    le_34 = joblib.load(ENCODER_34_PATH)
    classes = list(le_34.classes_)
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_19 = json.load(f)

    df_train = pd.read_parquet(TRAIN_PARQUET)
    df_val = pd.read_parquet(VAL_PARQUET)

    h_model = HierarchicalIDS(selected_19)
    h_model.fit_specialists(df_train, df_val)

    # 1. Validation evaluation
    print("\n[EXP D] Running hierarchical inference on validation set ...")
    preds_val = h_model.predict(df_val)
    y_val = df_val["label_34"].to_numpy(dtype=np.int64)
    res_val = evaluate_34class(y_val, preds_val, classes, split_name="Validation")

    # 2. Test evaluation
    res_test = None
    if eval_test:
        print("[EXP D] Running hierarchical inference on test set ...")
        df_test = pd.read_parquet(TEST_PARQUET)
        preds_test = h_model.predict(df_test)
        y_test = df_test["label_34"].to_numpy(dtype=np.int64)
        res_test = evaluate_34class(y_test, preds_test, classes, split_name="Test")

    print_experiment_result("D: Hierarchical IDS (Stage 1 Family + Stage 2 Specialists)", res_val, res_test)
    return {"val": res_val, "test": res_test, "model": h_model}


if __name__ == "__main__":
    run_exp_d(eval_test=True)
