"""
experiments/exp_a_baseline.py
=============================
Experiment A: Baseline
Evaluates the currently saved 34-class weighted RF+XGB ensemble on 19 features.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VAL_PARQUET, TEST_PARQUET, ENCODER_34_PATH, SELECTED_FEATURES_PATH, MODELS_DIR
from models.ensemble import EnsembleClassifier
from experiments.eval_utils import evaluate_34class, print_experiment_result


def run_exp_a(eval_test: bool = True):
    le_34 = joblib.load(ENCODER_34_PATH)
    classes = list(le_34.classes_)
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_19 = json.load(f)

    # 1. Load trained baseline ensemble
    ens = EnsembleClassifier.load(MODELS_DIR, target_name="34class")

    # 2. Validation evaluation
    df_val = pd.read_parquet(VAL_PARQUET)
    X_val = df_val[selected_19]
    y_val = df_val["label_34"].to_numpy(dtype=np.int64)

    preds_val, _ = ens.predict(X_val, mode="weighted")
    res_val = evaluate_34class(y_val, preds_val, classes, split_name="Validation")

    # 3. Test evaluation
    res_test = None
    if eval_test:
        df_test = pd.read_parquet(TEST_PARQUET)
        X_test = df_test[selected_19]
        y_test = df_test["label_34"].to_numpy(dtype=np.int64)
        preds_test, _ = ens.predict(X_test, mode="weighted")
        res_test = evaluate_34class(y_test, preds_test, classes, split_name="Test")

    print_experiment_result("A: Baseline (Weighted RF+XGB Ensemble, 19 Features)", res_val, res_test)
    return {"val": res_val, "test": res_test}


if __name__ == "__main__":
    run_exp_a(eval_test=True)
