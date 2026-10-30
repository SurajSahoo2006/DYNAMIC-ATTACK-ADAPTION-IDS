#!/usr/bin/env python3
"""
federated/fedavg.py - FedAvg tree pooling aggregation and test evaluation.
"""

import copy
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEDERATED_MODELS_DIR = PROJECT_ROOT / "federated" / "models"
FEDERATED_RESULTS_DIR = PROJECT_ROOT / "federated" / "results"


def load_scaler():
    scaler_paths = [
        PROJECT_ROOT / "artifacts" / "scaler.pkl",
        PROJECT_ROOT / "artifacts" / "standard_scaler.pkl",
    ]
    for p in scaler_paths:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError("Could not find scaler in candidate paths.")


def load_centralized_model():
    candidates = [
        PROJECT_ROOT / "models" / "rf_8class_b2.joblib",
        PROJECT_ROOT / "models" / "rf_model.joblib",
        PROJECT_ROOT / "models" / "rf_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            print(f"[FEDAVG] Loaded centralized baseline from {p.name}")
            return joblib.load(p)
    raise FileNotFoundError("Could not find centralized baseline model.")


def main():
    FEDERATED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FEDERATED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 3a. Load all 5 client models and pool trees
    client_models = [
        joblib.load(FEDERATED_MODELS_DIR / f"client_{i}_rf.joblib") for i in range(5)
    ]

    global_rf = copy.deepcopy(client_models[0])
    for cm in client_models[1:]:
        global_rf.estimators_ += copy.deepcopy(cm.estimators_)
    global_rf.n_estimators = len(global_rf.estimators_)

    global_model_path = FEDERATED_MODELS_DIR / "global_rf.joblib"
    joblib.dump(global_rf, global_model_path)
    print(f"[FEDAVG] Global model saved to {global_model_path.name} (pooled {global_rf.n_estimators} trees)")

    # 3b. Load test dataset and scale
    test_paths = [
        PROJECT_ROOT / "data" / "processed" / "test.parquet",
        PROJECT_ROOT / "data" / "processed" / "test_phase2.parquet",
    ]
    df_test = None
    for p in test_paths:
        if p.exists():
            print(f"[FEDAVG] Loading test data from {p.name}")
            df_test = pd.read_parquet(p)
            break
    if df_test is None:
        raise FileNotFoundError("Could not find test.parquet or test_phase2.parquet")

    feat_path = PROJECT_ROOT / "artifacts" / "selected_features.json"
    with open(feat_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if isinstance(meta, dict) and "selected_features" in meta:
        feature_cols = meta["selected_features"]
    elif isinstance(meta, list):
        feature_cols = meta
    else:
        raise ValueError(f"Unexpected format in {feat_path}")

    scaler = load_scaler()
    X = df_test[feature_cols]
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(feature_cols):
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.to_numpy()

    label_col = "label_8" if "label_8" in df_test.columns else "label"
    y_test = df_test[label_col].to_numpy()

    # Evaluate clients
    client_evals = []
    dominant_labels = ["DDoS", "DoS", "Recon", "Balanced", "Benign"]

    for cid in range(5):
        cm = client_models[cid]
        preds = cm.predict(X_scaled)
        acc = float(accuracy_score(y_test, preds))
        macro_f1 = float(f1_score(y_test, preds, average="macro"))
        client_evals.append({
            "client_id": cid,
            "test_accuracy": round(acc, 6),
            "test_macro_f1": round(macro_f1, 6),
        })

    # Evaluate global federated model
    g_preds = global_rf.predict(X_scaled)
    g_acc = float(accuracy_score(y_test, g_preds))
    g_macro_f1 = float(f1_score(y_test, g_preds, average="macro"))

    global_eval = {
        "test_accuracy": round(g_acc, 6),
        "test_macro_f1": round(g_macro_f1, 6),
        "n_trees_total": int(global_rf.n_estimators),
    }

    # 3c. Evaluate centralized baseline
    central_model = load_centralized_model()
    c_preds = central_model.predict(X_scaled)
    c_acc = float(accuracy_score(y_test, c_preds))
    c_macro_f1 = float(f1_score(y_test, c_preds, average="macro"))

    central_eval = {
        "test_accuracy": round(c_acc, 6),
        "test_macro_f1": round(c_macro_f1, 6),
    }

    # 3d. Write federated/results/fedavg_results.json
    results = {
        "client_models": client_evals,
        "global_federated_model": global_eval,
        "centralized_baseline": central_eval,
    }

    out_json = FEDERATED_RESULTS_DIR / "fedavg_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[FEDAVG] Results saved to {out_json.name}")

    # 3e. Print comparison table
    print("\n" + "=" * 55)
    print("      FEDERATED vs CENTRALIZED EVALUATION COMPARISON")
    print("=" * 55)
    print(f"{'Model':<22} | {'Test Accuracy':<14} | {'Macro F1':<10}")
    print("-" * 55)
    for cid in range(5):
        dom = dominant_labels[cid]
        c_acc_pct = client_evals[cid]["test_accuracy"] * 100
        c_f1_pct = client_evals[cid]["test_macro_f1"] * 100
        print(f"Client {cid} ({dom:<8}) | {c_acc_pct:>6.2f}%         | {c_f1_pct:>6.2f}%")
    print(f"Global FedAvg          | {g_acc * 100:>6.2f}%         | {g_macro_f1 * 100:>6.2f}%")
    print(f"Centralized (Phase3)   | {c_acc * 100:>6.2f}%         | {c_macro_f1 * 100:>6.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
