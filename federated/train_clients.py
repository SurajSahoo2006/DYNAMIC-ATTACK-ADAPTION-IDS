#!/usr/bin/env python3
"""
federated/train_clients.py - Local client training for federated learning simulation.
"""

import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEDERATED_DATA_DIR = PROJECT_ROOT / "federated" / "data"
FEDERATED_MODELS_DIR = PROJECT_ROOT / "federated" / "models"
FEDERATED_RESULTS_DIR = PROJECT_ROOT / "federated" / "results"


def load_artifacts():
    scaler_path = PROJECT_ROOT / "artifacts" / "scaler.pkl"
    if not scaler_path.exists():
        scaler_path = PROJECT_ROOT / "artifacts" / "standard_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError("Could not find artifacts/scaler.pkl")
    scaler = joblib.load(scaler_path)

    feat_path = PROJECT_ROOT / "artifacts" / "selected_features.json"
    with open(feat_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if isinstance(meta, dict) and "selected_features" in meta:
        features = meta["selected_features"]
    elif isinstance(meta, list):
        features = meta
    else:
        raise ValueError(f"Unexpected format in {feat_path}")

    return scaler, features


def train_clients():
    FEDERATED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FEDERATED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scaler, feature_cols = load_artifacts()

    clients_results = []

    print("\n" + "=" * 65)
    print("           LOCAL FEDERATED CLIENT TRAINING (RF)")
    print("=" * 65)
    print(f"{'Client ID':<10} | {'Rows':<8} | {'Train Accuracy':<16} | {'Train Time':<12}")
    print("-" * 65)

    for cid in range(5):
        client_file = FEDERATED_DATA_DIR / f"client_{cid}_train.parquet"
        if not client_file.exists():
            raise FileNotFoundError(f"Missing client partition file: {client_file}")

        df_client = pd.read_parquet(client_file)
        n_rows = len(df_client)

        X = df_client[feature_cols]
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(feature_cols):
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.to_numpy()

        label_col = "label_8" if "label_8" in df_client.columns else "label"
        y = df_client[label_col].to_numpy()

        n_estimators = 50
        max_depth = 12

        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(X_scaled, y)
        elapsed = time.time() - t0

        # If client exceeds 4 minutes, retry once with n_estimators=20
        if elapsed > 240.0:
            print(f"[WARN] Client {cid} took {elapsed:.1f}s (> 240s). Retrying with n_estimators=20...")
            t0 = time.time()
            rf = RandomForestClassifier(
                n_estimators=20,
                max_depth=max_depth,
                random_state=42,
                class_weight="balanced",
            )
            rf.fit(X_scaled, y)
            elapsed = time.time() - t0

        train_preds = rf.predict(X_scaled)
        train_acc = float(accuracy_score(y, train_preds))

        model_out = FEDERATED_MODELS_DIR / f"client_{cid}_rf.joblib"
        joblib.dump(rf, model_out)

        print(f"Client {cid:<3} | {n_rows:<8} | {train_acc * 100:.2f}%          | {elapsed:.2f}s")

        clients_results.append({
            "client_id": cid,
            "n_rows": int(n_rows),
            "local_train_accuracy": round(train_acc, 6),
        })

    results_data = {"clients": clients_results}
    results_path = FEDERATED_RESULTS_DIR / "local_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print("-" * 65)
    print(f"[TRAIN] Local training complete. Results saved to {results_path.name}")


def main():
    train_clients()


if __name__ == "__main__":
    main()
