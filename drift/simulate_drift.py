"""
drift/simulate_drift.py
=======================
Simulates concept drift across 5 temporal windows from the test partition.
In window 2, injects synthetic drift on DDoS flows (flow_duration x8.0, IAT x6.0).
Saves window_0..4.parquet and drift_metadata.json to data/drift/.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DRIFT_DIR = PROJECT_ROOT / "data" / "drift"


def main():
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    # 1a. Load test partition, selected features, and label encoder
    test_path = PROJECT_ROOT / "data" / "processed" / "test.parquet"
    if not test_path.exists():
        test_path = PROJECT_ROOT / "data" / "processed" / "test_phase2.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Neither test.parquet nor test_phase2.parquet found in {test_path.parent}")

    df_test = pd.read_parquet(test_path)

    feat_path = PROJECT_ROOT / "artifacts" / "selected_features.json"
    with open(feat_path, "r", encoding="utf-8") as f:
        features_meta = json.load(f)
    if isinstance(features_meta, dict) and "selected_features" in features_meta:
        feature_cols = features_meta["selected_features"]
    elif isinstance(features_meta, list):
        feature_cols = features_meta
    else:
        raise ValueError(f"Unrecognized format in {feat_path}")

    enc_path = PROJECT_ROOT / "artifacts" / "label_encoder_8.pkl"
    if not enc_path.exists():
        enc_path = PROJECT_ROOT / "artifacts" / "label_encoder_8class.pkl"
    label_encoder = joblib.load(enc_path)

    # Find DDoS label index
    ddos_idx = None
    for idx, cname in enumerate(label_encoder.classes_):
        if str(cname).strip().lower() == "ddos":
            ddos_idx = idx
            break
    if ddos_idx is None:
        ddos_idx = 2  # standard canonical mapping

    # 1b. Split into 5 equal temporal windows (window 0..4)
    split_indices = np.array_split(range(len(df_test)), 5)
    windows = [df_test.iloc[idx].copy().reset_index(drop=True) for idx in split_indices]
    rows_per_window = [len(w) for w in windows]

    # In window 2 only: inject synthetic drift on DDoS flows
    w2 = windows[2]
    if "label_8" in w2.columns:
        ddos_mask = (w2["label_8"] == ddos_idx) | (w2["label_8"] == "DDoS")
    elif "label_8class" in w2.columns:
        ddos_mask = (w2["label_8class"] == ddos_idx) | (w2["label_8class"] == "DDoS")
    else:
        ddos_mask = pd.Series(False, index=w2.index)

    drifted_count = int(ddos_mask.sum())
    w2.loc[ddos_mask, "flow_duration"] = w2.loc[ddos_mask, "flow_duration"] * 8.0
    w2.loc[ddos_mask, "IAT"] = w2.loc[ddos_mask, "IAT"] * 6.0
    windows[2] = w2

    # 1c. Save each window
    for i, w in enumerate(windows):
        out_w = DRIFT_DIR / f"window_{i}.parquet"
        w.to_parquet(out_w, index=False)

    # 1d. Write drift_metadata.json
    meta = {
        "n_windows": 5,
        "drift_window": 2,
        "drift_description": "DDoS flow_duration x8, IAT x6 in window 2",
        "rows_per_window": rows_per_window,
        "drifted_rows_in_window_2": drifted_count,
    }

    meta_path = DRIFT_DIR / "drift_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Windows saved to {DRIFT_DIR}:")
    for i, count in enumerate(rows_per_window):
        extra = f" (injected drift into {drifted_count} DDoS rows)" if i == 2 else ""
        print(f"  window_{i}.parquet: {count} rows{extra}")
    print(f"Metadata saved to {meta_path}")
    print("[DRIFT SIM] Done")


if __name__ == "__main__":
    main()
