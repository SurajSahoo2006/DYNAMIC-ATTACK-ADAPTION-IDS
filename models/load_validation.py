"""
models/load_validation.py
=========================
Phase 3 - Leak-Free Validation Data Loader

Loads validation data strictly from VALIDATION_CSV without touching
or carving rows from the SMOTE-balanced training set.
Applies:
  1. Cleaning (drop NaN, Inf, exact duplicates)
  2. Saved label encoders (label_34 and label_8)
  3. Saved StandardScaler fitted on Phase 2 training data (46 features)
  4. Feature reduction to the 19 selected features
  5. NO SMOTE applied (real distribution preserved)

Caches processed validation data to `data/processed/val_phase2.parquet`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VALIDATION_CSV,
    VAL_PARQUET,
    SCALER_PATH,
    ENCODER_34_PATH,
    ENCODER_8_PATH,
    SELECTED_FEATURES_PATH,
    LABEL_TO_CLASS,
    SAMPLE_PER_CLASS,
    CHUNK_SIZE,
    RANDOM_SEED,
)
from ingest.loader import load_ciciot2023, FEATURE_COLS
from preprocess.clean import clean

log = logging.getLogger(__name__)


def load_validation_data(
    sample_per_class: int = 5000,
    force_reload: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Loads and preprocesses validation data.

    Parameters
    ----------
    sample_per_class : int, cap per fine-grained class (matches Phase 1 cap, default 5000)
    force_reload     : bool, if True reprocesses validation.csv even if val_phase2.parquet exists

    Returns
    -------
    X_val : pd.DataFrame with 19 scaled features
    y_val_8 : np.ndarray of shape (N,) int64
    y_val_34 : np.ndarray of shape (N,) int64
    """
    if VAL_PARQUET.exists() and not force_reload:
        print(f"[VAL] Loading cached validation data from {VAL_PARQUET.name} ...")
        df_val_proc = pd.read_parquet(VAL_PARQUET)
        with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
            features_19 = json.load(f)
        X_val = df_val_proc[features_19].astype("float32")
        y_val_8 = df_val_proc["label_8"].to_numpy(dtype=np.int64)
        y_val_34 = df_val_proc["label_34"].to_numpy(dtype=np.int64)
        print(f"[VAL] Loaded {len(df_val_proc):,} validation rows across {len(features_19)} features.")
        return X_val, y_val_8, y_val_34

    print(f"\n[VAL] Processing validation data from {VALIDATION_CSV} (cap: {sample_per_class:,}/class) ...")
    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(f"Validation CSV not found at {VALIDATION_CSV}")

    # 1. Chunked reservoir sampling
    df_raw = load_ciciot2023(
        csv_paths=[VALIDATION_CSV],
        sample_per_class=sample_per_class,
        chunk_size=CHUNK_SIZE,
        seed=RANDOM_SEED,
    )

    # 2. Clean bad rows
    df_cleaned, _ = clean(df_raw)
    del df_raw

    # 3. Apply saved encoders
    if not ENCODER_34_PATH.exists() or not ENCODER_8_PATH.exists():
        raise FileNotFoundError("Saved encoders not found. Run Phase 1 & 2 first.")

    le_34 = joblib.load(ENCODER_34_PATH)
    le_8 = joblib.load(ENCODER_8_PATH)

    df_cleaned["_class8"] = df_cleaned["label"].map(LABEL_TO_CLASS)
    unmapped = df_cleaned["_class8"].isnull().sum()
    if unmapped > 0:
        print(f"[VAL] WARNING: Dropping {unmapped} unmapped labels.")
        df_cleaned = df_cleaned.dropna(subset=["_class8"])

    df_cleaned["label_34"] = le_34.transform(df_cleaned["label"])
    df_cleaned["label_8"] = le_8.transform(df_cleaned["_class8"])

    # 4. Apply saved StandardScaler on all 46 features
    if not SCALER_PATH.exists():
        raise FileNotFoundError("Saved scaler not found. Run Phase 1 & 2 first.")
    scaler = joblib.load(SCALER_PATH)

    X_val_all = df_cleaned[FEATURE_COLS].astype("float64")
    X_val_scaled_np = scaler.transform(X_val_all)
    X_val_scaled_df = pd.DataFrame(X_val_scaled_np.astype("float32"), columns=FEATURE_COLS)

    # 5. Select 19 features
    if not SELECTED_FEATURES_PATH.exists():
        raise FileNotFoundError("Selected features list not found. Run Phase 1 & 2 first.")
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        features_19 = json.load(f)

    X_val_19 = X_val_scaled_df[features_19]

    # Combine into parquet for caching
    df_val_parquet = X_val_19.copy()
    df_val_parquet["label_34"] = df_cleaned["label_34"].values.astype(np.int64)
    df_val_parquet["label_8"] = df_cleaned["label_8"].values.astype(np.int64)

    df_val_parquet.to_parquet(VAL_PARQUET, index=False)
    print(f"[VAL] Cached processed validation set to {VAL_PARQUET} ({len(df_val_parquet):,} rows).")

    y_val_8 = df_val_parquet["label_8"].to_numpy(dtype=np.int64)
    y_val_34 = df_val_parquet["label_34"].to_numpy(dtype=np.int64)

    return X_val_19, y_val_8, y_val_34


if __name__ == "__main__":
    X_val, y_val_8, y_val_34 = load_validation_data(sample_per_class=5000)
    print(f"Validation features shape: {X_val.shape}")
    print(f"y_val_8 shape: {y_val_8.shape}, unique classes: {np.unique(y_val_8)}")
    print(f"y_val_34 shape: {y_val_34.shape}, unique classes: {np.unique(y_val_34)}")
