"""
experiments/prepare_experiment_data.py
======================================
Prepares and caches full 46-feature and un-SMOTEd datasets for Experiments B, C, D, and E.
Ensures 100% bit-for-bit index and label alignment with Phase 2 train/test splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROCESSED_DIR,
    RANDOM_SEED,
    SCALER_PATH,
    ENCODER_34_PATH,
    ENCODER_8_PATH,
    SELECTED_FEATURES_PATH,
    VALIDATION_CSV,
    LABEL_TO_CLASS,
    CHUNK_SIZE,
)
from ingest.loader import load_ciciot2023, FEATURE_COLS
from preprocess.clean import clean
from preprocess.encode import encode_labels
from preprocess.split import split
from preprocess.balance import balance

def prepare_all_datasets():
    print("[PREPARE] Checking experiment dataset caches ...")
    p_train_unsmote_19 = DATA_PROCESSED_DIR / "train_unsmoted_19.parquet"
    p_train_smote_46   = DATA_PROCESSED_DIR / "train_smote_46.parquet"
    p_train_unsmote_46 = DATA_PROCESSED_DIR / "train_unsmoted_46.parquet"
    p_test_46          = DATA_PROCESSED_DIR / "test_46.parquet"
    p_val_46           = DATA_PROCESSED_DIR / "val_46.parquet"

    all_exist = all(p.exists() for p in [p_train_unsmote_19, p_train_smote_46, p_train_unsmote_46, p_test_46, p_val_46])
    if all_exist:
        print("[PREPARE] All experiment datasets already cached in data/processed/.")
        return

    # 1. Ingest raw train/test samples (matching Phase 1 & 2 run)
    print("[PREPARE] Ingesting raw train and test data (sample_per_class=5000) ...")
    df_raw = load_ciciot2023(sample_per_class=5000, seed=RANDOM_SEED)
    df_clean, _ = clean(df_raw)
    del df_raw

    # 2. Encode labels
    df_encoded, le_34, le_8 = encode_labels(df_clean)
    del df_clean

    # 3. Stratified split (80/20)
    X_train_raw, X_test_raw, y34_train, y34_test, y8_train, y8_test = split(df_encoded, FEATURE_COLS)
    del df_encoded

    # 4. Scale with saved scaler
    scaler = joblib.load(SCALER_PATH)
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_19 = json.load(f)

    X_train_s_np = scaler.transform(X_train_raw.astype(np.float64))
    X_test_s_np  = scaler.transform(X_test_raw.astype(np.float64))

    X_train_s_df = pd.DataFrame(X_train_s_np.astype(np.float32), columns=FEATURE_COLS)
    X_test_s_df  = pd.DataFrame(X_test_s_np.astype(np.float32), columns=FEATURE_COLS)

    # Cache test_46.parquet
    test_46_df = X_test_s_df.copy()
    test_46_df["label_34"] = y34_test.values.astype(np.int64)
    test_46_df["label_8"]  = y8_test.values.astype(np.int64)
    test_46_df.to_parquet(p_test_46, index=False)
    print(f"[PREPARE] Saved test_46.parquet: {test_46_df.shape}")

    # Cache train_unsmoted_19.parquet and train_unsmoted_46.parquet
    train_unsmote_19_df = X_train_s_df[selected_19].copy()
    train_unsmote_19_df["label_34"] = y34_train.values.astype(np.int64)
    train_unsmote_19_df["label_8"]  = y8_train.values.astype(np.int64)
    train_unsmote_19_df.to_parquet(p_train_unsmote_19, index=False)
    print(f"[PREPARE] Saved train_unsmoted_19.parquet: {train_unsmote_19_df.shape}")

    train_unsmote_46_df = X_train_s_df.copy()
    train_unsmote_46_df["label_34"] = y34_train.values.astype(np.int64)
    train_unsmote_46_df["label_8"]  = y8_train.values.astype(np.int64)
    train_unsmote_46_df.to_parquet(p_train_unsmote_46, index=False)
    print(f"[PREPARE] Saved train_unsmoted_46.parquet: {train_unsmote_46_df.shape}")

    # 5. Create SMOTE version on 46 features
    max_train_count = int(y34_train.value_counts().max())
    smote_target = min(max_train_count, 5000)
    print(f"[PREPARE] Applying SMOTE on 46 features (target={smote_target}) ...")
    X_train_b_46, y34_train_b, _ = balance(
        X_train_s_df,
        y34_train,
        target_count=smote_target,
        max_synthetic_multiplier=10,
        label_encoder=le_34,
    )
    le_34_classes = list(le_34.classes_)
    y8_train_b = y34_train_b.map(lambda i: le_8.transform([LABEL_TO_CLASS[le_34_classes[i]]])[0])

    train_smote_46_df = pd.DataFrame(X_train_b_46, columns=FEATURE_COLS)
    train_smote_46_df["label_34"] = y34_train_b.values.astype(np.int64)
    train_smote_46_df["label_8"]  = y8_train_b.values.astype(np.int64)
    train_smote_46_df.to_parquet(p_train_smote_46, index=False)
    print(f"[PREPARE] Saved train_smote_46.parquet: {train_smote_46_df.shape}")

    # 6. Prepare val_46.parquet
    print("[PREPARE] Preparing val_46.parquet from validation.csv ...")
    df_val_raw = load_ciciot2023(csv_paths=[VALIDATION_CSV], sample_per_class=5000, chunk_size=CHUNK_SIZE, seed=RANDOM_SEED)
    df_val_clean, _ = clean(df_val_raw)
    del df_val_raw

    df_val_clean["_class8"] = df_val_clean["label"].map(LABEL_TO_CLASS)
    df_val_clean = df_val_clean.dropna(subset=["_class8"])
    y34_val = le_34.transform(df_val_clean["label"])
    y8_val  = le_8.transform(df_val_clean["_class8"])

    X_val_all = df_val_clean[FEATURE_COLS].astype(np.float64)
    X_val_s_np = scaler.transform(X_val_all)
    val_46_df = pd.DataFrame(X_val_s_np.astype(np.float32), columns=FEATURE_COLS)
    val_46_df["label_34"] = y34_val.astype(np.int64)
    val_46_df["label_8"]  = y8_val.astype(np.int64)
    val_46_df.to_parquet(p_val_46, index=False)
    print(f"[PREPARE] Saved val_46.parquet: {val_46_df.shape}")
    print("[PREPARE] Done! All experiment datasets ready.")

if __name__ == "__main__":
    prepare_all_datasets()
