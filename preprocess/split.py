"""
preprocess/split.py
===================
Phase 2 - Step 3: Train / Test Split
Stratified 80/20 split performed BEFORE scaling or SMOTE.
This is the correct order to prevent data leakage.

LEAKAGE PREVENTION
------------------
- Scaler is fit ONLY on X_train  (done in scale.py).
- SMOTE is applied ONLY to X_train / y_train  (done in balance.py).
- Test set is never touched by any fitting step.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TEST_SIZE, RANDOM_SEED

log = logging.getLogger(__name__)

LABEL_34_COL = "label_34"
LABEL_8_COL  = "label_8"
LABEL_COL    = "label"


def split(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
           pd.Series, pd.Series]:
    """
    Stratified split on label_34 (the finer-grained column gives better
    stratification for rare classes).

    Parameters
    ----------
    df           : full cleaned+encoded DataFrame
    feature_cols : list of the 46 numeric feature column names

    Returns
    -------
    X_train, X_test  : DataFrames with `feature_cols` columns
    y34_train, y34_test  : Series with 34-class integer labels
    y8_train,  y8_test   : Series with 8-class integer labels
    """
    X = df[feature_cols]
    y34 = df[LABEL_34_COL]
    y8  = df[LABEL_8_COL]

    print(f"\n[SPLIT] Total rows: {len(df):,}  -->  "
          f"{int(len(df) * (1 - TEST_SIZE)):,} train / "
          f"{int(len(df) * TEST_SIZE):,} test  (seed={RANDOM_SEED})")

    # Stratify on the fine-grained label
    X_train, X_test, y34_train, y34_test, y8_train, y8_test = (
        train_test_split(
            X, y34, y8,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y34,
        )
    )

    # ---- Sanity check: no overlap ----
    train_idx = set(X_train.index)
    test_idx  = set(X_test.index)
    overlap   = train_idx & test_idx
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} rows appear in both sets!"
    print("[SPLIT] [OK]  Sanity check passed: 0 overlapping indices between train and test.")

    print(f"[SPLIT] X_train: {X_train.shape}   X_test: {X_test.shape}")

    # Class counts in train
    print("\n[SPLIT] 8-class distribution in TRAIN set:")
    for cls_int, cnt in y8_train.value_counts().sort_index().items():
        print(f"         class {cls_int:>2}: {cnt:>8,}")

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y34_train.reset_index(drop=True),
        y34_test.reset_index(drop=True),
        y8_train.reset_index(drop=True),
        y8_test.reset_index(drop=True),
    )
