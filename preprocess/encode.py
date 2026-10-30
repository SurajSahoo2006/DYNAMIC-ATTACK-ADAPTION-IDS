"""
preprocess/encode.py
====================
Phase 2 - Step 2: Label Encoding
Creates two encoded label columns:
  - label_34  : integer 0-33 (fine-grained, 34 classes)
  - label_8   : integer 0-7  (8 super-classes)

Saves both LabelEncoders as pickle files to ARTIFACTS_DIR.
"""

from __future__ import annotations

import joblib
import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ENCODER_34_PATH,
    ENCODER_8_PATH,
    LABEL_TO_CLASS,
    CLASS_NAMES,
)

log = logging.getLogger(__name__)

LABEL_COL    = "label"
LABEL_34_COL = "label_34"
LABEL_8_COL  = "label_8"


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Add label_34 and label_8 integer columns to df.

    Steps
    -----
    1. Map raw string labels --> 8-class strings using LABEL_TO_CLASS.
    2. Fit LabelEncoder for 34-class labels on the data.
    3. Fit LabelEncoder for 8-class labels (alphabetical = deterministic).
    4. Save both encoders.

    Returns
    -------
    (df_with_new_cols, le_34, le_8)
    """
    print(f"\n[ENCODE] Input rows: {len(df):,}")

    # ---- 8-class mapping ----
    df["_class8"] = df[LABEL_COL].map(LABEL_TO_CLASS)
    unmapped = df["_class8"].isnull().sum()
    if unmapped > 0:
        log.warning("[ENCODE] %d rows have unmapped labels:", unmapped)
        print(f"  WARNING  {unmapped} unmapped labels (will be dropped):")
        print(df[df["_class8"].isnull()][LABEL_COL].value_counts())
        df = df.dropna(subset=["_class8"])

    # ---- 34-class encoder ----
    le_34 = LabelEncoder()
    df[LABEL_34_COL] = le_34.fit_transform(df[LABEL_COL])
    joblib.dump(le_34, ENCODER_34_PATH)
    print(f"[ENCODE] 34-class encoder saved --> {ENCODER_34_PATH.name}")
    print("         Classes:", list(le_34.classes_))

    # ---- 8-class encoder ----
    le_8 = LabelEncoder()
    le_8.fit(CLASS_NAMES)                          # fixed order
    df[LABEL_8_COL] = le_8.transform(df["_class8"])
    joblib.dump(le_8, ENCODER_8_PATH)
    print(f"[ENCODE] 8-class encoder saved  --> {ENCODER_8_PATH.name}")
    print("         Classes:", list(le_8.classes_))

    # Mapping table
    print("\n  34-label --> 8-class mapping (as loaded from data):")
    print(f"  {'Fine-grained label':<42} {'int34':>5}  {'8-class':<14} {'int8':>4}")
    print(f"  {'-'*42} {'-'*5}  {'-'*14} {'-'*4}")
    for i, cls in enumerate(le_34.classes_):
        cls8  = LABEL_TO_CLASS.get(cls, "UNKNOWN")
        int8  = le_8.transform([cls8])[0]
        print(f"  {cls:<42} {i:>5}  {cls8:<14} {int8:>4}")

    df = df.drop(columns=["_class8"])
    return df, le_34, le_8
