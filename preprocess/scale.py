"""
preprocess/scale.py
===================
Phase 2 - Step 4: Normalisation
Fits StandardScaler on X_TRAIN only, then transforms both splits.
Saves the scaler to ARTIFACTS_DIR.

Why StandardScaler?
  Flow features have very different ranges (e.g. flow_duration ~0-10,
  ack_count ~0-1e6).  Standard scaling (zero mean, unit variance)
  prevents large-magnitude features from dominating distance-based
  algorithms and speeds up DNN training (Phase 3).
"""

from __future__ import annotations

import joblib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SCALER_PATH

log = logging.getLogger(__name__)


def scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit scaler on X_train, apply to both.

    Parameters
    ----------
    X_train, X_test : DataFrames of float32 features

    Returns
    -------
    (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    print(f"\n[SCALE] Fitting StandardScaler on {len(X_train):,} train rows ...")

    scaler = StandardScaler()
    cols   = X_train.columns.tolist()

    X_train_np = scaler.fit_transform(X_train.astype("float64"))
    X_test_np  = scaler.transform(X_test.astype("float64"))

    X_train_scaled = pd.DataFrame(X_train_np.astype("float32"),
                                  columns=cols)
    X_test_scaled  = pd.DataFrame(X_test_np.astype("float32"),
                                  columns=cols)

    joblib.dump(scaler, SCALER_PATH)
    print(f"[SCALE] Scaler saved --> {SCALER_PATH.name}")
    print(f"[SCALE] Feature means (first 5): "
          f"{np.round(scaler.mean_[:5], 4)}")
    print(f"[SCALE] Feature stds  (first 5): "
          f"{np.round(scaler.scale_[:5], 4)}")
    print("[SCALE] [OK]  Only train set used for fitting.")

    return X_train_scaled, X_test_scaled, scaler
