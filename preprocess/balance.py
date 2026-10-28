"""
preprocess/balance.py
=====================
Phase 2 - Step 5: Class Balancing with SMOTE
Applied to TRAIN set ONLY.  Test set is never resampled.

Why SMOTE instead of random oversampling?
  Random duplication of minority rows risks overfitting to those exact
  points.  SMOTE generates NEW synthetic points by interpolating between
  existing minority-class neighbours, producing a more generalisable
  training distribution.

Memory note
-----------
On a full 47 M-row dataset, even SMOTE on a 1 M-row train split may
need >32 GB RAM.  This module uses a CAPPED strategy:

  sampling_strategy = {cls: min(current_count, SMOTE_TARGET_COUNT)
                       for cls in minority_classes}

This means we only oversample classes that are genuinely under-represented
(below SMOTE_TARGET_COUNT).  Majority classes are left as-is.
If even that is too slow, you can further reduce SAMPLE_PER_CLASS
in config.py or switch to RandomOverSampler which is ~10x faster
(at the cost of exactduplicates in the minority).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SMOTE_TARGET_COUNT, RANDOM_SEED

log = logging.getLogger(__name__)


def balance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_count: int = SMOTE_TARGET_COUNT,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Oversample minority classes in (X_train, y_train) using SMOTE.

    Uses a CAPPED strategy: classes with fewer than `target_count` rows
    are upsampled to `target_count`.  Classes already above the cap are
    left unchanged.

    Falls back to RandomOverSampler if SMOTE fails (e.g. a class has
    fewer than k_neighbors+1 samples).

    Parameters
    ----------
    X_train      : scaled feature DataFrame
    y_train      : integer label Series (34-class or 8-class)
    target_count : maximum count to oversample TO
    seed         : random seed

    Returns
    -------
    (X_resampled, y_resampled)
    """
    counts = y_train.value_counts()
    print(f"\n[BALANCE] Class counts BEFORE SMOTE:")
    for cls, cnt in counts.sort_index().items():
        bar = "#" * min(int(cnt / max(counts) * 30), 30)
        print(f"  class {int(cls):>3}: {cnt:>8,}  {bar}")

    # Build sampling strategy: only oversample below cap
    strategy = {
        cls: target_count
        for cls, cnt in counts.items()
        if cnt < target_count
    }

    if not strategy:
        print("[BALANCE] All classes already meet or exceed target. "
              "No oversampling needed.")
        return X_train, y_train

    print(f"\n[BALANCE] SMOTE target count per class: {target_count:,}")
    print(f"[BALANCE] Classes to oversample: {len(strategy)}")

    # Try SMOTE first
    try:
        smote = SMOTE(
            sampling_strategy=strategy,
            random_state=seed,
            k_neighbors=min(5, counts.min() - 1),
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        method_used = "SMOTE"
    except Exception as exc:
        log.warning("SMOTE failed (%s). Falling back to RandomOverSampler.", exc)
        print(f"[BALANCE] WARNING  SMOTE failed: {exc}")
        print("[BALANCE] Falling back to RandomOverSampler ...")
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
        X_res, y_res = ros.fit_resample(X_train, y_train)
        method_used = "RandomOverSampler"

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    counts_after = y_res.value_counts()
    print(f"\n[BALANCE] Class counts AFTER {method_used}:")
    for cls, cnt in counts_after.sort_index().items():
        bar = "#" * min(int(cnt / max(counts_after) * 30), 30)
        print(f"  class {int(cls):>3}: {cnt:>8,}  {bar}")

    print(f"\n[BALANCE] Rows: {len(X_train):,} --> {len(X_res):,}  "
          f"(+{len(X_res) - len(X_train):,} synthetic rows, "
          f"method={method_used})")
    print("[BALANCE] [OK]  Test set was NOT resampled.")

    return X_res, y_res
