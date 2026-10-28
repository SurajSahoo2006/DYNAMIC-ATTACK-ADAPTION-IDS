"""
preprocess/clean.py
===================
Phase 2 - Step 1: Cleaning
Drops NaN rows, Inf rows, and exact duplicates.
Returns a cleaned DataFrame and a summary dict.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LABEL_COL = "label"


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove bad rows from `df`.

    Order of operations
    -------------------
    1. Drop rows with any NaN  (safer than imputing flow stats).
    2. Replace +/-Inf with NaN, then drop those rows too.
    3. Drop exact duplicate rows.

    Returns
    -------
    (cleaned_df, summary_dict)
    """
    original_len = len(df)
    print(f"\n[CLEAN] Input rows: {original_len:,}")

    # Step 1 - NaN
    df = df.dropna()
    after_nan = len(df)
    n_nan_dropped = original_len - after_nan
    print(f"[CLEAN] Dropped {n_nan_dropped:,} rows with NaN")

    # Step 2 - Inf  (only on numeric cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols].values).any(axis=1)
    df = df[~inf_mask]
    after_inf = len(df)
    n_inf_dropped = after_nan - after_inf
    print(f"[CLEAN] Dropped {n_inf_dropped:,} rows with +/-Inf")

    # Step 3 - Exact duplicates
    df = df.drop_duplicates()
    after_dedup = len(df)
    n_dupes_dropped = after_inf - after_dedup
    print(f"[CLEAN] Dropped {n_dupes_dropped:,} exact duplicate rows")

    print(f"[CLEAN] Output rows: {after_dedup:,}  "
          f"({after_dedup / original_len * 100:.1f}% retained)")

    summary = {
        "input_rows":        original_len,
        "nan_rows_dropped":  n_nan_dropped,
        "inf_rows_dropped":  n_inf_dropped,
        "dupe_rows_dropped": n_dupes_dropped,
        "output_rows":       after_dedup,
    }
    return df.reset_index(drop=True), summary
