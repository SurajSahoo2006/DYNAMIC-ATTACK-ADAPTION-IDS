"""
ingest/validate.py
==================
Phase 1 - Validation Report
Prints a human-readable summary of the raw ingested DataFrame and
saves the class distribution bar chart to reports/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import REPORTS_DIR, LABEL_TO_CLASS

log = logging.getLogger(__name__)

LABEL_COL = "label"


def run_validation_report(df: pd.DataFrame) -> dict:
    """
    Print and return a validation summary dict.
    Saves class_distribution.png to REPORTS_DIR.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1 -- VALIDATION REPORT")
    print("=" * 60)

    # ---- basic shape ----
    n_rows, n_cols = df.shape
    print(f"\n{'Row count:':<35} {n_rows:>12,}")
    print(f"{'Column count (features + label):':<35} {n_cols:>12,}")

    # ---- missing values ----
    missing = df.isnull().sum()
    n_missing_cols = (missing > 0).sum()
    total_missing  = missing.sum()
    print(f"\n{'Columns with NaN:':<35} {n_missing_cols:>12,}")
    print(f"{'Total NaN cells:':<35} {total_missing:>12,}")
    if n_missing_cols > 0:
        print("  Columns with missing values:")
        for col, cnt in missing[missing > 0].items():
            print(f"    {col:<30} {cnt:>10,}")

    # ---- infinite values ----
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols].values)
    total_inf = int(inf_mask.sum())
    inf_cols  = int((inf_mask.any(axis=0)).sum())
    print(f"\n{'Columns with Inf:':<35} {inf_cols:>12,}")
    print(f"{'Total Inf cells:':<35} {total_inf:>12,}")

    # ---- duplicates ----
    n_dupes = int(df.duplicated().sum())
    print(f"\n{'Exact duplicate rows:':<35} {n_dupes:>12,}")

    # ---- dtypes ----
    dtype_counts = df.dtypes.value_counts()
    print(f"\nData-type breakdown:")
    for dtype, cnt in dtype_counts.items():
        print(f"  {str(dtype):<15} {cnt:>5} columns")

    # ---- label counts ----
    label_counts = df[LABEL_COL].value_counts()
    print(f"\nFine-grained label counts  ({len(label_counts)} unique):")
    print(f"  {'Label':<40} {'Count':>10}  {'8-class':<12}")
    print(f"  {'-'*40} {'-'*10}  {'-'*12}")
    for lbl, cnt in label_counts.items():
        cls8 = LABEL_TO_CLASS.get(str(lbl), "UNKNOWN")
        print(f"  {str(lbl):<40} {cnt:>10,}  {cls8:<12}")

    # ---- 8-class summary ----
    df["_class8"] = df[LABEL_COL].map(LABEL_TO_CLASS)
    unmapped = df["_class8"].isnull().sum()
    if unmapped > 0:
        log.warning("%d rows have labels not in LABEL_TO_CLASS mapping!", unmapped)
        print(f"\n  WARNING: {unmapped} rows could not be mapped to an 8-class!")
    class8_counts = df["_class8"].value_counts()
    print(f"\n8-class distribution:")
    for cls, cnt in class8_counts.items():
        print(f"  {str(cls):<15} {cnt:>10,}")
    df.drop(columns=["_class8"], inplace=True)

    # ---- save bar chart ----
    _plot_class_distribution(label_counts, class8_counts)

    summary = {
        "n_rows":        n_rows,
        "n_cols":        n_cols,
        "total_missing": int(total_missing),
        "total_inf":     total_inf,
        "n_duplicates":  n_dupes,
        "n_labels":      len(label_counts),
        "label_counts":  label_counts.to_dict(),
        "class8_counts": class8_counts.to_dict(),
    }
    print("\n" + "=" * 60 + "\n")
    return summary


def _plot_class_distribution(
    label_counts: pd.Series,
    class8_counts: pd.Series,
) -> None:
    """Save a two-panel bar chart: fine-grained labels + 8 classes."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("CICIoT2023 - Class Distribution (sampled subset)", fontsize=14)

    # Left: 34 fine-grained labels
    ax = axes[0]
    sns.barplot(x=label_counts.values, y=label_counts.index, ax=ax,
                palette="tab20", orient="h",
                hue=label_counts.index, legend=False)
    ax.set_title("34 Fine-grained Labels")
    ax.set_xlabel("Row count")
    ax.set_ylabel("")
    for bar, val in zip(ax.patches, label_counts.values):
        ax.text(bar.get_width() + max(label_counts.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=7)

    # Right: 8 super-classes
    ax2 = axes[1]
    palette = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db",
               "#9b59b6", "#f39c12", "#1abc9c", "#e91e63"]
    sns.barplot(x=class8_counts.index, y=class8_counts.values, ax=ax2,
                palette=palette[:len(class8_counts)],
                hue=class8_counts.index, legend=False)
    ax2.set_title("8 Super-classes")
    ax2.set_xlabel("Attack class")
    ax2.set_ylabel("Row count")
    ax2.tick_params(axis='x', rotation=30)
    for bar in ax2.patches:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(class8_counts.values) * 0.01,
                 f"{int(bar.get_height()):,}", ha="center", fontsize=8)

    plt.tight_layout()
    out = REPORTS_DIR / "class_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Chart saved] --> {out}")
