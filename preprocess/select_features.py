"""
preprocess/select_features.py
==============================
Phase 2 - Step 6: Feature Selection via Recursive Feature Elimination
Uses a Random Forest estimator inside RFE to rank and select the top 19
features from the 46 CICIoT2023 columns.

WARNING  NOTE TO STUDENT (flag from the pipeline):
---------------------------------------------
Your slides mention "115 --> 19 features".  That figure comes from an
OLDER dataset (CIC-IDS-2017 / NSL-KDD combined) that had 115 features
after one-hot encoding.  CICIoT2023 has only 46 numeric features.
The correct statement for this project is:
    "RFE reduces 46 features --> top 19 features."
Please update your presentation accordingly.

Speed note
----------
RFE with a full RF on 1 M+ rows can take 30-60 min.  We use
RFE_SUBSAMPLE rows (default 50 k) to estimate feature importances
quickly.  The same 19 features are then used for the full dataset.
This is a standard practice in the literature.

Outputs
-------
  artifacts/selected_features.json     - ordered list of 19 feature names
  artifacts/rfe_feature_ranking.csv    - all 46 features with their rank
  reports/rfe_ranking.png              - horizontal bar chart
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    N_FEATURES_TO_SELECT,
    RFE_ESTIMATOR_TREES,
    RFE_SUBSAMPLE,
    RANDOM_SEED,
    SELECTED_FEATURES_PATH,
    RFE_RANKING_PATH,
    REPORTS_DIR,
)

log = logging.getLogger(__name__)


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = N_FEATURES_TO_SELECT,
    subsample: int = RFE_SUBSAMPLE,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], pd.DataFrame]:
    """
    Run RFE with RandomForest on a subsample of the training data.

    Parameters
    ----------
    X_train   : scaled feature DataFrame (all 46 cols)
    y_train   : integer label Series
    n_features: number of features to keep
    subsample : rows to use for RFE (for speed)
    seed      : random seed

    Returns
    -------
    (selected_feature_names, ranking_df)
    """
    total = len(X_train)
    n_sub = min(subsample, total)
    print(f"\n[RFE] Starting feature selection.")
    print(f"[RFE] Input: {total:,} rows x {X_train.shape[1]} features")
    print(f"[RFE] Using subsample of {n_sub:,} rows for speed.")
    print(f"[RFE] Target: {n_features} features   RF trees: {RFE_ESTIMATOR_TREES}")
    print("[RFE] [wait]  This may take 2-5 minutes ...")

    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n_sub, replace=False)
    X_sub = X_train.iloc[idx]
    y_sub = y_train.iloc[idx]

    estimator = RandomForestClassifier(
        n_estimators=RFE_ESTIMATOR_TREES,
        max_depth=12,
        n_jobs=-1,
        random_state=seed,
    )

    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=1,              # eliminate one feature per round
        verbose=0,
    )
    rfe.fit(X_sub, y_sub)

    cols    = X_train.columns.tolist()
    ranking = rfe.ranking_          # 1 = selected, higher = eliminated earlier
    support = rfe.support_          # True = in final set

    # In scikit-learn RFE, rfe.estimator_.feature_importances_ corresponds to
    # the selected features in their ORIGINAL column order (cols[support]).
    selected_in_original_order = [c for c, s in zip(cols, support) if s]
    rf_imp = rfe.estimator_.feature_importances_
    importance_map = dict(zip(selected_in_original_order, rf_imp))

    ranking_df = pd.DataFrame({
        "feature": cols,
        "rfe_rank": ranking,
        "selected": support,
        "rf_importance": [importance_map.get(c, float("nan")) for c in cols],
    }).sort_values("rfe_rank")

    selected = ranking_df[ranking_df["selected"]]["feature"].tolist()

    # Save ranking with importance column and selected features list
    ranking_df.to_csv(RFE_RANKING_PATH, index=False)
    with open(SELECTED_FEATURES_PATH, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\n[RFE] Done.")
    print(f"[RFE] Ranking saved  --> {RFE_RANKING_PATH.name}")
    print(f"[RFE] Features saved --> {SELECTED_FEATURES_PATH.name}")
    print(f"\n[RFE] Top {n_features} selected features (ranked):")
    for rank_i, feat in enumerate(selected, 1):
        fi = importance_map.get(feat, 0.0)
        print(f"  {rank_i:>2}. {feat:<35} (RF importance ~= {fi:.4f})")

    _plot_rfe_ranking(ranking_df, n_features)

    print(f"\n  SLIDE CORRECTION FLAG:")
    print(f"     Your slides say '115 -> 19 features'.")
    print(f"     CICIoT2023 has 46 numeric features.")
    print(f"     Please update your presentation to say '46 -> 19 features'.")

    return selected, ranking_df


def _plot_rfe_ranking(ranking_df: pd.DataFrame, n_features: int) -> None:
    """Save a horizontal bar chart of feature rankings."""
    fig, ax = plt.subplots(figsize=(10, 12))

    colors = [
        "#2ecc71" if sel else "#95a5a6"
        for sel in ranking_df["selected"]
    ]
    ax.barh(ranking_df["feature"], ranking_df["rfe_rank"],
            color=colors, edgecolor="white", height=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("RFE Rank  (1 = most important, kept)")
    ax.set_title(
        f"RFE Feature Ranking  -  CICIoT2023  |  "
        f"Green = top {n_features} selected",
        fontsize=12,
    )
    ax.axvline(x=1.5, color="#e74c3c", linestyle="--", linewidth=1.2,
               label="Selection boundary")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = REPORTS_DIR / "rfe_ranking.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RFE] [Chart saved] --> {out}")
