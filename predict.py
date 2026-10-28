"""
predict.py
==========
Lightweight inference module for Phase 3 Ensemble Classifier.
Takes a DataFrame of the 19 scaled flow features, executes the trained ensemble,
and returns predictions, class labels, confidence scores, and severity ratings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import (
    MODELS_DIR,
    SELECTED_FEATURES_PATH,
    ENCODER_8_PATH,
    ENCODER_34_PATH,
    get_flow_severity,
)
from models.ensemble import EnsembleClassifier

_LOADED_MODELS: dict[str, EnsembleClassifier] = {}
_FEATURE_LIST: Optional[list[str]] = None


def get_selected_features() -> list[str]:
    """Returns the ordered list of 19 selected features."""
    global _FEATURE_LIST
    if _FEATURE_LIST is None:
        with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
            _FEATURE_LIST = json.load(f)
    return _FEATURE_LIST


def get_or_load_model(target: str = "8class") -> EnsembleClassifier:
    """Loads and caches the trained ensemble model."""
    global _LOADED_MODELS
    if target not in _LOADED_MODELS:
        _LOADED_MODELS[target] = EnsembleClassifier.load(MODELS_DIR, target_name=target)
    return _LOADED_MODELS[target]


def predict_flow(
    df_19_scaled: pd.DataFrame,
    target: str = "8class",
    mode: str = "weighted",
) -> pd.DataFrame:
    """
    Run ensemble inference on scaled network flows.

    Parameters
    ----------
    df_19_scaled : pd.DataFrame containing the 19 selected scaled features
    target       : '8class' or '34class'
    mode         : 'weighted' or 'soft'

    Returns
    -------
    pd.DataFrame with columns:
      - predicted_class_idx (int)
      - predicted_class (str)
      - confidence (float)
      - severity (str: Low, Medium, High, Critical)
    """
    features = get_selected_features()
    missing = [c for c in features if c not in df_19_scaled.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required features: {missing}")

    X_in = df_19_scaled[features]
    model = get_or_load_model(target)

    preds, conf = model.predict(X_in, mode=mode)
    class_names = model.class_names

    labels = [class_names[idx] for idx in preds]
    severities = [get_flow_severity(lbl, c) for lbl, c in zip(labels, conf)]

    results = pd.DataFrame({
        "predicted_class_idx": preds,
        "predicted_class": labels,
        "confidence": np.round(conf, 4),
        "severity": severities,
    }, index=df_19_scaled.index)

    return results


if __name__ == "__main__":
    import argparse
    from config import TEST_PARQUET

    parser = argparse.ArgumentParser(description="Run lightweight ensemble inference")
    parser.add_argument("--target", choices=["8class", "34class"], default="8class")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples to infer")
    args = parser.parse_args()

    print(f"[PREDICT] Loading test parquet: {TEST_PARQUET} ...")
    df_test = pd.read_parquet(TEST_PARQUET)
    sample_df = df_test.head(args.samples)

    features = get_selected_features()
    preds_df = predict_flow(sample_df[features], target=args.target)

    print("\nInference Results:")
    for idx, row in preds_df.iterrows():
        actual_label = sample_df.loc[idx, f"label_{args.target[:len(args.target)-5]}"] if f"label_{args.target[:len(args.target)-5]}" in sample_df else "N/A"
        print(f"  Flow #{idx:>3} | Predicted: {row['predicted_class']:<14} (conf: {row['confidence']:.4f}) | Severity: {row['severity']:<8}")
