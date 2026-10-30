"""
xai/shap_explain.py
===================
Phase 4 - SHAP Explainability Module

Provides TreeExplainer SHAP values for the tree components of the B2 Ensemble.
Notice on Ensemble Explainability:
  - The B2 Ensemble is a weighted soft voting mixture of:
      Random Forest (weight = 0.60)
      XGBoost       (weight = 0.40)
      PyTorch DNN   (weight = 0.00)
  - Because TreeExplainer computes exact Shapley values in tree log-odds margin space,
    a heterogeneous weighted probability mixture cannot be explained as a single tree.
  - This module explains XGBoost (weight 0.40) and Random Forest (weight 0.60) directly,
    providing global feature importances, per-class top features, and local sample attributions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENCODER_8_PATH, SELECTED_FEATURES_PATH
from orchestrator.llm_orchestrator import FeatureUnscaler
from predict import get_or_load_model, get_selected_features


class SHAPExplainer:
    """
    Manages SHAP explanations for tree-based intrusion detection models.
    Explains XGBoost and Random Forest from the B2 ensemble.
    """

    def __init__(self, target: str = "8class", model_type: str = "b2"):
        self.target = target
        self.model_type = model_type
        self.ensemble = get_or_load_model(target=target, model_type=model_type)
        self.selected_features = get_selected_features()
        self.unscaler = FeatureUnscaler()

        # Class names
        le_8 = joblib.load(ENCODER_8_PATH)
        self.class_names: List[str] = list(le_8.classes_)

        # Initialize TreeExplainer for XGBoost (fast and primary)
        self.xgb_explainer = shap.TreeExplainer(self.ensemble.xgb_model)
        self._rf_explainer: Optional[shap.TreeExplainer] = None

    @property
    def rf_explainer(self) -> shap.TreeExplainer:
        """Lazily initialize TreeExplainer for Random Forest."""
        if self._rf_explainer is None:
            self._rf_explainer = shap.TreeExplainer(self.ensemble.rf_model)
        return self._rf_explainer

    def explain_xgb(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes SHAP values using XGBoost TreeExplainer.
        Returns:
            np.ndarray of shape (n_samples, n_features, n_classes)
        """
        vals = self.xgb_explainer.shap_values(X[self.selected_features])
        if isinstance(vals, list):
            # scikit-learn tree shap returns list of (n_samples, n_features)
            vals = np.stack(vals, axis=-1)
        return vals

    def explain_rf(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes SHAP values using Random Forest TreeExplainer.
        Returns:
            np.ndarray of shape (n_samples, n_features, n_classes)
        """
        vals = self.rf_explainer.shap_values(X[self.selected_features])
        if isinstance(vals, list):
            vals = np.stack(vals, axis=-1)
        return vals

    def get_global_importance(self, shap_values: np.ndarray) -> pd.Series:
        """
        Computes mean absolute SHAP value across all samples and classes.
        Returns:
            pd.Series sorted descending by importance.
        """
        # shap_values shape: (N, n_features, n_classes)
        mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))
        return pd.Series(mean_abs, index=self.selected_features).sort_values(ascending=False)

    def get_per_class_importance(self, shap_values: np.ndarray) -> Dict[str, pd.Series]:
        """
        Computes mean absolute SHAP value per attack class.
        Returns:
            Dict mapping class_name -> pd.Series sorted descending.
        """
        res = {}
        for c_idx, c_name in enumerate(self.class_names):
            class_mean_abs = np.mean(np.abs(shap_values[:, :, c_idx]), axis=0)
            res[c_name] = pd.Series(class_mean_abs, index=self.selected_features).sort_values(ascending=False)
        return res

    def get_local_explanation(
        self,
        shap_values_single: np.ndarray,
        unscaled_features_dict: Dict[str, float],
        target_class_idx: int,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extracts top-K features driving the prediction for a single flow sample.

        Args:
            shap_values_single: array of shape (n_features, n_classes)
            unscaled_features_dict: dict of feature_name -> raw physical value
            target_class_idx: integer index of the predicted class
            top_k: number of top features to return

        Returns:
            List of dicts: [
              {
                "feature": str,
                "shap_value": float,
                "direction": "increases_risk" | "decreases_risk",
                "unscaled_value": float
              }, ...
            ]
        """
        class_shap = shap_values_single[:, target_class_idx]
        abs_order = np.argsort(np.abs(class_shap))[::-1]

        top_features = []
        for idx in abs_order[:top_k]:
            feat_name = self.selected_features[idx]
            val = float(class_shap[idx])
            top_features.append({
                "feature": feat_name,
                "shap_value": round(val, 4),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
                "unscaled_value": round(float(unscaled_features_dict.get(feat_name, 0.0)), 3),
            })
        return top_features

    def check_stability(
        self,
        X_sample: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Checks stability of feature importances between XGBoost and Random Forest.
        Computes Spearman rank correlation and top-5 overlap.
        """
        shap_xgb = self.explain_xgb(X_sample)
        shap_rf = self.explain_rf(X_sample)

        imp_xgb = self.get_global_importance(shap_xgb)
        imp_rf = self.get_global_importance(shap_rf)

        # Align ranks
        ranks_xgb = imp_xgb.rank(ascending=False)
        ranks_rf = imp_rf.reindex(imp_xgb.index).rank(ascending=False)

        rho, pval = spearmanr(ranks_xgb, ranks_rf)

        top5_xgb = set(imp_xgb.index[:5])
        top5_rf = set(imp_rf.index[:5])
        overlap_top5 = list(top5_xgb.intersection(top5_rf))

        return {
            "spearman_rho": round(float(rho), 4),
            "spearman_pval": round(float(pval), 6),
            "top5_xgb": list(imp_xgb.index[:5]),
            "top5_rf": list(imp_rf.index[:5]),
            "top5_overlap": overlap_top5,
            "top5_overlap_count": len(overlap_top5),
            "top5_overlap_pct": round(len(overlap_top5) / 5.0 * 100.0, 1),
            "importance_xgb": imp_xgb.to_dict(),
            "importance_rf": imp_rf.to_dict(),
        }
