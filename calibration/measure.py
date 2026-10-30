"""
calibration/measure.py
======================
Phase 12: Raw Probability Quality Measurement.
Measures the 8-class calibration quality of the Phase 3 B2 ensemble
(RF 0.60 + XGBoost 0.40) on the validation set and on the test set.

Computes:
  - ECE (15 equal-width bins, top-label confidence)
  - Multiclass Brier score
  - Negative Log-Likelihood (NLL)
  - Reliability table (bin confidence, bin accuracy, bin count)
  - 200-sample bootstrap 95% confidence intervals (seed 42) for all metrics

Saves results to calibration/results/raw_metrics.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TEST_PARQUET
from models.load_validation import load_validation_data
from predict import get_or_load_model, get_selected_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibration_measure")

RESULTS_DIR = PROJECT_ROOT / "calibration" / "results"
RAW_METRICS_PATH = RESULTS_DIR / "raw_metrics.json"


def compute_ece(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) with equal-width bins on top-label confidence.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    rel_table: List[Dict[str, Any]] = []

    for b in range(n_bins):
        lower = float(bin_edges[b])
        upper = float(bin_edges[b + 1])
        if b == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (count / n) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = 0.0

        rel_table.append({
            "bin_idx": b,
            "bin_lower": round(lower, 4),
            "bin_upper": round(upper, 4),
            "bin_count": count,
            "bin_confidence": round(bin_conf, 4),
            "bin_accuracy": round(bin_acc, 4),
        })

    return float(ece), rel_table


def compute_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes multiclass Brier score: mean squared difference across all classes.
    """
    n, k = probs.shape
    y_one_hot = np.zeros((n, k), dtype=np.float32)
    y_one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))


def compute_nll(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-15) -> float:
    """
    Computes negative log-likelihood (cross-entropy loss).
    """
    n = len(y_true)
    p_true = probs[np.arange(n), y_true]
    return float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))


def compute_metrics_with_bootstrap_ci(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_resamples: int = 200,
    seed: int = 42,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Computes point estimates and 200-sample bootstrap 95% CIs for ECE, Brier, NLL.
    """
    point_ece, rel_table = compute_ece(probs, y_true, n_bins=n_bins)
    point_brier = compute_brier(probs, y_true)
    point_nll = compute_nll(probs, y_true)
    accuracy = float(np.mean(np.argmax(probs, axis=1) == y_true))

    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_eces = []
    boot_briers = []
    boot_nlls = []

    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        p_samp = probs[idx]
        y_samp = y_true[idx]

        b_ece, _ = compute_ece(p_samp, y_samp, n_bins=n_bins)
        b_brier = compute_brier(p_samp, y_samp)
        b_nll = compute_nll(p_samp, y_samp)

        boot_eces.append(b_ece)
        boot_briers.append(b_brier)
        boot_nlls.append(b_nll)

    return {
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "ece": round(point_ece, 4),
        "ece_ci": [round(float(np.percentile(boot_eces, 2.5)), 4), round(float(np.percentile(boot_eces, 97.5)), 4)],
        "brier_score": round(point_brier, 4),
        "brier_score_ci": [round(float(np.percentile(boot_briers, 2.5)), 4), round(float(np.percentile(boot_briers, 97.5)), 4)],
        "nll": round(point_nll, 4),
        "nll_ci": [round(float(np.percentile(boot_nlls, 2.5)), 4), round(float(np.percentile(boot_nlls, 97.5)), 4)],
        "reliability_table": rel_table,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log.info("Loading Phase 3 B2 Ensemble model...")
    model = get_or_load_model(target="8class", model_type="b2")
    selected_features = get_selected_features()

    # 1. Validation Set
    log.info("Loading pure validation data (models/load_validation.py)...")
    X_val, y_val_8, _ = load_validation_data()
    y_val_8 = np.asarray(y_val_8, dtype=np.int64)
    log.info(f"Validation data: {X_val.shape[0]:,} rows across {X_val.shape[1]} features.")

    # 2. Test Set
    log.info(f"Loading untouched test data ({TEST_PARQUET.name})...")
    df_test = pd.read_parquet(TEST_PARQUET)
    X_test = df_test[selected_features]
    y_test_8 = df_test["label_8"].to_numpy(dtype=np.int64)
    log.info(f"Test data: {len(df_test):,} rows.")

    # 3. Get raw probabilities
    log.info("Computing 8-class probabilities for validation set with B2 ensemble...")
    t_v = time.time()
    val_probs_dict = model.predict_probabilities(X_val)
    P_val = val_probs_dict["weighted"].astype(np.float32)
    log.info(f"Validation probabilities computed in {time.time() - t_v:.2f} s. Shape: {P_val.shape}")

    log.info("Computing 8-class probabilities for test set with B2 ensemble...")
    t_t = time.time()
    test_probs_dict = model.predict_probabilities(X_test)
    P_test = test_probs_dict["weighted"].astype(np.float32)
    log.info(f"Test probabilities computed in {time.time() - t_t:.2f} s. Shape: {P_test.shape}")

    # Cache probability arrays for downstream steps
    np.save(RESULTS_DIR / "val_probs_raw.npy", P_val)
    np.save(RESULTS_DIR / "test_probs_raw.npy", P_test)
    np.save(RESULTS_DIR / "y_val_8.npy", y_val_8)
    np.save(RESULTS_DIR / "y_test_8.npy", y_test_8)

    # 4. Measure metrics with bootstrap CIs
    log.info("Computing raw probability quality metrics for validation set (200 bootstrap resamples)...")
    val_metrics = compute_metrics_with_bootstrap_ci(P_val, y_val_8, n_resamples=200, seed=42)

    log.info("Computing raw probability quality metrics for test set (200 bootstrap resamples)...")
    test_metrics = compute_metrics_with_bootstrap_ci(P_test, y_test_8, n_resamples=200, seed=42)

    output_data = {
        "model": "Phase 3 B2 Ensemble (rf_8class_b2 0.60 + xgb_8class_b2 0.40)",
        "num_classes": 8,
        "n_bins": 15,
        "bootstrap_resamples": 200,
        "random_seed": 42,
        "validation": val_metrics,
        "test": test_metrics,
    }

    with open(RAW_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    log.info(f"Raw metrics saved to {RAW_METRICS_PATH.resolve()}")

    print("\n" + "=" * 70)
    print("       RAW PROBABILITY QUALITY (PHASE 3 B2 ENSEMBLE)")
    print("=" * 70)
    for split_name, m in [("Validation Set", val_metrics), ("Test Set", test_metrics)]:
        print(f"\n--- {split_name} (N = {m['n_samples']:,}) ---")
        print(f"  Accuracy:      {m['accuracy'] * 100:.2f}%")
        print(f"  ECE (15 bins): {m['ece']:.4f}  [95% CI: {m['ece_ci'][0]:.4f}, {m['ece_ci'][1]:.4f}]")
        print(f"  Brier Score:   {m['brier_score']:.4f}  [95% CI: {m['brier_score_ci'][0]:.4f}, {m['brier_score_ci'][1]:.4f}]")
        print(f"  NLL:           {m['nll']:.4f}  [95% CI: {m['nll_ci'][0]:.4f}, {m['nll_ci'][1]:.4f}]")

    print("\nTotal execution time: " + f"{time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
