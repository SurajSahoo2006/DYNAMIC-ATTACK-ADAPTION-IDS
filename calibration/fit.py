"""
calibration/fit.py
==================
Phase 12: Fit Calibrators on Validation Set Only.

Splits the validation set 50/50, stratified by class, seed 42:
  - "calib" half: used to fit calibrators
  - "select" half: used to evaluate and compare calibrators

Fits on "calib" half only:
  (a) Temperature scaling: one scalar T on log-probabilities, optimized by minimizing NLL
  (b) One-vs-rest isotonic regression per class with renormalization

Evaluates both on the "select" half (ECE, Brier, NLL) and selects the better
method by lower select-half NLL.
Saves fitted calibrators to calibration/models/ and the comparison to
calibration/results/fit.json.

Then refits the chosen method on the FULL validation set for final use and
records that in fit.json. NEVER fits on test rows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration.measure import compute_ece, compute_brier, compute_nll, compute_metrics_with_bootstrap_ci

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibration_fit")

RESULTS_DIR = PROJECT_ROOT / "calibration" / "results"
MODELS_DIR = PROJECT_ROOT / "calibration" / "models"
FIT_JSON_PATH = RESULTS_DIR / "fit.json"


class TemperatureScaler:
    """
    Temperature scaling calibrator: single scalar T optimizing NLL over log-probabilities.
    """

    def __init__(self, T: float = 1.0):
        self.T = float(T)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        z = np.log(np.clip(probs, 1e-15, 1.0))
        n = len(y)

        def nll_loss(t_val: float) -> float:
            t = max(1e-4, float(t_val))
            sz = z / t
            sz -= np.max(sz, axis=1, keepdims=True)
            ez = np.exp(sz)
            pz = ez / np.sum(ez, axis=1, keepdims=True)
            p_true = pz[np.arange(n), y]
            return float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0))))

        res = minimize_scalar(nll_loss, bounds=(0.01, 10.0), method="bounded")
        self.T = float(res.x)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        z = np.log(np.clip(probs, 1e-15, 1.0)) / max(1e-4, self.T)
        z -= np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return (ez / np.sum(ez, axis=1, keepdims=True)).astype(np.float32)


class IsotonicCalibrator:
    """
    One-vs-Rest Isotonic Regression calibrator with row renormalization.
    """

    def __init__(self, num_classes: int = 8):
        self.num_classes = num_classes
        self.regressors: List[IsotonicRegression] = []

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.regressors = []
        for k in range(self.num_classes):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs[:, k], (y == k).astype(float))
            self.regressors.append(iso)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        n = len(probs)
        out = np.zeros((n, self.num_classes), dtype=np.float32)
        for k in range(self.num_classes):
            out[:, k] = self.regressors[k].predict(probs[:, k])
        s = out.sum(axis=1, keepdims=True)
        zero_mask = (s.squeeze() <= 1e-9)
        out[~zero_mask] /= s[~zero_mask]
        if np.any(zero_mask):
            out[zero_mask] = probs[zero_mask]
        return out


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Load validation data & test data
    val_probs_path = RESULTS_DIR / "val_probs_raw.npy"
    y_val_path = RESULTS_DIR / "y_val_8.npy"
    test_probs_path = RESULTS_DIR / "test_probs_raw.npy"
    y_test_path = RESULTS_DIR / "y_test_8.npy"

    if not (val_probs_path.exists() and y_val_path.exists()):
        log.info("Raw validation probabilities not found. Running calibration/measure.py...")
        from calibration.measure import main as run_measure
        run_measure()

    P_val = np.load(val_probs_path)
    y_val = np.load(y_val_path)
    P_test = np.load(test_probs_path)
    y_test = np.load(y_test_path)
    num_classes = P_val.shape[1]

    log.info(f"Loaded validation set: N = {len(y_val):,}, Classes = {num_classes}")
    log.info(f"Loaded test set: N = {len(y_test):,}")

    # 2. Stratified 50/50 split of validation set
    idx_calib, idx_select = train_test_split(
        np.arange(len(y_val)),
        test_size=0.5,
        stratify=y_val,
        random_state=42,
    )

    P_calib, y_calib = P_val[idx_calib], y_val[idx_calib]
    P_select, y_select = P_val[idx_select], y_val[idx_select]
    log.info(f"Split validation set: Calib half = {len(y_calib):,} rows, Select half = {len(y_select):,} rows")

    # 3. Fit on calib half only
    log.info("[1/3] Fitting Temperature Scaling on calib half...")
    temp_calib = TemperatureScaler()
    temp_calib.fit(P_calib, y_calib)
    log.info(f"  Fitted Temperature T = {temp_calib.T:.4f}")

    log.info("[2/3] Fitting One-vs-Rest Isotonic Regression on calib half...")
    iso_calib = IsotonicCalibrator(num_classes=num_classes)
    iso_calib.fit(P_calib, y_calib)
    log.info(f"  Fitted {num_classes} Isotonic Regressors.")

    # 4. Evaluate both on select half
    log.info("[3/3] Evaluating on select half...")
    P_raw_select = P_select
    P_temp_select = temp_calib.predict_proba(P_select)
    P_iso_select = iso_calib.predict_proba(P_select)

    ece_raw, _ = compute_ece(P_raw_select, y_select, n_bins=15)
    brier_raw = compute_brier(P_raw_select, y_select)
    nll_raw = compute_nll(P_raw_select, y_select)

    ece_temp, _ = compute_ece(P_temp_select, y_select, n_bins=15)
    brier_temp = compute_brier(P_temp_select, y_select)
    nll_temp = compute_nll(P_temp_select, y_select)

    ece_iso, _ = compute_ece(P_iso_select, y_select, n_bins=15)
    brier_iso = compute_brier(P_iso_select, y_select)
    nll_iso = compute_nll(P_iso_select, y_select)

    select_eval = {
        "raw": {"ece": round(ece_raw, 4), "brier_score": round(brier_raw, 4), "nll": round(nll_raw, 4)},
        "temperature_scaling": {"T": round(temp_calib.T, 4), "ece": round(ece_temp, 4), "brier_score": round(brier_temp, 4), "nll": round(nll_temp, 4)},
        "isotonic_regression": {"ece": round(ece_iso, 4), "brier_score": round(brier_iso, 4), "nll": round(nll_iso, 4)},
    }

    # Selection by select-half NLL
    if nll_temp < nll_iso:
        chosen_method = "temperature_scaling"
        reason = f"Temperature Scaling achieved lower select-half NLL ({nll_temp:.4f} vs {nll_iso:.4f})"
    else:
        chosen_method = "isotonic_regression"
        reason = f"Isotonic Regression achieved lower select-half NLL ({nll_iso:.4f} vs {nll_temp:.4f})"

    log.info(f"Chosen method by select-half NLL: {chosen_method} ({reason})")

    # Save models fitted on calib half
    joblib.dump(temp_calib, MODELS_DIR / "temperature_scaler_calib_half.joblib")
    joblib.dump(iso_calib, MODELS_DIR / "isotonic_calibrator_calib_half.joblib")

    # 5. Refit chosen method on the FULL validation set (NEVER on test rows)
    log.info(f"Refitting chosen method '{chosen_method}' on FULL validation set (N = {len(y_val):,})...")
    if chosen_method == "temperature_scaling":
        final_calibrator = TemperatureScaler()
        final_calibrator.fit(P_val, y_val)
        final_params = {"T": round(final_calibrator.T, 4)}
        log.info(f"  Final Full Validation Temperature T = {final_calibrator.T:.4f}")
    else:
        final_calibrator = IsotonicCalibrator(num_classes=num_classes)
        final_calibrator.fit(P_val, y_val)
        final_params = {"num_classes": num_classes}
        log.info(f"  Final Full Validation Isotonic Regressors refitted.")

    joblib.dump(final_calibrator, MODELS_DIR / "chosen_calibrator.joblib")
    log.info(f"Saved chosen calibrator to {MODELS_DIR / 'chosen_calibrator.joblib'}")

    # Evaluate final calibrator on full validation set and test set
    P_val_calibrated = final_calibrator.predict_proba(P_val)
    P_test_calibrated = final_calibrator.predict_proba(P_test)

    # Cache calibrated probabilities
    np.save(RESULTS_DIR / "val_probs_calibrated.npy", P_val_calibrated)
    np.save(RESULTS_DIR / "test_probs_calibrated.npy", P_test_calibrated)

    val_calib_metrics = compute_metrics_with_bootstrap_ci(P_val_calibrated, y_val, n_resamples=200, seed=42)
    test_calib_metrics = compute_metrics_with_bootstrap_ci(P_test_calibrated, y_test, n_resamples=200, seed=42)

    fit_results = {
        "split_protocol": {
            "validation_rows": len(y_val),
            "calib_half_rows": len(y_calib),
            "select_half_rows": len(y_select),
            "seed": 42,
            "stratified": True,
        },
        "select_half_comparison": select_eval,
        "chosen_method": chosen_method,
        "selection_reason": reason,
        "final_model_parameters": final_params,
        "full_validation_calibrated_metrics": val_calib_metrics,
        "test_calibrated_metrics": test_calib_metrics,
    }

    with open(FIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(fit_results, f, indent=2)

    log.info(f"Fit results saved to {FIT_JSON_PATH.resolve()}")

    print("\n" + "=" * 70)
    print("           CALIBRATOR SELECTION AND EVALUATION")
    print("=" * 70)
    print(f"\nSelect Half Comparison (N = {len(y_select):,}):")
    print(f"{'Method':<25} {'ECE (15 bins)':>15} {'Brier Score':>15} {'NLL':>12}")
    print("-" * 70)
    for mname in ["raw", "temperature_scaling", "isotonic_regression"]:
        stats = select_eval[mname]
        print(f"{mname:<25} {stats['ece']:>15.4f} {stats['brier_score']:>15.4f} {stats['nll']:>12.4f}")
    print("-" * 70)
    print(f"Chosen Method: {chosen_method} ({reason})")

    print(f"\nFinal Calibrated Performance (Untouched Test Set, N = {len(y_test):,}):")
    print(f"  Accuracy:      {test_calib_metrics['accuracy'] * 100:.2f}%")
    print(f"  ECE (15 bins): {test_calib_metrics['ece']:.4f}  [95% CI: {test_calib_metrics['ece_ci'][0]:.4f}, {test_calib_metrics['ece_ci'][1]:.4f}]")
    print(f"  Brier Score:   {test_calib_metrics['brier_score']:.4f}  [95% CI: {test_calib_metrics['brier_score_ci'][0]:.4f}, {test_calib_metrics['brier_score_ci'][1]:.4f}]")
    print(f"  NLL:           {test_calib_metrics['nll']:.4f}  [95% CI: {test_calib_metrics['nll_ci'][0]:.4f}, {test_calib_metrics['nll_ci'][1]:.4f}]")
    print(f"\nExecution time: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
