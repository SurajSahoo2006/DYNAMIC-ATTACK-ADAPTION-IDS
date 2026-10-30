"""
calibration/openset_calibrated.py
=================================
Phase 12: Open-Set Evaluation at Matched False-Positive Budget (Raw vs Calibrated).

Compares RAW vs CALIBRATED confidence-only rule ("low confidence = UNKNOWN"),
each with its cutoff chosen on validation so that at most 10% of validation
Benign flows are flagged (identical matched budget for both).

Includes:
  4a. Known-class control: flagged rate per class on the test set for the full B2 ensemble.
  4b. Leave-one-class-out (7 attack classes), reusing Phase 10 protocol:
      - Fresh RF (100 trees, depth 15, balanced class weights, seed 42, 50k row cap).
      - Fits calibrator on validation rows that EXCLUDE the held-out class.
      - Records flagged, silently-Benign, and silently-other-attack counts with Wilson 95% CIs.
  4c. Paired bootstrap 95% CI (1,000 resamples, seed 42) on:
      (calibrated flagged rate - raw flagged rate) per held-out class.
  4d. Benign false-positive rate on the TEST set at each method's validation-chosen cutoff.

Supports --resume; writes calibration/results/openset_<class>.json after each held-out class.
Time limit: 6 min (360 s) per class; marks "truncated": true if exceeded.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER_8_PATH, SELECTED_FEATURES_PATH, TEST_PARQUET
from models.load_validation import load_validation_data
from predict import get_or_load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("openset_calibrated")

RESULTS_DIR = PROJECT_ROOT / "calibration" / "results"
SUMMARY_JSON_PATH = RESULTS_DIR / "openset_calibrated_summary.json"


def wilson_score_interval(x: int, n: int, confidence: float = 0.95) -> Dict[str, Union[int, float, str]]:
    """
    Computes 95% Wilson score confidence interval for binomial proportion x / n.
    """
    if n <= 0:
        return {
            "x": int(x),
            "n": int(n),
            "percentage": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "formatted": "0 of 0 (0.00% [0.00%, 0.00%])",
        }

    p = float(x) / float(n)
    z = 1.9599639845400542 if abs(confidence - 0.95) < 1e-4 else 1.96

    denominator = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denominator
    half_width = (z * math.sqrt((p * (1.0 - p) / n) + (z ** 2) / (4.0 * (n ** 2)))) / denominator

    lower = max(0.0, center - half_width) * 100.0
    upper = min(1.0, center + half_width) * 100.0
    pct = p * 100.0

    return {
        "x": int(x),
        "n": int(n),
        "percentage": round(pct, 2),
        "ci_lower": round(lower, 2),
        "ci_upper": round(upper, 2),
        "formatted": f"{x} of {n} ({pct:.2f}%) [{lower:.2f}%, {upper:.2f}%]",
    }


def paired_bootstrap_diff(
    flagged_cal: np.ndarray,
    flagged_raw: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Paired bootstrap 95% CI on (calibrated flagged rate - raw flagged rate) in percentage points.
    """
    rng = np.random.RandomState(seed)
    n = len(flagged_cal)
    point_diff = float(np.mean(flagged_cal) - np.mean(flagged_raw)) * 100.0

    boot_diffs = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        diff = (np.mean(flagged_cal[idx]) - np.mean(flagged_raw[idx])) * 100.0
        boot_diffs.append(diff)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return {
        "point_diff": round(point_diff, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "formatted": f"{point_diff:+.2f}% [{ci_lower:+.2f}%, {ci_upper:+.2f}%]",
        "significant_gain": ci_lower > 0,
        "significant_loss": ci_upper < 0,
        "about_the_same": ci_lower <= 0 and ci_upper >= 0,
    }


class SubsetClassIsotonicCalibrator:
    """
    One-vs-Rest Isotonic Regression calibrator for models with a subset of classes.
    """

    def __init__(self, known_classes: np.ndarray):
        self.known_classes = np.asarray(known_classes, dtype=np.int64)
        self.regressors: List[IsotonicRegression] = []

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "SubsetClassIsotonicCalibrator":
        self.regressors = []
        for k, cls_idx in enumerate(self.known_classes):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs[:, k], (y == cls_idx).astype(float))
            self.regressors.append(iso)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        n = len(probs)
        out = np.zeros_like(probs, dtype=np.float32)
        for k in range(len(self.known_classes)):
            out[:, k] = self.regressors[k].predict(probs[:, k])
        s = out.sum(axis=1, keepdims=True)
        zero_mask = (s.squeeze() <= 1e-9)
        out[~zero_mask] /= s[~zero_mask]
        if np.any(zero_mask):
            out[zero_mask] = probs[zero_mask]
        return out


def main():
    parser = argparse.ArgumentParser(description="Phase 12 Calibrated Open-Set Evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume completed held-out class evaluations")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_pipeline = time.time()

    # 1. Load Encoders and Feature List
    le_8 = joblib.load(ENCODER_8_PATH)
    class_names = list(le_8.classes_)
    benign_idx = int(np.where(le_8.classes_ == "Benign")[0][0])
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features = json.load(f)

    log.info(f"Loaded class mapping: {list(enumerate(class_names))}")
    log.info(f"Benign class index: {benign_idx}")

    # 2. Load Validation Data
    log.info("Loading validation data...")
    X_val, y_val_8, _ = load_validation_data()
    y_val_8 = np.asarray(y_val_8, dtype=np.int64)
    X_val_np = X_val[selected_features].to_numpy(dtype=np.float32)

    # 3. Load Test Data
    log.info("Loading test data...")
    df_test = pd.read_parquet(TEST_PARQUET)
    y_test_8 = df_test["label_8"].to_numpy(dtype=np.int64)
    X_test_np = df_test[selected_features].to_numpy(dtype=np.float32)

    # 4. Load Training Data for Leave-One-Class-Out
    train_path = PROJECT_ROOT / "data" / "processed" / "train_unsmoted_19.parquet"
    if not train_path.exists():
        train_path = PROJECT_ROOT / "data" / "processed" / "train_phase2.parquet"
    log.info(f"Loading training data from {train_path.name}...")
    df_train = pd.read_parquet(train_path)
    tr_label_col = "label_8" if "label_8" in df_train.columns else "label"

    # =========================================================================
    # STEP 4A: KNOWN-CLASS CONTROL (FULL B2 ENSEMBLE)
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 4A: KNOWN-CLASS CONTROL (FULL B2 ENSEMBLE)")
    log.info("=" * 70)

    # Load pre-computed validation and test probabilities from Step 1 & 2
    val_probs_raw = np.load(RESULTS_DIR / "val_probs_raw.npy")
    val_probs_cal = np.load(RESULTS_DIR / "val_probs_calibrated.npy")
    test_probs_raw = np.load(RESULTS_DIR / "test_probs_raw.npy")
    test_probs_cal = np.load(RESULTS_DIR / "test_probs_calibrated.npy")

    # Matched budget on validation Benign flows (at most 10% false positive rate)
    val_benign_mask = (y_val_8 == benign_idx)
    max_p_val_benign_raw = np.max(val_probs_raw[val_benign_mask], axis=1)
    max_p_val_benign_cal = np.max(val_probs_cal[val_benign_mask], axis=1)

    tau_raw_control = float(np.percentile(max_p_val_benign_raw, 10.0))
    tau_cal_control = float(np.percentile(max_p_val_benign_cal, 10.0))

    val_fpr_raw_control = float(np.mean(max_p_val_benign_raw < tau_raw_control))
    val_fpr_cal_control = float(np.mean(max_p_val_benign_cal < tau_cal_control))

    log.info(f"Validation Matched Budget Cutoffs: Raw = {tau_raw_control:.4f} (Val Benign FPR: {val_fpr_raw_control*100:.2f}%), Calibrated = {tau_cal_control:.4f} (Val Benign FPR: {val_fpr_cal_control*100:.2f}%)")

    known_control_results: Dict[str, Any] = {}
    test_max_raw = np.max(test_probs_raw, axis=1)
    test_max_cal = np.max(test_probs_cal, axis=1)

    for idx, cname in enumerate(class_names):
        mask_c = (y_test_8 == idx)
        n_c = int(np.sum(mask_c))

        flagged_raw_c = (test_max_raw[mask_c] < tau_raw_control)
        flagged_cal_c = (test_max_cal[mask_c] < tau_cal_control)

        diff_stat = paired_bootstrap_diff(flagged_cal_c, flagged_raw_c, n_resamples=1000, seed=42)

        known_control_results[cname] = {
            "n_test": n_c,
            "raw_flagged": wilson_score_interval(int(np.sum(flagged_raw_c)), n_c),
            "calibrated_flagged": wilson_score_interval(int(np.sum(flagged_cal_c)), n_c),
            "paired_difference": diff_stat,
        }

    # =========================================================================
    # STEP 4B: LEAVE-ONE-CLASS-OUT EVALUATION (7 ATTACK CLASSES)
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 4B: LEAVE-ONE-CLASS-OUT EVALUATION (7 ATTACK CLASSES)")
    log.info("=" * 70)

    attack_classes = [c for c in class_names if c != "Benign"]
    leave_one_out_results: Dict[str, Any] = {}
    headline_counts = {
        "significantly_more_flagged": 0,
        "about_the_same": 0,
        "significantly_fewer_flagged": 0,
    }

    test_benign_mask = (y_test_8 == benign_idx)
    X_test_benign = X_test_np[test_benign_mask]
    n_test_benign = len(X_test_benign)

    for cname in attack_classes:
        class_json_path = RESULTS_DIR / f"openset_{cname}.json"
        cidx = int(np.where(le_8.classes_ == cname)[0][0])

        if args.resume and class_json_path.exists():
            try:
                with open(class_json_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if not cached_data.get("truncated", False):
                    log.info(f"Loaded cached results for held-out class '{cname}' (--resume).")
                    leave_one_out_results[cname] = cached_data
                    diff_info = cached_data["heldout_test_evaluation"]["paired_difference"]
                    if diff_info["significant_gain"]:
                        headline_counts["significantly_more_flagged"] += 1
                    elif diff_info["significant_loss"]:
                        headline_counts["significantly_fewer_flagged"] += 1
                    else:
                        headline_counts["about_the_same"] += 1
                    continue
            except Exception as e:
                log.warning(f"Error loading cached {class_json_path.name}: {e}. Rerunning...")

        log.info(f"\n--- Held-Out Attack Class: {cname} (Index = {cidx}) ---")
        t_start_class = time.time()

        # Filter training data to exclude class cidx
        df_sub = df_train[df_train[tr_label_col] != cidx].copy()
        if len(df_sub) > 50_000:
            df_sub, _ = train_test_split(
                df_sub,
                train_size=50_000,
                stratify=df_sub[tr_label_col],
                random_state=42,
            )

        X_tr_sub = df_sub[selected_features].to_numpy(dtype=np.float32)
        y_tr_sub = df_sub[tr_label_col].to_numpy(dtype=np.int64)

        # Train fresh RF (100 trees, depth 15, balanced, seed 42)
        log.info(f"  Training fresh RF on {len(df_sub):,} rows (excluding {cname})...")
        rf_fresh = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_fresh.fit(X_tr_sub, y_tr_sub)
        known_sub_classes = rf_fresh.classes_

        # Validation rows excluding held-out class
        val_mask_sub = (y_val_8 != cidx)
        X_val_sub = X_val_np[val_mask_sub]
        y_val_sub = y_val_8[val_mask_sub]

        log.info(f"  Fitting calibrator on validation rows excluding {cname} (N = {len(y_val_sub):,})...")
        P_val_raw_sub = rf_fresh.predict_proba(X_val_sub).astype(np.float32)

        calibrator_fresh = SubsetClassIsotonicCalibrator(known_classes=known_sub_classes)
        calibrator_fresh.fit(P_val_raw_sub, y_val_sub)
        P_val_cal_sub = calibrator_fresh.predict_proba(P_val_raw_sub)

        # Validation cutoffs for fresh model (at most 10% FPR on validation Benign)
        val_benign_sub_mask = (y_val_sub == benign_idx)
        max_p_val_benign_raw_sub = np.max(P_val_raw_sub[val_benign_sub_mask], axis=1)
        max_p_val_benign_cal_sub = np.max(P_val_cal_sub[val_benign_sub_mask], axis=1)

        tau_raw_c = float(np.percentile(max_p_val_benign_raw_sub, 10.0))
        tau_cal_c = float(np.percentile(max_p_val_benign_cal_sub, 10.0))

        val_benign_fpr_raw_c = float(np.mean(max_p_val_benign_raw_sub < tau_raw_c))
        val_benign_fpr_cal_c = float(np.mean(max_p_val_benign_cal_sub < tau_cal_c))

        # Held-out test flows of class cidx
        test_heldout_mask = (y_test_8 == cidx)
        X_heldout = X_test_np[test_heldout_mask]
        n_heldout = len(X_heldout)

        log.info(f"  Evaluating on held-out test flows (N = {n_heldout:,})...")
        P_heldout_raw = rf_fresh.predict_proba(X_heldout).astype(np.float32)
        P_heldout_cal = calibrator_fresh.predict_proba(P_heldout_raw)

        # Raw evaluation
        max_raw = np.max(P_heldout_raw, axis=1)
        pred_raw = known_sub_classes[np.argmax(P_heldout_raw, axis=1)]

        flagged_raw = (max_raw < tau_raw_c)
        unflagged_raw = ~flagged_raw
        silently_benign_raw = unflagged_raw & (pred_raw == benign_idx)
        silently_other_raw = unflagged_raw & (pred_raw != benign_idx)

        # Calibrated evaluation
        max_cal = np.max(P_heldout_cal, axis=1)
        pred_cal = known_sub_classes[np.argmax(P_heldout_cal, axis=1)]

        flagged_cal = (max_cal < tau_cal_c)
        unflagged_cal = ~flagged_cal
        silently_benign_cal = unflagged_cal & (pred_cal == benign_idx)
        silently_other_cal = unflagged_cal & (pred_cal != benign_idx)

        # Paired bootstrap difference
        diff_stat = paired_bootstrap_diff(flagged_cal, flagged_raw, n_resamples=1000, seed=42)

        # 4d. Test Benign FPR
        P_test_benign_raw = rf_fresh.predict_proba(X_test_benign).astype(np.float32)
        P_test_benign_cal = calibrator_fresh.predict_proba(P_test_benign_raw)

        test_benign_flagged_raw = (np.max(P_test_benign_raw, axis=1) < tau_raw_c)
        test_benign_flagged_cal = (np.max(P_test_benign_cal, axis=1) < tau_cal_c)

        elapsed = time.time() - t_start_class
        truncated = (elapsed > 360.0)

        class_result = {
            "class_name": cname,
            "class_idx": cidx,
            "truncated": truncated,
            "elapsed_sec": round(elapsed, 2),
            "validation_budget": {
                "budget_benign_fpr": 0.10,
                "tau_raw": round(tau_raw_c, 4),
                "tau_calibrated": round(tau_cal_c, 4),
                "val_benign_fpr_raw": round(val_benign_fpr_raw_c, 4),
                "val_benign_fpr_calibrated": round(val_benign_fpr_cal_c, 4),
            },
            "heldout_test_evaluation": {
                "n_samples": n_heldout,
                "raw": {
                    "flagged_unknown": wilson_score_interval(int(np.sum(flagged_raw)), n_heldout),
                    "silently_benign": wilson_score_interval(int(np.sum(silently_benign_raw)), n_heldout),
                    "silently_other_attack": wilson_score_interval(int(np.sum(silently_other_raw)), n_heldout),
                },
                "calibrated": {
                    "flagged_unknown": wilson_score_interval(int(np.sum(flagged_cal)), n_heldout),
                    "silently_benign": wilson_score_interval(int(np.sum(silently_benign_cal)), n_heldout),
                    "silently_other_attack": wilson_score_interval(int(np.sum(silently_other_cal)), n_heldout),
                },
                "paired_difference": diff_stat,
            },
            "test_benign_fpr": {
                "n_samples": n_test_benign,
                "raw": wilson_score_interval(int(np.sum(test_benign_flagged_raw)), n_test_benign),
                "calibrated": wilson_score_interval(int(np.sum(test_benign_flagged_cal)), n_test_benign),
            },
        }

        # Write result immediately for this class
        with open(class_json_path, "w", encoding="utf-8") as f:
            json.dump(class_result, f, indent=2)
        log.info(f"Saved {class_json_path.name} in {elapsed:.2f} s.")

        leave_one_out_results[cname] = class_result

        if diff_stat["significant_gain"]:
            headline_counts["significantly_more_flagged"] += 1
        elif diff_stat["significant_loss"]:
            headline_counts["significantly_fewer_flagged"] += 1
        else:
            headline_counts["about_the_same"] += 1

        print(
            f"  [{cname}] Flagged Raw: {class_result['heldout_test_evaluation']['raw']['flagged_unknown']['formatted']} | "
            f"Cal: {class_result['heldout_test_evaluation']['calibrated']['flagged_unknown']['formatted']} | "
            f"Diff: {diff_stat['formatted']}"
        )

    # Save summary
    summary_data = {
        "experiment": "Phase 12 Calibrated Open-Set Detection at Matched 10% Benign FP Budget",
        "budget": 0.10,
        "n_attack_classes": len(attack_classes),
        "headline_counts": headline_counts,
        "known_class_control": known_control_results,
        "leave_one_out": leave_one_out_results,
        "total_elapsed_sec": round(time.time() - t_pipeline, 2),
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    log.info(f"Open-set summary saved to {SUMMARY_JSON_PATH.resolve()}")

    print("\n" + "=" * 70)
    print("      PHASE 12 OPEN-SET CALIBRATION EVALUATION SUMMARY")
    print("=" * 70)
    print(
        f"Headline: In {headline_counts['significantly_more_flagged']} of the 7 held-out classes, "
        f"the calibrated rule flagged significantly more flows than the raw rule; "
        f"in {headline_counts['about_the_same']} classes differences were about the same; "
        f"and in {headline_counts['significantly_fewer_flagged']} classes it flagged significantly fewer flows."
    )
    print(f"Total pipeline time: {time.time() - t_pipeline:.2f} s")


if __name__ == "__main__":
    main()
