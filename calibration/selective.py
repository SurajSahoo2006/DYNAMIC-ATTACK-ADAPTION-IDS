"""
calibration/selective.py
========================
Phase 12: Selective Prediction & Risk-Coverage Analysis.

For raw and calibrated max-probability:
  1. Computes the risk-coverage curve on the TEST set (coverage vs error among accepted flows).
  2. Computes AURC (Area Under the Risk-Coverage Curve) with 200-sample bootstrap 95% CI (seed 42).
  3. Selects an operating point on VALIDATION only: lowest confidence cutoff whose
     accepted-flow accuracy is at least 99.0%.
  4. Evaluates that validation cutoff on the TEST set:
     reports test coverage and accepted-flow accuracy with Wilson 95% CIs.

Saves results to calibration/results/selective.json.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibration_selective")

RESULTS_DIR = PROJECT_ROOT / "calibration" / "results"
SELECTIVE_JSON_PATH = RESULTS_DIR / "selective.json"


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


def compute_risk_coverage_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes coverage, risk (error rate among accepted flows), and AURC.
    Samples are sorted in descending order of maximum probability (confidence).
    """
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    errors = (preds != y_true).astype(float)
    n = len(y_true)

    sort_idx = np.argsort(-conf)
    sorted_errors = errors[sort_idx]
    cum_errors = np.cumsum(sorted_errors)
    counts = np.arange(1, n + 1)

    risks = cum_errors / counts
    coverages = counts / float(n)
    aurc = float(np.mean(risks))

    return coverages, risks, aurc


def compute_aurc_with_ci(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_resamples: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes AURC point estimate and 200-sample bootstrap 95% CI.
    """
    coverages, risks, point_aurc = compute_risk_coverage_curve(probs, y_true)

    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aurcs = []

    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        _, _, b_aurc = compute_risk_coverage_curve(probs[idx], y_true[idx])
        boot_aurcs.append(b_aurc)

    ci_lower = float(np.percentile(boot_aurcs, 2.5))
    ci_upper = float(np.percentile(boot_aurcs, 97.5))

    # Downsample curve to 20 representative coverage points for clean reporting
    stride = max(1, n // 20)
    sampled_points = []
    for i in range(0, n, stride):
        sampled_points.append({
            "coverage": round(float(coverages[i]), 4),
            "risk": round(float(risks[i]), 4),
            "accuracy": round(float(1.0 - risks[i]), 4),
        })
    if sampled_points[-1]["coverage"] != round(float(coverages[-1]), 4):
        sampled_points.append({
            "coverage": round(float(coverages[-1]), 4),
            "risk": round(float(risks[-1]), 4),
            "accuracy": round(float(1.0 - risks[-1]), 4),
        })

    return {
        "aurc": round(point_aurc, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "formatted": f"{point_aurc:.6f} [{ci_lower:.6f}, {ci_upper:.6f}]",
        "sampled_curve": sampled_points,
    }


def compute_paired_aurc_ci(
    probs_raw: np.ndarray,
    probs_cal: np.ndarray,
    y_true: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes paired difference on AURC (calibrated minus raw) with a 1,000-resample bootstrap 95% CI.
    """
    _, _, aurc_raw = compute_risk_coverage_curve(probs_raw, y_true)
    _, _, aurc_cal = compute_risk_coverage_curve(probs_cal, y_true)
    point_diff = aurc_cal - aurc_raw

    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_diffs = []

    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        _, _, r = compute_risk_coverage_curve(probs_raw[idx], y_true[idx])
        _, _, c = compute_risk_coverage_curve(probs_cal[idx], y_true[idx])
        boot_diffs.append(c - r)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    excludes_zero = bool((ci_upper < 0) or (ci_lower > 0))

    return {
        "point_diff": round(float(point_diff), 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "ci": [round(float(ci_lower), 6), round(float(ci_upper), 6)],
        "formatted": f"{point_diff:+.6f} [{ci_lower:+.6f}, {ci_upper:+.6f}]",
        "n_resamples": n_resamples,
        "seed": seed,
        "excludes_zero": excludes_zero,
    }



def find_validation_operating_point(
    probs_val: np.ndarray,
    y_val: np.ndarray,
    target_accuracy: float = 0.990,
) -> Dict[str, Any]:
    """
    Finds the lowest confidence cutoff on validation data whose accepted-flow
    accuracy is at least target_accuracy (default: 99.0%).
    """
    conf = np.max(probs_val, axis=1)
    preds = np.argmax(probs_val, axis=1)
    correct = (preds == y_val).astype(float)

    # Unique confidence cutoffs sorted ascending
    cutoffs = np.sort(np.unique(conf))
    chosen_cutoff = 1.0
    achieved_val_acc = 1.0
    val_cov = 0.0

    for c in cutoffs:
        mask = (conf >= c)
        acc = float(np.mean(correct[mask]))
        if acc >= target_accuracy:
            chosen_cutoff = float(c)
            achieved_val_acc = acc
            val_cov = float(np.mean(mask))
            break

    return {
        "target_accuracy": target_accuracy,
        "chosen_cutoff": round(chosen_cutoff, 4),
        "val_accepted_accuracy": round(achieved_val_acc, 4),
        "val_coverage": round(val_cov, 4),
    }


def evaluate_operating_point_on_test(
    probs_test: np.ndarray,
    y_test: np.ndarray,
    cutoff: float,
) -> Dict[str, Any]:
    """
    Applies the validation-chosen cutoff to the test set and computes test
    coverage and accepted-flow accuracy with Wilson 95% CIs.
    """
    conf = np.max(probs_test, axis=1)
    preds = np.argmax(probs_test, axis=1)
    n_total = len(y_test)

    accepted_mask = (conf >= cutoff)
    n_accepted = int(np.sum(accepted_mask))
    n_correct = int(np.sum((preds == y_test) & accepted_mask))

    cov_info = wilson_score_interval(n_accepted, n_total)
    acc_info = wilson_score_interval(n_correct, n_accepted)

    return {
        "cutoff": round(cutoff, 4),
        "total_test_flows": n_total,
        "accepted_flows": n_accepted,
        "rejected_flows": n_total - n_accepted,
        "coverage": cov_info,
        "accepted_accuracy": acc_info,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load cached probabilities
    val_raw_p = RESULTS_DIR / "val_probs_raw.npy"
    val_cal_p = RESULTS_DIR / "val_probs_calibrated.npy"
    test_raw_p = RESULTS_DIR / "test_probs_raw.npy"
    test_cal_p = RESULTS_DIR / "test_probs_calibrated.npy"
    y_val_p = RESULTS_DIR / "y_val_8.npy"
    y_test_p = RESULTS_DIR / "y_test_8.npy"

    for p in [val_raw_p, val_cal_p, test_raw_p, test_cal_p, y_val_p, y_test_p]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required cached probability array: {p.name}. Run measure.py and fit.py first.")

    P_val_raw = np.load(val_raw_p)
    P_val_cal = np.load(val_cal_p)
    P_test_raw = np.load(test_raw_p)
    P_test_cal = np.load(test_cal_p)
    y_val = np.load(y_val_p)
    y_test = np.load(y_test_p)

    log.info(f"Loaded validation set: N = {len(y_val):,}")
    log.info(f"Loaded test set: N = {len(y_test):,}")

    # 1. Test set AURC with 200-sample bootstrap CI
    log.info("Computing Test AURC for Raw probabilities (200 bootstrap resamples)...")
    aurc_raw = compute_aurc_with_ci(P_test_raw, y_test, n_resamples=200, seed=42)

    log.info("Computing Test AURC for Calibrated probabilities (200 bootstrap resamples)...")
    aurc_cal = compute_aurc_with_ci(P_test_cal, y_test, n_resamples=200, seed=42)

    log.info("Computing paired bootstrap 95% CI on Test AURC difference (1,000 resamples)...")
    paired_aurc = compute_paired_aurc_ci(P_test_raw, P_test_cal, y_test, n_resamples=1000, seed=42)
    log.info(f"  Paired AURC Delta: {paired_aurc['formatted']} (excludes zero: {paired_aurc['excludes_zero']})")

    # 2. Operating point chosen on VALIDATION only (lowest cutoff with accepted accuracy >= 99.0%)
    log.info("Selecting 99.0% accuracy operating point on VALIDATION set...")
    op_val_raw = find_validation_operating_point(P_val_raw, y_val, target_accuracy=0.990)
    op_val_cal = find_validation_operating_point(P_val_cal, y_val, target_accuracy=0.990)

    log.info(f"  Raw: Cutoff = {op_val_raw['chosen_cutoff']} (Val Acc: {op_val_raw['val_accepted_accuracy']*100:.2f}%, Val Cov: {op_val_raw['val_coverage']*100:.2f}%)")
    log.info(f"  Calibrated: Cutoff = {op_val_cal['chosen_cutoff']} (Val Acc: {op_val_cal['val_accepted_accuracy']*100:.2f}%, Val Cov: {op_val_cal['val_coverage']*100:.2f}%)")

    # 3. Test set evaluation at validation-chosen operating points
    log.info("Evaluating validation-chosen operating points on TEST set...")
    test_eval_raw = evaluate_operating_point_on_test(P_test_raw, y_test, op_val_raw["chosen_cutoff"])
    test_eval_cal = evaluate_operating_point_on_test(P_test_cal, y_test, op_val_cal["chosen_cutoff"])

    # Comparisons
    aurc_diff = round(aurc_cal["aurc"] - aurc_raw["aurc"], 6)
    cov_diff_pct = round(test_eval_cal["coverage"]["percentage"] - test_eval_raw["coverage"]["percentage"], 2)

    results = {
        "raw": {
            "aurc": aurc_raw,
            "validation_operating_point": op_val_raw,
            "test_evaluation": test_eval_raw,
        },
        "calibrated": {
            "aurc": aurc_cal,
            "validation_operating_point": op_val_cal,
            "test_evaluation": test_eval_cal,
        },
        "comparison": {
            "aurc_delta": aurc_diff,
            "paired_aurc_ci": paired_aurc,
            "coverage_delta_pct": cov_diff_pct,
            "coverage_gain": cov_diff_pct > 0,
        },
    }

    with open(SELECTIVE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info(f"Selective prediction results saved to {SELECTIVE_JSON_PATH.resolve()}")

    print("\n" + "=" * 70)
    print("      SELECTIVE PREDICTION EVALUATION (RAW VS CALIBRATED)")
    print("=" * 70)
    print("\n1. Risk-Coverage Quality (Untouched Test Set, N = 26,292):")
    print(f"  Raw AURC:        {aurc_raw['formatted']}")
    print(f"  Calibrated AURC: {aurc_cal['formatted']}")
    print(f"  AURC Delta:      {aurc_diff:+.6f} ({'Improved' if aurc_diff < 0 else 'Degraded'})")
    print(f"  Paired 95% CI:   {paired_aurc['formatted']} (excludes zero: {paired_aurc['excludes_zero']})")

    print("\n2. Validation-Chosen 99.0% Accuracy Operating Point on Test Set:")
    print(f"  Raw (Cutoff = {op_val_raw['chosen_cutoff']}):")
    print(f"    Test Coverage: {test_eval_raw['coverage']['formatted']}")
    print(f"    Accepted Acc:  {test_eval_raw['accepted_accuracy']['formatted']}")
    print(f"  Calibrated (Cutoff = {op_val_cal['chosen_cutoff']}):")
    print(f"    Test Coverage: {test_eval_cal['coverage']['formatted']}")
    print(f"    Accepted Acc:  {test_eval_cal['accepted_accuracy']['formatted']}")
    print(f"  Coverage Delta:  {cov_diff_pct:+.2f}% ({test_eval_cal['accepted_flows'] - test_eval_raw['accepted_flows']:+d} flows)")

    print(f"\nExecution time: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
