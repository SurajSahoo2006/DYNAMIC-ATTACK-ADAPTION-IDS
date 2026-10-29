"""
marl/stats.py
=============
Statistical utilities for Phase 5 MARL evaluation.
Implements:
  1. Wilson score confidence interval for binomial proportions (with counts "x of y").
  2. Non-parametric bootstrap confidence interval for means (10,000 resamples).
  3. Paired difference bootstrap confidence interval for pre-registered decision rules.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def wilson_score_interval(x: int, n: int, confidence: float = 0.95) -> Dict[str, Union[int, float]]:
    """
    Computes 95% Wilson score confidence interval for binomial count x of n.
    Returns dictionary with count, total, percentage, and CI bounds [lower, upper].
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
    # Standard normal quantile: 1.95996 for 95%
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
        "formatted": f"{x} of {n} ({pct:.2f}% [{lower:.2f}%, {upper:.2f}%])",
    }


def bootstrap_ci_mean(
    values: Union[List[float], np.ndarray],
    num_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes 95% bootstrap confidence interval for the sample mean.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    rng = np.random.default_rng(seed)
    # Vectorized bootstrap resampling
    resample_indices = rng.integers(0, n, size=(num_resamples, n))
    resampled_means = np.mean(arr[resample_indices], axis=1)

    alpha = (1.0 - confidence) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    ci_lower = float(np.percentile(resampled_means, lower_pct))
    ci_upper = float(np.percentile(resampled_means, upper_pct))

    return {
        "mean": round(mean_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std": round(std_val, 4),
    }


def bootstrap_ci_paired_difference(
    treatment_values: Union[List[float], np.ndarray],
    control_values: Union[List[float], np.ndarray],
    num_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Union[float, str]]:
    """
    Computes 95% bootstrap CI for the paired difference (treatment - control).
    Applies the pre-registered decision rule strictly.
    """
    t_arr = np.asarray(treatment_values, dtype=np.float64)
    c_arr = np.asarray(control_values, dtype=np.float64)
    if len(t_arr) != len(c_arr):
        raise ValueError(f"Arrays must have identical length, got {len(t_arr)} and {len(c_arr)}")

    diffs = t_arr - c_arr
    ci_res = bootstrap_ci_mean(diffs, num_resamples=num_resamples, confidence=confidence, seed=seed)

    ci_lower = ci_res["ci_lower"]
    ci_upper = ci_res["ci_upper"]
    mean_diff = ci_res["mean"]

    # Pre-registered decision rule:
    # "MAPPO is reported as better than the severity-rule baseline only if the paired-difference interval
    #  excludes zero in the positive direction for that reward setting; if it excludes zero in the negative
    #  direction it is reported as worse; otherwise it is reported as not distinguishable."
    if ci_lower > 0.0:
        outcome = "better"
    elif ci_upper < 0.0:
        outcome = "worse"
    else:
        outcome = "not distinguishable at this sample size"

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "decision_outcome": outcome,
    }
