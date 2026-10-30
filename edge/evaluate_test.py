"""
edge/evaluate_test.py
=====================
Phase 11 Final Test Set Evaluation.
Selects the SMALLEST compression variant whose validation macro F1 is within
1.0 percentage point of baseline. Evaluates Baseline, Chosen Variant, and V5
on the untouched test set ONCE:
  - 8-class accuracy and macro F1 with Wilson 95% CI on accuracy
  - Paired bootstrap 95% CI (1,000 resamples, seed 42) on accuracy difference vs baseline
  - Per-class F1 for the 3 rare 34-class labels (Uploading_Attack, Recon-PingSweep, Backdoor_Malware)
    for baseline vs chosen variant.
Saves results to edge/results/final_comparison.json.
"""

from __future__ import annotations

import os
# Single-thread execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODELS_DIR,
    TEST_PARQUET,
    ENCODER_34_PATH,
    SELECTED_FEATURES_PATH,
    RANDOM_SEED,
)
from edge.benchmark import RESULTS_DIR
from edge.compress import truncate_rf, slice_xgb, EDGE_MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RARE_34_CLASSES = ["Uploading_Attack", "Recon-PingSweep", "Backdoor_Malware"]


def wilson_score_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    half_width = (z * np.sqrt((p * (1.0 - p)) / total + (z**2) / (4.0 * (total**2)))) / denom
    return round(float(center - half_width), 4), round(float(center + half_width), 4)


def paired_bootstrap_acc_diff(
    y_true: np.ndarray,
    y_pred_candidate: np.ndarray,
    y_pred_base: np.ndarray,
    n_resamples: int = 1000,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """
    Computes paired bootstrap 95% CI on accuracy difference (candidate - baseline).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    correct_cand = (y_pred_candidate == y_true).astype(np.float64)
    correct_base = (y_pred_base == y_true).astype(np.float64)

    diffs = np.zeros(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        diffs[i] = np.mean(correct_cand[indices]) - np.mean(correct_base[indices])

    mean_diff = float(np.mean(diffs))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    return {
        "mean_diff": round(mean_diff, 5),
        "ci_lower": round(ci_lower, 5),
        "ci_upper": round(ci_upper, 5),
    }


def select_smallest_within_tolerance(
    baseline_val_macro_f1: float,
    tolerance: float = 0.010,
) -> Dict[str, Any]:
    """
    Finds the SMALLEST variant whose validation macro F1 is within 1.0 percentage point
    of the baseline (val_macro_f1 >= baseline_val_macro_f1 - 0.010).
    """
    variant_files = list(RESULTS_DIR.glob("variant_*.json"))
    if not variant_files:
        raise FileNotFoundError("No variant JSON files found in edge/results/")

    min_allowed_f1 = baseline_val_macro_f1 - tolerance
    candidates = []

    for fpath in variant_files:
        with open(fpath, "r", encoding="utf-8") as f:
            vdata = json.load(f)
        val_f1 = vdata.get("validation_macro_f1", 0.0)
        size_comp = vdata.get("size_mb_compressed", float("inf"))
        size_uncomp = vdata.get("size_mb_uncompressed", float("inf"))
        vname = vdata.get("variant", fpath.stem)

        if val_f1 >= min_allowed_f1:
            candidates.append({
                "variant": vname,
                "data": vdata,
                "size_compressed": size_comp,
                "size_uncompressed": size_uncomp,
                "val_macro_f1": val_f1,
            })

    if not candidates:
        raise ValueError(f"No variant found within {tolerance*100}% of baseline macro F1 ({baseline_val_macro_f1})")

    # Pick smallest by compressed size, tie-break by uncompressed size
    candidates.sort(key=lambda c: (c["size_compressed"], c["size_uncompressed"]))
    selected = candidates[0]
    log.info(f"Selected smallest variant: '{selected['variant']}' (size: {selected['size_compressed']}MB, val_f1: {selected['val_macro_f1']:.4f})")
    return selected["data"]


def evaluate_final():
    print("=" * 70)
    print("PHASE 11: FINAL TEST SET EVALUATION")
    print("=" * 70)

    # 1. Load baseline results
    baseline_path = RESULTS_DIR / "baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing {baseline_path}. Run edge/benchmark.py first.")
    with open(baseline_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    baseline_val_macro_f1 = base_data["validation_macro_f1"]
    log.info(f"Baseline validation macro F1: {baseline_val_macro_f1:.4f}")

    # 2. Select chosen variant using validation only
    chosen_variant_data = select_smallest_within_tolerance(baseline_val_macro_f1, tolerance=0.010)
    chosen_name = chosen_variant_data["variant"]

    # Load V5 data
    v5_path = RESULTS_DIR / "variant_v5_combined.json"
    if not v5_path.exists():
        raise FileNotFoundError(f"Missing {v5_path}. Run edge/compress.py first.")
    with open(v5_path, "r", encoding="utf-8") as f:
        v5_data = json.load(f)

    # 3. Load test set ONCE
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)
    df_test = pd.read_parquet(TEST_PARQUET)
    X_test = df_test[features].to_numpy(dtype=np.float32)
    y_test_8 = df_test["label_8"].to_numpy(dtype=np.int64)
    y_test_34 = df_test["label_34"].to_numpy(dtype=np.int64)
    n_test = len(y_test_8)
    log.info(f"Loaded test set with {n_test:,} flows across 19 features.")

    # 4. Load models for 8-class
    rf_base_8 = joblib.load(MODELS_DIR / "rf_8class_b2.joblib")
    xgb_base_8 = joblib.load(MODELS_DIR / "xgb_8class_b2.joblib")
    rf_base_8.n_jobs = 1
    xgb_base_8.set_params(n_jobs=1)

    # Predict baseline 8-class
    p_rf_base = rf_base_8.predict_proba(X_test)
    p_xgb_base = xgb_base_8.predict_proba(X_test)
    p_base_8 = 0.60 * p_rf_base + 0.40 * p_xgb_base
    preds_base_8 = np.argmax(p_base_8, axis=1)

    acc_base = float(accuracy_score(y_test_8, preds_base_8))
    f1_base = float(f1_score(y_test_8, preds_base_8, average="macro", zero_division=0))
    wilson_base = wilson_score_interval(int(np.sum(preds_base_8 == y_test_8)), n_test)

    # Build chosen models for 8-class
    rf_chosen_8, xgb_chosen_8 = build_models_from_variant(chosen_variant_data, rf_base_8, xgb_base_8)
    p_rf_chosen = rf_chosen_8.predict_proba(X_test)
    p_xgb_chosen = xgb_chosen_8.predict_proba(X_test)
    p_chosen_8 = 0.60 * p_rf_chosen + 0.40 * p_xgb_chosen
    preds_chosen_8 = np.argmax(p_chosen_8, axis=1)

    acc_chosen = float(accuracy_score(y_test_8, preds_chosen_8))
    f1_chosen = float(f1_score(y_test_8, preds_chosen_8, average="macro", zero_division=0))
    wilson_chosen = wilson_score_interval(int(np.sum(preds_chosen_8 == y_test_8)), n_test)
    diff_chosen = paired_bootstrap_acc_diff(y_test_8, preds_chosen_8, preds_base_8, n_resamples=1000)

    # Build V5 models for 8-class
    rf_v5_8, xgb_v5_8 = build_models_from_variant(v5_data, rf_base_8, xgb_base_8)
    p_rf_v5 = rf_v5_8.predict_proba(X_test)
    p_xgb_v5 = xgb_v5_8.predict_proba(X_test)
    p_v5_8 = 0.60 * p_rf_v5 + 0.40 * p_xgb_v5
    preds_v5_8 = np.argmax(p_v5_8, axis=1)

    acc_v5 = float(accuracy_score(y_test_8, preds_v5_8))
    f1_v5 = float(f1_score(y_test_8, preds_v5_8, average="macro", zero_division=0))
    wilson_v5 = wilson_score_interval(int(np.sum(preds_v5_8 == y_test_8)), n_test)
    diff_v5 = paired_bootstrap_acc_diff(y_test_8, preds_v5_8, preds_base_8, n_resamples=1000)

    # 5. Evaluate rare 34-class labels for Baseline vs Chosen Variant
    log.info("Evaluating rare 34-class labels for baseline vs chosen variant...")
    rf_base_34 = joblib.load(MODELS_DIR / "rf_34class_b2.joblib")
    xgb_base_34 = joblib.load(MODELS_DIR / "xgb_34class_b2.joblib")
    rf_base_34.n_jobs = 1
    xgb_base_34.set_params(n_jobs=1)

    p_rf_base_34 = rf_base_34.predict_proba(X_test)
    p_xgb_base_34 = xgb_base_34.predict_proba(X_test)
    p_base_34 = 0.60 * p_rf_base_34 + 0.40 * p_xgb_base_34
    preds_base_34 = np.argmax(p_base_34, axis=1)

    rf_chosen_34, xgb_chosen_34 = build_models_from_variant(chosen_variant_data, rf_base_34, xgb_base_34, target="34class")
    p_rf_chosen_34 = rf_chosen_34.predict_proba(X_test)
    p_xgb_chosen_34 = xgb_chosen_34.predict_proba(X_test)
    p_chosen_34 = 0.60 * p_rf_chosen_34 + 0.40 * p_xgb_chosen_34
    preds_chosen_34 = np.argmax(p_chosen_34, axis=1)

    le_34 = joblib.load(ENCODER_34_PATH)
    classes_34 = list(le_34.classes_)

    rare_class_results = []
    for rname in RARE_34_CLASSES:
        c_idx = classes_34.index(rname)
        sup = int(np.sum(y_test_34 == c_idx))
        f1_b = float(f1_score(y_test_34 == c_idx, preds_base_34 == c_idx, zero_division=0))
        f1_c = float(f1_score(y_test_34 == c_idx, preds_chosen_34 == c_idx, zero_division=0))
        diff_f1 = f1_c - f1_b
        rare_class_results.append({
            "class_name": rname,
            "class_index": c_idx,
            "support": sup,
            "baseline_f1": round(f1_b, 4),
            "chosen_f1": round(f1_c, 4),
            "f1_difference": round(diff_f1, 4),
        })

    # 6. Assemble output
    final_output = {
        "chosen_variant_name": chosen_name,
        "chosen_variant_selection_reason": (
            f"Smallest compressed size ({chosen_variant_data['size_mb_compressed']} MB) "
            f"with validation macro F1 ({chosen_variant_data['validation_macro_f1']:.4f}) within 1.0% "
            f"of baseline ({baseline_val_macro_f1:.4f})"
        ),
        "test_evaluations": {
            "baseline": {
                "name": "Baseline (B2 Ensemble)",
                "size_mb_uncompressed": base_data["size_mb_uncompressed"],
                "size_mb_compressed": base_data["size_mb_compressed"],
                "latency_p95_ms": base_data["latency_per_flow_ms"]["p95_ms"],
                "throughput_fps": base_data["throughput_fps"]["median_fps"],
                "test_accuracy": round(acc_base, 4),
                "test_accuracy_wilson_ci": wilson_base,
                "test_macro_f1": round(f1_base, 4),
            },
            "chosen_variant": {
                "name": chosen_name,
                "size_mb_uncompressed": chosen_variant_data["size_mb_uncompressed"],
                "size_mb_compressed": chosen_variant_data["size_mb_compressed"],
                "latency_p95_ms": chosen_variant_data["latency_per_flow_ms"]["p95_ms"],
                "throughput_fps": chosen_variant_data["throughput_fps"]["median_fps"],
                "test_accuracy": round(acc_chosen, 4),
                "test_accuracy_wilson_ci": wilson_chosen,
                "test_macro_f1": round(f1_chosen, 4),
                "paired_accuracy_diff_vs_baseline": diff_chosen,
            },
            "v5_combined": {
                "name": "v5_combined",
                "size_mb_uncompressed": v5_data["size_mb_uncompressed"],
                "size_mb_compressed": v5_data["size_mb_compressed"],
                "latency_p95_ms": v5_data["latency_per_flow_ms"]["p95_ms"],
                "throughput_fps": v5_data["throughput_fps"]["median_fps"],
                "test_accuracy": round(acc_v5, 4),
                "test_accuracy_wilson_ci": wilson_v5,
                "test_macro_f1": round(f1_v5, 4),
                "paired_accuracy_diff_vs_baseline": diff_v5,
            },
        },
        "rare_34class_f1": rare_class_results,
    }

    out_file = RESULTS_DIR / "final_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n[DONE] Final comparison saved to {out_file}")
    print(f"Chosen Variant: {chosen_name}")
    print(f"  Test Accuracy: {acc_chosen:.4f} (Wilson 95% CI: [{wilson_chosen[0]:.4f}, {wilson_chosen[1]:.4f}])")
    print(f"  Paired Acc Diff vs Baseline: {diff_chosen['mean_diff']:+.5f} (95% CI: [{diff_chosen['ci_lower']:+.5f}, {diff_chosen['ci_upper']:+.5f}])")
    print(f"  Test Macro F1: {f1_chosen:.4f}")
    print("\nRare Class F1 (Baseline vs Chosen):")
    for r in rare_class_results:
        print(f"  {r['class_name']:<20} | Support: {r['support']:>3} | Base F1: {r['baseline_f1']:.4f} | Chosen F1: {r['chosen_f1']:.4f} | Diff: {r['f1_difference']:+.4f}")

    return final_output


def build_models_from_variant(vdata: Dict[str, Any], rf_base: Any, xgb_base: Any, target: str = "8class") -> Tuple[Any, Any]:
    """Helper to reconstruct RF and XGBoost according to variant metadata."""
    vname = vdata.get("variant", "")
    rf_trees = vdata.get("rf_trees", 200)
    xgb_rounds = vdata.get("xgb_rounds", 250)
    rf_depth = vdata.get("rf_max_depth", None)

    # 1. XGBoost
    if xgb_rounds < 250:
        xgb_mod = slice_xgb(xgb_base, xgb_rounds)
    else:
        xgb_mod = xgb_base

    # 2. RF
    if rf_depth is not None:
        if target == "8class":
            depth_file = EDGE_MODELS_DIR / f"rf_v3_depth{rf_depth}.joblib"
            rf_mod = joblib.load(depth_file)
            rf_mod.n_jobs = 1
        else:
            # For 34class, if retrained depth model doesn't exist, use depth-capped trees or truncate
            rf_mod = truncate_rf(rf_base, rf_trees)
    elif rf_trees < 200:
        rf_mod = truncate_rf(rf_base, rf_trees)
    elif "v5_combined" in vname:
        # Check sub-components
        sub_rf = vdata.get("selected_rf_variant", "")
        sub_xgb = vdata.get("selected_xgb_variant", "")
        # Resolve RF
        if "depth" in sub_rf:
            d = 15 if "15" in sub_rf else 10
            if target == "8class":
                depth_file = EDGE_MODELS_DIR / f"rf_v3_depth{d}.joblib"
                rf_mod = joblib.load(depth_file)
                rf_mod.n_jobs = 1
            else:
                rf_mod = rf_base
        elif "rf_" in sub_rf:
            t = int(sub_rf.split("_")[-1])
            rf_mod = truncate_rf(rf_base, t)
        else:
            rf_mod = rf_base

        # Resolve XGB
        if "xgb_" in sub_xgb:
            r = int(sub_xgb.split("_")[-1])
            xgb_mod = slice_xgb(xgb_base, r)
        else:
            xgb_mod = xgb_base
    else:
        rf_mod = rf_base

    return rf_mod, xgb_mod


if __name__ == "__main__":
    evaluate_final()
