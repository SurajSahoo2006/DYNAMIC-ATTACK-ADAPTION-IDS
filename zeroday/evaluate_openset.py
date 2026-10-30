#!/usr/bin/env python3
"""
zeroday/evaluate_openset.py - Open-set evaluation with separated detectors, lift calculation, AUROC, and FP-budget operating point.
"""

from __future__ import annotations

import argparse
import copy
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SELECTED_FEATURES_PATH,
    ENCODER_8_PATH,
    SCALER_PATH,
)
from models.load_validation import load_validation_data
from zeroday.train_autoencoder import BenignAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_openset")

MODELS_DIR = PROJECT_ROOT / "zeroday" / "models"
RESULTS_DIR = PROJECT_ROOT / "zeroday" / "results"


def wilson_score_interval(x: int, n: int, confidence: float = 0.95) -> Dict[str, Union[int, float, str]]:
    """
    Computes 95% Wilson score confidence interval for binomial count x of n.
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


def bootstrap_difference_ci(
    y_heldout: np.ndarray,
    y_known: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Union[float, str]]:
    """
    Bootstrap 95% CI (paired resamples) for the difference: heldout_rate - known_rate.
    """
    rng = np.random.RandomState(seed)
    n = len(y_heldout)
    heldout_rate = float(np.mean(y_heldout)) * 100.0
    known_rate = float(np.mean(y_known)) * 100.0
    lift_val = heldout_rate - known_rate

    boot_diffs = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_diffs.append((np.mean(y_heldout[idx]) - np.mean(y_known[idx])) * 100.0)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    sign = "+" if lift_val >= 0 else ""
    return {
        "heldout_rate": round(heldout_rate, 2),
        "known_rate": round(known_rate, 2),
        "lift": round(lift_val, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "formatted": f"{sign}{lift_val:.2f}% [{ci_lower:+.2f}%, {ci_upper:+.2f}%]",
    }


def compute_auroc_with_ci(
    benign_scores: np.ndarray,
    attack_scores: np.ndarray,
    n_resamples: int = 200,
    seed: int = 42,
) -> Dict[str, Union[float, str]]:
    """
    Computes AUROC of AE reconstruction score (Benign vs Attack Class) with bootstrap 95% CI.
    """
    y_true = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(attack_scores))])
    y_score = np.concatenate([benign_scores, attack_scores])
    point_auroc = float(roc_auc_score(y_true, y_score))

    rng = np.random.RandomState(seed)
    n_total = len(y_true)
    boot_aurocs = []
    for _ in range(n_resamples):
        b_idx = rng.randint(0, n_total, size=n_total)
        if len(np.unique(y_true[b_idx])) < 2:
            continue
        boot_aurocs.append(float(roc_auc_score(y_true[b_idx], y_score[b_idx])))

    ci_lower = float(np.percentile(boot_aurocs, 2.5))
    ci_upper = float(np.percentile(boot_aurocs, 97.5))
    return {
        "auroc": round(point_auroc, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "formatted": f"{point_auroc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]",
    }


def find_fp_budget_operating_point(
    ae: nn.Module,
    selected_features: List[str],
    class_names: List[str],
    benign_idx: int,
) -> Dict[str, Any]:
    """
    FIX 3: Validation-only grid search over AE percentile and confidence cutoff.
    Keep settings with validation Benign FPR <= 10%, pick highest mean flagged rate on held-out classes.
    """
    log.info("Running Step FIX 3: Validation-only operating point grid search ...")
    rf_path = PROJECT_ROOT / "models" / "rf_8class_b2.joblib"
    xgb_path = PROJECT_ROOT / "models" / "xgb_8class_b2.joblib"
    rf = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)

    X_val_all, y_val_8, _ = load_validation_data()
    X_val = X_val_all[selected_features].to_numpy(dtype=np.float32)

    rf_val_prob = rf.predict_proba(X_val)
    xgb_val_prob = xgb.predict_proba(X_val)
    val_b2_prob = 0.6 * rf_val_prob + 0.4 * xgb_val_prob
    val_max_prob = np.max(val_b2_prob, axis=1)

    with torch.no_grad():
        x_val_t = torch.from_numpy(X_val.copy())
        rec = ae(x_val_t)
        val_ae_scores = ((x_val_t - rec) ** 2).mean(dim=1).cpu().numpy()

    val_benign_scores = val_ae_scores[y_val_8 == benign_idx]

    percentiles = [95.0, 97.5, 99.0]
    cutoffs = [0.30, 0.40, 0.50, 0.60]

    attack_indices = [i for i, c in enumerate(class_names) if c != "Benign"]
    candidates = []
    all_grid_results = []

    for p in percentiles:
        thresh = float(np.percentile(val_benign_scores, p))
        for c in cutoffs:
            is_flagged = (val_ae_scores > thresh) | (val_max_prob < c)
            benign_fpr = float((is_flagged[y_val_8 == benign_idx]).mean())

            attack_rates = []
            for aidx in attack_indices:
                mask = (y_val_8 == aidx)
                if mask.sum() > 0:
                    attack_rates.append(float((is_flagged[mask]).mean()))
            mean_attack_rate = float(np.mean(attack_rates))

            is_valid = bool(benign_fpr <= 0.10)
            entry = {
                "percentile": p,
                "threshold": round(thresh, 6),
                "confidence_cutoff": c,
                "val_benign_fpr": round(benign_fpr, 4),
                "val_mean_attack_rate": round(mean_attack_rate, 4),
                "valid_within_10pct_budget": is_valid,
            }
            all_grid_results.append(entry)
            if is_valid:
                candidates.append(entry)

    if not candidates:
        raise ValueError("No grid point met the 10% validation Benign FPR budget.")

    candidates.sort(key=lambda x: x["val_mean_attack_rate"], reverse=True)
    best_op = candidates[0]
    log.info(
        f"Chosen validation operating point: Percentile={best_op['percentile']}%, "
        f"Threshold={best_op['threshold']:.5f}, Confidence Cutoff={best_op['confidence_cutoff']} "
        f"(Val Benign FPR: {best_op['val_benign_fpr']*100:.2f}%, Val Mean Attack Rate: {best_op['val_mean_attack_rate']*100:.2f}%)"
    )

    return {
        "chosen_operating_point": best_op,
        "grid_evaluations": all_grid_results,
    }


def load_artifacts_and_ae():
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features = json.load(f)
    if isinstance(selected_features, dict) and "selected_features" in selected_features:
        selected_features = selected_features["selected_features"]

    le_8 = joblib.load(ENCODER_8_PATH)
    class_names = list(le_8.classes_)
    benign_idx = int(np.where(le_8.classes_ == "Benign")[0][0])

    ae_model_path = MODELS_DIR / "autoencoder.pt"
    if not ae_model_path.exists():
        raise FileNotFoundError(f"Autoencoder model not found at {ae_model_path}. Run train_autoencoder.py first.")

    ckpt = torch.load(ae_model_path, map_location="cpu")
    ae = BenignAutoencoder(in_features=len(selected_features))
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval()
    threshold = float(ckpt["threshold"])

    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)

    return selected_features, le_8, class_names, benign_idx, ae, threshold, scaler, ckpt


def main():
    parser = argparse.ArgumentParser(description="Phase 10 Open-Set Evaluation with fixes")
    parser.add_argument("--resume", action="store_true", help="Resume completed held-out evaluations")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    selected_features, le_8, class_names, benign_idx, ae, threshold_orig, scaler, ae_ckpt = load_artifacts_and_ae()

    # Step FIX 3: Validation-only grid search to select FP-budget operating point
    val_grid_data = find_fp_budget_operating_point(ae, selected_features, class_names, benign_idx)
    chosen_op = val_grid_data["chosen_operating_point"]

    thresh_op1 = threshold_orig  # 95th percentile
    conf_op1 = 0.60

    thresh_op2 = chosen_op["threshold"]  # e.g. 97.5th percentile
    conf_op2 = chosen_op["confidence_cutoff"]  # e.g. 0.40

    log.info(f"Operating Point 1 (Original): Threshold={thresh_op1:.6f} (95th), Conf Cutoff={conf_op1:.2f}")
    log.info(f"Operating Point 2 (FP Budget): Threshold={thresh_op2:.6f} ({chosen_op['percentile']}th), Conf Cutoff={conf_op2:.2f}")

    # Load full test set
    test_path = PROJECT_ROOT / "data" / "processed" / "test.parquet"
    if not test_path.exists():
        test_path = PROJECT_ROOT / "data" / "processed" / "test_phase2.parquet"
    if not test_path.exists():
        raise FileNotFoundError("Could not find test.parquet or test_phase2.parquet")

    df_test = pd.read_parquet(test_path)
    X_test = df_test[selected_features].to_numpy(dtype=np.float32)
    label_col = "label_8" if "label_8" in df_test.columns else "label"
    y_test = df_test[label_col].to_numpy(dtype=np.int64)

    if scaler is not None and hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(selected_features):
        X_test = scaler.transform(X_test).astype(np.float32)

    # Compute AE reconstruction scores on full test set
    with torch.no_grad():
        x_test_t = torch.from_numpy(X_test.copy())
        reconstructions = ae(x_test_t)
        ae_scores_test = ((x_test_t - reconstructions) ** 2).mean(dim=1).cpu().numpy()

    # Step FIX 2: Threshold-Free AE Quality (AUROC of AE reconstruction score on TEST set)
    log.info("Running Step FIX 2: Computing AUROC of AE score on TEST set ...")
    benign_mask = (y_test == benign_idx)
    benign_ae_scores = ae_scores_test[benign_mask]

    auroc_results = {}
    attack_classes = [c for c in class_names if c != "Benign"]
    for cname in attack_classes:
        cidx = int(np.where(le_8.classes_ == cname)[0][0])
        c_scores = ae_scores_test[y_test == cidx]
        stat_auroc = compute_auroc_with_ci(benign_ae_scores, c_scores, n_resamples=200, seed=42)
        auroc_results[cname] = stat_auroc
        log.info(f"  AUROC (Benign vs {cname:<11}): {stat_auroc['formatted']}")

    # Step FIX 1 & 2a: Known-Class Control (Full Test Set)
    log.info("Running Known-Class Control with B2 ensemble (reporting AE-only, conf-only, combined) ...")
    rf_path = PROJECT_ROOT / "models" / "rf_8class_b2.joblib"
    xgb_path = PROJECT_ROOT / "models" / "xgb_8class_b2.joblib"
    rf_b2 = joblib.load(rf_path)
    xgb_b2 = joblib.load(xgb_path)

    rf_probs = rf_b2.predict_proba(X_test)
    xgb_probs = xgb_b2.predict_proba(X_test)
    b2_probs = 0.6 * rf_probs + 0.4 * xgb_probs
    max_probs_b2 = np.max(b2_probs, axis=1)

    # Indicator masks for Known Control at both operating points
    known_indicators_op1 = {}
    known_indicators_op2 = {}
    known_control_op1 = {}
    known_control_op2 = {}

    for idx, cname in enumerate(class_names):
        mask = (y_test == idx)
        n_c = int(mask.sum())

        # OP 1
        flag_ae_1 = (ae_scores_test[mask] > thresh_op1)
        flag_conf_1 = (max_probs_b2[mask] < conf_op1)
        flag_either_1 = (flag_ae_1 | flag_conf_1)
        known_indicators_op1[cname] = flag_either_1
        known_control_op1[cname] = {
            "n_test": n_c,
            "flagged_unknown_by_ae": wilson_score_interval(int(flag_ae_1.sum()), n_c),
            "flagged_unknown_by_confidence": wilson_score_interval(int(flag_conf_1.sum()), n_c),
            "flagged_unknown_either": wilson_score_interval(int(flag_either_1.sum()), n_c),
        }

        # OP 2
        flag_ae_2 = (ae_scores_test[mask] > thresh_op2)
        flag_conf_2 = (max_probs_b2[mask] < conf_op2)
        flag_either_2 = (flag_ae_2 | flag_conf_2)
        known_indicators_op2[cname] = flag_either_2
        known_control_op2[cname] = {
            "n_test": n_c,
            "flagged_unknown_by_ae": wilson_score_interval(int(flag_ae_2.sum()), n_c),
            "flagged_unknown_by_confidence": wilson_score_interval(int(flag_conf_2.sum()), n_c),
            "flagged_unknown_either": wilson_score_interval(int(flag_either_2.sum()), n_c),
        }

    # Step FIX 1 & 2b: Leave-One-Class-Out Evaluation
    train_path = PROJECT_ROOT / "data" / "processed" / "train_unsmoted_19.parquet"
    if not train_path.exists():
        train_path = PROJECT_ROOT / "data" / "processed" / "train_phase2.parquet"
    df_train = pd.read_parquet(train_path)
    tr_label_col = "label_8" if "label_8" in df_train.columns else "label"

    leave_one_out_op1 = {}
    leave_one_out_op2 = {}
    lift_table_op1 = {}
    lift_table_op2 = {}
    truncated_classes = []

    print("\n" + "=" * 80)
    print("  STEP 2B: LEAVE-ONE-CLASS-OUT EVALUATION & LIFT VS KNOWN CONTROL")
    print("=" * 80)

    for cname in attack_classes:
        class_json_path = RESULTS_DIR / f"openset_{cname}.json"
        cidx = int(np.where(le_8.classes_ == cname)[0][0])
        test_mask = (y_test == cidx)
        X_heldout = X_test[test_mask]
        n_heldout = len(X_heldout)
        ae_scores_heldout = ae_scores_test[test_mask]

        t_start = time.time()
        # Train fresh RF excluding held-out class
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

        if scaler is not None and hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(selected_features):
            X_tr_sub = scaler.transform(X_tr_sub).astype(np.float32)

        rf_fresh = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_fresh.fit(X_tr_sub, y_tr_sub)

        probs_fresh = rf_fresh.predict_proba(X_heldout)
        max_probs_fresh = np.max(probs_fresh, axis=1)
        pred_classes_fresh = rf_fresh.predict(X_heldout)

        elapsed = time.time() - t_start
        is_truncated = bool(elapsed > 360.0)
        if is_truncated:
            truncated_classes.append(cname)

        # Evaluation at OP 1 (Original: 95th, 0.60)
        f_ae_1 = (ae_scores_heldout > thresh_op1)
        f_conf_1 = (max_probs_fresh < conf_op1)
        f_either_1 = (f_ae_1 | f_conf_1)
        s_benign_1 = (~f_either_1) & (pred_classes_fresh == benign_idx)
        s_other_1 = (~f_either_1) & (pred_classes_fresh != benign_idx)

        res_entry_op1 = {
            "class_name": cname,
            "n_test": int(n_heldout),
            "flagged_unknown_by_ae": wilson_score_interval(int(f_ae_1.sum()), n_heldout),
            "flagged_unknown_by_confidence": wilson_score_interval(int(f_conf_1.sum()), n_heldout),
            "flagged_unknown_either": wilson_score_interval(int(f_either_1.sum()), n_heldout),
            "silently_predicted_benign": wilson_score_interval(int(s_benign_1.sum()), n_heldout),
            "silently_predicted_other_attack": wilson_score_interval(int(s_other_1.sum()), n_heldout),
            "missed_rate_percentage": round(float((~f_either_1).mean() * 100.0), 2),
            "elapsed_seconds": round(elapsed, 2),
            "truncated": is_truncated,
        }
        leave_one_out_op1[cname] = res_entry_op1

        # Lift calculation for OP 1 (paired bootstrap against known control indicator)
        known_ind_1 = known_indicators_op1[cname]
        lift_entry_op1 = bootstrap_difference_ci(f_either_1, known_ind_1, n_resamples=1000, seed=42)
        lift_table_op1[cname] = lift_entry_op1

        # Evaluation at OP 2 (FP-Budget: 97.5th, 0.40)
        f_ae_2 = (ae_scores_heldout > thresh_op2)
        f_conf_2 = (max_probs_fresh < conf_op2)
        f_either_2 = (f_ae_2 | f_conf_2)
        s_benign_2 = (~f_either_2) & (pred_classes_fresh == benign_idx)
        s_other_2 = (~f_either_2) & (pred_classes_fresh != benign_idx)

        res_entry_op2 = {
            "class_name": cname,
            "n_test": int(n_heldout),
            "flagged_unknown_by_ae": wilson_score_interval(int(f_ae_2.sum()), n_heldout),
            "flagged_unknown_by_confidence": wilson_score_interval(int(f_conf_2.sum()), n_heldout),
            "flagged_unknown_either": wilson_score_interval(int(f_either_2.sum()), n_heldout),
            "silently_predicted_benign": wilson_score_interval(int(s_benign_2.sum()), n_heldout),
            "silently_predicted_other_attack": wilson_score_interval(int(s_other_2.sum()), n_heldout),
            "missed_rate_percentage": round(float((~f_either_2).mean() * 100.0), 2),
            "elapsed_seconds": round(elapsed, 2),
            "truncated": is_truncated,
        }
        leave_one_out_op2[cname] = res_entry_op2

        # Lift calculation for OP 2
        known_ind_2 = known_indicators_op2[cname]
        lift_entry_op2 = bootstrap_difference_ci(f_either_2, known_ind_2, n_resamples=1000, seed=42)
        lift_table_op2[cname] = lift_entry_op2

        # Save per-class JSON
        per_class_data = {
            "class_name": cname,
            "n_test": int(n_heldout),
            "operating_point_original": res_entry_op1,
            "operating_point_fp_budget": res_entry_op2,
            "lift_original": lift_entry_op1,
            "lift_fp_budget": lift_entry_op2,
            "auroc": auroc_results[cname],
        }
        with open(class_json_path, "w", encoding="utf-8") as f:
            json.dump(per_class_data, f, indent=2)

        print(f"[{cname}] (n={n_heldout:,}):")
        print(f"  OP1 Combined Flagged : {res_entry_op1['flagged_unknown_either']['formatted']}")
        print(f"  OP1 Lift vs Control  : {lift_entry_op1['formatted']}")
        print(f"  OP2 Combined Flagged : {res_entry_op2['flagged_unknown_either']['formatted']}")
        print(f"  OP2 Lift vs Control  : {lift_entry_op2['formatted']}")
        print(f"  AUROC (AE-only)      : {auroc_results[cname]['formatted']}")

    # Assemble comprehensive openset_summary.json
    summary_data = {
        "auroc": auroc_results,
        "lift": lift_table_op1,
        "lift_fp_budget": lift_table_op2,
        "known_class_control": known_control_op1,
        "known_class_control_fp_budget": known_control_op2,
        "leave_one_class_out": leave_one_out_op1,
        "leave_one_class_out_fp_budget": leave_one_out_op2,
        "operating_point_original": {
            "percentile": 95.0,
            "threshold": thresh_op1,
            "confidence_cutoff": conf_op1,
            "test_benign_fpr": known_control_op1["Benign"]["flagged_unknown_either"]["percentage"],
        },
        "operating_point_fp_budget": {
            "percentile": chosen_op["percentile"],
            "threshold": thresh_op2,
            "confidence_cutoff": conf_op2,
            "val_benign_fpr": chosen_op["val_benign_fpr"],
            "val_mean_attack_rate": chosen_op["val_mean_attack_rate"],
            "test_benign_fpr": known_control_op2["Benign"]["flagged_unknown_either"]["percentage"],
        },
        "validation_grid_search": val_grid_data,
        "truncated_classes": truncated_classes,
    }

    summary_path = RESULTS_DIR / "openset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    log.info(f"Updated openset_summary.json saved to {summary_path}")
    print("\n[OPEN-SET EVALUATION FIXES COMPLETE]")


if __name__ == "__main__":
    main()
