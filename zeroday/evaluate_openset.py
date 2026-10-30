#!/usr/bin/env python3
"""
zeroday/evaluate_openset.py - Open-set evaluation with known-class control and leave-one-class-out attacks.
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
from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
        "formatted": f"{x} of {n} ({pct:.2f}% [{lower:.2f}%, {upper:.2f}%])",
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

    return selected_features, le_8, class_names, benign_idx, ae, threshold, scaler


def run_known_class_control(
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    ae: nn.Module,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    log.info("Running Step 2a: Known-Class Control using existing B2 ensemble ...")
    rf_path = PROJECT_ROOT / "models" / "rf_8class_b2.joblib"
    xgb_path = PROJECT_ROOT / "models" / "xgb_8class_b2.joblib"

    if not rf_path.exists() or not xgb_path.exists():
        raise FileNotFoundError("Missing models/rf_8class_b2.joblib or xgb_8class_b2.joblib")

    rf = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)

    rf_probs = rf.predict_proba(X_test)
    xgb_probs = xgb.predict_proba(X_test)
    b2_probs = 0.6 * rf_probs + 0.4 * xgb_probs
    max_probs = np.max(b2_probs, axis=1)

    with torch.no_grad():
        x_tensor = torch.from_numpy(X_test.copy())
        reconstructions = ae(x_tensor)
        ae_scores = ((x_tensor - reconstructions) ** 2).mean(dim=1).cpu().numpy()

    is_flagged_unknown = (ae_scores > threshold) | (max_probs < 0.60)

    control_results = {}
    print("\n" + "=" * 70)
    print("      STEP 2A: KNOWN-CLASS CONTROL (FULL TEST SET EVALUATION)")
    print("=" * 70)
    print(f"{'Class Name':<14} | {'Flagged Unknown (95% Wilson CI)':<45}")
    print("-" * 70)

    for idx, cname in enumerate(class_names):
        mask = (y_test == idx)
        n_total = int(mask.sum())
        x_flagged = int(is_flagged_unknown[mask].sum())
        stat = wilson_score_interval(x_flagged, n_total)
        control_results[cname] = stat
        print(f"{cname:<14} | {stat['formatted']:<45}")

    print("=" * 70 + "\n")
    return control_results


def main():
    parser = argparse.ArgumentParser(description="Phase 10 Open-Set Evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume from existing per-class JSON files")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    selected_features, le_8, class_names, benign_idx, ae, threshold, scaler = load_artifacts_and_ae()

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

    # 2a. Known-Class Control
    known_control = run_known_class_control(X_test, y_test, class_names, ae, threshold)

    # 2b. Leave-One-Class-Out
    # 7 attack classes
    attack_classes = [c for c in class_names if c != "Benign"]

    # Load unsmoted real training data
    train_path = PROJECT_ROOT / "data" / "processed" / "train_unsmoted_19.parquet"
    if not train_path.exists():
        train_path = PROJECT_ROOT / "data" / "processed" / "train_phase2.parquet"
    if not train_path.exists():
        raise FileNotFoundError("Could not find unsmoted train parquet")

    log.info(f"Loading training data from {train_path.name} ...")
    df_train = pd.read_parquet(train_path)
    tr_label_col = "label_8" if "label_8" in df_train.columns else "label"

    leave_one_out_results = {}
    truncated_classes = []

    print("=" * 85)
    print("      STEP 2B: LEAVE-ONE-CLASS-OUT EVALUATION (7 UNSEEN ATTACK CLASSES)")
    print("=" * 85)

    for cname in attack_classes:
        class_json_path = RESULTS_DIR / f"openset_{cname}.json"
        if args.resume and class_json_path.exists():
            log.info(f"[RESUME] Loading completed result for {cname} from {class_json_path.name}")
            with open(class_json_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            leave_one_out_results[cname] = c_data
            continue

        cidx = int(np.where(le_8.classes_ == cname)[0][0])
        log.info(f"\nEvaluating held-out class: {cname} (label index: {cidx}) ...")
        t_start = time.time()

        # Filter out held-out class from training data
        df_sub = df_train[df_train[tr_label_col] != cidx].copy()

        # Stratified sampling capped at 50,000 rows total
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

        # Train fresh RF (100 trees, max_depth 15, class_weight balanced, seed 42)
        rf_fresh = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_fresh.fit(X_tr_sub, y_tr_sub)

        # Evaluate on test rows of held-out class
        test_mask = (y_test == cidx)
        X_heldout = X_test[test_mask]
        n_heldout = len(X_heldout)

        probs = rf_fresh.predict_proba(X_heldout)
        max_probs = np.max(probs, axis=1)
        pred_classes = rf_fresh.predict(X_heldout)

        with torch.no_grad():
            x_heldout_t = torch.from_numpy(X_heldout.copy())
            reconstructions = ae(x_heldout_t)
            ae_scores = ((x_heldout_t - reconstructions) ** 2).mean(dim=1).cpu().numpy()

        flagged_ae = (ae_scores > threshold)
        flagged_conf = (max_probs < 0.60)
        flagged_either = (flagged_ae | flagged_conf)

        silently_benign = (~flagged_either) & (pred_classes == benign_idx)
        silently_other = (~flagged_either) & (pred_classes != benign_idx)

        elapsed = time.time() - t_start

        # Check time limit (6 min = 360 s)
        is_truncated = elapsed > 360.0
        if is_truncated:
            truncated_classes.append(cname)
            log.warning(f"Class {cname} exceeded 6 min limit ({elapsed:.1f}s). Marked truncated.")

        res_entry = {
            "class_name": cname,
            "n_test": int(n_heldout),
            "flagged_unknown_by_ae": wilson_score_interval(int(flagged_ae.sum()), n_heldout),
            "flagged_unknown_by_confidence": wilson_score_interval(int(flagged_conf.sum()), n_heldout),
            "flagged_unknown_either": wilson_score_interval(int(flagged_either.sum()), n_heldout),
            "silently_predicted_benign": wilson_score_interval(int(silently_benign.sum()), n_heldout),
            "silently_predicted_other_attack": wilson_score_interval(int(silently_other.sum()), n_heldout),
            "elapsed_seconds": round(elapsed, 2),
            "truncated": is_truncated,
        }

        # Write per-class json
        with open(class_json_path, "w", encoding="utf-8") as f:
            json.dump(res_entry, f, indent=2)

        leave_one_out_results[cname] = res_entry

        print(f"[{cname}] (n={n_heldout:,}, time={elapsed:.2f}s):")
        print(f"  Flagged Unknown (Either)    : {res_entry['flagged_unknown_either']['formatted']}")
        print(f"  Flagged by AE               : {res_entry['flagged_unknown_by_ae']['formatted']}")
        print(f"  Flagged by Uncertainty (<60): {res_entry['flagged_unknown_by_confidence']['formatted']}")
        print(f"  Silently Predicted Benign   : {res_entry['silently_predicted_benign']['formatted']}")
        print(f"  Silently Other Attack       : {res_entry['silently_predicted_other_attack']['formatted']}")

    # 2c. Write summary JSON
    summary_data = {
        "known_class_control": known_control,
        "leave_one_class_out": leave_one_out_results,
        "truncated_classes": truncated_classes,
    }

    summary_path = RESULTS_DIR / "openset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    log.info(f"Open-set evaluation summary saved to {summary_path}")
    print("\n[OPEN-SET EVALUATION] Complete.")


if __name__ == "__main__":
    main()
