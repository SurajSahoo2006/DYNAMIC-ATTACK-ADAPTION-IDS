"""
adversarial/run_phase4_adversarial.py
=====================================
Phase 4 - Adversarial Robustness Assessment (Staged Execution Suite)

Evaluates the B2 Ensemble Classifier (RF 0.60 + XGBoost 0.40 + PyTorch DNN 0.00) under
strict physical domain constraints, scientific controls, and statistical rigor.

Supported Stages:
  --stage fgsm     : Zero-attack control, FGSM transfer across epsilons vs Random Noise (500 flows)
  --stage hsj      : HopSkipJump black-box decision attack (Budget 1: 100 flows, Budget 2: 50 flows)
  --stage zoo      : ZOO black-box score attack (Budget 1 & 2 on 40 flows)
  --stage defense  : 4-Arm comparison across Clean, FGSM, Random Noise, HSJ (50 flows/arm), and Held-out
  --stage combine  : Merge stage outputs into reports/adversarial_metrics.json and generate plot
  --stage all      : Execute all stages sequentially

Each stage immediately writes its own JSON file to reports/ upon completion and
logs progress every 10 flows to logs/adversarial_run.log.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from xgboost import XGBClassifier

from art.attacks.evasion import FastGradientMethod, HopSkipJump, ZooAttack
from art.estimators.classification import BlackBoxClassifierNeuralNetwork, PyTorchClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ARTIFACTS_DIR,
    ENCODER_8_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SCALER_PATH,
    TEST_PARQUET,
    TRAIN_PARQUET,
)
from adversarial.constraints import AdversarialConstraintEnforcer
from orchestrator.llm_orchestrator import FeatureUnscaler
from predict import get_or_load_model, get_selected_features

RUN_LOG_PATH = Path("logs/adversarial_run.log")
RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_progress(stage: str, current: int, total: int, t_start: float, extra: str = ""):
    """Logs progress to console and logs/adversarial_run.log."""
    elapsed = time.time() - t_start
    rate = current / elapsed if elapsed > 0 else 0.0
    msg = (
        f"[{time.strftime('%H:%M:%S')}] [{stage:<7}] "
        f"Flows {current:>3}/{total:<3} ({current / total * 100:>5.1f}%) | "
        f"Elapsed: {elapsed:>6.1f}s | Rate: {rate:>4.2f} flows/s"
    )
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes two-sided Wilson score confidence interval."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denom = 1.0 + (z**2) / n
    centre = (p + (z**2) / (2 * n)) / denom
    delta = (z * np.sqrt((p * (1.0 - p) / n) + ((z**2) / (4 * (n**2))))) / denom
    return float(max(0.0, centre - delta)), float(min(1.0, centre + delta))


def format_count_and_ci(k: int, n: int) -> str:
    """Formats count, percentage, and 95% Wilson confidence interval."""
    if n == 0:
        return "0/0 (0.0% [0.0%, 0.0%])"
    pct = (k / n) * 100.0
    ci_l, ci_u = wilson_ci(k, n)
    return f"{k:>3}/{n:<3} ({pct:>5.1f}% [{ci_l * 100:>4.1f}%, {ci_u * 100:>4.1f}%])"


def sample_stratified_subset(df: pd.DataFrame, n_samples: int = 500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Samples stratified flows ensuring small classes are represented."""
    rng = np.random.default_rng(seed)
    class_counts = df["label_8"].value_counts()
    total = len(df)
    indices = []
    for label, count in class_counts.items():
        quota = max(5, int(round((count / total) * n_samples)))
        sub = df[df["label_8"] == label]
        chosen = rng.choice(sub.index, size=min(len(sub), quota), replace=False)
        indices.extend(chosen.tolist())
    if len(indices) > n_samples:
        indices = rng.choice(indices, size=n_samples, replace=False).tolist()
    return df.loc[indices].copy()


def get_fast_ensemble_predictor(ensemble) -> Callable[[np.ndarray], np.ndarray]:
    """
    Direct NumPy prediction wrapper for B2 Ensemble (RF 0.60 + XGB 0.40).
    Avoids pd.DataFrame allocation on every query and bypasses DNN when weight is 0.0.
    Enforces n_jobs=1 on RF to eliminate multi-threading pool dispatch latency on small queries.
    """
    w_rf = float(ensemble.weights["rf"])
    w_xgb = float(ensemble.weights["xgb"])
    w_dnn = float(ensemble.weights.get("dnn", 0.0))
    rf_m = ensemble.rf_model
    xgb_m = ensemble.xgb_model
    dnn_m = ensemble.dnn_model

    if hasattr(rf_m, "n_jobs"):
        rf_m.n_jobs = 1

    def predict_probs_fast(x: np.ndarray) -> np.ndarray:
        x_arr = np.ascontiguousarray(x, dtype=np.float32)
        p_rf = rf_m.predict_proba(x_arr)
        p_xgb = xgb_m.predict_proba(x_arr)
        if w_dnn > 0.0 and dnn_m is not None:
            p_dnn = dnn_m.predict_proba(x_arr)
            return (w_rf * p_rf + w_xgb * p_xgb + w_dnn * p_dnn).astype(np.float32)
        return (w_rf * p_rf + w_xgb * p_xgb).astype(np.float32)

    return predict_probs_fast


def load_shared_environment():
    """Loads feature definitions, encoders, scalers, enforcer, and test set."""
    le_8 = joblib.load(ENCODER_8_PATH)
    class_names = list(le_8.classes_)
    selected_features = get_selected_features()
    scaler = joblib.load(SCALER_PATH)
    unscaler = FeatureUnscaler()
    enforcer = AdversarialConstraintEnforcer(selected_features, unscaler, scaler)
    df_test = pd.read_parquet(TEST_PARQUET)
    ensemble = get_or_load_model(target="8class", model_type="b2")
    return le_8, class_names, selected_features, scaler, unscaler, enforcer, df_test, ensemble


# ======================================================================
# STAGE 1: FGSM White-Box Surrogate Transfer & Random Noise Control
# ======================================================================
def run_stage_fgsm(fgsm_samples: int = 500) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  STAGE 1: FGSM TRANSFER & RANDOM NOISE CONTROL (500 flows)")
    print("=" * 80)
    t_start = time.time()
    _, class_names, selected_features, _, _, enforcer, df_test, ensemble = load_shared_environment()

    # PyTorch surrogate
    dnn_model = ensemble.dnn_model
    dnn_model.eval()
    criterion = nn.CrossEntropyLoss()
    classifier_dnn = PyTorchClassifier(
        model=dnn_model, loss=criterion, input_shape=(19,), nb_classes=8, clip_values=None
    )

    df_fgsm = sample_stratified_subset(df_test, n_samples=fgsm_samples, seed=RANDOM_SEED)
    X_clean = df_fgsm[selected_features].to_numpy(dtype=np.float32)
    y_clean = df_fgsm["label_8"].to_numpy(dtype=np.int64)

    # Zero-Attack Control (pipeline check on epsilon = 0.0)
    X_zero = enforcer.enforce_constraints(X_clean, X_clean)
    diff_zero = float(np.max(np.abs(X_zero - X_clean)))

    preds_dnn_clean = np.argmax(classifier_dnn.predict(X_clean), axis=1)
    k_dnn_clean = int(np.sum(preds_dnn_clean == y_clean))
    acc_dnn_clean = float(accuracy_score(y_clean, preds_dnn_clean))

    preds_dnn_zero = np.argmax(classifier_dnn.predict(X_zero), axis=1)
    acc_dnn_zero = float(accuracy_score(y_clean, preds_dnn_zero))

    fast_predict = get_fast_ensemble_predictor(ensemble)
    prob_ens_clean = fast_predict(X_clean)
    preds_ens_clean = np.argmax(prob_ens_clean, axis=1)
    k_ens_clean = int(np.sum(preds_ens_clean == y_clean))
    acc_ens_clean = float(accuracy_score(y_clean, preds_ens_clean))

    prob_ens_zero = fast_predict(X_zero)
    preds_ens_zero = np.argmax(prob_ens_zero, axis=1)
    acc_ens_zero = float(accuracy_score(y_clean, preds_ens_zero))

    print(f"  --- Zero-Attack Pipeline Invariance Control (eps = 0.0) ---")
    print(f"  Max Numerical Drift (|X_pipe - X_clean|): {diff_zero:.6f}")
    print(f"  DNN Clean Acc:         Raw={acc_dnn_clean * 100:.2f}% | Pipeline={acc_dnn_zero * 100:.2f}% ({k_dnn_clean}/{len(y_clean)})")
    print(f"  B2 Ensemble Clean Acc:  Raw={acc_ens_clean * 100:.2f}% | Pipeline={acc_ens_zero * 100:.2f}% ({k_ens_clean}/{len(y_clean)})")

    rng = np.random.default_rng(RANDOM_SEED)
    epsilons = [0.01, 0.05, 0.10, 0.20, 0.30]
    fgsm_results = []
    saved_adv_flows = None

    print("\n  Evaluating Evasion vs Epsilon (FGSM Transfer vs Random Noise Control):")
    print("  " + "-" * 88)
    print("  Eps    | Mean L2 | DNN FGSM | Ens FGSM (Count & 95% CI)        | Ens Rand (Count & 95% CI)")
    print("  " + "-" * 88)

    for eps in epsilons:
        t_eps = time.time()
        attack_fgsm = FastGradientMethod(estimator=classifier_dnn, eps=eps)
        X_adv_raw = attack_fgsm.generate(x=X_clean)
        X_adv_constrained = enforcer.enforce_constraints(X_adv_raw, X_clean)

        sample_l2 = np.linalg.norm(X_adv_constrained - X_clean, axis=1, keepdims=True)
        mean_l2_fgsm = float(np.mean(sample_l2))

        preds_dnn_adv = np.argmax(classifier_dnn.predict(X_adv_constrained), axis=1)
        acc_dnn_adv = float(accuracy_score(y_clean, preds_dnn_adv))

        prob_ens_adv = fast_predict(X_adv_constrained)
        preds_ens_adv = np.argmax(prob_ens_adv, axis=1)
        k_ens_adv = int(np.sum(preds_ens_adv == y_clean))
        acc_ens_adv = float(accuracy_score(y_clean, preds_ens_adv))

        correct_mask = (preds_ens_clean == y_clean)
        n_correct_clean = int(np.sum(correct_mask))
        n_transfer_success = int(np.sum(preds_ens_adv[correct_mask] != y_clean[correct_mask]))
        asr_ens = float(n_transfer_success / n_correct_clean) if n_correct_clean > 0 else 0.0

        # Random Noise Control (isotropic Gaussian, same L2 magnitude)
        v_rand = rng.standard_normal(size=X_clean.shape).astype(np.float32)
        v_rand[:, enforcer.fixed_indices] = 0.0
        v_norm = np.linalg.norm(v_rand, axis=1, keepdims=True) + 1e-12
        v_scaled = v_rand * (sample_l2 / v_norm)
        X_rand_constrained = enforcer.enforce_constraints(X_clean + v_scaled, X_clean)
        mean_l2_rand = float(np.mean(np.linalg.norm(X_rand_constrained - X_clean, axis=1)))

        preds_ens_rand = np.argmax(fast_predict(X_rand_constrained), axis=1)
        k_ens_rand = int(np.sum(preds_ens_rand == y_clean))
        acc_ens_rand = float(accuracy_score(y_clean, preds_ens_rand))

        n_rand_success = int(np.sum(preds_ens_rand[correct_mask] != y_clean[correct_mask]))
        asr_ens_rand = float(n_rand_success / n_correct_clean) if n_correct_clean > 0 else 0.0

        fgsm_ci_str = format_count_and_ci(k_ens_adv, len(y_clean))
        rand_ci_str = format_count_and_ci(k_ens_rand, len(y_clean))
        print(f"  {eps:<6.2f} | {mean_l2_fgsm:<7.3f} | {acc_dnn_adv * 100:>6.2f}%  | {fgsm_ci_str:<32} | {rand_ci_str}")

        ci_ens_l, ci_ens_u = wilson_ci(k_ens_adv, len(y_clean))
        ci_rand_l, ci_rand_u = wilson_ci(k_ens_rand, len(y_clean))
        ci_asr_l, ci_asr_u = wilson_ci(n_transfer_success, n_correct_clean)

        fgsm_results.append({
            "epsilon": eps,
            "mean_l2_fgsm": round(mean_l2_fgsm, 4),
            "mean_l2_random": round(mean_l2_rand, 4),
            "dnn_clean_acc": round(acc_dnn_clean, 4),
            "dnn_adversarial_acc": round(acc_dnn_adv, 4),
            "ensemble_clean_acc": round(acc_ens_clean, 4),
            "ensemble_clean_count": f"{k_ens_clean}/{len(y_clean)}",
            "ensemble_adversarial_acc": round(acc_ens_adv, 4),
            "ensemble_adversarial_count": f"{k_ens_adv}/{len(y_clean)}",
            "ensemble_adversarial_ci_95": [round(ci_ens_l, 4), round(ci_ens_u, 4)],
            "transfer_asr": round(asr_ens, 4),
            "transfer_asr_count": f"{n_transfer_success}/{n_correct_clean}",
            "transfer_asr_ci_95": [round(ci_asr_l, 4), round(ci_asr_u, 4)],
            "ensemble_random_noise_acc": round(acc_ens_rand, 4),
            "ensemble_random_noise_count": f"{k_ens_rand}/{len(y_clean)}",
            "ensemble_random_noise_ci_95": [round(ci_rand_l, 4), round(ci_rand_u, 4)],
            "random_noise_asr": round(asr_ens_rand, 4),
            "random_noise_asr_count": f"{n_rand_success}/{n_correct_clean}",
            "runtime_sec": round(time.time() - t_eps, 2),
        })

        if eps == 0.10:
            saved_adv_flows = (df_fgsm.copy(), pd.DataFrame(X_adv_constrained, columns=selected_features), preds_ens_adv, y_clean)

    print("  " + "-" * 88)

    if saved_adv_flows is not None:
        clean_df, adv_df, adv_preds, y_clean_arr = saved_adv_flows
        adv_records = adv_df.copy()
        adv_records["true_class_idx"] = y_clean_arr
        adv_records["adversarial_pred_class_idx"] = adv_preds
        adv_records["true_class"] = [class_names[idx] for idx in y_clean_arr]
        adv_records["adversarial_pred_class"] = [class_names[idx] for idx in adv_preds]
        adv_records_path = ARTIFACTS_DIR / "adversarial_demo_flows.parquet"
        adv_records.to_parquet(adv_records_path, index=True)
        print(f"  [SAVE] Adversarial sample flows saved --> {adv_records_path.name}")

    out_data = {
        "zero_attack_control": {
            "samples": len(df_fgsm),
            "max_numerical_drift": diff_zero,
            "dnn_clean_raw_acc": round(acc_dnn_clean, 4),
            "dnn_clean_pipeline_acc": round(acc_dnn_zero, 4),
            "ensemble_clean_raw_acc": round(acc_ens_clean, 4),
            "ensemble_clean_pipeline_acc": round(acc_ens_zero, 4),
        },
        "fgsm_and_random_noise_results": fgsm_results,
        "runtime_sec": round(time.time() - t_start, 2),
    }

    out_file = REPORTS_DIR / "adversarial_fgsm.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"  [STAGE DONE] FGSM & Zero-Attack saved --> {out_file.name} ({out_data['runtime_sec']:.2f}s)")
    return out_data


# ======================================================================
# STAGE 2: HopSkipJump Decision-Based Attack (Multi-Budget)
# ======================================================================
def run_stage_hsj(hsj1_samples: int = 100, hsj2_samples: int = 50) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  STAGE 2: HOPSKIPJUMP DECISION-BASED ATTACK (Budget 1: 100 flows | Budget 2: 50 flows)")
    print("=" * 80)
    t_stage_start = time.time()
    _, _, selected_features, _, _, enforcer, df_test, ensemble = load_shared_environment()
    fast_predict = get_fast_ensemble_predictor(ensemble)

    classifier_ensemble = BlackBoxClassifierNeuralNetwork(
        predict_fn=fast_predict, input_shape=(19,), nb_classes=8
    )

    # Fixed 100 flows test set
    df_hsj = sample_stratified_subset(df_test, n_samples=hsj1_samples, seed=RANDOM_SEED + 10)
    X_hsj_clean = df_hsj[selected_features].to_numpy(dtype=np.float32)
    y_hsj_clean = df_hsj["label_8"].to_numpy(dtype=np.int64)

    prob_hsj_clean = fast_predict(X_hsj_clean)
    preds_hsj_clean = np.argmax(prob_hsj_clean, axis=1)
    k_hsj_clean = int(np.sum(preds_hsj_clean == y_hsj_clean))
    acc_hsj_clean = float(accuracy_score(y_hsj_clean, preds_hsj_clean))
    ci_hsj_cl, ci_hsj_cu = wilson_ci(k_hsj_clean, len(y_hsj_clean))

    print(f"  Clean Accuracy on HSJ test set (100 flows): {format_count_and_ci(k_hsj_clean, len(y_hsj_clean))}")

    hsj_results = {}

    # Budget 1: Low Budget (3 iters, 10 evals) on 100 flows
    print(f"\n  Running Budget 1 (Low: 3 iters, 10 evals) on {hsj1_samples} flows in chunks of 10 ...")
    t_b1 = time.time()
    attack_hsj1 = HopSkipJump(
        classifier=classifier_ensemble, targeted=False, max_iter=3, max_eval=10, init_eval=5, verbose=False
    )
    chunk_size = 10
    adv_chunks = []
    for i in range(0, len(X_hsj_clean), chunk_size):
        chunk = X_hsj_clean[i : i + chunk_size]
        adv_chunks.append(attack_hsj1.generate(x=chunk))
        log_progress("HSJ-B1", min(i + chunk_size, len(X_hsj_clean)), len(X_hsj_clean), t_b1)
    X_hsj1_raw = np.vstack(adv_chunks)
    X_hsj1_constrained = enforcer.enforce_constraints(X_hsj1_raw, X_hsj_clean)
    t_hsj1_total = time.time() - t_b1

    l2_b1 = float(np.mean(np.linalg.norm(X_hsj1_constrained - X_hsj_clean, axis=1)))
    preds_b1 = np.argmax(fast_predict(X_hsj1_constrained), axis=1)
    k_b1 = int(np.sum(preds_b1 == y_hsj_clean))
    acc_b1 = float(accuracy_score(y_hsj_clean, preds_b1))
    ci_b1_l, ci_b1_u = wilson_ci(k_b1, len(y_hsj_clean))

    m_b1 = (preds_hsj_clean == y_hsj_clean)
    n_corr_b1 = int(np.sum(m_b1))
    n_evaded_b1 = int(np.sum(preds_b1[m_b1] != y_hsj_clean[m_b1]))
    asr_b1 = float(n_evaded_b1 / n_corr_b1) if n_corr_b1 > 0 else 0.0
    ci_asr_b1_l, ci_asr_b1_u = wilson_ci(n_evaded_b1, n_corr_b1)

    print(f"  Budget 1 Results:")
    print(f"    Perturbed Acc: {format_count_and_ci(k_b1, len(y_hsj_clean))}")
    print(f"    Attack Success Rate: {format_count_and_ci(n_evaded_b1, n_corr_b1)} | Mean L2: {l2_b1:.4f} ({t_hsj1_total:.2f}s)")

    hsj_results["Budget 1 (Low: 3 iters, 10 evals)"] = {
        "samples": hsj1_samples,
        "max_iter": 3,
        "max_eval": 10,
        "clean_accuracy": round(acc_hsj_clean, 4),
        "clean_count": f"{k_hsj_clean}/{len(y_hsj_clean)}",
        "clean_ci_95": [round(ci_hsj_cl, 4), round(ci_hsj_cu, 4)],
        "adversarial_accuracy": round(acc_b1, 4),
        "adversarial_count": f"{k_b1}/{len(y_hsj_clean)}",
        "adversarial_ci_95": [round(ci_b1_l, 4), round(ci_b1_u, 4)],
        "attack_success_rate": round(asr_b1, 4),
        "attack_success_count": f"{n_evaded_b1}/{n_corr_b1}",
        "asr_ci_95": [round(ci_asr_b1_l, 4), round(ci_asr_b1_u, 4)],
        "mean_l2_perturbation": round(l2_b1, 4),
        "runtime_sec": round(t_hsj1_total, 2),
    }

    # Budget 2: High Budget (6 iters, 20 evals) on 50 flows subset
    X_hsj2_clean = X_hsj_clean[:hsj2_samples]
    y_hsj2_clean = y_hsj_clean[:hsj2_samples]
    preds_hsj2_clean = preds_hsj_clean[:hsj2_samples]
    k_hsj2_clean = int(np.sum(preds_hsj2_clean == y_hsj2_clean))
    acc_hsj2_clean = float(accuracy_score(y_hsj2_clean, preds_hsj2_clean))
    ci_hsj2_cl, ci_hsj2_cu = wilson_ci(k_hsj2_clean, len(y_hsj2_clean))

    print(f"\n  Running Budget 2 (High: 6 iters, 20 evals) on {hsj2_samples} flows in chunks of 10 ...")
    t_b2 = time.time()
    attack_hsj2 = HopSkipJump(
        classifier=classifier_ensemble, targeted=False, max_iter=6, max_eval=20, init_eval=8, verbose=False
    )
    adv_chunks2 = []
    for i in range(0, len(X_hsj2_clean), chunk_size):
        chunk = X_hsj2_clean[i : i + chunk_size]
        adv_chunks2.append(attack_hsj2.generate(x=chunk))
        log_progress("HSJ-B2", min(i + chunk_size, len(X_hsj2_clean)), len(X_hsj2_clean), t_b2)
    X_hsj2_raw = np.vstack(adv_chunks2)
    X_hsj2_constrained = enforcer.enforce_constraints(X_hsj2_raw, X_hsj2_clean)
    t_hsj2_total = time.time() - t_b2

    l2_b2 = float(np.mean(np.linalg.norm(X_hsj2_constrained - X_hsj2_clean, axis=1)))
    preds_b2 = np.argmax(fast_predict(X_hsj2_constrained), axis=1)
    k_b2 = int(np.sum(preds_b2 == y_hsj2_clean))
    acc_b2 = float(accuracy_score(y_hsj2_clean, preds_b2))
    ci_b2_l, ci_b2_u = wilson_ci(k_b2, len(y_hsj2_clean))

    m_b2 = (preds_hsj2_clean == y_hsj2_clean)
    n_corr_b2 = int(np.sum(m_b2))
    n_evaded_b2 = int(np.sum(preds_b2[m_b2] != y_hsj2_clean[m_b2]))
    asr_b2 = float(n_evaded_b2 / n_corr_b2) if n_corr_b2 > 0 else 0.0
    ci_asr_b2_l, ci_asr_b2_u = wilson_ci(n_evaded_b2, n_corr_b2)

    print(f"  Budget 2 Results ({hsj2_samples} flows):")
    print(f"    Perturbed Acc: {format_count_and_ci(k_b2, len(y_hsj2_clean))}")
    print(f"    Attack Success Rate: {format_count_and_ci(n_evaded_b2, n_corr_b2)} | Mean L2: {l2_b2:.4f} ({t_hsj2_total:.2f}s)")

    hsj_results["Budget 2 (High: 6 iters, 20 evals)"] = {
        "samples": hsj2_samples,
        "max_iter": 6,
        "max_eval": 20,
        "clean_accuracy": round(acc_hsj2_clean, 4),
        "clean_count": f"{k_hsj2_clean}/{len(y_hsj2_clean)}",
        "clean_ci_95": [round(ci_hsj2_cl, 4), round(ci_hsj2_cu, 4)],
        "adversarial_accuracy": round(acc_b2, 4),
        "adversarial_count": f"{k_b2}/{len(y_hsj2_clean)}",
        "adversarial_ci_95": [round(ci_b2_l, 4), round(ci_b2_u, 4)],
        "attack_success_rate": round(asr_b2, 4),
        "attack_success_count": f"{n_evaded_b2}/{n_corr_b2}",
        "asr_ci_95": [round(ci_asr_b2_l, 4), round(ci_asr_b2_u, 4)],
        "mean_l2_perturbation": round(l2_b2, 4),
        "runtime_sec": round(t_hsj2_total, 2),
    }

    out_file = REPORTS_DIR / "adversarial_hsj.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(hsj_results, f, indent=2)
    print(f"  [STAGE DONE] HopSkipJump metrics saved --> {out_file.name} (Total: {time.time() - t_stage_start:.2f}s)")
    return hsj_results


# ======================================================================
# STAGE 3: ZOO Score-Based Black-Box Attack (Multi-Budget)
# ======================================================================
def run_stage_zoo(zoo_samples: int = 40) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"  STAGE 3: ZOO SCORE-BASED ATTACK (Fixed {zoo_samples} flows, Multi-Budget)")
    print("=" * 80)
    t_stage_start = time.time()
    _, _, selected_features, _, _, enforcer, df_test, ensemble = load_shared_environment()
    fast_predict = get_fast_ensemble_predictor(ensemble)

    classifier_ensemble = BlackBoxClassifierNeuralNetwork(
        predict_fn=fast_predict, input_shape=(19,), nb_classes=8
    )

    df_zoo = sample_stratified_subset(df_test, n_samples=zoo_samples, seed=RANDOM_SEED + 20)
    X_zoo_clean = df_zoo[selected_features].to_numpy(dtype=np.float32)
    y_zoo_clean = df_zoo["label_8"].to_numpy(dtype=np.int64)

    prob_zoo_clean = fast_predict(X_zoo_clean)
    preds_zoo_clean = np.argmax(prob_zoo_clean, axis=1)
    k_zoo_clean = int(np.sum(preds_zoo_clean == y_zoo_clean))
    acc_zoo_clean = float(accuracy_score(y_zoo_clean, preds_zoo_clean))
    ci_zoo_cl, ci_zoo_cu = wilson_ci(k_zoo_clean, len(y_zoo_clean))

    print(f"  Clean Accuracy on ZOO test set ({zoo_samples} flows): {format_count_and_ci(k_zoo_clean, len(y_zoo_clean))}")

    zoo_results = {}
    chunk_size = 10

    for budget_name, max_iter in [
        ("Budget 1 (Low: 4 iters)", 4),
        ("Budget 2 (High: 8 iters)", 8),
    ]:
        print(f"\n  Running {budget_name} on {zoo_samples} flows in chunks of 10 ...")
        t_z = time.time()
        attack_zoo = ZooAttack(
            classifier=classifier_ensemble,
            targeted=False,
            max_iter=max_iter,
            learning_rate=0.05,
            batch_size=1,
            nb_parallel=min(10, len(selected_features)),
            verbose=False,
        )
        adv_chunks = []
        for i in range(0, len(X_zoo_clean), chunk_size):
            chunk = X_zoo_clean[i : i + chunk_size]
            adv_chunks.append(attack_zoo.generate(x=chunk))
            log_progress(f"ZOO-{max_iter}", min(i + chunk_size, len(X_zoo_clean)), len(X_zoo_clean), t_z)
        X_zoo_raw = np.vstack(adv_chunks)
        X_zoo_constrained = enforcer.enforce_constraints(X_zoo_raw, X_zoo_clean)
        t_zoo = time.time() - t_z

        l2_zoo = float(np.mean(np.linalg.norm(X_zoo_constrained - X_zoo_clean, axis=1)))
        preds_zoo_adv = np.argmax(fast_predict(X_zoo_constrained), axis=1)
        k_zoo_adv = int(np.sum(preds_zoo_adv == y_zoo_clean))
        acc_zoo_adv = float(accuracy_score(y_zoo_clean, preds_zoo_adv))
        ci_zoo_l, ci_zoo_u = wilson_ci(k_zoo_adv, len(y_zoo_clean))

        m_zoo = (preds_zoo_clean == y_zoo_clean)
        n_corr_z = int(np.sum(m_zoo))
        n_evaded_z = int(np.sum(preds_zoo_adv[m_zoo] != y_zoo_clean[m_zoo]))
        asr_zoo = float(n_evaded_z / n_corr_z) if n_corr_z > 0 else 0.0
        ci_asr_zl, ci_asr_zu = wilson_ci(n_evaded_z, n_corr_z)

        print(f"  {budget_name} Results:")
        print(f"    Perturbed Acc: {format_count_and_ci(k_zoo_adv, len(y_zoo_clean))}")
        print(f"    Attack Success Rate: {format_count_and_ci(n_evaded_z, n_corr_z)} | Mean L2: {l2_zoo:.4f} ({t_zoo:.2f}s)")

        zoo_results[budget_name] = {
            "samples": zoo_samples,
            "max_iter": max_iter,
            "clean_accuracy": round(acc_zoo_clean, 4),
            "clean_count": f"{k_zoo_clean}/{len(y_zoo_clean)}",
            "clean_ci_95": [round(ci_zoo_cl, 4), round(ci_zoo_cu, 4)],
            "adversarial_accuracy": round(acc_zoo_adv, 4),
            "adversarial_count": f"{k_zoo_adv}/{len(y_zoo_clean)}",
            "adversarial_ci_95": [round(ci_zoo_l, 4), round(ci_zoo_u, 4)],
            "attack_success_rate": round(asr_zoo, 4),
            "attack_success_count": f"{n_evaded_z}/{n_corr_z}",
            "asr_ci_95": [round(ci_asr_zl, 4), round(ci_asr_zu, 4)],
            "mean_l2_perturbation": round(l2_zoo, 4),
            "runtime_sec": round(t_zoo, 2),
        }

    out_file = REPORTS_DIR / "adversarial_zoo.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(zoo_results, f, indent=2)
    print(f"  [STAGE DONE] ZOO metrics saved --> {out_file.name} (Total: {time.time() - t_stage_start:.2f}s)")
    return zoo_results


# ======================================================================
# STAGE 4: 4-Arm Defense Benchmark Across 4 Conditions + HSJ (50 flows/arm)
# ======================================================================
def run_stage_defense(test_samples: int = 500, hsj_defense_samples: int = 50) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"  STAGE 4: 4-ARM DEFENSE MATRIX (500 Fresh Flows, HSJ {hsj_defense_samples} flows/arm)")
    print("=" * 80)
    t_stage_start = time.time()
    _, _, selected_features, _, _, enforcer, df_test, ensemble = load_shared_environment()
    fast_b2_predict = get_fast_ensemble_predictor(ensemble)

    # PyTorch surrogate for training set FGSM generation
    dnn_model = ensemble.dnn_model
    dnn_model.eval()
    criterion = nn.CrossEntropyLoss()
    classifier_dnn = PyTorchClassifier(
        model=dnn_model, loss=criterion, input_shape=(19,), nb_classes=8, clip_values=None
    )

    df_train = pd.read_parquet(TRAIN_PARQUET)

    # Training slice 1: 3,000 clean training flows
    df_tr_3k = sample_stratified_subset(df_train, n_samples=3000, seed=RANDOM_SEED + 30)
    X_tr_3k = df_tr_3k[selected_features].to_numpy(dtype=np.float32)
    y_tr_3k = df_tr_3k["label_8"].to_numpy(dtype=np.int64)

    # Training slice 2: 6,000 clean training flows (volume control)
    df_tr_6k = sample_stratified_subset(df_train, n_samples=6000, seed=RANDOM_SEED + 31)
    X_tr_6k = df_tr_6k[selected_features].to_numpy(dtype=np.float32)
    y_tr_6k = df_tr_6k["label_8"].to_numpy(dtype=np.int64)

    # Generate FGSM perturbations on 3k training flows
    print("  Generating FGSM perturbations on 3,000 training flows (eps=0.10) ...")
    attack_tr_fgsm = FastGradientMethod(estimator=classifier_dnn, eps=0.10)
    X_tr_adv_raw = attack_tr_fgsm.generate(x=X_tr_3k)
    X_tr_adv = enforcer.enforce_constraints(X_tr_adv_raw, X_tr_3k)

    X_combined = np.vstack([X_tr_3k, X_tr_adv])
    y_combined = np.concatenate([y_tr_3k, y_tr_3k])

    print("  Fitting Arm B: Retrained 1x Clean (3,000 flows) ...")
    rf_b = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)
    rf_b.fit(X_tr_3k, y_tr_3k)
    xgb_b = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=1)
    xgb_b.fit(X_tr_3k, y_tr_3k)

    print("  Fitting Arm C: Retrained 2x Clean Volume Control (6,000 flows) ...")
    rf_c = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)
    rf_c.fit(X_tr_6k, y_tr_6k)
    xgb_c = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=1)
    xgb_c.fit(X_tr_6k, y_tr_6k)

    print("  Fitting Arm D: Retrained Clean + Adversarial (3k clean + 3k FGSM) ...")
    rf_d = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)
    rf_d.fit(X_combined, y_combined)
    xgb_d = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=1)
    xgb_d.fit(X_combined, y_combined)

    # Fresh test slice: 500 flows
    df_fresh_test = sample_stratified_subset(df_test, n_samples=test_samples, seed=RANDOM_SEED + 99)
    X_te_clean = df_fresh_test[selected_features].to_numpy(dtype=np.float32)
    y_te = df_fresh_test["label_8"].to_numpy(dtype=np.int64)

    # Condition 1: Clean (pipeline passed)
    X_te_clean_pipe = enforcer.enforce_constraints(X_te_clean, X_te_clean)

    # Condition 2: Fresh FGSM eps=0.10
    X_te_fgsm_raw = FastGradientMethod(estimator=classifier_dnn, eps=0.10).generate(x=X_te_clean)
    X_te_fgsm = enforcer.enforce_constraints(X_te_fgsm_raw, X_te_clean)

    # Condition 3: Random Noise matching FGSM eps=0.10 L2
    l2_te_fgsm = np.linalg.norm(X_te_fgsm - X_te_clean, axis=1, keepdims=True)
    rng = np.random.default_rng(RANDOM_SEED)
    v_rand_te = rng.standard_normal(size=X_te_clean.shape).astype(np.float32)
    v_rand_te[:, enforcer.fixed_indices] = 0.0
    v_norm_te = np.linalg.norm(v_rand_te, axis=1, keepdims=True) + 1e-12
    X_te_rand = enforcer.enforce_constraints(X_te_clean + v_rand_te * (l2_te_fgsm / v_norm_te), X_te_clean)

    # Condition 4: Held-out attack (unseen FGSM eps=0.20)
    X_te_fgsm020 = enforcer.enforce_constraints(
        FastGradientMethod(estimator=classifier_dnn, eps=0.20).generate(x=X_te_clean), X_te_clean
    )

    # Condition 5: HopSkipJump Budget 1 (3 iters, 10 evals) evaluated on 50 flows per arm
    X_te_hsj_sub = X_te_clean[:hsj_defense_samples]
    y_te_hsj_sub = y_te[:hsj_defense_samples]

    def make_arm_predictor(rf_m, xgb_m):
        def pred_fn(x: np.ndarray) -> np.ndarray:
            x_arr = np.ascontiguousarray(x, dtype=np.float32)
            return (0.60 * rf_m.predict_proba(x_arr) + 0.40 * xgb_m.predict_proba(x_arr)).astype(np.float32)
        return pred_fn

    arms_predictors = {
        "Arm A: Original B2 (Full Data)": fast_b2_predict,
        "Arm B: Retrained Clean 1x (3k)": make_arm_predictor(rf_b, xgb_b),
        "Arm C: Retrained Clean 2x (6k)": make_arm_predictor(rf_c, xgb_c),
        "Arm D: Retrained Clean + Adv (3k+3k)": make_arm_predictor(rf_d, xgb_d),
    }

    arms_results = {}
    chunk_size = 10

    for arm_name, pred_fn in arms_predictors.items():
        print(f"\n  Evaluating {arm_name} across test conditions & HSJ ({hsj_defense_samples} flows) ...")
        t_arm = time.time()
        p_clean = np.argmax(pred_fn(X_te_clean_pipe), axis=1)
        p_fgsm = np.argmax(pred_fn(X_te_fgsm), axis=1)
        p_rand = np.argmax(pred_fn(X_te_rand), axis=1)
        p_fgsm020 = np.argmax(pred_fn(X_te_fgsm020), axis=1)

        k_cl = int(np.sum(p_clean == y_te))
        k_fg = int(np.sum(p_fgsm == y_te))
        k_rn = int(np.sum(p_rand == y_te))
        k_fg2 = int(np.sum(p_fgsm020 == y_te))

        # HopSkipJump Budget 1 on 50 flows for this specific arm
        arm_bbox = BlackBoxClassifierNeuralNetwork(predict_fn=pred_fn, input_shape=(19,), nb_classes=8)
        arm_hsj_attack = HopSkipJump(
            classifier=arm_bbox, targeted=False, max_iter=3, max_eval=10, init_eval=5, verbose=False
        )
        hsj_chunks = []
        for i in range(0, len(X_te_hsj_sub), chunk_size):
            chunk = X_te_hsj_sub[i : i + chunk_size]
            hsj_chunks.append(arm_hsj_attack.generate(x=chunk))
            log_progress(f"DEF-{arm_name[:5]}", min(i + chunk_size, len(X_te_hsj_sub)), len(X_te_hsj_sub), t_arm)
        X_te_hsj_arm_raw = np.vstack(hsj_chunks)
        X_te_hsj_arm = enforcer.enforce_constraints(X_te_hsj_arm_raw, X_te_hsj_sub)

        p_hsj = np.argmax(pred_fn(X_te_hsj_arm), axis=1)
        k_hs = int(np.sum(p_hsj == y_te_hsj_sub))

        arms_results[arm_name] = {
            "clean": {"count": f"{k_cl}/{len(y_te)}", "acc": round(k_cl / len(y_te), 4), "ci_95": [round(c, 4) for c in wilson_ci(k_cl, len(y_te))]},
            "fgsm_eps010": {"count": f"{k_fg}/{len(y_te)}", "acc": round(k_fg / len(y_te), 4), "ci_95": [round(c, 4) for c in wilson_ci(k_fg, len(y_te))]},
            "random_noise": {"count": f"{k_rn}/{len(y_te)}", "acc": round(k_rn / len(y_te), 4), "ci_95": [round(c, 4) for c in wilson_ci(k_rn, len(y_te))]},
            "heldout_fgsm_eps020": {"count": f"{k_fg2}/{len(y_te)}", "acc": round(k_fg2 / len(y_te), 4), "ci_95": [round(c, 4) for c in wilson_ci(k_fg2, len(y_te))]},
            "hopskipjump_50": {"count": f"{k_hs}/{len(y_te_hsj_sub)}", "acc": round(k_hs / len(y_te_hsj_sub), 4), "ci_95": [round(c, 4) for c in wilson_ci(k_hs, len(y_te_hsj_sub))]},
        }

    t_def = time.time() - t_stage_start
    print(f"\n  Defense 4-Arm Evaluation Matrix (completed in {t_def:.2f}s):")
    print("  " + "-" * 98)
    print("  Model Arm                        | Clean (500) | FGSM 0.10 (500) | Rand Noise (500) | HSJ (50 flows)")
    print("  " + "-" * 98)
    for arm_name, m in arms_results.items():
        cl_str = f"{m['clean']['acc'] * 100:.1f}% [{m['clean']['ci_95'][0] * 100:.1f}%, {m['clean']['ci_95'][1] * 100:.1f}%]"
        fg_str = f"{m['fgsm_eps010']['acc'] * 100:.1f}% [{m['fgsm_eps010']['ci_95'][0] * 100:.1f}%, {m['fgsm_eps010']['ci_95'][1] * 100:.1f}%]"
        rn_str = f"{m['random_noise']['acc'] * 100:.1f}% [{m['random_noise']['ci_95'][0] * 100:.1f}%, {m['random_noise']['ci_95'][1] * 100:.1f}%]"
        hs_str = f"{m['hopskipjump_50']['acc'] * 100:.1f}% [{m['hopskipjump_50']['ci_95'][0] * 100:.1f}%, {m['hopskipjump_50']['ci_95'][1] * 100:.1f}%]"
        print(f"  {arm_name:<32} | {cl_str:<12} | {fg_str:<15} | {rn_str:<16} | {hs_str}")
    print("  " + "-" * 98)

    print("\n  Arm D Generalization Audit on Held-Out Attack Conditions:")
    for arm_name, m in arms_results.items():
        fg2_str = f"{m['heldout_fgsm_eps020']['count']} ({m['heldout_fgsm_eps020']['acc'] * 100:.1f}% [{m['heldout_fgsm_eps020']['ci_95'][0] * 100:.1f}%, {m['heldout_fgsm_eps020']['ci_95'][1] * 100:.1f}%])"
        print(f"  {arm_name:<32} -> Held-out FGSM eps=0.20: {fg2_str}")

    defense_summary = {
        "test_sample_size": len(y_te),
        "hsj_sample_size": len(y_te_hsj_sub),
        "arms": arms_results,
        "runtime_sec": round(t_def, 2),
    }

    out_file = REPORTS_DIR / "adversarial_defense.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(defense_summary, f, indent=2)
    print(f"  [STAGE DONE] Defense metrics saved --> {out_file.name} (Total: {t_def:.2f}s)")
    return defense_summary


# ======================================================================
# STAGE 5: Combine Staged Metrics & Generate Plots
# ======================================================================
def run_stage_combine() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  STAGE 5: COMBINING ALL STAGE METRICS INTO ADVERSARIAL REPORT")
    print("=" * 80)
    fgsm_file = REPORTS_DIR / "adversarial_fgsm.json"
    hsj_file = REPORTS_DIR / "adversarial_hsj.json"
    zoo_file = REPORTS_DIR / "adversarial_zoo.json"
    def_file = REPORTS_DIR / "adversarial_defense.json"

    if not fgsm_file.exists():
        raise FileNotFoundError(f"Missing {fgsm_file}. Run --stage fgsm first.")

    with open(fgsm_file, "r", encoding="utf-8") as f:
        fgsm_data = json.load(f)

    hsj_data = json.load(open(hsj_file, "r", encoding="utf-8")) if hsj_file.exists() else {}
    zoo_data = json.load(open(zoo_file, "r", encoding="utf-8")) if zoo_file.exists() else {}
    def_data = json.load(open(def_file, "r", encoding="utf-8")) if def_file.exists() else {}

    combined = {
        "zero_attack_control": fgsm_data.get("zero_attack_control", {}),
        "fgsm_and_random_noise_results": fgsm_data.get("fgsm_and_random_noise_results", []),
        "hopskipjump_budgets": hsj_data,
        "zoo_budgets": zoo_data,
        "defense_experiment": def_data,
        "stage_runtimes_sec": {
            "fgsm": fgsm_data.get("runtime_sec", 0.0),
            "hsj": sum(b.get("runtime_sec", 0.0) for b in hsj_data.values()) if hsj_data else 0.0,
            "zoo": sum(b.get("runtime_sec", 0.0) for b in zoo_data.values()) if zoo_data else 0.0,
            "defense": def_data.get("runtime_sec", 0.0),
        },
    }

    # Plot Accuracy vs Epsilon
    fgsm_results = fgsm_data.get("fgsm_and_random_noise_results", [])
    if fgsm_results:
        plot_path = REPORTS_DIR / "adversarial_accuracy_vs_epsilon.png"
        plt.figure(figsize=(10, 6))
        eps_vals = [r["epsilon"] for r in fgsm_results]
        dnn_adv_accs = [r["dnn_adversarial_acc"] * 100 for r in fgsm_results]
        ens_adv_accs = [r["ensemble_adversarial_acc"] * 100 for r in fgsm_results]
        ens_rand_accs = [r["ensemble_random_noise_acc"] * 100 for r in fgsm_results]
        acc_ens_clean = fgsm_data["zero_attack_control"]["ensemble_clean_pipeline_acc"]
        acc_dnn_clean = fgsm_data["zero_attack_control"]["dnn_clean_pipeline_acc"]

        plt.plot(eps_vals, [acc_ens_clean * 100] * len(eps_vals), "k--", label=f"Clean B2 Ensemble Baseline ({acc_ens_clean * 100:.1f}%)")
        plt.plot(eps_vals, [acc_dnn_clean * 100] * len(eps_vals), "gray", linestyle=":", label=f"Clean PyTorch DNN Baseline ({acc_dnn_clean * 100:.1f}%)")
        plt.plot(eps_vals, ens_rand_accs, "g-^", linewidth=2, label="Random Noise Control (Ensemble)")
        plt.plot(eps_vals, dnn_adv_accs, "r-o", linewidth=2, label="PyTorch DNN (White-Box FGSM)")
        plt.plot(eps_vals, ens_adv_accs, "b-s", linewidth=2, label="B2 Tree Ensemble (Black-Box FGSM Transfer)")

        plt.xlabel(r"Perturbation Magnitude (Epsilon $\epsilon$)")
        plt.ylabel("Classification Accuracy (%)")
        plt.title("Adversarial Evasion Robustness: Attacks vs Scientific Controls (Constrained)")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"  [PLOT] Robustness plot saved --> {plot_path.name}")

    out_metrics = REPORTS_DIR / "adversarial_metrics.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"  [SAVE] Unified adversarial metrics saved --> {out_metrics.name}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 4 Adversarial Robustness Assessment")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["fgsm", "hsj", "zoo", "defense", "combine", "all"],
        help="Stage to execute: fgsm | hsj | zoo | defense | combine | all",
    )
    parser.add_argument("--fgsm-samples", type=int, default=500, help="Test flows for FGSM transfer")
    parser.add_argument("--hsj1-samples", type=int, default=100, help="Test flows for HopSkipJump Budget 1")
    parser.add_argument("--hsj2-samples", type=int, default=50, help="Test flows for HopSkipJump Budget 2")
    parser.add_argument("--zoo-samples", type=int, default=40, help="Test flows for ZOO")
    parser.add_argument("--defense-samples", type=int, default=500, help="Test flows for defense matrix")
    parser.add_argument("--defense-hsj-samples", type=int, default=50, help="Test flows per arm for HSJ in defense")
    args = parser.parse_args()

    proc = psutil.Process(os.getpid())
    t0_main = time.time()

    if args.stage in ["fgsm", "all"]:
        run_stage_fgsm(fgsm_samples=args.fgsm_samples)

    if args.stage in ["hsj", "all"]:
        run_stage_hsj(hsj1_samples=args.hsj1_samples, hsj2_samples=args.hsj2_samples)

    if args.stage in ["zoo", "all"]:
        run_stage_zoo(zoo_samples=args.zoo_samples)

    if args.stage in ["defense", "all"]:
        run_stage_defense(test_samples=args.defense_samples, hsj_defense_samples=args.defense_hsj_samples)

    if args.stage in ["combine", "all"]:
        run_stage_combine()

    total_time = time.time() - t0_main
    peak_ram = proc.memory_info().rss / (1024.0 * 1024.0)
    print(f"\nExecution finished in {total_time:.2f}s ({total_time / 60.0:.2f} min) | Peak RAM: {peak_ram:.1f} MB")
