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
import concurrent.futures
from datetime import datetime
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
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
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


def sample_stratified_subset(
    df: pd.DataFrame,
    n_samples: int = 2000,
    min_benign: int = 200,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Samples stratified flows ensuring:
      - Total flows sampled: min(n_samples, available test rows)
      - Benign flows: min(min_benign, available benign test rows)
    """
    rng = np.random.default_rng(seed)
    total_available = len(df)
    n_target = min(n_samples, total_available)

    benign_sub = df[df["label_8"] == 0]
    attack_sub = df[df["label_8"] != 0]

    if n_target >= min_benign:
        n_benign_target = min(min_benign, len(benign_sub))
    else:
        n_benign_target = max(5, int(round((len(benign_sub) / total_available) * n_target)))

    n_attack_target = min(n_target - n_benign_target, len(attack_sub))
    chosen_benign = rng.choice(benign_sub.index, size=n_benign_target, replace=False).tolist()

    attack_class_counts = attack_sub["label_8"].value_counts()
    attack_total = len(attack_sub)
    chosen_attacks = []

    for label, count in attack_class_counts.items():
        quota = max(5, int(round((count / attack_total) * n_attack_target)))
        sub = attack_sub[attack_sub["label_8"] == label]
        c = rng.choice(sub.index, size=min(len(sub), quota), replace=False)
        chosen_attacks.extend(c.tolist())

    if len(chosen_attacks) > n_attack_target:
        chosen_attacks = rng.choice(chosen_attacks, size=n_attack_target, replace=False).tolist()
    elif len(chosen_attacks) < n_attack_target:
        remaining_indices = list(set(attack_sub.index) - set(chosen_attacks))
        needed = n_attack_target - len(chosen_attacks)
        chosen_attacks.extend(rng.choice(remaining_indices, size=min(len(remaining_indices), needed), replace=False).tolist())

    all_indices = chosen_benign + chosen_attacks
    rng.shuffle(all_indices)
    return df.loc[all_indices].copy()


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
def run_stage_fgsm(fgsm_samples: int = 2000) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"  STAGE 1: FGSM TRANSFER & RANDOM NOISE CONTROL ({fgsm_samples} flows)")
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

    ADV_RESULTS_DIR = PROJECT_ROOT / "adversarial" / "results"
    ADV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_data = {
        "pool_info": {
            "total_flows": len(df_fgsm),
            "benign_flows": int((df_fgsm["label_8"] == 0).sum()),
        },
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

    out_file_adv = ADV_RESULTS_DIR / "fgsm_results.json"
    with open(out_file_adv, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"  [STAGE DONE] FGSM & Zero-Attack saved --> {out_file_adv} ({out_data['runtime_sec']:.2f}s)")
    return out_data


# ======================================================================
# STAGE 2: HopSkipJump Decision-Based Attack (Multi-Budget on SAME 100 flows)
# ======================================================================
class FlowTimeoutError(Exception):
    """Raised when HopSkipJump per-flow evaluation exceeds the allowed time limit."""
    pass


class FlowQueryCapError(Exception):
    """Raised when HopSkipJump per-flow evaluation exceeds the allowed query cap."""
    pass


def generate_hsj_flow_by_flow(
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    clean_preds: np.ndarray,
    predictor: Callable[[np.ndarray], np.ndarray],
    enforcer: AdversarialConstraintEnforcer,
    budget_name: str,
    max_iter: int,
    max_eval: int,
    init_eval: int,
    checkpoint_file: Optional[Path] = None,
    resume: bool = False,
    per_flow_timeout: float = 30.0,
    max_queries: int = 1000,
    log_prefix: str = "HSJ",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Executes HopSkipJump iteratively on each individual flow with:
    1. Per-flow timeout limit (default 30s)
    2. Per-flow query cap (default 1000 queries)
    3. Checkpointing every 10 flows to JSON with --resume capability
    4. Progress logging every 5 flows with elapsed time, rate, and slowest flow
    5. Warning if any flow exceeds 60s
    6. Isolated execution per sample avoiding multi-sample line-search deadlocks
    """
    N = len(X_clean)
    completed_flows: Dict[str, Any] = {}

    if resume and checkpoint_file and checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                chk_all = json.load(f)
            if budget_name in chk_all and "flows" in chk_all[budget_name]:
                completed_flows = chk_all[budget_name]["flows"]
                print(f"  [RESUME] Loaded {len(completed_flows)}/{N} completed flows from checkpoint for '{budget_name}'.")
        except Exception as e:
            print(f"  [RESUME WARNING] Could not read checkpoint: {e}")

    t_budget_start = time.time()
    slowest_sec = 0.0
    slowest_idx = -1
    slowest_queries = 0

    for idx_str, entry in completed_flows.items():
        if entry.get("seconds", 0.0) > slowest_sec:
            slowest_sec = entry["seconds"]
            slowest_idx = entry["flow_idx"]
            slowest_queries = entry.get("queries", 0)

    for i in range(N):
        idx_key = str(i)
        if idx_key in completed_flows:
            continue

        flow_x = X_clean[i : i + 1]
        flow_y = y_clean[i]
        clean_pred = int(clean_preds[i])

        flow_state = {
            "flow_idx": i,
            "start_time": time.time(),
            "queries": 0,
        }

        def guarded_predict(x: np.ndarray) -> np.ndarray:
            now = time.time()
            if now - flow_state["start_time"] > per_flow_timeout:
                raise FlowTimeoutError(f"Flow {i} exceeded {per_flow_timeout}s timeout")
            if flow_state["queries"] + len(x) > max_queries:
                raise FlowQueryCapError(f"Flow {i} exceeded query cap {max_queries}")
            flow_state["queries"] += len(x)
            return predictor(x)

        bbox = BlackBoxClassifierNeuralNetwork(
            predict_fn=guarded_predict, input_shape=(19,), nb_classes=8
        )
        attack = HopSkipJump(
            classifier=bbox,
            targeted=False,
            max_iter=max_iter,
            max_eval=max_eval,
            init_eval=init_eval,
            verbose=False,
        )

        t_flow_start = time.time()
        outcome = "normal"
        adv_flow_raw = flow_x.copy()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(attack.generate, x=flow_x)
            try:
                adv_cand = future.result(timeout=per_flow_timeout + 2.0)
                adv_flow_raw = adv_cand
                raw_pred = int(np.argmax(predictor(adv_flow_raw), axis=1)[0])
                outcome = "evaded_raw" if raw_pred != flow_y else "evasion_failed"
            except (FlowTimeoutError, concurrent.futures.TimeoutError):
                outcome = "timeout"
                raw_pred = clean_pred
            except FlowQueryCapError:
                outcome = "query_cap"
                raw_pred = clean_pred
            except Exception as ex:
                outcome = f"error_{type(ex).__name__}"
                raw_pred = clean_pred

        t_flow_elapsed = time.time() - t_flow_start
        if t_flow_elapsed > 60.0:
            print(f"  [WARNING] Flow {i} took {t_flow_elapsed:.2f}s (> 60s)!", flush=True)

        if t_flow_elapsed > slowest_sec:
            slowest_sec = t_flow_elapsed
            slowest_idx = i
            slowest_queries = flow_state["queries"]

        adv_flow_constrained = enforcer.enforce_constraints(adv_flow_raw, flow_x)
        pred_constrained = int(np.argmax(predictor(adv_flow_constrained), axis=1)[0])
        l2_raw = float(np.linalg.norm(adv_flow_raw - flow_x))
        l2_constrained = float(np.linalg.norm(adv_flow_constrained - flow_x))

        completed_flows[idx_key] = {
            "flow_idx": i,
            "outcome": outcome,
            "true_label": int(flow_y),
            "clean_pred": clean_pred,
            "raw_pred": raw_pred,
            "constrained_pred": pred_constrained,
            "queries": flow_state["queries"],
            "seconds": round(t_flow_elapsed, 3),
            "l2_raw": round(l2_raw, 4),
            "l2_constrained": round(l2_constrained, 4),
            "raw_adv": adv_flow_raw[0].tolist(),
            "constrained_adv": adv_flow_constrained[0].tolist(),
        }

        # Checkpoint every 10 flows or on last flow
        if checkpoint_file and ((i + 1) % 10 == 0 or (i + 1) == N):
            try:
                chk_all = {}
                if checkpoint_file.exists():
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        chk_all = json.load(f)
                chk_all[budget_name] = {
                    "completed": (len(completed_flows) == N),
                    "total_flows": N,
                    "completed_count": len(completed_flows),
                    "flows": completed_flows,
                }
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(chk_all, f, indent=2)
            except Exception as e:
                print(f"  [CHECKPOINT ERROR] Failed to write checkpoint: {e}")

        # Progress logging every 5 flows or on last flow
        if (i + 1) % 5 == 0 or (i + 1) == N:
            elapsed = time.time() - t_budget_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            log_line = (
                f"[{log_prefix}] Flows {i + 1:3d}/{N} ({(i + 1) / N * 100:5.1f}%) | "
                f"Elapsed: {elapsed:6.1f}s | Rate: {rate:4.2f} flows/s | "
                f"Slowest: Flow {slowest_idx} ({slowest_sec:.2f}s, {slowest_queries} q)"
            )
            print("  " + log_line, flush=True)
            with open(RUN_LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(f"[{datetime.now().strftime('%H:%M:%S')}] {log_line}\n")

    # Assemble arrays
    X_raw_list = [completed_flows[str(i)]["raw_adv"] for i in range(N)]
    X_constrained_list = [completed_flows[str(i)]["constrained_adv"] for i in range(N)]
    X_raw = np.array(X_raw_list, dtype=np.float32)
    X_constrained = np.array(X_constrained_list, dtype=np.float32)
    t_budget_total = time.time() - t_budget_start

    # Metrics computation
    entries = [completed_flows[str(i)] for i in range(N)]
    m_clean = (clean_preds == y_clean)
    n_corr_clean = int(np.sum(m_clean))
    k_clean = int(np.sum(clean_preds == y_clean))
    acc_clean = float(k_clean / N)
    ci_cl_l, ci_cl_u = wilson_ci(k_clean, N)

    n_timeouts = sum(1 for e in entries if e["outcome"] in ["timeout", "query_cap"])
    cand_entries = [e for e in entries if e["clean_pred"] == e["true_label"]]
    t_cand = sum(1 for e in cand_entries if e["outcome"] in ["timeout", "query_cap"])
    n_attacked_clean = n_corr_clean - t_cand

    # Raw evaluation
    l2_raw = float(np.mean(np.linalg.norm(X_raw - X_clean, axis=1)))
    preds_raw = np.array([e["raw_pred"] for e in entries], dtype=np.int64)
    k_raw = int(np.sum(preds_raw == y_clean))
    acc_raw = float(k_raw / N)

    evaded_raw_mask = [(e["clean_pred"] == e["true_label"] and e["outcome"] == "evaded_raw") for e in entries]
    n_evaded_raw = int(sum(evaded_raw_mask))
    asr_raw = float(n_evaded_raw / n_attacked_clean) if n_attacked_clean > 0 else 0.0

    # Constrained evaluation
    l2_constrained = float(np.mean(np.linalg.norm(X_constrained - X_clean, axis=1)))
    preds_constrained = np.array([e["constrained_pred"] for e in entries], dtype=np.int64)
    k_constrained = int(np.sum(preds_constrained == y_clean))
    acc_constrained = float(k_constrained / N)
    ci_c_l, ci_c_u = wilson_ci(k_constrained, N)

    evaded_constrained_mask = [
        (e["clean_pred"] == e["true_label"] and e["constrained_pred"] != e["true_label"] and e["outcome"] not in ["timeout", "query_cap"])
        for e in entries
    ]
    n_evaded_constrained = int(sum(evaded_constrained_mask))
    asr_constrained = float(n_evaded_constrained / n_attacked_clean) if n_attacked_clean > 0 else 0.0
    ci_asr_l, ci_asr_u = wilson_ci(n_evaded_constrained, n_attacked_clean if n_attacked_clean > 0 else 1)

    n_reversed = int(sum(
        (e["clean_pred"] == e["true_label"] and e["outcome"] == "evaded_raw" and e["constrained_pred"] == e["true_label"])
        for e in entries
    ))
    pct_reversed = float(n_reversed / n_evaded_raw) if n_evaded_raw > 0 else 0.0

    asr_count_str = (
        f"{n_evaded_constrained}/{n_attacked_clean} attacked ({t_cand} timed out)"
        if t_cand > 0
        else f"{n_evaded_constrained}/{n_corr_clean}"
    )

    metrics = {
        "samples": N,
        "max_iter": max_iter,
        "max_eval": max_eval,
        "clean_accuracy": round(acc_clean, 4),
        "clean_count": f"{k_clean}/{N}",
        "clean_ci_95": [round(ci_cl_l, 4), round(ci_cl_u, 4)],
        "raw_adversarial_accuracy": round(acc_raw, 4),
        "raw_adversarial_count": f"{k_raw}/{N}",
        "raw_asr": round(asr_raw, 4),
        "raw_asr_count": f"{n_evaded_raw}/{n_attacked_clean}",
        "raw_mean_l2": round(l2_raw, 4),
        "adversarial_accuracy": round(acc_constrained, 4),
        "adversarial_count": f"{k_constrained}/{N}",
        "adversarial_ci_95": [round(ci_c_l, 4), round(ci_c_u, 4)],
        "attack_success_rate": round(asr_constrained, 4),
        "attack_success_count": asr_count_str,
        "asr_ci_95": [round(ci_asr_l, 4), round(ci_asr_u, 4)],
        "mean_l2_perturbation": round(l2_constrained, 4),
        "constraint_reversed_count": f"{n_reversed}/{n_evaded_raw}",
        "constraint_reversed_share": round(pct_reversed, 4),
        "timeouts": n_timeouts,
        "timeouts_in_candidates": t_cand,
        "runtime_sec": round(t_budget_total, 2),
    }

    print(f"  {budget_name} Results on same {N} flows:")
    print(f"    Raw (Unconstrained): Acc={k_raw}/{N} ({acc_raw*100:.1f}%) | ASR={n_evaded_raw}/{n_attacked_clean} ({asr_raw*100:.1f}%) | Mean L2: {l2_raw:.4f}")
    print(f"    Constrained:         Acc={format_count_and_ci(k_constrained, N)}")
    print(f"    Constrained ASR:     {format_count_and_ci(n_evaded_constrained, n_attacked_clean)} | Mean L2: {l2_constrained:.4f} ({t_budget_total:.2f}s)")
    if t_cand > 0:
        print(f"    Timeouts:            {t_cand} flows timed out (excluded from denominator)")
    print(f"    Constraint Effect:   {n_reversed}/{n_evaded_raw} ({pct_reversed*100:.1f}%) evasions reversed by protocol/range bounds")

    return X_raw, X_constrained, metrics


def run_stage_hsj(
    hsj_samples: int = 100,
    resume: bool = False,
    per_flow_timeout: float = 30.0,
    max_queries: int = 1000,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"  STAGE 2: HOPSKIPJUMP DECISION-BASED ATTACK (SAME {hsj_samples} flows for both budgets)")
    print("=" * 80)
    t_stage_start = time.time()
    _, _, selected_features, _, _, enforcer, df_test, ensemble = load_shared_environment()
    fast_predict = get_fast_ensemble_predictor(ensemble)

    df_hsj = sample_stratified_subset(df_test, n_samples=hsj_samples, seed=RANDOM_SEED + 10)
    X_hsj_clean = df_hsj[selected_features].to_numpy(dtype=np.float32)
    y_hsj_clean = df_hsj["label_8"].to_numpy(dtype=np.int64)

    prob_hsj_clean = fast_predict(X_hsj_clean)
    preds_hsj_clean = np.argmax(prob_hsj_clean, axis=1)
    k_hsj_clean = int(np.sum(preds_hsj_clean == y_hsj_clean))
    print(f"  Clean Accuracy on HSJ test set ({hsj_samples} flows): {format_count_and_ci(k_hsj_clean, len(y_hsj_clean))}")

    checkpoint_file = REPORTS_DIR / "adversarial_hsj_partial.json"
    hsj_results = {}

    budgets = [
        ("Budget 1 (Low: 3 iters, 10 evals)", 3, 10, 5),
        ("Budget 2 (High: 6 iters, 20 evals)", 6, 20, 8),
    ]

    for budget_name, max_iter, max_eval, init_eval in budgets:
        print(f"\n  Running {budget_name} on {hsj_samples} flows flow-by-flow (timeout: {per_flow_timeout}s, cap: {max_queries} q) ...")
        _, _, metrics = generate_hsj_flow_by_flow(
            X_clean=X_hsj_clean,
            y_clean=y_hsj_clean,
            clean_preds=preds_hsj_clean,
            predictor=fast_predict,
            enforcer=enforcer,
            budget_name=budget_name,
            max_iter=max_iter,
            max_eval=max_eval,
            init_eval=init_eval,
            checkpoint_file=checkpoint_file,
            resume=resume,
            per_flow_timeout=per_flow_timeout,
            max_queries=max_queries,
            log_prefix=f"HSJ-I{max_iter}",
        )
        hsj_results[budget_name] = metrics

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

        # HopSkipJump Budget 1 on hsj_defense_samples flows for this specific arm
        arm_preds_clean = np.argmax(pred_fn(X_te_hsj_sub), axis=1)
        _, _, hsj_arm_metrics = generate_hsj_flow_by_flow(
            X_clean=X_te_hsj_sub,
            y_clean=y_te_hsj_sub,
            clean_preds=arm_preds_clean,
            predictor=pred_fn,
            enforcer=enforcer,
            budget_name=f"Defense-{arm_name[:5]}",
            max_iter=3,
            max_eval=10,
            init_eval=5,
            checkpoint_file=None,
            resume=False,
            per_flow_timeout=30.0,
            max_queries=1000,
            log_prefix=f"DEF-{arm_name[:5]}",
        )
        k_hs = int(hsj_arm_metrics["adversarial_count"].split("/")[0])

        arms_results[arm_name] = {
            "clean": {"count": f"{k_cl}/{len(y_te)}", "acc": round(k_cl / len(y_te), 4), "ci_95": [round(c, 4) for c in wilson_ci(k_cl, len(y_te))]},
            "fgsm_eps010": {"count": f"{k_fg}/{len(y_te)}", "acc": round(k_fg / len(y_te), 4), "ci_95": [round(c, 4) for c in wilson_ci(k_fg, len(y_te))]},
            "random_noise": {"count": f"{k_rn}/{len(y_te)}", "acc": round(k_rn / len(y_te), 4), "ci_95": [round(c, 4) for c in wilson_ci(k_rn, len(y_te))]},
            "heldout_fgsm_eps020": {"count": f"{k_fg2}/{len(y_te)}", "acc": round(k_fg2 / len(y_te), 4), "ci_95": [round(c, 4) for c in wilson_ci(k_fg2, len(y_te))]},
            f"hopskipjump_{hsj_defense_samples}": {"count": f"{k_hs}/{len(y_te_hsj_sub)}", "acc": round(k_hs / len(y_te_hsj_sub), 4), "ci_95": [round(c, 4) for c in wilson_ci(k_hs, len(y_te_hsj_sub))]},
        }

    t_def = time.time() - t_stage_start
    hs_key = f"hopskipjump_{hsj_defense_samples}"
    print(f"\n  Defense 4-Arm Evaluation Matrix (completed in {t_def:.2f}s):")
    print("  " + "-" * 100)
    print(f"  Model Arm                        | Clean (500) | FGSM 0.10 (500) | Rand Noise (500) | HSJ ({hsj_defense_samples} flows)")
    print("  " + "-" * 100)
    for arm_name, m in arms_results.items():
        cl_str = f"{m['clean']['acc'] * 100:.1f}% [{m['clean']['ci_95'][0] * 100:.1f}%, {m['clean']['ci_95'][1] * 100:.1f}%]"
        fg_str = f"{m['fgsm_eps010']['acc'] * 100:.1f}% [{m['fgsm_eps010']['ci_95'][0] * 100:.1f}%, {m['fgsm_eps010']['ci_95'][1] * 100:.1f}%]"
        rn_str = f"{m['random_noise']['acc'] * 100:.1f}% [{m['random_noise']['ci_95'][0] * 100:.1f}%, {m['random_noise']['ci_95'][1] * 100:.1f}%]"
        hs_str = f"{m[hs_key]['acc'] * 100:.1f}% [{m[hs_key]['ci_95'][0] * 100:.1f}%, {m[hs_key]['ci_95'][1] * 100:.1f}%]"
        print(f"  {arm_name:<32} | {cl_str:<12} | {fg_str:<15} | {rn_str:<16} | {hs_str}")
    print("  " + "-" * 100)

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
    parser.add_argument("--fgsm-samples", type=int, default=2000, help="Test flows for FGSM transfer")
    parser.add_argument("--hsj-samples", type=int, default=100, help="Test flows for HopSkipJump (both budgets on same set)")
    parser.add_argument("--zoo-samples", type=int, default=40, help="Test flows for ZOO")
    parser.add_argument("--defense-samples", type=int, default=500, help="Test flows for defense matrix")
    parser.add_argument("--defense-hsj-samples", type=int, default=100, help="Test flows per arm for HSJ in defense")
    parser.add_argument("--resume", action="store_true", help="Resume HopSkipJump from partial checkpoint")
    parser.add_argument("--per-flow-timeout", type=float, default=30.0, help="Per-flow timeout limit in seconds")
    parser.add_argument("--max-queries", type=int, default=1000, help="Hard query cap per flow")
    args = parser.parse_args()

    proc = psutil.Process(os.getpid())
    t0_main = time.time()

    if args.stage in ["fgsm", "all"]:
        run_stage_fgsm(fgsm_samples=args.fgsm_samples)

    if args.stage in ["hsj", "all"]:
        run_stage_hsj(
            hsj_samples=args.hsj_samples,
            resume=args.resume,
            per_flow_timeout=args.per_flow_timeout,
            max_queries=args.max_queries,
        )

    if args.stage in ["zoo", "all"]:
        run_stage_zoo(zoo_samples=args.zoo_samples)

    if args.stage in ["defense", "all"]:
        run_stage_defense(test_samples=args.defense_samples, hsj_defense_samples=args.defense_hsj_samples)

    if args.stage in ["combine", "all"]:
        run_stage_combine()

    total_time = time.time() - t0_main
    peak_ram = proc.memory_info().rss / (1024.0 * 1024.0)
    print(f"\nExecution finished in {total_time:.2f}s ({total_time / 60.0:.2f} min) | Peak RAM: {peak_ram:.1f} MB")
