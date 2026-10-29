"""
adversarial/run_phase4_adversarial.py
=====================================
Phase 4 - Adversarial Robustness Assessment (Rigorous Control & Defense Suite)

Evaluates the B2 Ensemble Classifier (RF 0.60 + XGBoost 0.40) and PyTorch DNN under
strict physical domain constraints and scientific controls:
1. Zero-Attack Control (epsilon = 0): Verifies that passing clean flows through the
   unscale -> clip -> round -> rescale constraint pipeline produces 0.0 numerical drift
   and preserves clean accuracy bit-for-bit.
2. Random-Noise Control: Generates non-adversarial Gaussian perturbations matching the exact
   per-sample L2 magnitude of FGSM perturbations, bounded by the exact same domain constraints.
   Confirms whether adversarial attacks beat random noise.
3. White-Box FGSM Surrogate: Uses the properly loaded PyTorch DNN (76.8% clean baseline)
   to compute white-box gradients, assessing black-box transfer evasion against B2 Tree Ensemble.
4. Black-Box Decision-Based (HopSkipJump) & Score-Based (ZOO): Evaluated with clean flows
   passed through the constraint pipeline across two distinct query budgets.
5. 4-Arm Defense Comparison: Compares (a) Original B2, (b) Retrained Clean (1x),
   (c) Retrained Clean (2x volume control), and (d) Retrained Clean + Adversarial (FGSM eps=0.10)
   on fresh test flows across Clean, FGSM, HopSkipJump, and Random Noise conditions.
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
from typing import Any, Dict, List, Tuple

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
from adversarial.constraints import AdversarialConstraintEnforcer, get_constraint_summary_table
from orchestrator.llm_orchestrator import FeatureUnscaler
from predict import get_or_load_model, get_selected_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("art").setLevel(logging.ERROR)
log = logging.getLogger("run_phase4_adversarial")


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


def run_adversarial_suite(
    fgsm_samples: int = 500,
    hsj_b1_samples: int = 20,
    hsj_b2_samples: int = 10,
    zoo_samples: int = 10,
    run_defense: bool = True,
) -> Dict[str, Any]:
    proc = psutil.Process(os.getpid())
    t0_global = time.time()
    init_ram = proc.memory_info().rss / (1024.0 * 1024.0)

    print("=" * 80)
    print("  PHASE 4: ADVERSARIAL ROBUSTNESS ASSESSMENT (ART + SCIENTIFIC CONTROLS)")
    print("=" * 80)
    print(f"Initial Process RAM: {init_ram:.1f} MB (Safety ceiling: < 6.0 GB)")

    # 1. Load artifacts, models, and constraints
    print("\n[1/6] Loading B2 models, encoders, and domain constraint enforcer ...")
    le_8 = joblib.load(ENCODER_8_PATH)
    class_names = list(le_8.classes_)
    selected_features = get_selected_features()
    scaler = joblib.load(SCALER_PATH)
    unscaler = FeatureUnscaler()
    enforcer = AdversarialConstraintEnforcer(selected_features, unscaler, scaler)

    ensemble = get_or_load_model(target="8class", model_type="b2")
    if hasattr(ensemble.rf_model, "n_jobs"):
        ensemble.rf_model.n_jobs = 1
    df_test = pd.read_parquet(TEST_PARQUET)

    # 2. Setup PyTorch DNN Wrapper (white-box surrogate)
    print("\n[2/6] Initializing PyTorch DNN surrogate for gradient-based FGSM ...")
    dnn_model = ensemble.dnn_model
    dnn_model.eval()
    criterion = nn.CrossEntropyLoss()
    classifier_dnn = PyTorchClassifier(
        model=dnn_model,
        loss=criterion,
        input_shape=(19,),
        nb_classes=8,
        clip_values=None,
    )

    # 3. Setup Black-box Wrapper for Ensemble (HopSkipJump & ZOO)
    def ensemble_predict_fn(x: np.ndarray) -> np.ndarray:
        df_in = pd.DataFrame(x, columns=selected_features)
        prob_dict = ensemble.predict_probabilities(df_in)
        return prob_dict["weighted"].astype(np.float32)

    classifier_ensemble = BlackBoxClassifierNeuralNetwork(
        predict_fn=ensemble_predict_fn,
        input_shape=(19,),
        nb_classes=8,
    )

    # 4. Zero-Attack Control & FGSM Evasion vs Random Noise Control
    print(f"\n[3/6] Running FGSM Transfer & Random Noise Control (Sample: {fgsm_samples} flows) ...")
    df_fgsm = sample_stratified_subset(df_test, n_samples=fgsm_samples, seed=RANDOM_SEED)
    X_fgsm_clean = df_fgsm[selected_features].to_numpy(dtype=np.float32)
    y_fgsm_clean = df_fgsm["label_8"].to_numpy(dtype=np.int64)

    # Zero-attack control (pipeline check on epsilon = 0)
    X_fgsm_zero = enforcer.enforce_constraints(X_fgsm_clean, X_fgsm_clean)
    diff_zero = np.max(np.abs(X_fgsm_zero - X_fgsm_clean))

    preds_dnn_clean = np.argmax(classifier_dnn.predict(X_fgsm_clean), axis=1)
    acc_dnn_clean = float(accuracy_score(y_fgsm_clean, preds_dnn_clean))

    preds_dnn_zero = np.argmax(classifier_dnn.predict(X_fgsm_zero), axis=1)
    acc_dnn_zero = float(accuracy_score(y_fgsm_clean, preds_dnn_zero))

    prob_ens_clean = ensemble.predict_probabilities(df_fgsm[selected_features])["weighted"]
    preds_ens_clean = np.argmax(prob_ens_clean, axis=1)
    acc_ens_clean = float(accuracy_score(y_fgsm_clean, preds_ens_clean))

    prob_ens_zero = ensemble.predict_probabilities(pd.DataFrame(X_fgsm_zero, columns=selected_features))["weighted"]
    preds_ens_zero = np.argmax(prob_ens_zero, axis=1)
    acc_ens_zero = float(accuracy_score(y_fgsm_clean, preds_ens_zero))

    print(f"  --- Zero-Attack Pipeline Invariance Control (eps = 0.0) ---")
    print(f"  Max Numerical Drift (|X_pipe - X_clean|): {diff_zero:.6f}")
    print(f"  DNN Clean Acc:        Raw={acc_dnn_clean*100:.2f}% | Pipeline={acc_dnn_zero*100:.2f}%")
    print(f"  B2 Ensemble Clean Acc: Raw={acc_ens_clean*100:.2f}% | Pipeline={acc_ens_zero*100:.2f}%")

    rng = np.random.default_rng(RANDOM_SEED)
    epsilons = [0.01, 0.05, 0.10, 0.20, 0.30]
    fgsm_results = []
    saved_adv_flows = None

    print("\n  Evaluating Evasion vs Epsilon (FGSM Transfer vs Random Noise Control):")
    print("  " + "-" * 76)
    print("  Eps    | Mean L2 | DNN FGSM | Ens FGSM | Transfer ASR | Ens Rand Acc | Rand ASR")
    print("  " + "-" * 76)

    for eps in epsilons:
        t_eps = time.time()
        # FGSM attack
        attack_fgsm = FastGradientMethod(estimator=classifier_dnn, eps=eps)
        X_adv_raw = attack_fgsm.generate(x=X_fgsm_clean)
        X_adv_constrained = enforcer.enforce_constraints(X_adv_raw, X_fgsm_clean)

        # L2 perturbation magnitude per sample
        sample_l2 = np.linalg.norm(X_adv_constrained - X_fgsm_clean, axis=1, keepdims=True)
        mean_l2_fgsm = float(np.mean(sample_l2))

        # DNN white-box evaluation
        preds_dnn_adv = np.argmax(classifier_dnn.predict(X_adv_constrained), axis=1)
        acc_dnn_adv = float(accuracy_score(y_fgsm_clean, preds_dnn_adv))

        # B2 Ensemble Transfer evaluation
        df_adv = pd.DataFrame(X_adv_constrained, columns=selected_features)
        prob_ens_adv = ensemble.predict_probabilities(df_adv)["weighted"]
        preds_ens_adv = np.argmax(prob_ens_adv, axis=1)
        acc_ens_adv = float(accuracy_score(y_fgsm_clean, preds_ens_adv))

        correct_mask = (preds_ens_clean == y_fgsm_clean)
        asr_ens = float(np.mean(preds_ens_adv[correct_mask] != y_fgsm_clean[correct_mask])) if np.sum(correct_mask) > 0 else 0.0

        # Random Noise Control: exact same L2 magnitude per sample, random direction on modifiable features
        v_rand = rng.standard_normal(size=X_fgsm_clean.shape).astype(np.float32)
        v_rand[:, enforcer.fixed_indices] = 0.0
        v_norm = np.linalg.norm(v_rand, axis=1, keepdims=True) + 1e-12
        v_scaled = v_rand * (sample_l2 / v_norm)
        X_rand_raw = X_fgsm_clean + v_scaled
        X_rand_constrained = enforcer.enforce_constraints(X_rand_raw, X_fgsm_clean)
        mean_l2_rand = float(np.mean(np.linalg.norm(X_rand_constrained - X_fgsm_clean, axis=1)))

        df_rand = pd.DataFrame(X_rand_constrained, columns=selected_features)
        preds_ens_rand = np.argmax(ensemble.predict_probabilities(df_rand)["weighted"], axis=1)
        acc_ens_rand = float(accuracy_score(y_fgsm_clean, preds_ens_rand))
        asr_ens_rand = float(np.mean(preds_ens_rand[correct_mask] != y_fgsm_clean[correct_mask])) if np.sum(correct_mask) > 0 else 0.0

        elapsed = time.time() - t_eps
        print(f"  {eps:<6.2f} | {mean_l2_fgsm:<7.3f} | {acc_dnn_adv*100:>6.2f}%  | {acc_ens_adv*100:>6.2f}%  | {asr_ens*100:>10.2f}% | {acc_ens_rand*100:>10.2f}% | {asr_ens_rand*100:>6.2f}%")

        fgsm_results.append({
            "epsilon": eps,
            "mean_l2_fgsm": round(mean_l2_fgsm, 4),
            "mean_l2_random": round(mean_l2_rand, 4),
            "dnn_clean_acc": round(acc_dnn_clean, 4),
            "dnn_adversarial_acc": round(acc_dnn_adv, 4),
            "ensemble_clean_acc": round(acc_ens_clean, 4),
            "ensemble_adversarial_acc": round(acc_ens_adv, 4),
            "transfer_asr": round(asr_ens, 4),
            "ensemble_random_noise_acc": round(acc_ens_rand, 4),
            "random_noise_asr": round(asr_ens_rand, 4),
            "runtime_sec": round(elapsed, 2),
        })

        if eps == 0.10:
            saved_adv_flows = (df_fgsm.copy(), df_adv.copy(), preds_ens_adv, y_fgsm_clean)

    print("  " + "-" * 76)

    # 5. HopSkipJump Black-Box Attack (Multi-Budget Comparison)
    print("\n[4/6] Running HopSkipJump Black-Box Attack (Multi-Budget Comparison) ...")
    hsj_results = {}

    for budget_name, max_iter, max_eval, init_eval, n_flows in [
        ("Budget 1 (Low: 3 iters, 10 evals)", 3, 10, 5, hsj_b1_samples),
        ("Budget 2 (High: 6 iters, 20 evals)", 6, 20, 8, hsj_b2_samples),
    ]:
        print(f"  Evaluating {budget_name} on {n_flows} stratified test flows ...")
        df_hsj = sample_stratified_subset(df_test, n_samples=n_flows, seed=RANDOM_SEED + 10)
        X_hsj_clean = df_hsj[selected_features].to_numpy(dtype=np.float32)
        y_hsj_clean = df_hsj["label_8"].to_numpy(dtype=np.int64)

        t_hsj_s = time.time()
        attack_hsj = HopSkipJump(
            classifier=classifier_ensemble,
            targeted=False,
            max_iter=max_iter,
            max_eval=max_eval,
            init_eval=init_eval,
            verbose=False,
        )
        X_hsj_raw = attack_hsj.generate(x=X_hsj_clean)
        X_hsj_constrained = enforcer.enforce_constraints(X_hsj_raw, X_hsj_clean)
        t_hsj = time.time() - t_hsj_s

        l2_hsj = float(np.mean(np.linalg.norm(X_hsj_constrained - X_hsj_clean, axis=1)))
        prob_hsj_clean = ensemble.predict_probabilities(df_hsj[selected_features])["weighted"]
        preds_hsj_clean = np.argmax(prob_hsj_clean, axis=1)
        acc_hsj_clean = float(accuracy_score(y_hsj_clean, preds_hsj_clean))

        prob_hsj_adv = ensemble.predict_probabilities(pd.DataFrame(X_hsj_constrained, columns=selected_features))["weighted"]
        preds_hsj_adv = np.argmax(prob_hsj_adv, axis=1)
        acc_hsj_adv = float(accuracy_score(y_hsj_clean, preds_hsj_adv))

        m_hsj = (preds_hsj_clean == y_hsj_clean)
        asr_hsj = float(np.mean(preds_hsj_adv[m_hsj] != y_hsj_clean[m_hsj])) if np.sum(m_hsj) > 0 else 0.0

        print(f"    Clean Acc: {acc_hsj_clean*100:.2f}% | Perturbed Acc: {acc_hsj_adv*100:.2f}% | ASR: {asr_hsj*100:.2f}% | Mean L2: {l2_hsj:.4f} ({t_hsj:.2f}s)")
        hsj_results[budget_name] = {
            "samples": n_flows,
            "max_iter": max_iter,
            "max_eval": max_eval,
            "clean_accuracy": round(acc_hsj_clean, 4),
            "adversarial_accuracy": round(acc_hsj_adv, 4),
            "attack_success_rate": round(asr_hsj, 4),
            "mean_l2_perturbation": round(l2_hsj, 4),
            "runtime_sec": round(t_hsj, 2),
        }

    # 6. Zeroth-Order Optimization (ZOO) Attack (Multi-Budget Comparison)
    print("\n[5/6] Running ZOO Score-Based Black-Box Attack (Multi-Budget Comparison) ...")
    zoo_results = {}

    for budget_name, max_iter, n_flows in [
        ("Budget 1 (Low: 4 iters)", 4, zoo_samples),
        ("Budget 2 (High: 8 iters)", 8, zoo_samples),
    ]:
        print(f"  Evaluating {budget_name} on {n_flows} flows (nb_parallel=10) ...")
        df_zoo = sample_stratified_subset(df_test, n_samples=n_flows, seed=RANDOM_SEED + 20)
        X_zoo_clean = df_zoo[selected_features].to_numpy(dtype=np.float32)
        y_zoo_clean = df_zoo["label_8"].to_numpy(dtype=np.int64)

        t_zoo_s = time.time()
        attack_zoo = ZooAttack(
            classifier=classifier_ensemble,
            targeted=False,
            max_iter=max_iter,
            learning_rate=0.05,
            batch_size=1,
            nb_parallel=min(10, len(selected_features)),
            verbose=False,
        )
        X_zoo_raw = attack_zoo.generate(x=X_zoo_clean)
        X_zoo_constrained = enforcer.enforce_constraints(X_zoo_raw, X_zoo_clean)
        t_zoo = time.time() - t_zoo_s

        l2_zoo = float(np.mean(np.linalg.norm(X_zoo_constrained - X_zoo_clean, axis=1)))
        prob_zoo_clean = ensemble.predict_probabilities(df_zoo[selected_features])["weighted"]
        preds_zoo_clean = np.argmax(prob_zoo_clean, axis=1)
        acc_zoo_clean = float(accuracy_score(y_zoo_clean, preds_zoo_clean))

        prob_zoo_adv = ensemble.predict_probabilities(pd.DataFrame(X_zoo_constrained, columns=selected_features))["weighted"]
        preds_zoo_adv = np.argmax(prob_zoo_adv, axis=1)
        acc_zoo_adv = float(accuracy_score(y_zoo_clean, preds_zoo_adv))

        m_zoo = (preds_zoo_clean == y_zoo_clean)
        asr_zoo = float(np.mean(preds_zoo_adv[m_zoo] != y_zoo_clean[m_zoo])) if np.sum(m_zoo) > 0 else 0.0

        print(f"    Clean Acc: {acc_zoo_clean*100:.2f}% | Perturbed Acc: {acc_zoo_adv*100:.2f}% | ASR: {asr_zoo*100:.2f}% | Mean L2: {l2_zoo:.4f} ({t_zoo:.2f}s)")
        zoo_results[budget_name] = {
            "samples": n_flows,
            "max_iter": max_iter,
            "clean_accuracy": round(acc_zoo_clean, 4),
            "adversarial_accuracy": round(acc_zoo_adv, 4),
            "attack_success_rate": round(asr_zoo, 4),
            "mean_l2_perturbation": round(l2_zoo, 4),
            "runtime_sec": round(t_zoo, 2),
        }

    # 7. Defense Evaluation: 4-Arm Comparison Across 4 Test Conditions
    defense_summary = {}
    if run_defense:
        print("\n[6/6] Executing Defense Experiment: 4-Arm Comparison across 4 Perturbation Conditions ...")
        t_def_start = time.time()
        df_train = pd.read_parquet(TRAIN_PARQUET)

        # Slice 1: 3,000 clean training flows
        df_tr_3k = sample_stratified_subset(df_train, n_samples=3000, seed=RANDOM_SEED + 30)
        X_tr_3k = df_tr_3k[selected_features].to_numpy(dtype=np.float32)
        y_tr_3k = df_tr_3k["label_8"].to_numpy(dtype=np.int64)

        # Slice 2: 6,000 clean training flows (volume control for 50/50 dataset)
        df_tr_6k = sample_stratified_subset(df_train, n_samples=6000, seed=RANDOM_SEED + 31)
        X_tr_6k = df_tr_6k[selected_features].to_numpy(dtype=np.float32)
        y_tr_6k = df_tr_6k["label_8"].to_numpy(dtype=np.int64)

        # Generate FGSM perturbations on 3k training flows
        print("  Generating FGSM perturbations on 3,000 training flows (eps=0.10) ...")
        attack_tr_fgsm = FastGradientMethod(estimator=classifier_dnn, eps=0.10)
        X_tr_adv_raw = attack_tr_fgsm.generate(x=X_tr_3k)
        X_tr_adv = enforcer.enforce_constraints(X_tr_adv_raw, X_tr_3k)

        # Combined adversarial training set: 3,000 clean + 3,000 FGSM
        X_combined = np.vstack([X_tr_3k, X_tr_adv])
        y_combined = np.concatenate([y_tr_3k, y_tr_3k])

        print("  Fitting Arm B: Retrained 1x Clean (3,000 flows) ...")
        rf_clean1x = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)
        rf_clean1x.fit(X_tr_3k, y_tr_3k)
        xgb_clean1x = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=1)
        xgb_clean1x.fit(X_tr_3k, y_tr_3k)

        print("  Fitting Arm C: Retrained 2x Clean Volume Control (6,000 flows) ...")
        rf_clean2x = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)
        rf_clean2x.fit(X_tr_6k, y_tr_6k)
        xgb_clean2x = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=1)
        xgb_clean2x.fit(X_tr_6k, y_tr_6k)

        print("  Fitting Arm D: Retrained Clean + Adversarial (3k clean + 3k FGSM) ...")
        rf_adv = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)
        rf_adv.fit(X_combined, y_combined)
        xgb_adv = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=1)
        xgb_adv.fit(X_combined, y_combined)

        # Fresh test slice: 400 test flows
        df_fresh_test = sample_stratified_subset(df_test, n_samples=400, seed=RANDOM_SEED + 99)
        X_te_clean = df_fresh_test[selected_features].to_numpy(dtype=np.float32)
        y_te = df_fresh_test["label_8"].to_numpy(dtype=np.int64)

        # Perturbation 1: Clean (pipeline-passed)
        X_te_clean_pipe = enforcer.enforce_constraints(X_te_clean, X_te_clean)

        # Perturbation 2: Fresh FGSM eps=0.10
        X_te_fgsm_raw = FastGradientMethod(estimator=classifier_dnn, eps=0.10).generate(x=X_te_clean)
        X_te_fgsm = enforcer.enforce_constraints(X_te_fgsm_raw, X_te_clean)

        # Perturbation 3: Random Noise matching FGSM eps=0.10 L2
        l2_te_fgsm = np.linalg.norm(X_te_fgsm - X_te_clean, axis=1, keepdims=True)
        v_rand_te = rng.standard_normal(size=X_te_clean.shape).astype(np.float32)
        v_rand_te[:, enforcer.fixed_indices] = 0.0
        v_norm_te = np.linalg.norm(v_rand_te, axis=1, keepdims=True) + 1e-12
        X_te_rand = enforcer.enforce_constraints(X_te_clean + v_rand_te * (l2_te_fgsm / v_norm_te), X_te_clean)

        # Perturbation 4: HopSkipJump on 15 test flows
        X_te_hsj_sub = X_te_clean[:15]
        y_te_hsj_sub = y_te[:15]
        attack_hsj_eval = HopSkipJump(classifier=classifier_ensemble, targeted=False, max_iter=3, max_eval=10, init_eval=5, verbose=False)
        X_te_hsj_sub_adv = enforcer.enforce_constraints(attack_hsj_eval.generate(x=X_te_hsj_sub), X_te_hsj_sub)

        # Helper to predict with 0.60 RF + 0.40 XGB
        def eval_arm(name: str, rf_m, xgb_m, is_orig_b2: bool = False) -> Dict[str, float]:
            def predict_probs(X: np.ndarray) -> np.ndarray:
                if is_orig_b2:
                    return ensemble.predict_probabilities(pd.DataFrame(X, columns=selected_features))["weighted"]
                return 0.60 * rf_m.predict_proba(X) + 0.40 * xgb_m.predict_proba(X)

            p_clean = np.argmax(predict_probs(X_te_clean_pipe), axis=1)
            p_fgsm = np.argmax(predict_probs(X_te_fgsm), axis=1)
            p_rand = np.argmax(predict_probs(X_te_rand), axis=1)
            p_hsj = np.argmax(predict_probs(X_te_hsj_sub_adv), axis=1)

            return {
                "clean_acc": round(float(accuracy_score(y_te, p_clean)), 4),
                "fgsm_eps010_acc": round(float(accuracy_score(y_te, p_fgsm)), 4),
                "random_noise_acc": round(float(accuracy_score(y_te, p_rand)), 4),
                "hopskipjump_acc": round(float(accuracy_score(y_te_hsj_sub, p_hsj)), 4),
            }

        arms_results = {
            "Arm A: Original B2 (Full)": eval_arm("Original B2", None, None, is_orig_b2=True),
            "Arm B: Retrained Clean 1x (3k)": eval_arm("Clean 1x", rf_clean1x, xgb_clean1x),
            "Arm C: Retrained Clean 2x (6k)": eval_arm("Clean 2x", rf_clean2x, xgb_clean2x),
            "Arm D: Retrained Clean + Adv (3k+3k)": eval_arm("Clean+Adv", rf_adv, xgb_adv),
        }

        t_def = time.time() - t_def_start
        print(f"\n  Defense 4-Arm Evaluation Matrix (completed in {t_def:.2f}s):")
        print("  " + "-" * 76)
        print("  Model Arm                             | Clean Acc | FGSM Acc | Rand Acc | HSJ Acc")
        print("  " + "-" * 76)
        for arm_name, metrics in arms_results.items():
            print(f"  {arm_name:<37} | {metrics['clean_acc']*100:>8.2f}% | {metrics['fgsm_eps010_acc']*100:>7.2f}% | {metrics['random_noise_acc']*100:>7.2f}% | {metrics['hopskipjump_acc']*100:>6.2f}%")
        print("  " + "-" * 76)

        defense_summary = {
            "test_sample_size": len(y_te),
            "hsj_subsample_size": len(y_te_hsj_sub),
            "arms": arms_results,
            "runtime_sec": round(t_def, 2),
        }

    # Plot Accuracy vs Epsilon
    plot_path = REPORTS_DIR / "adversarial_accuracy_vs_epsilon.png"
    plt.figure(figsize=(10, 6))
    eps_vals = [r["epsilon"] for r in fgsm_results]
    dnn_adv_accs = [r["dnn_adversarial_acc"] * 100 for r in fgsm_results]
    ens_adv_accs = [r["ensemble_adversarial_acc"] * 100 for r in fgsm_results]
    ens_rand_accs = [r["ensemble_random_noise_acc"] * 100 for r in fgsm_results]

    plt.plot(eps_vals, [acc_ens_clean * 100] * len(eps_vals), "k--", label=f"Clean B2 Ensemble Baseline ({acc_ens_clean*100:.1f}%)")
    plt.plot(eps_vals, [acc_dnn_clean * 100] * len(eps_vals), "gray", linestyle=":", label=f"Clean PyTorch DNN Baseline ({acc_dnn_clean*100:.1f}%)")
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
    print(f"\n[SAVE] Robustness plot saved --> {plot_path.name}")

    # Save sample adversarial flows for alerts / analysis
    if saved_adv_flows is not None:
        clean_df, adv_df, adv_preds, y_clean = saved_adv_flows
        adv_records = adv_df.copy()
        adv_records["true_class_idx"] = y_clean
        adv_records["adversarial_pred_class_idx"] = adv_preds
        adv_records["true_class"] = [class_names[idx] for idx in y_clean]
        adv_records["adversarial_pred_class"] = [class_names[idx] for idx in adv_preds]
        adv_records_path = ARTIFACTS_DIR / "adversarial_demo_flows.parquet"
        adv_records.to_parquet(adv_records_path, index=True)
        print(f"[SAVE] Adversarial sample flows saved --> {adv_records_path.name}")

    total_time = time.time() - t0_global
    peak_ram = proc.memory_info().rss / (1024.0 * 1024.0)

    print("\n" + "=" * 80)
    print("  PHASE 4 ADVERSARIAL EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total Execution Time: {total_time:.2f} s ({total_time / 60.0:.2f} min)")
    print(f"Peak Process RAM:     {peak_ram:.1f} MB ({peak_ram / 1024.0:.2f} GB) [Ceiling: < 6.0 GB]")
    print("=" * 80)

    summary_metrics = {
        "zero_attack_control": {
            "samples": len(df_fgsm),
            "max_numerical_drift": float(diff_zero),
            "dnn_clean_raw_acc": round(acc_dnn_clean, 4),
            "dnn_clean_pipeline_acc": round(acc_dnn_zero, 4),
            "ensemble_clean_raw_acc": round(acc_ens_clean, 4),
            "ensemble_clean_pipeline_acc": round(acc_ens_zero, 4),
        },
        "fgsm_and_random_noise_results": fgsm_results,
        "hopskipjump_budgets": hsj_results,
        "zoo_budgets": zoo_results,
        "defense_experiment": defense_summary,
        "runtime_total_sec": round(total_time, 2),
        "peak_ram_mb": round(peak_ram, 1),
    }

    metrics_file = REPORTS_DIR / "adversarial_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"[SAVE] Adversarial metrics saved --> {metrics_file.name}")

    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 4 Adversarial Robustness Assessment")
    parser.add_argument("--fgsm-samples", type=int, default=500, help="Test flows for FGSM transfer")
    parser.add_argument("--hsj-b1-samples", type=int, default=20, help="Test flows for HopSkipJump Budget 1")
    parser.add_argument("--hsj-b2-samples", type=int, default=10, help="Test flows for HopSkipJump Budget 2")
    parser.add_argument("--zoo-samples", type=int, default=10, help="Test flows for ZOO")
    parser.add_argument("--no-defense", action="store_true", help="Skip defense retraining experiment")
    args = parser.parse_args()

    run_adversarial_suite(
        fgsm_samples=args.fgsm_samples,
        hsj_b1_samples=args.hsj_b1_samples,
        hsj_b2_samples=args.hsj_b2_samples,
        zoo_samples=args.zoo_samples,
        run_defense=not args.no_defense,
    )
