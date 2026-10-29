"""
adversarial/run_phase4_adversarial.py
=====================================
Phase 4 - Adversarial Robustness Assessment

Evaluates the B2 Ensemble Classifier and PyTorch DNN under realistic adversarial attacks:
1. Fast Gradient Sign Method (FGSM): White-box gradient attack on the differentiable PyTorch DNN,
   tested for evasion TRANSFER against the B2 Tree Ensemble (RF 0.60 + XGBoost 0.40).
2. HopSkipJump: Decision-based black-box attack on the B2 ensemble across decision boundaries.
3. Zeroth-Order Optimization (ZOO): Coordinate-wise score-based black-box attack on the B2 ensemble.
4. Domain Constraints: All perturbations are strictly bounded, fixed flags/protocols are frozen,
   and features are clipped to empirical training intervals via AdversarialConstraintEnforcer.
5. Defense Experiment: Evaluates Adversarial Retraining (clean + FGSM) on unseen test flows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

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
    hopskipjump_samples: int = 100,
    zoo_samples: int = 40,
    run_defense: bool = True,
) -> Dict[str, Any]:
    proc = psutil.Process(os.getpid())
    t0_global = time.time()
    init_ram = proc.memory_info().rss / (1024.0 * 1024.0)

    print("=" * 70)
    print("  PHASE 4: ADVERSARIAL ROBUSTNESS EVALUATION (ART + DOMAIN CONSTRAINTS)")
    print("=" * 70)
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
    df_test = pd.read_parquet(TEST_PARQUET)

    # 2. Setup PyTorch DNN Wrapper (for white-box gradient access)
    print("\n[2/6] Initializing PyTorch DNN surrogate for gradient-based FGSM ...")
    dnn_model = ensemble.dnn_model
    dnn_model.eval()
    criterion = nn.CrossEntropyLoss()
    classifier_dnn = PyTorchClassifier(
        model=dnn_model,
        loss=criterion,
        input_shape=(19,),
        nb_classes=8,
        clip_values=None,  # Domain constraints applied post-generation
    )

    # 3. Setup Black-box Wrapper for Ensemble (for HopSkipJump & ZOO)
    def ensemble_predict_fn(x: np.ndarray) -> np.ndarray:
        df_in = pd.DataFrame(x, columns=selected_features)
        prob_dict = ensemble.predict_probabilities(df_in)
        return prob_dict["weighted"].astype(np.float32)

    classifier_ensemble = BlackBoxClassifierNeuralNetwork(
        predict_fn=ensemble_predict_fn,
        input_shape=(19,),
        nb_classes=8,
    )

    # 4. FGSM Attack & Transfer Evaluation
    print(f"\n[3/6] Running FGSM Attack on DNN and Transfer to B2 Tree Ensemble (Sample: {fgsm_samples} flows) ...")
    df_fgsm = sample_stratified_subset(df_test, n_samples=fgsm_samples, seed=RANDOM_SEED)
    X_fgsm_clean = df_fgsm[selected_features].to_numpy(dtype=np.float32)
    y_fgsm_clean = df_fgsm["label_8"].to_numpy(dtype=np.int64)

    # Clean accuracies
    preds_dnn_clean = np.argmax(classifier_dnn.predict(X_fgsm_clean), axis=1)
    acc_dnn_clean = accuracy_score(y_fgsm_clean, preds_dnn_clean)

    prob_ens_clean = ensemble.predict_probabilities(df_fgsm[selected_features])["weighted"]
    preds_ens_clean = np.argmax(prob_ens_clean, axis=1)
    acc_ens_clean = accuracy_score(y_fgsm_clean, preds_ens_clean)

    print(f"  Clean DNN Accuracy:      {acc_dnn_clean * 100:.2f}%")
    print(f"  Clean B2 Ensemble Acc:   {acc_ens_clean * 100:.2f}%")

    epsilons = [0.01, 0.05, 0.10, 0.20, 0.30]
    fgsm_results = []
    saved_adv_flows = None

    for eps in epsilons:
        t_eps = time.time()
        attack_fgsm = FastGradientMethod(estimator=classifier_dnn, eps=eps)
        X_adv_raw = attack_fgsm.generate(x=X_fgsm_clean)

        # Enforce physical constraints: freeze flags/protocols, clip sizes/timing
        X_adv_constrained = enforcer.enforce_constraints(X_adv_raw, X_fgsm_clean)

        # Average L2 perturbation size
        l2_pert = float(np.mean(np.linalg.norm(X_adv_constrained - X_fgsm_clean, axis=1)))

        # DNN white-box evaluation
        preds_dnn_adv = np.argmax(classifier_dnn.predict(X_adv_constrained), axis=1)
        acc_dnn_adv = float(accuracy_score(y_fgsm_clean, preds_dnn_adv))

        # B2 Ensemble Transfer evaluation
        df_adv = pd.DataFrame(X_adv_constrained, columns=selected_features)
        prob_ens_adv = ensemble.predict_probabilities(df_adv)["weighted"]
        preds_ens_adv = np.argmax(prob_ens_adv, axis=1)
        acc_ens_adv = float(accuracy_score(y_fgsm_clean, preds_ens_adv))

        # Attack Success Rate (ASR) on previously correct flows
        correct_mask = (preds_ens_clean == y_fgsm_clean)
        asr_ens = float(np.mean(preds_ens_adv[correct_mask] != y_fgsm_clean[correct_mask])) if np.sum(correct_mask) > 0 else 0.0

        elapsed = time.time() - t_eps
        print(f"  eps={eps:<4} | Perturbation L2={l2_pert:.3f} | DNN Acc: {acc_dnn_adv*100:.2f}% | Ensemble Acc: {acc_ens_adv*100:.2f}% | Transfer ASR: {asr_ens*100:.2f}% ({elapsed:.2f}s)")

        fgsm_results.append({
            "epsilon": eps,
            "l2_perturbation": round(l2_pert, 4),
            "dnn_clean_acc": round(float(acc_dnn_clean), 4),
            "dnn_adversarial_acc": round(acc_dnn_adv, 4),
            "ensemble_clean_acc": round(float(acc_ens_clean), 4),
            "ensemble_adversarial_acc": round(acc_ens_adv, 4),
            "transfer_asr": round(asr_ens, 4),
            "runtime_sec": round(elapsed, 2),
        })

        if eps == 0.10:
            saved_adv_flows = (df_fgsm.copy(), df_adv.copy(), preds_ens_adv, y_fgsm_clean)

    # 5. HopSkipJump Black-Box Attack
    print(f"\n[4/6] Running HopSkipJump Black-Box Attack on B2 Ensemble (Sample: {hopskipjump_samples} flows) ...")
    df_hsj = sample_stratified_subset(df_test, n_samples=hopskipjump_samples, seed=RANDOM_SEED + 1)
    X_hsj_clean = df_hsj[selected_features].to_numpy(dtype=np.float32)
    y_hsj_clean = df_hsj["label_8"].to_numpy(dtype=np.int64)

    t_hsj_start = time.time()
    # Constrained budget: max_iter=5, max_eval=20 to respect laptop RAM & CPU time
    attack_hsj = HopSkipJump(
        classifier=classifier_ensemble,
        targeted=False,
        max_iter=5,
        max_eval=20,
        init_eval=10,
        verbose=False,
    )
    X_hsj_raw = attack_hsj.generate(x=X_hsj_clean)
    X_hsj_constrained = enforcer.enforce_constraints(X_hsj_raw, X_hsj_clean)
    t_hsj = time.time() - t_hsj_start

    l2_hsj = float(np.mean(np.linalg.norm(X_hsj_constrained - X_hsj_clean, axis=1)))
    prob_hsj_clean = ensemble.predict_probabilities(df_hsj[selected_features])["weighted"]
    preds_hsj_clean = np.argmax(prob_hsj_clean, axis=1)
    acc_hsj_clean = float(accuracy_score(y_hsj_clean, preds_hsj_clean))

    prob_hsj_adv = ensemble.predict_probabilities(pd.DataFrame(X_hsj_constrained, columns=selected_features))["weighted"]
    preds_hsj_adv = np.argmax(prob_hsj_adv, axis=1)
    acc_hsj_adv = float(accuracy_score(y_hsj_clean, preds_hsj_adv))

    mask_hsj = (preds_hsj_clean == y_hsj_clean)
    asr_hsj = float(np.mean(preds_hsj_adv[mask_hsj] != y_hsj_clean[mask_hsj])) if np.sum(mask_hsj) > 0 else 0.0

    print(f"HopSkipJump completed in {t_hsj:.2f} s")
    print(f"  Clean Accuracy:        {acc_hsj_clean * 100:.2f}%")
    print(f"  Perturbed Accuracy:    {acc_hsj_adv * 100:.2f}%")
    print(f"  Attack Success Rate:   {asr_hsj * 100:.2f}%")
    print(f"  Mean L2 Perturbation:  {l2_hsj:.4f}")

    # 6. Zeroth-Order Optimization (ZOO) Score-Based Attack
    print(f"\n[5/6] Running ZOO Score-Based Black-Box Attack (Sample: {zoo_samples} flows) ...")
    df_zoo = sample_stratified_subset(df_test, n_samples=zoo_samples, seed=RANDOM_SEED + 2)
    X_zoo_clean = df_zoo[selected_features].to_numpy(dtype=np.float32)
    y_zoo_clean = df_zoo["label_8"].to_numpy(dtype=np.int64)

    t_zoo_start = time.time()
    attack_zoo = ZooAttack(
        classifier=classifier_ensemble,
        targeted=False,
        max_iter=10,
        learning_rate=0.05,
        batch_size=1,
        nb_parallel=min(10, len(selected_features)),
        verbose=False,
    )
    X_zoo_raw = attack_zoo.generate(x=X_zoo_clean)
    X_zoo_constrained = enforcer.enforce_constraints(X_zoo_raw, X_zoo_clean)
    t_zoo = time.time() - t_zoo_start

    l2_zoo = float(np.mean(np.linalg.norm(X_zoo_constrained - X_zoo_clean, axis=1)))
    prob_zoo_clean = ensemble.predict_probabilities(df_zoo[selected_features])["weighted"]
    preds_zoo_clean = np.argmax(prob_zoo_clean, axis=1)
    acc_zoo_clean = float(accuracy_score(y_zoo_clean, preds_zoo_clean))

    prob_zoo_adv = ensemble.predict_probabilities(pd.DataFrame(X_zoo_constrained, columns=selected_features))["weighted"]
    preds_zoo_adv = np.argmax(prob_zoo_adv, axis=1)
    acc_zoo_adv = float(accuracy_score(y_zoo_clean, preds_zoo_adv))

    mask_zoo = (preds_zoo_clean == y_zoo_clean)
    asr_zoo = float(np.mean(preds_zoo_adv[mask_zoo] != y_zoo_clean[mask_zoo])) if np.sum(mask_zoo) > 0 else 0.0

    print(f"ZOO completed in {t_zoo:.2f} s")
    print(f"  Clean Accuracy:        {acc_zoo_clean * 100:.2f}%")
    print(f"  Perturbed Accuracy:    {acc_zoo_adv * 100:.2f}%")
    print(f"  Attack Success Rate:   {asr_zoo * 100:.2f}%")
    print(f"  Mean L2 Perturbation:  {l2_zoo:.4f}")

    # 7. Defense Experiment: Adversarial Retraining (AT)
    defense_results = {}
    if run_defense:
        print("\n[6/6] Executing Defense Experiment: Adversarial Retraining on Training Split ...")
        t_def_start = time.time()
        # Train on modest slice to avoid memory spike: 5,000 clean training flows
        df_train = pd.read_parquet(TRAIN_PARQUET)
        df_tr_slice = sample_stratified_subset(df_train, n_samples=3000, seed=RANDOM_SEED)
        X_tr = df_tr_slice[selected_features].to_numpy(dtype=np.float32)
        y_tr = df_tr_slice["label_8"].to_numpy(dtype=np.int64)

        # Generate FGSM perturbations on training slice
        attack_tr_fgsm = FastGradientMethod(estimator=classifier_dnn, eps=0.10)
        X_tr_adv_raw = attack_tr_fgsm.generate(x=X_tr)
        X_tr_adv = enforcer.enforce_constraints(X_tr_adv_raw, X_tr)

        # Combined training data: 50% clean, 50% adversarial
        X_combined = np.vstack([X_tr, X_tr_adv])
        y_combined = np.concatenate([y_tr, y_tr])

        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier

        print("  Retraining Random Forest and XGBoost on clean + adversarial training data ...")
        rf_robust = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
        rf_robust.fit(X_combined, y_combined)

        xgb_robust = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=-1)
        xgb_robust.fit(X_combined, y_combined)

        # Evaluate robust models on UNSEEN test samples (clean test vs test adversarial generated earlier)
        # B2 weights: 0.60 RF + 0.40 XGB
        prob_rf_test_clean = rf_robust.predict_proba(X_fgsm_clean)
        prob_xgb_test_clean = xgb_robust.predict_proba(X_fgsm_clean)
        ens_test_clean_prob = 0.60 * prob_rf_test_clean + 0.40 * prob_xgb_test_clean
        acc_robust_clean = float(accuracy_score(y_fgsm_clean, np.argmax(ens_test_clean_prob, axis=1)))

        # Fresh adversarial test flows (eps=0.10 constrained)
        X_test_adv_fresh = enforcer.enforce_constraints(FastGradientMethod(estimator=classifier_dnn, eps=0.10).generate(x=X_fgsm_clean), X_fgsm_clean)
        prob_rf_test_adv = rf_robust.predict_proba(X_test_adv_fresh)
        prob_xgb_test_adv = xgb_robust.predict_proba(X_test_adv_fresh)
        ens_test_adv_prob = 0.60 * prob_rf_test_adv + 0.40 * prob_xgb_test_adv
        acc_robust_adv = float(accuracy_score(y_fgsm_clean, np.argmax(ens_test_adv_prob, axis=1)))

        t_def = time.time() - t_def_start
        print(f"Defense Experiment completed in {t_def:.2f} s")
        print(f"  Baseline Model  -> Clean Acc: {acc_ens_clean*100:.2f}% | Under FGSM eps=0.10: {fgsm_results[2]['ensemble_adversarial_acc']*100:.2f}%")
        print(f"  Retrained Model -> Clean Acc: {acc_robust_clean*100:.2f}% | Under FGSM eps=0.10: {acc_robust_adv*100:.2f}%")

        defense_results = {
            "training_samples_clean": len(X_tr),
            "training_samples_adv": len(X_tr_adv),
            "baseline_clean_acc": round(float(acc_ens_clean), 4),
            "baseline_adv_acc_eps010": round(fgsm_results[2]["ensemble_adversarial_acc"], 4),
            "retrained_clean_acc": round(acc_robust_clean, 4),
            "retrained_adv_acc_eps010": round(acc_robust_adv, 4),
            "clean_acc_delta": round(acc_robust_clean - float(acc_ens_clean), 4),
            "adv_acc_gain": round(acc_robust_adv - fgsm_results[2]["ensemble_adversarial_acc"], 4),
            "runtime_sec": round(t_def, 2),
        }

    # Plot Accuracy vs Epsilon
    plot_path = REPORTS_DIR / "adversarial_accuracy_vs_epsilon.png"
    plt.figure(figsize=(10, 6))
    eps_vals = [r["epsilon"] for r in fgsm_results]
    dnn_adv_accs = [r["dnn_adversarial_acc"] * 100 for r in fgsm_results]
    ens_adv_accs = [r["ensemble_adversarial_acc"] * 100 for r in fgsm_results]

    plt.plot(eps_vals, [acc_ens_clean * 100] * len(eps_vals), "k--", label="Clean B2 Ensemble Baseline")
    plt.plot(eps_vals, [acc_dnn_clean * 100] * len(eps_vals), "gray", linestyle=":", label="Clean PyTorch DNN Baseline")
    plt.plot(eps_vals, dnn_adv_accs, "r-o", linewidth=2, label="PyTorch DNN (White-box FGSM)")
    plt.plot(eps_vals, ens_adv_accs, "b-s", linewidth=2, label="B2 Tree Ensemble (Black-box Transfer)")

    plt.xlabel(r"Perturbation Magnitude (Epsilon $\epsilon$)")
    plt.ylabel("Classification Accuracy (%)")
    plt.title("Adversarial Evasion Robustness: White-Box DNN vs Ensemble Transfer (Constrained)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[SAVE] Robustness plot saved --> {plot_path.name}")

    # Save sample adversarial flows for alert logging
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

    print("\n" + "=" * 70)
    print("  PHASE 4 ADVERSARIAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Execution Time: {total_time:.2f} s ({total_time / 60.0:.2f} min)")
    print(f"Peak Process RAM:     {peak_ram:.1f} MB ({peak_ram / 1024.0:.2f} GB) [Ceiling: < 6.0 GB]")
    print("=" * 70)

    summary_metrics = {
        "fgsm_results": fgsm_results,
        "hopskipjump": {
            "samples": hopskipjump_samples,
            "clean_accuracy": round(acc_hsj_clean, 4),
            "adversarial_accuracy": round(acc_hsj_adv, 4),
            "attack_success_rate": round(asr_hsj, 4),
            "mean_l2_perturbation": round(l2_hsj, 4),
            "runtime_sec": round(t_hsj, 2),
        },
        "zoo": {
            "samples": zoo_samples,
            "clean_accuracy": round(acc_zoo_clean, 4),
            "adversarial_accuracy": round(acc_zoo_adv, 4),
            "attack_success_rate": round(asr_zoo, 4),
            "mean_l2_perturbation": round(l2_zoo, 4),
            "runtime_sec": round(t_zoo, 2),
        },
        "defense_experiment": defense_results,
        "runtime_total_sec": round(total_time, 2),
        "peak_ram_mb": round(peak_ram, 1),
    }

    metrics_file = REPORTS_DIR / "adversarial_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"[SAVE] Adversarial metrics saved --> {metrics_file.name}")

    # Return sample adversarial flows for alert logging
    return {
        "metrics": summary_metrics,
        "saved_adv_sample": saved_adv_flows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 4 Adversarial Robustness Assessment")
    parser.add_argument("--fgsm-samples", type=int, default=500, help="Test flows for FGSM transfer")
    parser.add_argument("--hsj-samples", type=int, default=100, help="Test flows for HopSkipJump")
    parser.add_argument("--zoo-samples", type=int, default=40, help="Test flows for ZOO")
    parser.add_argument("--no-defense", action="store_true", help="Skip defense retraining experiment")
    args = parser.parse_args()

    run_adversarial_suite(
        fgsm_samples=args.fgsm_samples,
        hopskipjump_samples=args.hsj_samples,
        zoo_samples=args.zoo_samples,
        run_defense=not args.no_defense,
    )
