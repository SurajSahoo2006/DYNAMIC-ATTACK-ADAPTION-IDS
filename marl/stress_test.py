"""
marl/stress_test.py
===================
Stage 6: Adversarial Perturbation Stress Test (Paired Like-for-Like Protocol).

Re-uses Phase 4 FGSM transfer attack (epsilon = 0.10) with physical domain constraint
enforcement and the repaired DNN surrogate over 500 test flows.

Like-for-like protocol:
  Evaluates every policy (all 7 baselines and MAPPO) on:
    1. The clean version of the 500 test flows.
    2. The physically-constrained FGSM perturbed version of those SAME 500 flows.
  Both evaluated over the same 500 episodes with paired bootstrap differences (adv - clean).

Audits the exact failure mechanism:
  - Misclassification of benign flows into attack classes and elevated severity levels.
  - Benign traffic disruption rate vs attack containment rate.
  - Escalation budget depletion under evasion.

Outputs:
  - reports/marl_stress_test.json
"""

from __future__ import annotations

import collections
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from adversarial.run_phase4_adversarial import (
    load_shared_environment,
    sample_stratified_subset,
)
from config import get_flow_severity
from config_marl import (
    EVAL_JSON_PATH,
    FINAL_EVAL_EPISODES,
    MARL_MODELS_DIR,
    NON_LEARNED_BASELINES,
    PILOT_JSON_PATH,
    REFERENCE_LABEL_AWARE_BASELINES,
    STRESS_TEST_EPSILON,
    STRESS_TEST_JSON_PATH,
    STRESS_TEST_SAMPLE_SIZE,
    STRESS_TEST_TIME_LIMIT_SEC,
    TRAIN_SEEDS,
)
from marl.baselines import (
    AlwaysNonePolicy,
    BudgetAwareLabelAwareRulePolicy,
    BudgetAwareSeverityRulePolicy,
    LabelAwareRulePolicy,
    MaxContainmentPolicy,
    OraclePolicy,
    RandomPolicy,
    SeverityRulePolicy,
)
from marl.env import BasePolicy, TacticalDefenseEnv
from marl.evaluate import MAPPOEvaluatorPolicy
from marl.stats import (
    bootstrap_ci_mean,
    bootstrap_ci_paired_difference,
    wilson_score_interval,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger("marl_stress_test")


def get_peak_ram_mb() -> float:
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def generate_paired_test_caches(
    sample_size: int = STRESS_TEST_SAMPLE_SIZE,
    epsilon: float = STRESS_TEST_EPSILON,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, Dict]:
    """
    Generates paired clean and physically-constrained FGSM perturbed test caches
    from the exact same stratified test sample.
    """
    log.info(f"Generating paired test caches ({sample_size} flows, FGSM eps={epsilon})...")
    le_8, class_names, selected_features, scaler, unscaler, enforcer, df_test, ensemble = load_shared_environment()

    # Stratified test sample
    df_sample = sample_stratified_subset(df_test, n_samples=sample_size, seed=42)
    X_clean = df_sample[selected_features].to_numpy(dtype=np.float32)
    y_clean = df_sample["label_8"].to_numpy(dtype=np.int64)

    # 1. Clean Predictions
    clean_probs = ensemble.predict_probabilities(pd.DataFrame(X_clean, columns=selected_features))["weighted"]
    clean_preds = np.argmax(clean_probs, axis=1)
    clean_confs = np.max(clean_probs, axis=1)
    clean_acc = float((clean_preds == y_clean).mean())

    clean_cache = pd.DataFrame(index=df_sample.index)
    for idx, cname in enumerate(class_names):
        clean_cache[f"prob_{cname}"] = clean_probs[:, idx].astype(np.float32)
    clean_cache["pred_class_idx"] = clean_preds.astype(np.int32)
    clean_cache["pred_class"] = [class_names[i] for i in clean_preds]
    clean_cache["confidence"] = np.round(clean_confs, 4).astype(np.float32)
    clean_severities = [get_flow_severity(cname, conf) for cname, conf in zip(clean_cache["pred_class"], clean_confs)]
    clean_cache["severity"] = clean_severities
    clean_cache["true_label_idx"] = y_clean.astype(np.int32)
    clean_cache["true_label"] = [class_names[i] for i in y_clean]

    # 2. FGSM Perturbation via DNN surrogate
    dnn_model = ensemble.dnn_model
    dnn_model.eval()
    classifier_dnn = PyTorchClassifier(
        model=dnn_model, loss=nn.CrossEntropyLoss(), input_shape=(19,), nb_classes=8, clip_values=None
    )
    attack = FastGradientMethod(estimator=classifier_dnn, eps=epsilon, batch_size=128)
    X_adv_raw = attack.generate(x=X_clean)

    # Physical domain constraints enforcement (unscale -> clip -> round -> rescale)
    X_adv_constrained = enforcer.enforce_constraints(X_adv_raw, X_clean)

    # Recompute B2 ensemble probabilities on perturbed flows
    adv_probs = ensemble.predict_probabilities(pd.DataFrame(X_adv_constrained, columns=selected_features))["weighted"]
    adv_preds = np.argmax(adv_probs, axis=1)
    adv_confs = np.max(adv_probs, axis=1)
    adv_acc = float((adv_preds == y_clean).mean())

    adv_cache = pd.DataFrame(index=df_sample.index)
    for idx, cname in enumerate(class_names):
        adv_cache[f"prob_{cname}"] = adv_probs[:, idx].astype(np.float32)
    adv_cache["pred_class_idx"] = adv_preds.astype(np.int32)
    adv_cache["pred_class"] = [class_names[i] for i in adv_preds]
    adv_cache["confidence"] = np.round(adv_confs, 4).astype(np.float32)
    adv_severities = [get_flow_severity(cname, conf) for cname, conf in zip(adv_cache["pred_class"], adv_confs)]
    adv_cache["severity"] = adv_severities
    adv_cache["true_label_idx"] = y_clean.astype(np.int32)
    adv_cache["true_label"] = [class_names[i] for i in y_clean]

    log.info(f"Ensemble B2 accuracy under FGSM eps={epsilon}: {adv_acc * 100:.2f}% (Clean: {clean_acc * 100:.2f}%)")

    # 3. Mechanism Audit: Analyze Benign vs Attack flow shifts
    benign_mask = np.array([c == "Benign" for c in clean_cache["true_label"]])
    total_benign = int(benign_mask.sum())
    total_attack = int((~benign_mask).sum())

    clean_benign_correct = int((clean_preds[benign_mask] == y_clean[benign_mask]).sum())
    adv_benign_correct = int((adv_preds[benign_mask] == y_clean[benign_mask]).sum())
    adv_benign_flipped_to_attack = int((adv_preds[benign_mask] != y_clean[benign_mask]).sum())

    clean_benign_sevs = collections.Counter([clean_severities[i] for i in range(len(y_clean)) if benign_mask[i]])
    adv_benign_sevs = collections.Counter([adv_severities[i] for i in range(len(y_clean)) if benign_mask[i]])
    adv_benign_pred_classes = collections.Counter([adv_cache["pred_class"].iloc[i] for i in range(len(y_clean)) if benign_mask[i]])

    clean_attack_correct = int((clean_preds[~benign_mask] == y_clean[~benign_mask]).sum())
    adv_attack_correct = int((adv_preds[~benign_mask] == y_clean[~benign_mask]).sum())
    adv_attack_evading_as_benign = int((adv_preds[~benign_mask] == 0).sum())

    mechanism_audit = {
        "total_flows": sample_size,
        "total_benign_flows": total_benign,
        "total_attack_flows": total_attack,
        "benign_accuracy_clean": round(clean_benign_correct / max(1, total_benign) * 100, 2),
        "benign_accuracy_adv": round(adv_benign_correct / max(1, total_benign) * 100, 2),
        "benign_flipped_to_attack_count": adv_benign_flipped_to_attack,
        "benign_flipped_to_attack_pct": round(adv_benign_flipped_to_attack / max(1, total_benign) * 100, 2),
        "clean_benign_severities": dict(clean_benign_sevs),
        "adv_benign_severities": dict(adv_benign_sevs),
        "adv_benign_predicted_classes": dict(adv_benign_pred_classes),
        "attack_accuracy_clean": round(clean_attack_correct / max(1, total_attack) * 100, 2),
        "attack_accuracy_adv": round(adv_attack_correct / max(1, total_attack) * 100, 2),
        "attack_evading_as_benign_count": adv_attack_evading_as_benign,
        "attack_evading_as_benign_pct": round(adv_attack_evading_as_benign / max(1, total_attack) * 100, 2),
    }

    return clean_cache, adv_cache, clean_acc, adv_acc, mechanism_audit


def evaluate_policy_on_cache_env(
    policy: BasePolicy,
    env: TacticalDefenseEnv,
    num_episodes: int = FINAL_EVAL_EPISODES,
    base_seed: int = 4000,
) -> Dict:
    episode_returns: List[float] = []
    total_attacks = 0
    contained_attacks = 0
    total_benigns = 0
    disrupted_benigns = 0
    total_notifications = 0

    for ep in range(num_episodes):
        obs = env.reset(seed=base_seed + ep)
        done = False
        ep_ret = 0.0
        info: Optional[Dict] = None

        while not done:
            if isinstance(policy, (LabelAwareRulePolicy, BudgetAwareLabelAwareRulePolicy)):
                flow_idx = env.episode_flow_indices[env.current_step]
                is_att = (env.true_labels[flow_idx] != "Benign")
                actions = policy.get_actions(
                    obs,
                    info={"is_attack": is_att, "true_label": env.true_labels[flow_idx]},
                )
            else:
                actions = policy.get_actions(obs, info=info)

            obs, reward, done, info = env.step(actions)
            ep_ret += reward

            if info["is_attack"]:
                total_attacks += 1
                if info["attack_contained"]:
                    contained_attacks += 1
            else:
                total_benigns += 1
                if info["benign_disrupted"]:
                    disrupted_benigns += 1

            if info["analyst_notified"]:
                total_notifications += 1

        episode_returns.append(ep_ret)

    ret_ci = bootstrap_ci_mean(episode_returns, num_resamples=10_000, seed=base_seed)
    att_wilson = wilson_score_interval(contained_attacks, total_attacks)
    ben_wilson = wilson_score_interval(disrupted_benigns, total_benigns)

    return {
        "num_episodes": num_episodes,
        "mean_return": ret_ci["mean"],
        "return_ci_lower": ret_ci["ci_lower"],
        "return_ci_upper": ret_ci["ci_upper"],
        "return_std": ret_ci["std"],
        "attacks_contained": att_wilson,
        "benigns_disrupted": ben_wilson,
        "notifications_per_episode": round(float(total_notifications) / float(num_episodes), 2),
        "episode_returns": [round(float(r), 4) for r in episode_returns],
    }


def run_stress_test() -> Dict:
    start_time = time.time()
    log.info("Starting Stage 6 Adversarial Stress Test (Paired Like-for-Like Protocol)...")

    with open(PILOT_JSON_PATH, "r", encoding="utf-8") as f:
        pilot_meta = json.load(f)
    setting = pilot_meta["active_settings"][0]  # balanced

    # Generate paired caches from identical 500 test flows
    clean_cache_df, adv_cache_df, clean_clf_acc, adv_clf_acc, mechanism_audit = generate_paired_test_caches(
        sample_size=STRESS_TEST_SAMPLE_SIZE,
        epsilon=STRESS_TEST_EPSILON,
    )

    clean_env = TacticalDefenseEnv(
        cache_mode="custom",
        custom_cache_df=clean_cache_df,
        reward_setting=setting,
        seed=42,
    )

    adv_env = TacticalDefenseEnv(
        cache_mode="custom",
        custom_cache_df=adv_cache_df,
        reward_setting=setting,
        seed=42,
    )

    baselines_to_eval = {
        "random": RandomPolicy(seed=42),
        "always_none": AlwaysNonePolicy(),
        "max_containment": MaxContainmentPolicy(),
        "severity_rule": SeverityRulePolicy(),
        "severity_rule_budget_aware": BudgetAwareSeverityRulePolicy(),
        "label_aware_rule": LabelAwareRulePolicy(),
        "label_aware_rule_budget_aware": BudgetAwareLabelAwareRulePolicy(),
    }

    stress_results = {
        "metadata": {
            "description": "Stage 6 Adversarial Stress Test with Constrained FGSM Evasion (Paired Like-for-Like)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epsilon": STRESS_TEST_EPSILON,
            "sample_size": STRESS_TEST_SAMPLE_SIZE,
            "clean_classifier_acc": round(clean_clf_acc, 4),
            "adversarial_classifier_acc": round(adv_clf_acc, 4),
            "episodes": FINAL_EVAL_EPISODES,
            "reward_setting": setting,
        },
        "mechanism_audit": mechanism_audit,
        "clean_evaluations": {},
        "adversarial_evaluations": {},
        "paired_comparisons": {},
        "mappo_seeds": {},
        "mappo_aggregate": {},
    }

    # 1. Evaluate All 7 Baselines on Clean and Adversarial Environments
    for b_name, b_policy in baselines_to_eval.items():
        log.info(f"Evaluating baseline: {b_name} on clean & adversarial paired caches...")
        clean_res = evaluate_policy_on_cache_env(b_policy, clean_env, num_episodes=FINAL_EVAL_EPISODES, base_seed=4000)
        adv_res = evaluate_policy_on_cache_env(b_policy, adv_env, num_episodes=FINAL_EVAL_EPISODES, base_seed=4000)

        stress_results["clean_evaluations"][b_name] = clean_res
        stress_results["adversarial_evaluations"][b_name] = adv_res

        # Paired difference (adversarial - clean) per episode
        paired_diff = bootstrap_ci_paired_difference(
            adv_res["episode_returns"],
            clean_res["episode_returns"],
            seed=42,
        )

        stress_results["paired_comparisons"][b_name] = {
            "clean_return": clean_res["mean_return"],
            "clean_return_ci": [clean_res["return_ci_lower"], clean_res["return_ci_upper"]],
            "adv_return": adv_res["mean_return"],
            "adv_return_ci": [adv_res["return_ci_lower"], adv_res["return_ci_upper"]],
            "paired_delta_return": paired_diff["mean_diff"],
            "paired_delta_ci": [paired_diff["ci_lower"], paired_diff["ci_upper"]],
            "decision_outcome": paired_diff["decision_outcome"],
            "clean_attacks_contained": clean_res["attacks_contained"],
            "adv_attacks_contained": adv_res["attacks_contained"],
            "delta_attacks_contained_pct": round(
                adv_res["attacks_contained"]["percentage"] - clean_res["attacks_contained"]["percentage"], 2
            ),
            "clean_benigns_disrupted": clean_res["benigns_disrupted"],
            "adv_benigns_disrupted": adv_res["benigns_disrupted"],
            "delta_benigns_disrupted_pct": round(
                adv_res["benigns_disrupted"]["percentage"] - clean_res["benigns_disrupted"]["percentage"], 2
            ),
            "clean_notifs_per_ep": clean_res["notifications_per_episode"],
            "adv_notifs_per_ep": adv_res["notifications_per_episode"],
        }

        log.info(
            f"  [{b_name:<30}] Clean: {clean_res['mean_return']:>7.2f} | Adv: {adv_res['mean_return']:>7.2f} | "
            f"Paired Diff: {paired_diff['mean_diff']:>6.2f} [{paired_diff['ci_lower']:>6.2f}, {paired_diff['ci_upper']:>6.2f}] | "
            f"Attacks: {clean_res['attacks_contained']['percentage']:>5.1f}% -> {adv_res['attacks_contained']['percentage']:>5.1f}%"
        )

    # 2. Evaluate MAPPO Checkpoints on Clean and Adversarial Environments
    plan = pilot_meta.get("training_plan", {})
    seeds_to_eval = plan.get(
        setting,
        [s for s in TRAIN_SEEDS if (MARL_MODELS_DIR / f"mappo_{setting}_seed{s}_best.pt").exists()],
    )

    clean_seed_matrix = []
    adv_seed_matrix = []

    for seed in seeds_to_eval:
        ckpt_path = MARL_MODELS_DIR / f"mappo_{setting}_seed{seed}_best.pt"
        log.info(f"Stress-testing MAPPO seed {seed} from {ckpt_path} ...")
        mappo_policy = MAPPOEvaluatorPolicy(ckpt_path)

        clean_seed_res = evaluate_policy_on_cache_env(mappo_policy, clean_env, num_episodes=FINAL_EVAL_EPISODES, base_seed=4000)
        adv_seed_res = evaluate_policy_on_cache_env(mappo_policy, adv_env, num_episodes=FINAL_EVAL_EPISODES, base_seed=4000)

        clean_seed_matrix.append(clean_seed_res["episode_returns"])
        adv_seed_matrix.append(adv_seed_res["episode_returns"])

        paired_seed_diff = bootstrap_ci_paired_difference(
            adv_seed_res["episode_returns"],
            clean_seed_res["episode_returns"],
            seed=seed,
        )

        seed_entry = {
            "clean_eval": clean_seed_res,
            "adv_eval": adv_seed_res,
            "paired_difference": paired_seed_diff,
        }
        stress_results["mappo_seeds"][f"seed_{seed}"] = seed_entry
        stress_results["paired_comparisons"][f"mappo_seed_{seed}"] = {
            "clean_return": clean_seed_res["mean_return"],
            "clean_return_ci": [clean_seed_res["return_ci_lower"], clean_seed_res["return_ci_upper"]],
            "adv_return": adv_seed_res["mean_return"],
            "adv_return_ci": [adv_seed_res["return_ci_lower"], adv_seed_res["return_ci_upper"]],
            "paired_delta_return": paired_seed_diff["mean_diff"],
            "paired_delta_ci": [paired_seed_diff["ci_lower"], paired_seed_diff["ci_upper"]],
            "decision_outcome": paired_seed_diff["decision_outcome"],
            "clean_attacks_contained": clean_seed_res["attacks_contained"],
            "adv_attacks_contained": adv_seed_res["attacks_contained"],
            "delta_attacks_contained_pct": round(
                adv_seed_res["attacks_contained"]["percentage"] - clean_seed_res["attacks_contained"]["percentage"], 2
            ),
            "clean_benigns_disrupted": clean_seed_res["benigns_disrupted"],
            "adv_benigns_disrupted": adv_seed_res["benigns_disrupted"],
            "delta_benigns_disrupted_pct": round(
                adv_seed_res["benigns_disrupted"]["percentage"] - clean_seed_res["benigns_disrupted"]["percentage"], 2
            ),
            "clean_notifs_per_ep": clean_seed_res["notifications_per_episode"],
            "adv_notifs_per_ep": adv_seed_res["notifications_per_episode"],
        }

        log.info(
            f"  [MAPPO Seed {seed:<4}] Clean: {clean_seed_res['mean_return']:>7.2f} | Adv: {adv_seed_res['mean_return']:>7.2f} | "
            f"Paired Diff: {paired_seed_diff['mean_diff']:>6.2f} [{paired_seed_diff['ci_lower']:>6.2f}, {paired_seed_diff['ci_upper']:>6.2f}]"
        )

    # 3. Aggregate across MAPPO seeds
    all_clean_arr = np.array(clean_seed_matrix)
    all_adv_arr = np.array(adv_seed_matrix)
    mean_clean_per_ep = np.mean(all_clean_arr, axis=0)
    mean_adv_per_ep = np.mean(all_adv_arr, axis=0)

    agg_clean_ci = bootstrap_ci_mean(mean_clean_per_ep, seed=42)
    agg_adv_ci = bootstrap_ci_mean(mean_adv_per_ep, seed=42)
    agg_paired = bootstrap_ci_paired_difference(mean_adv_per_ep, mean_clean_per_ep, seed=42)

    agg_clean_att_x = sum(stress_results["mappo_seeds"][f"seed_{s}"]["clean_eval"]["attacks_contained"]["x"] for s in seeds_to_eval)
    agg_clean_att_y = sum(stress_results["mappo_seeds"][f"seed_{s}"]["clean_eval"]["attacks_contained"]["n"] for s in seeds_to_eval)
    agg_adv_att_x = sum(stress_results["mappo_seeds"][f"seed_{s}"]["adv_eval"]["attacks_contained"]["x"] for s in seeds_to_eval)
    agg_adv_att_y = sum(stress_results["mappo_seeds"][f"seed_{s}"]["adv_eval"]["attacks_contained"]["n"] for s in seeds_to_eval)

    agg_clean_ben_x = sum(stress_results["mappo_seeds"][f"seed_{s}"]["clean_eval"]["benigns_disrupted"]["x"] for s in seeds_to_eval)
    agg_clean_ben_y = sum(stress_results["mappo_seeds"][f"seed_{s}"]["clean_eval"]["benigns_disrupted"]["n"] for s in seeds_to_eval)
    agg_adv_ben_x = sum(stress_results["mappo_seeds"][f"seed_{s}"]["adv_eval"]["benigns_disrupted"]["x"] for s in seeds_to_eval)
    agg_adv_ben_y = sum(stress_results["mappo_seeds"][f"seed_{s}"]["adv_eval"]["benigns_disrupted"]["n"] for s in seeds_to_eval)

    agg_clean_notifs = float(np.mean([stress_results["mappo_seeds"][f"seed_{s}"]["clean_eval"]["notifications_per_episode"] for s in seeds_to_eval]))
    agg_adv_notifs = float(np.mean([stress_results["mappo_seeds"][f"seed_{s}"]["adv_eval"]["notifications_per_episode"] for s in seeds_to_eval]))

    clean_att_wilson = wilson_score_interval(agg_clean_att_x, agg_clean_att_y)
    adv_att_wilson = wilson_score_interval(agg_adv_att_x, agg_adv_att_y)
    clean_ben_wilson = wilson_score_interval(agg_clean_ben_x, agg_clean_ben_y)
    adv_ben_wilson = wilson_score_interval(agg_adv_ben_x, agg_adv_ben_y)

    stress_results["mappo_aggregate"] = {
        "clean_eval": {
            "mean_return": agg_clean_ci["mean"],
            "return_ci_lower": agg_clean_ci["ci_lower"],
            "return_ci_upper": agg_clean_ci["ci_upper"],
            "return_std": agg_clean_ci["std"],
            "attacks_contained": clean_att_wilson,
            "benigns_disrupted": clean_ben_wilson,
            "notifications_per_episode": round(agg_clean_notifs, 2),
            "episode_returns": [round(float(r), 4) for r in mean_clean_per_ep],
        },
        "adv_eval": {
            "mean_return": agg_adv_ci["mean"],
            "return_ci_lower": agg_adv_ci["ci_lower"],
            "return_ci_upper": agg_adv_ci["ci_upper"],
            "return_std": agg_adv_ci["std"],
            "attacks_contained": adv_att_wilson,
            "benigns_disrupted": adv_ben_wilson,
            "notifications_per_episode": round(agg_adv_notifs, 2),
            "episode_returns": [round(float(r), 4) for r in mean_adv_per_ep],
        },
        "paired_difference": agg_paired,
    }

    stress_results["paired_comparisons"]["mappo_aggregate"] = {
        "clean_return": agg_clean_ci["mean"],
        "clean_return_ci": [agg_clean_ci["ci_lower"], agg_clean_ci["ci_upper"]],
        "adv_return": agg_adv_ci["mean"],
        "adv_return_ci": [agg_adv_ci["ci_lower"], agg_adv_ci["ci_upper"]],
        "paired_delta_return": agg_paired["mean_diff"],
        "paired_delta_ci": [agg_paired["ci_lower"], agg_paired["ci_upper"]],
        "decision_outcome": agg_paired["decision_outcome"],
        "clean_attacks_contained": clean_att_wilson,
        "adv_attacks_contained": adv_att_wilson,
        "delta_attacks_contained_pct": round(adv_att_wilson["percentage"] - clean_att_wilson["percentage"], 2),
        "clean_benigns_disrupted": clean_ben_wilson,
        "adv_benigns_disrupted": adv_ben_wilson,
        "delta_benigns_disrupted_pct": round(adv_ben_wilson["percentage"] - clean_ben_wilson["percentage"], 2),
        "clean_notifs_per_ep": round(agg_clean_notifs, 2),
        "adv_notifs_per_ep": round(agg_adv_notifs, 2),
    }

    # Best non-learned baseline under adversarial conditions
    best_non_learned_clean = max(
        NON_LEARNED_BASELINES,
        key=lambda b: stress_results["clean_evaluations"][b]["mean_return"],
    )
    best_non_learned_adv = max(
        NON_LEARNED_BASELINES,
        key=lambda b: stress_results["adversarial_evaluations"][b]["mean_return"],
    )

    stress_results["comparative_summary"] = {
        "best_non_learned_clean": best_non_learned_clean,
        "best_non_learned_adv": best_non_learned_adv,
        "mappo_vs_always_none_under_attack": {
            "mappo_adv_return": agg_adv_ci["mean"],
            "always_none_adv_return": stress_results["adversarial_evaluations"]["always_none"]["mean_return"],
            "mappo_minus_always_none": round(
                agg_adv_ci["mean"] - stress_results["adversarial_evaluations"]["always_none"]["mean_return"], 2
            ),
        },
        "mappo_vs_best_non_learned_under_attack": {
            "mappo_adv_return": agg_adv_ci["mean"],
            "best_non_learned_name": best_non_learned_adv,
            "best_non_learned_adv_return": stress_results["adversarial_evaluations"][best_non_learned_adv]["mean_return"],
            "mappo_minus_best_non_learned": round(
                agg_adv_ci["mean"] - stress_results["adversarial_evaluations"][best_non_learned_adv]["mean_return"], 2
            ),
        },
    }

    log.info(
        f"=== MAPPO Aggregate Paired Diff: {agg_paired['mean_diff']:>6.2f} "
        f"[{agg_paired['ci_lower']:>6.2f}, {agg_paired['ci_upper']:>6.2f}] | "
        f"Clean: {agg_clean_ci['mean']:>7.2f} -> Adv: {agg_adv_ci['mean']:>7.2f} ==="
    )

    runtime = round(time.time() - start_time, 2)
    peak_ram = round(get_peak_ram_mb(), 2)
    stress_results["metadata"]["runtime_seconds"] = runtime
    stress_results["metadata"]["peak_ram_mb"] = peak_ram

    with open(STRESS_TEST_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(stress_results, f, indent=2)
    log.info(f"Saved adversarial stress test results to {STRESS_TEST_JSON_PATH}")
    log.info(f"Stage 6 completed in {runtime}s, peak RAM {peak_ram} MB.")
    return stress_results


if __name__ == "__main__":
    run_stress_test()
