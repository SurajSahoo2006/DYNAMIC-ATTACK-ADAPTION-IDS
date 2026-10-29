"""
marl/stress_test.py
===================
Stage 6: Adversarial Perturbation Stress Test.

Re-uses Phase 4 FGSM transfer attack (epsilon = 0.10) with physical domain constraint
enforcement and the repaired DNN surrogate over 500 test flows.
Evaluates the robustness of the trained MAPPO team and baselines when classifier inferences
suffer adversarial evasion degradation.

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
    PILOT_JSON_PATH,
    STRESS_TEST_EPSILON,
    STRESS_TEST_JSON_PATH,
    STRESS_TEST_SAMPLE_SIZE,
    STRESS_TEST_TIME_LIMIT_SEC,
    TRAIN_SEEDS,
)
from marl.baselines import (
    AlwaysNonePolicy,
    MaxContainmentPolicy,
    OraclePolicy,
    RandomPolicy,
    SeverityRulePolicy,
)
from marl.env import BasePolicy, TacticalDefenseEnv
from marl.evaluate import MAPPOEvaluatorPolicy
from marl.stats import bootstrap_ci_mean, wilson_score_interval

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger("marl_stress_test")


def get_peak_ram_mb() -> float:
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def generate_adversarial_test_cache(
    sample_size: int = STRESS_TEST_SAMPLE_SIZE,
    epsilon: float = STRESS_TEST_EPSILON,
) -> Tuple[pd.DataFrame, float, float]:
    """Generates physically constrained FGSM perturbed test cache."""
    log.info(f"Generating adversarial test flows ({sample_size} flows, eps={epsilon})...")
    le_8, class_names, selected_features, scaler, unscaler, enforcer, df_test, ensemble = load_shared_environment()

    # Stratified test sample
    df_sample = sample_stratified_subset(df_test, n_samples=sample_size, seed=42)
    X_clean = df_sample[selected_features].to_numpy(dtype=np.float32)
    y_clean = df_sample["label_8"].to_numpy(dtype=np.int64)

    # Clean accuracy
    clean_probs = ensemble.predict_probabilities(pd.DataFrame(X_clean, columns=selected_features))["weighted"]
    clean_preds = np.argmax(clean_probs, axis=1)
    clean_acc = float((clean_preds == y_clean).mean())

    # FGSM perturbation via DNN surrogate
    dnn_model = ensemble.dnn_model
    dnn_model.eval()
    classifier_dnn = PyTorchClassifier(
        model=dnn_model, loss=nn.CrossEntropyLoss(), input_shape=(19,), nb_classes=8, clip_values=None
    )
    attack = FastGradientMethod(estimator=classifier_dnn, eps=epsilon, batch_size=128)
    X_adv_raw = attack.generate(x=X_clean)

    # Physical domain constraints enforcement (unscale -> clip -> round -> rescale)
    X_adv_constrained = enforcer.enforce_constraints(X_adv_raw, X_clean)

    # Recompute B2 ensemble probabilities
    adv_probs = ensemble.predict_probabilities(pd.DataFrame(X_adv_constrained, columns=selected_features))["weighted"]
    adv_preds = np.argmax(adv_probs, axis=1)
    adv_confs = np.max(adv_probs, axis=1)
    adv_acc = float((adv_preds == y_clean).mean())

    log.info(f"Ensemble B2 accuracy under FGSM eps={epsilon}: {adv_acc * 100:.2f}% (Clean: {clean_acc * 100:.2f}%)")

    # Build cache DataFrame
    adv_cache = pd.DataFrame(index=df_sample.index)
    for idx, cname in enumerate(class_names):
        adv_cache[f"prob_{cname}"] = adv_probs[:, idx].astype(np.float32)

    adv_cache["pred_class_idx"] = adv_preds.astype(np.int32)
    adv_cache["pred_class"] = [class_names[i] for i in adv_preds]
    adv_cache["confidence"] = np.round(adv_confs, 4).astype(np.float32)

    # Severities based on degraded inferences
    severities = [get_flow_severity(cname, conf) for cname, conf in zip(adv_cache["pred_class"], adv_confs)]
    adv_cache["severity"] = severities

    adv_cache["true_label_idx"] = y_clean.astype(np.int32)
    adv_cache["true_label"] = [class_names[i] for i in y_clean]

    return adv_cache, clean_acc, adv_acc


def evaluate_policy_on_adv_env(
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
            if isinstance(policy, OraclePolicy):
                flow_idx = env.episode_flow_indices[env.current_step]
                is_att = (env.true_labels[flow_idx] != "Benign")
                actions = policy.get_actions(obs, info={"is_attack": is_att})
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
    log.info("Starting Stage 6 Adversarial Stress Test...")

    # Load clean eval results for comparison
    with open(EVAL_JSON_PATH, "r", encoding="utf-8") as f:
        clean_eval = json.load(f)

    with open(PILOT_JSON_PATH, "r", encoding="utf-8") as f:
        pilot_meta = json.load(f)
    setting = pilot_meta["active_settings"][0]  # balanced

    # Generate adversarial cache
    adv_cache_df, clean_clf_acc, adv_clf_acc = generate_adversarial_test_cache(
        sample_size=STRESS_TEST_SAMPLE_SIZE,
        epsilon=STRESS_TEST_EPSILON,
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
        "oracle": OraclePolicy(),
    }

    stress_results = {
        "metadata": {
            "description": "Stage 6 Adversarial Stress Test with Constrained FGSM Evasion",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epsilon": STRESS_TEST_EPSILON,
            "sample_size": STRESS_TEST_SAMPLE_SIZE,
            "clean_classifier_acc": round(clean_clf_acc, 4),
            "adversarial_classifier_acc": round(adv_clf_acc, 4),
            "episodes": FINAL_EVAL_EPISODES,
            "reward_setting": setting,
        },
        "baselines": {},
        "mappo_seeds": {},
        "mappo_aggregate": {},
        "comparison_vs_clean": {},
    }

    # 1. Evaluate Baselines under adversarial environment
    for b_name, b_policy in baselines_to_eval.items():
        log.info(f"Stress-testing baseline: {b_name} ...")
        res = evaluate_policy_on_adv_env(b_policy, adv_env, num_episodes=FINAL_EVAL_EPISODES, base_seed=3000)
        stress_results["baselines"][b_name] = res

        clean_b = clean_eval["settings"][setting]["baselines"][b_name]
        ret_diff = round(res["mean_return"] - clean_b["mean_return"], 4)
        att_diff = round(res["attacks_contained"]["percentage"] - clean_b["attacks_contained"]["percentage"], 2)

        stress_results["comparison_vs_clean"][b_name] = {
            "clean_return": clean_b["mean_return"],
            "adv_return": res["mean_return"],
            "delta_return": ret_diff,
            "clean_attacks_contained_pct": clean_b["attacks_contained"]["percentage"],
            "adv_attacks_contained_pct": res["attacks_contained"]["percentage"],
            "delta_attacks_contained_pct": att_diff,
        }
        log.info(
            f"  [{b_name:<16}] Adv Return: {res['mean_return']:>8.2f} (Clean: {clean_b['mean_return']:>8.2f}, Delta: {ret_diff:>6.2f}) | "
            f"Attacks: {res['attacks_contained']['formatted']}"
        )

    # 2. Evaluate MAPPO Checkpoints under adversarial environment
    seed_returns_matrix = []
    for seed in TRAIN_SEEDS:
        ckpt_path = MARL_MODELS_DIR / f"mappo_{setting}_seed{seed}_best.pt"
        log.info(f"Stress-testing MAPPO seed {seed} ...")
        mappo_policy = MAPPOEvaluatorPolicy(ckpt_path)
        seed_res = evaluate_policy_on_adv_env(mappo_policy, adv_env, num_episodes=FINAL_EVAL_EPISODES, base_seed=3000)
        stress_results["mappo_seeds"][f"seed_{seed}"] = seed_res
        seed_returns_matrix.append(seed_res["episode_returns"])

        clean_seed = clean_eval["settings"][setting]["mappo_seeds"][f"seed_{seed}"]
        ret_diff = round(seed_res["mean_return"] - clean_seed["mean_return"], 4)
        att_diff = round(seed_res["attacks_contained"]["percentage"] - clean_seed["attacks_contained"]["percentage"], 2)

        stress_results["comparison_vs_clean"][f"mappo_seed_{seed}"] = {
            "clean_return": clean_seed["mean_return"],
            "adv_return": seed_res["mean_return"],
            "delta_return": ret_diff,
            "clean_attacks_contained_pct": clean_seed["attacks_contained"]["percentage"],
            "adv_attacks_contained_pct": seed_res["attacks_contained"]["percentage"],
            "delta_attacks_contained_pct": att_diff,
        }
        log.info(
            f"  [MAPPO Seed {seed:<4}] Adv Return: {seed_res['mean_return']:>8.2f} (Clean: {clean_seed['mean_return']:>8.2f}, Delta: {ret_diff:>6.2f}) | "
            f"Attacks: {seed_res['attacks_contained']['formatted']}"
        )

    # 3. Aggregate across MAPPO seeds
    all_adv_returns = np.array(seed_returns_matrix)
    mean_adv_returns_per_ep = np.mean(all_adv_returns, axis=0)
    agg_adv_ci = bootstrap_ci_mean(mean_adv_returns_per_ep, seed=42)

    agg_att_x = sum(stress_results["mappo_seeds"][f"seed_{s}"]["attacks_contained"]["x"] for s in TRAIN_SEEDS)
    agg_att_y = sum(stress_results["mappo_seeds"][f"seed_{s}"]["attacks_contained"]["n"] for s in TRAIN_SEEDS)
    agg_ben_x = sum(stress_results["mappo_seeds"][f"seed_{s}"]["benigns_disrupted"]["x"] for s in TRAIN_SEEDS)
    agg_ben_y = sum(stress_results["mappo_seeds"][f"seed_{s}"]["benigns_disrupted"]["n"] for s in TRAIN_SEEDS)
    agg_notifs = np.mean([stress_results["mappo_seeds"][f"seed_{s}"]["notifications_per_episode"] for s in TRAIN_SEEDS])

    stress_results["mappo_aggregate"] = {
        "mean_return": agg_adv_ci["mean"],
        "return_ci_lower": agg_adv_ci["ci_lower"],
        "return_ci_upper": agg_adv_ci["ci_upper"],
        "return_std": agg_adv_ci["std"],
        "attacks_contained": wilson_score_interval(agg_att_x, agg_att_y),
        "benigns_disrupted": wilson_score_interval(agg_ben_x, agg_ben_y),
        "notifications_per_episode": round(float(agg_notifs), 2),
        "episode_returns": [round(float(r), 4) for r in mean_adv_returns_per_ep],
    }

    clean_agg = clean_eval["settings"][setting]["mappo_aggregate"]
    agg_delta_ret = round(agg_adv_ci["mean"] - clean_agg["mean_return"], 4)
    agg_delta_att = round(
        stress_results["mappo_aggregate"]["attacks_contained"]["percentage"]
        - clean_agg["attacks_contained"]["percentage"],
        2,
    )

    stress_results["comparison_vs_clean"]["mappo_aggregate"] = {
        "clean_return": clean_agg["mean_return"],
        "adv_return": agg_adv_ci["mean"],
        "delta_return": agg_delta_ret,
        "clean_attacks_contained_pct": clean_agg["attacks_contained"]["percentage"],
        "adv_attacks_contained_pct": stress_results["mappo_aggregate"]["attacks_contained"]["percentage"],
        "delta_attacks_contained_pct": agg_delta_att,
    }

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
