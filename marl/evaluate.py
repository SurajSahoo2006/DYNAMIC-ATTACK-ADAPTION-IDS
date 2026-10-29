"""
marl/evaluate.py
================
Stage 5: Final Evaluation on Test Cache (Single Evaluation Protocol).

Evaluates the 5 selected MAPPO checkpoints (best rl_val return per seed) alongside
all 5 baselines across the SAME 500 test episodes.

Calculates:
  - Mean episode return per method with 95% bootstrap CI (10,000 resamples).
  - Paired difference (MAPPO - severity_rule) with 95% bootstrap CI and decision outcome.
  - Attacks contained (x of y, pct, 95% Wilson CI).
  - Benign flows disrupted (x of y, pct, 95% Wilson CI).
  - Analyst notifications per episode.
  - Per-attack-family containment rates with Wilson CIs.

Generates:
  - reports/marl_eval.json
  - reports/marl_returns.png
  - reports/marl_learning_curves.png
"""

from __future__ import annotations

import collections
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config_marl import (
    DECISION_RULE_TEXT,
    EVAL_JSON_PATH,
    FINAL_EVAL_EPISODES,
    FINAL_EVAL_SEEDS,
    LEARNING_CURVES_PLOT_PATH,
    MARL_MODELS_DIR,
    PILOT_JSON_PATH,
    REPORTS_DIR,
    RETURNS_PLOT_PATH,
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
from marl.mappo import MAPPOTeam
from marl.stats import (
    bootstrap_ci_mean,
    bootstrap_ci_paired_difference,
    wilson_score_interval,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger("marl_eval")


def get_peak_ram_mb() -> float:
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


class MAPPOEvaluatorPolicy(BasePolicy):
    """Wraps a trained PyTorch MAPPO team as a BasePolicy."""

    def __init__(self, checkpoint_path: Path):
        self.team = MAPPOTeam()
        self.meta = self.team.load_checkpoint(checkpoint_path)

    def get_actions(self, observations: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int, int]:
        actions, _, _ = self.team.select_actions(observations, deterministic=True)
        return actions


def run_evaluation_on_test(
    policy: BasePolicy,
    env: TacticalDefenseEnv,
    num_episodes: int = FINAL_EVAL_EPISODES,
    base_seed: int = 2000,
) -> Dict:
    episode_returns: List[float] = []
    total_attacks = 0
    contained_attacks = 0
    total_benigns = 0
    disrupted_benigns = 0
    total_notifications = 0

    family_stats = collections.defaultdict(lambda: {"contained": 0, "total": 0})

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        obs = env.reset(seed=ep_seed)
        done = False
        ep_return = 0.0
        info: Optional[Dict] = None

        while not done:
            if isinstance(policy, OraclePolicy):
                flow_idx = env.episode_flow_indices[env.current_step]
                is_att = (env.true_labels[flow_idx] != "Benign")
                actions = policy.get_actions(obs, info={"is_attack": is_att})
            else:
                actions = policy.get_actions(obs, info=info)

            obs, reward, done, info = env.step(actions)
            ep_return += reward

            if info["is_attack"]:
                total_attacks += 1
                fam = info["campaign_family"]
                family_stats[fam]["total"] += 1
                if info["attack_contained"]:
                    contained_attacks += 1
                    family_stats[fam]["contained"] += 1
            else:
                total_benigns += 1
                if info["benign_disrupted"]:
                    disrupted_benigns += 1

            if info["analyst_notified"]:
                total_notifications += 1

        episode_returns.append(ep_return)

    return_ci = bootstrap_ci_mean(episode_returns, num_resamples=10_000, seed=base_seed)
    attacks_wilson = wilson_score_interval(contained_attacks, total_attacks)
    benigns_wilson = wilson_score_interval(disrupted_benigns, total_benigns)

    # Per-family breakdowns
    per_family: Dict[str, Dict] = {}
    for fam in sorted(family_stats.keys()):
        c = family_stats[fam]["contained"]
        t = family_stats[fam]["total"]
        per_family[fam] = wilson_score_interval(c, t)

    return {
        "num_episodes": num_episodes,
        "mean_return": return_ci["mean"],
        "return_ci_lower": return_ci["ci_lower"],
        "return_ci_upper": return_ci["ci_upper"],
        "return_std": return_ci["std"],
        "attacks_contained": attacks_wilson,
        "benigns_disrupted": benigns_wilson,
        "notifications_per_episode": round(float(total_notifications) / float(num_episodes), 2),
        "per_family_containment": per_family,
        "episode_returns": [round(float(r), 4) for r in episode_returns],
    }


def generate_plots(eval_results: Dict, active_settings: List[str]):
    """Generates returns comparison and training learning curve plots."""
    # 1. Returns Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = []
    means = []
    ci_err_low = []
    ci_err_high = []

    for setting in active_settings:
        set_data = eval_results["settings"][setting]
        # Baselines
        for b_name in ["random", "always_none", "max_containment", "severity_rule", "oracle"]:
            b_info = set_data["baselines"][b_name]
            methods.append(f"{b_name}\n({setting})")
            m = b_info["mean_return"]
            means.append(m)
            ci_err_low.append(m - b_info["return_ci_lower"])
            ci_err_high.append(b_info["return_ci_upper"] - m)

        # MAPPO Aggregate
        agg_info = set_data["mappo_aggregate"]
        methods.append(f"MAPPO Mean\n({setting})")
        m = agg_info["mean_return"]
        means.append(m)
        ci_err_low.append(m - agg_info["return_ci_lower"])
        ci_err_high.append(agg_info["return_ci_upper"] - m)

    x = np.arange(len(methods))
    yerr = [ci_err_low, ci_err_high]

    colors = ["#7f7f7f", "#bcbd22", "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    bar_colors = (colors * ((len(methods) // len(colors)) + 1))[: len(methods)]

    ax.bar(x, means, yerr=yerr, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black")
    ax.set_ylabel("Mean Episode Return (T=50)")
    ax.set_title("Tactical Defense Policy Returns on Test Cache (500 Episodes, 95% Bootstrap CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.axhline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(RETURNS_PLOT_PATH, dpi=150)
    plt.close()
    log.info(f"Saved returns plot to {RETURNS_PLOT_PATH}")

    # 2. Learning Curves Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for seed in TRAIN_SEEDS:
        train_log = REPORTS_DIR / f"marl_train_balanced_seed{seed}.json"
        if train_log.exists():
            with open(train_log, "r", encoding="utf-8") as f:
                d = json.load(f)
            lc = d.get("learning_curve", [])
            steps = [e["step"] for e in lc]
            rets = [e["rl_val_return"] for e in lc]
            ax.plot(steps, rets, marker="o", label=f"Seed {seed} (best: {d['best_val_return']:.2f})")

    ax.set_xlabel("Environment Training Steps")
    ax.set_ylabel("Validation Return (rl_val, 20 Episodes)")
    ax.set_title("MAPPO Learning Curves Across 5 Random Seeds ('balanced' Setting)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(LEARNING_CURVES_PLOT_PATH, dpi=150)
    plt.close()
    log.info(f"Saved learning curves plot to {LEARNING_CURVES_PLOT_PATH}")


def main():
    start_time = time.time()
    log.info("Starting Stage 5 Final Test Cache Evaluation...")

    with open(PILOT_JSON_PATH, "r", encoding="utf-8") as f:
        pilot_meta = json.load(f)
    active_settings = pilot_meta["active_settings"]
    log.info(f"Evaluating settings: {active_settings}")

    eval_output = {
        "metadata": {
            "description": "Stage 5 Final Test Cache Evaluation (500 Episodes)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_test_episodes": FINAL_EVAL_EPISODES,
            "decision_rule": DECISION_RULE_TEXT,
            "active_settings": active_settings,
            "skipped_settings_reason": pilot_meta.get("skipped_settings_reason"),
        },
        "settings": {},
    }

    baselines_to_eval = {
        "random": RandomPolicy(seed=42),
        "always_none": AlwaysNonePolicy(),
        "max_containment": MaxContainmentPolicy(),
        "severity_rule": SeverityRulePolicy(),
        "oracle": OraclePolicy(),
    }

    for setting in active_settings:
        log.info(f"=== Evaluating Setting: '{setting}' on Test Cache ===")
        setting_record: Dict[str, Dict] = {
            "baselines": {},
            "mappo_seeds": {},
            "mappo_aggregate": {},
            "paired_differences": {},
        }
        env = TacticalDefenseEnv(cache_mode="test", reward_setting=setting, seed=42)

        # 1. Evaluate Baselines
        for b_name, b_policy in baselines_to_eval.items():
            log.info(f"Evaluating baseline: {b_name} ...")
            b_res = run_evaluation_on_test(b_policy, env, num_episodes=FINAL_EVAL_EPISODES, base_seed=3000)
            setting_record["baselines"][b_name] = b_res
            log.info(
                f"  [{b_name:<16}] Mean Return: {b_res['mean_return']:>8.2f} "
                f"[{b_res['return_ci_lower']:>8.2f}, {b_res['return_ci_upper']:>8.2f}] | "
                f"Attacks: {b_res['attacks_contained']['formatted']}"
            )

        # 2. Evaluate Selected MAPPO Checkpoints (5 seeds)
        seed_returns_matrix = []
        for seed in TRAIN_SEEDS:
            ckpt_path = MARL_MODELS_DIR / f"mappo_{setting}_seed{seed}_best.pt"
            log.info(f"Evaluating MAPPO seed {seed} from {ckpt_path} ...")
            mappo_policy = MAPPOEvaluatorPolicy(ckpt_path)
            seed_res = run_evaluation_on_test(mappo_policy, env, num_episodes=FINAL_EVAL_EPISODES, base_seed=3000)
            setting_record["mappo_seeds"][f"seed_{seed}"] = seed_res
            seed_returns_matrix.append(seed_res["episode_returns"])

            # Compute paired difference for this seed against severity_rule
            sev_returns = setting_record["baselines"]["severity_rule"]["episode_returns"]
            paired_seed = bootstrap_ci_paired_difference(seed_res["episode_returns"], sev_returns, seed=seed)
            setting_record["paired_differences"][f"seed_{seed}"] = paired_seed
            log.info(
                f"  [MAPPO Seed {seed:<4}] Mean Return: {seed_res['mean_return']:>8.2f} "
                f"[{seed_res['return_ci_lower']:>8.2f}, {seed_res['return_ci_upper']:>8.2f}] | "
                f"Attacks: {seed_res['attacks_contained']['formatted']} | "
                f"Paired Diff vs Severity: {paired_seed['mean_diff']:>6.2f} "
                f"[{paired_seed['ci_lower']:>6.2f}, {paired_seed['ci_upper']:>6.2f}] -> {paired_seed['decision_outcome']}"
            )

        # 3. Aggregate across MAPPO seeds
        all_seed_returns_arr = np.array(seed_returns_matrix)  # shape (5, 500)
        mean_returns_per_episode = np.mean(all_seed_returns_arr, axis=0)  # shape (500,)
        agg_ci = bootstrap_ci_mean(mean_returns_per_episode, seed=42)

        # Total attacks & benigns across seeds
        agg_attacks_x = sum(setting_record["mappo_seeds"][f"seed_{s}"]["attacks_contained"]["x"] for s in TRAIN_SEEDS)
        agg_attacks_y = sum(setting_record["mappo_seeds"][f"seed_{s}"]["attacks_contained"]["n"] for s in TRAIN_SEEDS)
        agg_benigns_x = sum(setting_record["mappo_seeds"][f"seed_{s}"]["benigns_disrupted"]["x"] for s in TRAIN_SEEDS)
        agg_benigns_y = sum(setting_record["mappo_seeds"][f"seed_{s}"]["benigns_disrupted"]["n"] for s in TRAIN_SEEDS)
        agg_notifs = np.mean([setting_record["mappo_seeds"][f"seed_{s}"]["notifications_per_episode"] for s in TRAIN_SEEDS])

        setting_record["mappo_aggregate"] = {
            "mean_return": agg_ci["mean"],
            "return_ci_lower": agg_ci["ci_lower"],
            "return_ci_upper": agg_ci["ci_upper"],
            "return_std": agg_ci["std"],
            "attacks_contained": wilson_score_interval(agg_attacks_x, agg_attacks_y),
            "benigns_disrupted": wilson_score_interval(agg_benigns_x, agg_benigns_y),
            "notifications_per_episode": round(float(agg_notifs), 2),
            "episode_returns": [round(float(r), 4) for r in mean_returns_per_episode],
        }

        # Paired difference of Aggregate MAPPO vs Severity Rule
        sev_returns = setting_record["baselines"]["severity_rule"]["episode_returns"]
        overall_paired = bootstrap_ci_paired_difference(mean_returns_per_episode, sev_returns, seed=42)
        setting_record["paired_differences"]["overall"] = overall_paired
        log.info(
            f"=== MAPPO Overall Aggregate Return: {agg_ci['mean']:>8.2f} "
            f"[{agg_ci['ci_lower']:>8.2f}, {agg_ci['ci_upper']:>8.2f}] | "
            f"Paired Diff vs Severity: {overall_paired['mean_diff']:>6.2f} "
            f"[{overall_paired['ci_lower']:>6.2f}, {overall_paired['ci_upper']:>6.2f}] -> {overall_paired['decision_outcome']} ==="
        )

        eval_output["settings"][setting] = setting_record

    runtime = round(time.time() - start_time, 2)
    peak_ram = round(get_peak_ram_mb(), 2)
    eval_output["metadata"]["runtime_seconds"] = runtime
    eval_output["metadata"]["peak_ram_mb"] = peak_ram

    with open(EVAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)
    log.info(f"Saved final evaluation JSON to {EVAL_JSON_PATH}")

    # Generate plots
    generate_plots(eval_output, active_settings)
    log.info(f"Stage 5 completed in {runtime}s, peak RAM {peak_ram} MB.")


if __name__ == "__main__":
    main()
