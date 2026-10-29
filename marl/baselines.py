"""
marl/baselines.py
=================
Stage 3: Heuristic, Rule-Based, and Oracle Baselines for Tactical Defense.

Implements and benchmarks 5 baseline policies across reward settings:
  1. Random: Uniform random action selection across all 3 agent action spaces.
  2. Always None: Inaction baseline (accept_verdict, none, none).
  3. Max Containment: Extreme posture baseline (accept_verdict, isolate, forensic_capture).
  4. Severity Rule: Standard SOC playbook rule mapped from ensemble severity:
       - Critical -> (flag_for_review, isolate, notify_analyst)
       - High     -> (flag_for_review, block, notify_analyst)
       - Medium   -> (accept_verdict, rate_limit, none)
       - Low      -> (accept_verdict, none, none)
  5. Oracle: Theoretical upper bound accessing ground truth attack labels (non-deployable).

Saves benchmark metrics to reports/marl_baselines.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config_marl import (
    BASELINE_EVAL_EPISODES,
    BASELINES_JSON_PATH,
    CONTAINMENT_ACTIONS,
    CONTAINMENT_ACTION_TO_IDX,
    EPISODE_STEPS,
    ESCALATION_ACTIONS,
    ESCALATION_ACTION_TO_IDX,
    REWARD_SETTINGS,
    TRIAGE_ACTIONS,
    TRIAGE_ACTION_TO_IDX,
)
from marl.env import BasePolicy, TacticalDefenseEnv
from marl.stats import bootstrap_ci_mean, wilson_score_interval

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger("marl_baselines")


def get_peak_ram_mb() -> float:
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


class RandomPolicy(BasePolicy):
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def get_actions(self, observations: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int, int]:
        a_t = int(self.rng.integers(0, len(TRIAGE_ACTIONS)))
        a_c = int(self.rng.integers(0, len(CONTAINMENT_ACTIONS)))
        a_e = int(self.rng.integers(0, len(ESCALATION_ACTIONS)))
        return (a_t, a_c, a_e)


class AlwaysNonePolicy(BasePolicy):
    def get_actions(self, observations: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int, int]:
        return (
            TRIAGE_ACTION_TO_IDX["accept_verdict"],
            CONTAINMENT_ACTION_TO_IDX["none"],
            ESCALATION_ACTION_TO_IDX["none"],
        )


class MaxContainmentPolicy(BasePolicy):
    def get_actions(self, observations: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int, int]:
        return (
            TRIAGE_ACTION_TO_IDX["accept_verdict"],
            CONTAINMENT_ACTION_TO_IDX["isolate"],
            ESCALATION_ACTION_TO_IDX["forensic_capture"],
        )


class SeverityRulePolicy(BasePolicy):
    def get_actions(self, observations: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int, int]:
        # Observation format: dims 0..7=probs, 8=conf, 9..12=sev_onehot [Low, Medium, High, Critical]
        sev_onehot = observations[0, 9:13]
        sev_idx = int(np.argmax(sev_onehot))  # 0: Low, 1: Medium, 2: High, 3: Critical

        if sev_idx == 3:  # Critical
            return (
                TRIAGE_ACTION_TO_IDX["flag_for_review"],
                CONTAINMENT_ACTION_TO_IDX["isolate"],
                ESCALATION_ACTION_TO_IDX["notify_analyst"],
            )
        elif sev_idx == 2:  # High
            return (
                TRIAGE_ACTION_TO_IDX["flag_for_review"],
                CONTAINMENT_ACTION_TO_IDX["block"],
                ESCALATION_ACTION_TO_IDX["notify_analyst"],
            )
        elif sev_idx == 1:  # Medium
            return (
                TRIAGE_ACTION_TO_IDX["accept_verdict"],
                CONTAINMENT_ACTION_TO_IDX["rate_limit"],
                ESCALATION_ACTION_TO_IDX["none"],
            )
        else:  # Low
            return (
                TRIAGE_ACTION_TO_IDX["accept_verdict"],
                CONTAINMENT_ACTION_TO_IDX["none"],
                ESCALATION_ACTION_TO_IDX["none"],
            )


class OraclePolicy(BasePolicy):
    """
    Theoretical upper bound with perfect knowledge of the ground-truth attack flow label.
    Clearly marked as not deployable in real environments.
    """
    def get_actions(self, observations: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int, int]:
        if info and info.get("is_attack", False):
            return (
                TRIAGE_ACTION_TO_IDX["flag_for_review"],
                CONTAINMENT_ACTION_TO_IDX["isolate"],
                ESCALATION_ACTION_TO_IDX["notify_analyst"],
            )
        else:
            return (
                TRIAGE_ACTION_TO_IDX["accept_verdict"],
                CONTAINMENT_ACTION_TO_IDX["none"],
                ESCALATION_ACTION_TO_IDX["none"],
            )


def evaluate_policy_on_env(
    policy: BasePolicy,
    env: TacticalDefenseEnv,
    num_episodes: int = BASELINE_EVAL_EPISODES,
    base_seed: int = 1000,
) -> Dict:
    episode_returns: List[float] = []
    total_attacks = 0
    contained_attacks = 0
    total_benigns = 0
    disrupted_benigns = 0
    total_notifications = 0

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        obs = env.reset(seed=ep_seed)
        done = False
        ep_return = 0.0
        info: Optional[Dict] = None

        while not done:
            # Oracle checks ground truth from env state
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
                if info["attack_contained"]:
                    contained_attacks += 1
            else:
                total_benigns += 1
                if info["benign_disrupted"]:
                    disrupted_benigns += 1

            if info["analyst_notified"]:
                total_notifications += 1

        episode_returns.append(ep_return)

    # Compute bootstrap CI for return
    return_ci = bootstrap_ci_mean(episode_returns, num_resamples=10_000, seed=base_seed)
    attacks_wilson = wilson_score_interval(contained_attacks, total_attacks)
    benigns_wilson = wilson_score_interval(disrupted_benigns, total_benigns)

    return {
        "num_episodes": num_episodes,
        "mean_return": return_ci["mean"],
        "return_ci_lower": return_ci["ci_lower"],
        "return_ci_upper": return_ci["ci_upper"],
        "return_std": return_ci["std"],
        "attacks_contained_x": contained_attacks,
        "attacks_contained_y": total_attacks,
        "attacks_contained_pct": attacks_wilson["percentage"],
        "attacks_contained_ci_lower": attacks_wilson["ci_lower"],
        "attacks_contained_ci_upper": attacks_wilson["ci_upper"],
        "attacks_contained_formatted": attacks_wilson["formatted"],
        "benigns_disrupted_x": disrupted_benigns,
        "benigns_disrupted_y": total_benigns,
        "benigns_disrupted_pct": benigns_wilson["percentage"],
        "benigns_disrupted_ci_lower": benigns_wilson["ci_lower"],
        "benigns_disrupted_ci_upper": benigns_wilson["ci_upper"],
        "benigns_disrupted_formatted": benigns_wilson["formatted"],
        "notifications_per_episode": round(float(total_notifications) / float(num_episodes), 2),
        "episode_returns": [round(float(r), 4) for r in episode_returns],
    }


def run_baselines_benchmark() -> Dict:
    start_time = time.time()
    log.info("Starting Stage 3 Baselines benchmark...")

    results: Dict[str, Dict] = {
        "metadata": {
            "description": "Stage 3 Heuristic and Oracle Baselines Benchmark",
            "episodes_per_setting": BASELINE_EVAL_EPISODES,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "splits": {},
    }

    policies: Dict[str, BasePolicy] = {
        "random": RandomPolicy(seed=42),
        "always_none": AlwaysNonePolicy(),
        "max_containment": MaxContainmentPolicy(),
        "severity_rule": SeverityRulePolicy(),
        "oracle": OraclePolicy(),
    }

    for split in ["rl_val", "test"]:
        log.info(f"--- Evaluating Baselines on partition: {split} ---")
        results["splits"][split] = {}

        for setting_name in REWARD_SETTINGS.keys():
            log.info(f"Reward Setting: {setting_name} (split: {split})")
            results["splits"][split][setting_name] = {}
            env = TacticalDefenseEnv(cache_mode=split, reward_setting=setting_name, seed=42)

            for policy_name, policy in policies.items():
                eval_metrics = evaluate_policy_on_env(
                    policy=policy,
                    env=env,
                    num_episodes=BASELINE_EVAL_EPISODES,
                    base_seed=1000 if split == "rl_val" else 2000,
                )
                results["splits"][split][setting_name][policy_name] = eval_metrics
                log.info(
                    f"  [{policy_name:<16}] Return: {eval_metrics['mean_return']:>8.2f} "
                    f"[{eval_metrics['return_ci_lower']:>8.2f}, {eval_metrics['return_ci_upper']:>8.2f}] | "
                    f"Attacks: {eval_metrics['attacks_contained_formatted']} | "
                    f"Disrupted: {eval_metrics['benigns_disrupted_formatted']}"
                )

    runtime_sec = round(time.time() - start_time, 2)
    peak_ram = round(get_peak_ram_mb(), 2)
    results["metadata"]["runtime_seconds"] = runtime_sec
    results["metadata"]["peak_ram_mb"] = peak_ram

    with open(BASELINES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved baseline benchmark results to {BASELINES_JSON_PATH}")
    log.info(f"Stage 3 completed in {runtime_sec}s, peak RAM {peak_ram} MB.")
    return results


if __name__ == "__main__":
    run_baselines_benchmark()
