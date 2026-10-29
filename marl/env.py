"""
marl/env.py
===========
Multi-Agent Cooperative Tactical Defense Environment for Network Threat Adaptation.

Implements a deterministic, vectorized simulation of tactical network defense
using real B2 ensemble inferences over CICIoT2023 flows.

Episode Dynamics:
  - Horizon T = 50 steps.
  - Each episode represents a targeted attack campaign: an attack family is selected,
    and a burst of 10 to 25 consecutive attack flows of that family is embedded within
    benign background flows.
  - Consecutive flows are temporally correlated around the campaign burst.

Three Cooperative Agents:
  1. Triage: {0: accept_verdict, 1: flag_for_review, 2: defer}
  2. Containment: {0: none, 1: rate_limit, 2: block, 3: isolate}
  3. Escalation: {0: none, 1: notify_analyst, 2: forensic_capture}

Reward Formula (Module Docstring & Report Specification):
---------------------------------------------------------
At step t with weights (w_D, w_C, w_cost, w_B) from the active reward setting:

If true class is an attack flow:
  damage_term = - w_D * SEVERITY_BASE_DAMAGE[severity] * CONTAINMENT_DAMAGE_MULTIPLIER[active_posture]
  collateral_term = 0.0
  bonus_term = + w_B * CONTAINED_AND_NOTIFIED_BONUS
               if (active_posture != 'none' and escalation_action != 'none') else 0.0

If true class is benign:
  damage_term = 0.0
  collateral_term = - w_C * CONTAINMENT_COLLATERAL_COST[active_posture]
  bonus_term = 0.0

Action Costs & Penalties:
  delay_cost = TRIAGE_DELAY_PENALTY[triage_action]
  containment_cost = CONTAINMENT_ACTION_COST[containment_action]
  esc_discount = TRIAGE_FLAG_DISCOUNT if triage_action == 'flag_for_review' else 0.0
  esc_cost = ESCALATION_ACTION_COST[escalation_action] * (1.0 - esc_discount)
  budget_penalty = EXCESS_BUDGET_PENALTY if (escalation_action != 'none' and escalations_remaining <= 0) else 0.0
  total_action_cost = w_cost * (delay_cost + containment_cost + esc_cost + budget_penalty)

Step Team Reward:
  R_t = damage_term + collateral_term - total_action_cost + bonus_term
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config_marl import (
    AGENT_ID_DIM,
    AGENT_OBS_DIM,
    CACHE_TEST_PATH,
    CACHE_VAL_PATH,
    CENTRAL_CRITIC_OBS_DIM,
    CONTAINED_AND_NOTIFIED_BONUS,
    CONTAINMENT_ACTION_COST,
    CONTAINMENT_ACTION_TO_IDX,
    CONTAINMENT_ACTIONS,
    CONTAINMENT_COLLATERAL_COST,
    CONTAINMENT_DAMAGE_MULTIPLIER,
    ENV_FEATURE_DIM,
    EPISODE_STEPS,
    ESCALATION_ACTION_COST,
    ESCALATION_ACTION_TO_IDX,
    ESCALATION_ACTIONS,
    ESCALATION_BUDGET,
    EXCESS_BUDGET_PENALTY,
    MAX_BURST_LENGTH,
    MIN_BURST_LENGTH,
    NUM_AGENTS,
    POSTURE_DURATION,
    REWARD_SETTINGS,
    RL_TRAIN_INDICES_PATH,
    RL_VAL_INDICES_PATH,
    SEVERITY_BASE_DAMAGE,
    TRIAGE_ACTION_TO_IDX,
    TRIAGE_ACTIONS,
    TRIAGE_DELAY_PENALTY,
    TRIAGE_FLAG_DISCOUNT,
)


class BasePolicy:
    """Interface for joint MARL and baseline policies."""

    def get_actions(self, observations: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int, int]:
        """
        Receives agent observations of shape (3, 28) and returns joint action tuple (a_triage, a_containment, a_escalation).
        """
        raise NotImplementedError


class TacticalDefenseEnv:
    """
    Vectorized cooperative multi-agent tactical defense environment.
    Strictly forbids observation leakage of true attack labels.
    """

    def __init__(
        self,
        cache_mode: str = "rl_train",
        reward_setting: str = "balanced",
        seed: Optional[int] = 42,
        custom_cache_df: Optional[pd.DataFrame] = None,
    ):
        """
        Parameters
        ----------
        cache_mode: 'rl_train', 'rl_val', 'test', or 'custom'
        reward_setting: 'balanced', 'security-first', or 'availability-first'
        seed: random seed for reproducibility
        custom_cache_df: optional custom DataFrame (e.g. for adversarial stress testing)
        """
        self.cache_mode = cache_mode
        self.reward_setting = reward_setting
        if reward_setting not in REWARD_SETTINGS:
            raise ValueError(f"Unknown reward setting '{reward_setting}'. Valid: {list(REWARD_SETTINGS.keys())}")
        self.reward_weights = REWARD_SETTINGS[reward_setting]

        # Load cache partition
        self._load_cache_partition(cache_mode, custom_cache_df)

        self.rng = np.random.default_rng(seed)
        self.seed_val = seed

        # Episode state variables
        self.current_step = 0
        self.episode_flow_indices: List[int] = []
        self.campaign_family = ""
        self.burst_start = 0
        self.burst_length = 0

        # Posture & Budget state
        self.active_posture: str = "none"
        self.posture_steps_remaining: int = 0
        self.escalations_remaining: int = ESCALATION_BUDGET
        self.recent_pred_classes: collections.deque = collections.deque(maxlen=5)

        # Agent IDs one-hot
        self.agent_ids = np.eye(NUM_AGENTS, dtype=np.float32)

    def _load_cache_partition(self, mode: str, custom_df: Optional[pd.DataFrame]):
        if mode == "custom":
            if custom_df is None:
                raise ValueError("custom_cache_df must be provided when mode='custom'")
            self.df = custom_df.copy().reset_index(drop=True)
        elif mode in ("rl_train", "rl_val"):
            df_full = pd.read_parquet(CACHE_VAL_PATH)
            if mode == "rl_train":
                with open(RL_TRAIN_INDICES_PATH, "r", encoding="utf-8") as f:
                    indices = json.load(f)
            else:
                with open(RL_VAL_INDICES_PATH, "r", encoding="utf-8") as f:
                    indices = json.load(f)
            self.df = df_full.iloc[indices].copy().reset_index(drop=True)
        elif mode == "test":
            self.df = pd.read_parquet(CACHE_TEST_PATH).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown cache mode: {mode}")

        # Precompute class index arrays for fast campaign flow retrieval
        self.prob_cols = [c for c in self.df.columns if c.startswith("prob_")]
        self.prob_matrix = self.df[self.prob_cols].to_numpy(dtype=np.float32)
        self.pred_indices = self.df["pred_class_idx"].to_numpy(dtype=np.int32)
        self.confidences = self.df["confidence"].to_numpy(dtype=np.float32)
        self.severities = self.df["severity"].to_numpy()
        self.true_labels = self.df["true_label"].to_numpy()

        self.class_indices: Dict[str, np.ndarray] = {}
        for cname in np.unique(self.true_labels):
            self.class_indices[str(cname)] = np.where(self.true_labels == cname)[0]

        # Available attack families (all non-benign classes present in split)
        self.attack_families = [c for c in self.class_indices.keys() if c != "Benign"]
        if not self.attack_families:
            raise ValueError(f"No attack families found in cache partition {mode}!")

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Resets environment and builds a deterministic campaign episode.
        Returns initial joint observation of shape (3, 28).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_val = seed

        self.current_step = 0
        self.active_posture = "none"
        self.posture_steps_remaining = 0
        self.escalations_remaining = ESCALATION_BUDGET
        self.recent_pred_classes.clear()

        # 1. Sample campaign dynamics
        self.campaign_family = str(self.rng.choice(self.attack_families))
        self.burst_length = int(self.rng.integers(MIN_BURST_LENGTH, MAX_BURST_LENGTH + 1))
        self.burst_start = int(self.rng.integers(0, EPISODE_STEPS - self.burst_length + 1))

        # 2. Sample flow indices for all T steps
        self.episode_flow_indices = []
        attack_pool = self.class_indices[self.campaign_family]
        benign_pool = self.class_indices["Benign"]

        for t in range(EPISODE_STEPS):
            if self.burst_start <= t < (self.burst_start + self.burst_length):
                flow_idx = int(self.rng.choice(attack_pool))
            else:
                flow_idx = int(self.rng.choice(benign_pool))
            self.episode_flow_indices.append(flow_idx)

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the observation vector for each of the 3 agents.
        NO true label is ever included in the observation.
        Shape: (3, 28).
        """
        flow_idx = self.episode_flow_indices[self.current_step]

        # 1. Ensemble probabilities (8 dims)
        probs = self.prob_matrix[flow_idx]

        # 2. Confidence (1 dim)
        conf = self.confidences[flow_idx]

        # 3. One-hot severity (4 dims: Low, Medium, High, Critical)
        sev_str = self.severities[flow_idx]
        sev_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        sev_onehot = np.zeros(4, dtype=np.float32)
        sev_onehot[sev_map.get(sev_str, 1)] = 1.0

        # 4. Normalized step fraction (1 dim)
        step_frac = float(self.current_step) / float(EPISODE_STEPS)

        # 5. Escalations left normalized (1 dim)
        esc_left_norm = max(0.0, float(self.escalations_remaining)) / float(ESCALATION_BUDGET)

        # 6. Containment posture active (1 dim: 0.0 or 1.0)
        posture_active = 1.0 if self.active_posture != "none" else 0.0

        # 7. Posture steps remaining normalized (1 dim)
        posture_rem_norm = float(self.posture_steps_remaining) / float(POSTURE_DURATION)

        # 8. Rolling history counts of last 5 predicted classes (8 dims, normalized by 5)
        history_counts = np.zeros(8, dtype=np.float32)
        for p_idx in self.recent_pred_classes:
            history_counts[p_idx] += 1.0
        history_counts /= 5.0

        # Concatenate 25-dim environment feature vector
        env_obs = np.concatenate([
            probs,
            np.array([conf], dtype=np.float32),
            sev_onehot,
            np.array([step_frac, esc_left_norm, posture_active, posture_rem_norm], dtype=np.float32),
            history_counts,
        ], dtype=np.float32)

        # Construct (3, 28) agent observation matrix
        obs_matrix = np.zeros((NUM_AGENTS, AGENT_OBS_DIM), dtype=np.float32)
        for agent_idx in range(NUM_AGENTS):
            obs_matrix[agent_idx, :ENV_FEATURE_DIM] = env_obs
            obs_matrix[agent_idx, ENV_FEATURE_DIM:] = self.agent_ids[agent_idx]

        return obs_matrix

    def step(self, joint_action: Union[Tuple[int, int, int], List[int], np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executes one joint action transition.
        joint_action: (triage_action, containment_action, escalation_action)
        Returns: (next_obs, reward, done, info)
        """
        a_triage, a_containment, a_escalation = int(joint_action[0]), int(joint_action[1]), int(joint_action[2])

        # Validate action bounds
        if not (0 <= a_triage < len(TRIAGE_ACTIONS)):
            raise ValueError(f"Invalid triage action: {a_triage}")
        if not (0 <= a_containment < len(CONTAINMENT_ACTIONS)):
            raise ValueError(f"Invalid containment action: {a_containment}")
        if not (0 <= a_escalation < len(ESCALATION_ACTIONS)):
            raise ValueError(f"Invalid escalation action: {a_escalation}")

        triage_act_str = TRIAGE_ACTIONS[a_triage]
        contain_act_str = CONTAINMENT_ACTIONS[a_containment]
        esc_act_str = ESCALATION_ACTIONS[a_escalation]

        # Current flow telemetry
        flow_idx = self.episode_flow_indices[self.current_step]
        true_label = self.true_labels[flow_idx]
        is_attack = (true_label != "Benign")
        severity = self.severities[flow_idx]
        pred_idx = self.pred_indices[flow_idx]

        # Update rolling predicted class history
        self.recent_pred_classes.append(pred_idx)

        # -------------------------------------------------------------
        # Containment State Transition
        # -------------------------------------------------------------
        if a_containment > 0:
            # New posture triggered
            self.active_posture = contain_act_str
            self.posture_steps_remaining = POSTURE_DURATION
            containment_action_cost = CONTAINMENT_ACTION_COST[contain_act_str]
        else:
            containment_action_cost = 0.0

        effective_posture = self.active_posture

        # -------------------------------------------------------------
        # Escalation State Transition & Discounting
        # -------------------------------------------------------------
        esc_discount = TRIAGE_FLAG_DISCOUNT if triage_act_str == "flag_for_review" else 0.0
        base_esc_cost = ESCALATION_ACTION_COST[esc_act_str]
        effective_esc_cost = base_esc_cost * (1.0 - esc_discount)

        excess_penalty = 0.0
        if a_escalation > 0:
            if self.escalations_remaining > 0:
                self.escalations_remaining -= 1
            else:
                excess_penalty = EXCESS_BUDGET_PENALTY

        # -------------------------------------------------------------
        # Reward Computation
        # -------------------------------------------------------------
        w_d = self.reward_weights["damage_weight"]
        w_c = self.reward_weights["collateral_weight"]
        w_cost = self.reward_weights["cost_weight"]
        w_b = self.reward_weights["bonus_weight"]

        delay_cost = TRIAGE_DELAY_PENALTY[triage_act_str]

        if is_attack:
            base_dmg = SEVERITY_BASE_DAMAGE.get(severity, 3.0)
            leak_multiplier = CONTAINMENT_DAMAGE_MULTIPLIER[effective_posture]
            damage = base_dmg * leak_multiplier
            damage_term = - (w_d * damage)
            collateral = 0.0
            collateral_term = 0.0

            # Bonus when attack is both contained (posture active) and escalated
            is_contained = (effective_posture != "none")
            is_escalated = (a_escalation > 0)
            if is_contained and is_escalated:
                bonus = CONTAINED_AND_NOTIFIED_BONUS
                bonus_term = w_b * bonus
            else:
                bonus = 0.0
                bonus_term = 0.0
        else:
            damage = 0.0
            damage_term = 0.0
            collateral = CONTAINMENT_COLLATERAL_COST[effective_posture]
            collateral_term = - (w_c * collateral)
            bonus = 0.0
            bonus_term = 0.0

        total_action_cost = w_cost * (delay_cost + containment_action_cost + effective_esc_cost + excess_penalty)
        reward = damage_term + collateral_term - total_action_cost + bonus_term

        # Step tracking metrics
        info = {
            "step": self.current_step,
            "true_label": true_label,
            "is_attack": is_attack,
            "severity": severity,
            "pred_idx": int(pred_idx),
            "effective_posture": effective_posture,
            "triage_action": triage_act_str,
            "containment_action": contain_act_str,
            "escalation_action": esc_act_str,
            "damage": float(damage),
            "collateral": float(collateral),
            "total_action_cost": float(total_action_cost),
            "delay_cost": float(delay_cost),
            "bonus": float(bonus),
            "attack_contained": bool(is_attack and effective_posture != "none"),
            "benign_disrupted": bool((not is_attack) and effective_posture != "none"),
            "analyst_notified": bool(a_escalation > 0),
            "campaign_family": self.campaign_family,
        }

        # Posture duration countdown at step end
        if self.posture_steps_remaining > 0:
            self.posture_steps_remaining -= 1
            if self.posture_steps_remaining == 0:
                self.active_posture = "none"

        self.current_step += 1
        done = (self.current_step >= EPISODE_STEPS)

        next_obs = self._get_observation() if not done else np.zeros((NUM_AGENTS, AGENT_OBS_DIM), dtype=np.float32)

        return next_obs, float(reward), done, info
