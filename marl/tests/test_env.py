"""
marl/tests/test_env.py
======================
Unit tests for TacticalDefenseEnv ensuring:
  (a) Observations never leak true labels.
  (b) Determinism: identical seed produces identical trajectories.
  (c) Mathematical correctness of reward arithmetic on 3 hand-computed cases.
  (d) Containment posture persistence (K=3) and escalation budget (B=5) enforcement.
  (e) Split boundary isolation between rl_train, rl_val, and test splits.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config_marl import (
    CONTAINED_AND_NOTIFIED_BONUS,
    CONTAINMENT_ACTION_COST,
    CONTAINMENT_COLLATERAL_COST,
    CONTAINMENT_DAMAGE_MULTIPLIER,
    EPISODE_STEPS,
    ESCALATION_ACTION_COST,
    ESCALATION_BUDGET,
    EXCESS_BUDGET_PENALTY,
    POSTURE_DURATION,
    RL_TRAIN_INDICES_PATH,
    RL_VAL_INDICES_PATH,
    SEVERITY_BASE_DAMAGE,
    TRIAGE_DELAY_PENALTY,
    TRIAGE_FLAG_DISCOUNT,
    get_true_class_severity,
)
from marl.env import TacticalDefenseEnv


def test_no_true_label_leakage():
    """Verify observations never contain or depend on the ground-truth attack label."""
    env = TacticalDefenseEnv(cache_mode="rl_val", seed=42)
    obs_original = env.reset(seed=42).copy()

    # Artificially alter ground-truth labels in cache
    flow_idx = env.episode_flow_indices[env.current_step]
    original_label = env.true_labels[flow_idx]
    env.true_labels[flow_idx] = "Mirai" if original_label == "Benign" else "Benign"

    # Re-generate observation
    obs_mutated = env._get_observation()

    # Observations must be strictly identical regardless of true label
    np.testing.assert_array_equal(
        obs_original,
        obs_mutated,
        err_msg="Observation matrix changed when ground truth label was modified! Label leakage detected."
    )


def test_deterministic_seeding():
    """Verify that identical random seeds generate bit-for-bit identical episodes."""
    env1 = TacticalDefenseEnv(cache_mode="rl_val", seed=101)
    env2 = TacticalDefenseEnv(cache_mode="rl_val", seed=101)

    obs1 = env1.reset(seed=101)
    obs2 = env2.reset(seed=101)

    assert env1.campaign_family == env2.campaign_family
    assert env1.burst_start == env2.burst_start
    assert env1.burst_length == env2.burst_length
    assert env1.episode_flow_indices == env2.episode_flow_indices
    np.testing.assert_array_equal(obs1, obs2)

    # Step through 10 actions
    actions = [(0, 1, 0), (1, 0, 1), (2, 2, 0), (0, 0, 0), (1, 3, 2)] * 2
    for act in actions:
        o1, r1, d1, info1 = env1.step(act)
        o2, r2, d2, info2 = env2.step(act)
        assert r1 == pytest.approx(r2, rel=1e-6)
        assert d1 == d2
        assert info1 == info2
        np.testing.assert_array_equal(o1, o2)


def test_hand_computed_rewards():
    """Verify reward arithmetic matches hand-computed cases across settings."""
    # -------------------------------------------------------------------------
    # Case 1: Benign flow, balanced setting (weights=1.0)
    # Actions: accept_verdict (0), none (0), none (0)
    # Expected: damage=0, collateral=0, action_costs=0, bonus=0 -> reward = 0.0
    # -------------------------------------------------------------------------
    env_bal = TacticalDefenseEnv(cache_mode="rl_val", reward_setting="balanced", seed=42)
    env_bal.reset(seed=42)
    # Force step 0 flow to be Benign
    benign_idx = env_bal.class_indices["Benign"][0]
    env_bal.episode_flow_indices[0] = benign_idx
    _, r1, _, info1 = env_bal.step((0, 0, 0))

    assert info1["damage"] == 0.0
    assert info1["collateral"] == 0.0
    assert info1["total_action_cost"] == 0.0
    assert r1 == pytest.approx(0.0, abs=1e-5)

    # -------------------------------------------------------------------------
    # Case 2: Critical Attack flow, security-first setting
    # Weights: w_D=2.0, w_C=1.0, w_cost=1.0, w_B=1.0
    # Actions: flag_for_review (1), block (2), notify_analyst (1)
    # Base dmg = 10.0, block leak = 0.1 -> damage = 1.0 -> dmg_term = - (2.0 * 1.0) = -2.0
    # Triage cost = 0.2
    # Containment cost = 0.3
    # Escalation cost = 0.5 * (1 - 0.5) = 0.25 (flag discount)
    # Action cost = 0.2 + 0.3 + 0.25 = 0.75 -> cost_term = -0.75
    # Bonus = +2.0 * 1.0 = +2.0 (contained and notified)
    # Expected reward = -2.0 + 0.0 - 0.75 + 2.0 = -0.75
    # -------------------------------------------------------------------------
    env_sec = TacticalDefenseEnv(cache_mode="rl_val", reward_setting="security-first", seed=42)
    env_sec.reset(seed=42)
    # Find an attack flow with true severity Critical (e.g. DDoS or Mirai)
    crit_indices = [idx for idx in env_sec.class_indices.get("DDoS", env_sec.class_indices[env_sec.campaign_family]) if get_true_class_severity(env_sec.true_labels[idx]) == "Critical"]
    target_idx = crit_indices[0] if crit_indices else env_sec.class_indices[env_sec.campaign_family][0]
    env_sec.episode_flow_indices[0] = target_idx
    expected_true_sev = get_true_class_severity(env_sec.true_labels[target_idx])
    expected_base_dmg = SEVERITY_BASE_DAMAGE[expected_true_sev]
    expected_dmg = expected_base_dmg * CONTAINMENT_DAMAGE_MULTIPLIER["block"]
    expected_dmg_term = -2.0 * expected_dmg
    expected_action_cost = TRIAGE_DELAY_PENALTY["flag_for_review"] + CONTAINMENT_ACTION_COST["block"] + (ESCALATION_ACTION_COST["notify_analyst"] * 0.5)
    expected_r2 = expected_dmg_term - expected_action_cost + CONTAINED_AND_NOTIFIED_BONUS

    _, r2, _, info2 = env_sec.step((1, 2, 1))
    assert r2 == pytest.approx(expected_r2, abs=1e-5)

    # -------------------------------------------------------------------------
    # Case 3: Benign flow under active 'isolate' posture, availability-first
    # Weights: w_D=1.0, w_C=2.0, w_cost=1.0, w_B=1.0
    # Active posture = isolate (collateral = 5.0)
    # Actions: defer (2), none (0), none (0)
    # Collateral term = - (2.0 * 5.0) = -10.0
    # Action cost = defer penalty = 0.3 -> cost_term = -0.3
    # Expected reward = -10.0 - 0.3 = -10.3
    # -------------------------------------------------------------------------
    env_avail = TacticalDefenseEnv(cache_mode="rl_val", reward_setting="availability-first", seed=42)
    env_avail.reset(seed=42)
    env_avail.episode_flow_indices[0] = benign_idx
    # Pre-set active posture to isolate
    env_avail.active_posture = "isolate"
    env_avail.posture_steps_remaining = 2

    _, r3, _, info3 = env_avail.step((2, 0, 0))
    expected_r3 = - (2.0 * CONTAINMENT_COLLATERAL_COST["isolate"]) - TRIAGE_DELAY_PENALTY["defer"]
    assert r3 == pytest.approx(expected_r3, abs=1e-5)


def test_posture_duration_and_escalation_budget():
    """Verify posture persistence across K=3 steps and budget penalty after B=5 escalations."""
    env = TacticalDefenseEnv(cache_mode="rl_val", seed=42)
    env.reset(seed=42)

    # Step 0: Enact rate_limit
    _, _, _, info0 = env.step((0, 1, 0))
    assert info0["effective_posture"] == "rate_limit"
    assert env.posture_steps_remaining == POSTURE_DURATION - 1  # 2 steps remaining

    # Step 1: action none -> rate_limit persists
    _, _, _, info1 = env.step((0, 0, 0))
    assert info1["effective_posture"] == "rate_limit"
    assert env.posture_steps_remaining == 1

    # Step 2: action none -> rate_limit persists
    _, _, _, info2 = env.step((0, 0, 0))
    assert info2["effective_posture"] == "rate_limit"
    assert env.posture_steps_remaining == 0
    assert env.active_posture == "none"

    # Step 3: action none -> posture is now none
    _, _, _, info3 = env.step((0, 0, 0))
    assert info3["effective_posture"] == "none"

    # Test Escalation Budget
    env.reset(seed=42)
    assert env.escalations_remaining == ESCALATION_BUDGET  # 5

    # Consume budget 5 times
    for i in range(5):
        _, _, _, info = env.step((0, 0, 1))
        assert env.escalations_remaining == ESCALATION_BUDGET - (i + 1)
        assert info["total_action_cost"] == pytest.approx(ESCALATION_ACTION_COST["notify_analyst"])

    # 6th escalation action: budget exhausted -> penalty applied
    _, _, _, info_excess = env.step((0, 0, 1))
    assert env.escalations_remaining == 0
    expected_penalized_cost = ESCALATION_ACTION_COST["notify_analyst"] + EXCESS_BUDGET_PENALTY
    assert info_excess["total_action_cost"] == pytest.approx(expected_penalized_cost)


def test_split_boundary_isolation():
    """Verify that environments draw flows exclusively from their configured data splits."""
    with open(RL_TRAIN_INDICES_PATH, "r", encoding="utf-8") as f:
        train_indices = set(json.load(f))
    with open(RL_VAL_INDICES_PATH, "r", encoding="utf-8") as f:
        val_indices = set(json.load(f))

    # Assert train and val index partitions are strictly disjoint
    assert train_indices.isdisjoint(val_indices), "rl_train and rl_val partition indices overlap!"

    # Verify rl_train env length matches train_indices
    env_train = TacticalDefenseEnv(cache_mode="rl_train", seed=42)
    assert len(env_train.df) == len(train_indices)

    # Verify rl_val env length matches val_indices
    env_val = TacticalDefenseEnv(cache_mode="rl_val", seed=42)
    assert len(env_val.df) == len(val_indices)

    # Verify test env length matches test cache
    env_test = TacticalDefenseEnv(cache_mode="test", seed=42)
    assert len(env_test.df) == 26292


def test_damage_invariant_to_predicted_class_mutation():
    """
    Verify that for the same true attack flow, mutating the cached predicted class
    or predicted severity does NOT change the ground-truth damage term.
    """
    env1 = TacticalDefenseEnv(cache_mode="rl_val", reward_setting="balanced", seed=42)
    env1.reset(seed=42)

    ddos_indices = env1.class_indices.get("DDoS", env1.class_indices[env1.campaign_family])
    target_idx = ddos_indices[0]
    true_label = env1.true_labels[target_idx]
    assert true_label != "Benign"

    env1.episode_flow_indices[0] = target_idx
    _, r1, _, info1 = env1.step((0, 0, 0))  # none, none, none -> damage leaks 100%

    # Second env with mutated cached prediction
    env2 = TacticalDefenseEnv(cache_mode="rl_val", reward_setting="balanced", seed=42)
    env2.reset(seed=42)
    env2.episode_flow_indices[0] = target_idx
    env2.severities[target_idx] = "Low"
    env2.pred_indices[target_idx] = 0
    _, r2, _, info2 = env2.step((0, 0, 0))

    # Damage must be strictly identical regardless of predicted class
    assert info1["damage"] == info2["damage"], (
        f"Damage changed when predicted class was mutated! "
        f"Original damage={info1['damage']}, Mutated damage={info2['damage']}"
    )
    assert r1 == pytest.approx(r2, abs=1e-5)
    assert info1["true_severity"] == info2["true_severity"]
    assert info1["pred_severity"] != info2["pred_severity"]
