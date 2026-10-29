"""
config_marl.py
==============
Central configuration and Single Source of Truth for Phase 5:
Multi-Agent Reinforcement Learning (MAPPO) and Operational Dashboard.

All numbers used by the environment, training, evaluation, baselines, and reports
are defined here. No number should be hard-coded elsewhere or typed by hand into reports.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# Inherit root directories and paths from central config
from config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    TEST_PARQUET,
    VAL_PARQUET,
    get_flow_severity,
)

# -----------------------------------------------------------------------------
# 1. Output Directories & Paths
# -----------------------------------------------------------------------------
MARL_MODELS_DIR = MODELS_DIR / "marl"
MARL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Data cache paths
CACHE_VAL_PATH = DATA_PROCESSED_DIR / "marl_cache_val.parquet"
CACHE_TEST_PATH = DATA_PROCESSED_DIR / "marl_cache_test.parquet"
RL_TRAIN_INDICES_PATH = DATA_PROCESSED_DIR / "marl_rl_train_indices.json"
RL_VAL_INDICES_PATH = DATA_PROCESSED_DIR / "marl_rl_val_indices.json"

# Report artifact paths
CACHE_METRICS_PATH = REPORTS_DIR / "marl_cache_metrics.json"
MARL_TESTS_TXT_PATH = REPORTS_DIR / "marl_tests.txt"
BASELINES_JSON_PATH = REPORTS_DIR / "marl_baselines.json"
PILOT_JSON_PATH = REPORTS_DIR / "marl_pilot.json"
EVAL_JSON_PATH = REPORTS_DIR / "marl_eval.json"
STRESS_TEST_JSON_PATH = REPORTS_DIR / "marl_stress_test.json"
PHASE5_SUMMARY_PATH = REPORTS_DIR / "phase5_summary.md"
RETURNS_PLOT_PATH = REPORTS_DIR / "marl_returns.png"
LEARNING_CURVES_PLOT_PATH = REPORTS_DIR / "marl_learning_curves.png"
DASHBOARD_HTML_PATH = REPORTS_DIR / "dashboard.html"

# Alert and logging paths
ALERTS_PHASE5_JSONL_PATH = LOGS_DIR / "alerts_phase5.jsonl"
ALERTS_PHASE5_TAIL_HASH_PATH = LOGS_DIR / "alerts_phase5_tail_hash.txt"
ALERTS_PHASE5_ECS_NDJSON_PATH = LOGS_DIR / "alerts_phase5_ecs.ndjson"
PHASE5_LOG_VERIFICATION_TXT_PATH = REPORTS_DIR / "phase5_log_verification.txt"
DOCKER_COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"

# -----------------------------------------------------------------------------
# 2. Reproducibility & Splits
# -----------------------------------------------------------------------------
CACHE_SPLIT_SEED = 42
RL_TRAIN_SPLIT_RATIO = 0.70
RL_VAL_SPLIT_RATIO = 0.30

TRAIN_SEEDS: List[int] = [42, 123, 456, 789, 1011]
BASELINE_EVAL_SEEDS: List[int] = [42, 123, 456, 789, 1011]
FINAL_EVAL_SEEDS: List[int] = [42, 123, 456, 789, 1011]

BASELINE_EVAL_EPISODES = 200
FINAL_EVAL_EPISODES = 500
BOOTSTRAP_ROUNDS = 10_000

# -----------------------------------------------------------------------------
# 3. Environment & Campaign Dynamics
# -----------------------------------------------------------------------------
EPISODE_STEPS = 50           # Horizon T = 50 steps
MIN_BURST_LENGTH = 10        # Burst attack flow lower bound
MAX_BURST_LENGTH = 25        # Burst attack flow upper bound

# Agent 1: Triage {0: accept_verdict, 1: flag_for_review, 2: defer}
TRIAGE_ACTIONS: List[str] = ["accept_verdict", "flag_for_review", "defer"]
TRIAGE_ACTION_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(TRIAGE_ACTIONS)}

# Agent 2: Containment {0: none, 1: rate_limit, 2: block, 3: isolate}
CONTAINMENT_ACTIONS: List[str] = ["none", "rate_limit", "block", "isolate"]
CONTAINMENT_ACTION_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CONTAINMENT_ACTIONS)}
POSTURE_DURATION = 3         # Active posture duration K = 3 steps

# Agent 3: Escalation {0: none, 1: notify_analyst, 2: forensic_capture}
ESCALATION_ACTIONS: List[str] = ["none", "notify_analyst", "forensic_capture"]
ESCALATION_ACTION_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(ESCALATION_ACTIONS)}
ESCALATION_BUDGET = 5        # Per-episode budget B = 5 notifications/captures

# -----------------------------------------------------------------------------
# 4. Reward Parameters & Penalty Formulations
# -----------------------------------------------------------------------------
# Base attack damage scaled by ensemble-derived severity
SEVERITY_BASE_DAMAGE: Dict[str, float] = {
    "Benign": 0.0,
    "Low": 1.0,
    "Medium": 3.0,
    "High": 6.0,
    "Critical": 10.0,
}


def get_true_class_severity(true_class: str) -> str:
    """
    Maps ground-truth class to severity using the canonical severity table:
      - Benign: "Benign" (damage = 0.0)
      - DDoS, Mirai: "Critical"
      - DoS, Web-based, BruteForce: "High"
      - Spoofing, Recon: "Medium"
    Matches get_flow_severity(true_class, confidence=1.0).
    """
    tc = str(true_class).strip()
    if tc == "Benign":
        return "Benign"
    return get_flow_severity(tc, 1.0)

# Multiplier for residual damage that breaches containment posture
CONTAINMENT_DAMAGE_MULTIPLIER: Dict[str, float] = {
    "none": 1.0,         # 100% damage penetrates
    "rate_limit": 0.6,   # 40% mitigated
    "block": 0.1,        # 90% mitigated
    "isolate": 0.0,      # 100% mitigated
}

# Collateral penalty incurred on benign traffic per step while posture is active
CONTAINMENT_COLLATERAL_COST: Dict[str, float] = {
    "none": 0.0,
    "rate_limit": 0.5,
    "block": 2.0,
    "isolate": 5.0,
}

# One-time action execution costs (incurred at the step action is enacted)
CONTAINMENT_ACTION_COST: Dict[str, float] = {
    "none": 0.0,
    "rate_limit": 0.1,
    "block": 0.3,
    "isolate": 0.5,
}

# Triage delay penalty per step
TRIAGE_DELAY_PENALTY: Dict[str, float] = {
    "accept_verdict": 0.0,
    "flag_for_review": 0.2,
    "defer": 0.3,
}

# Escalation base costs
ESCALATION_ACTION_COST: Dict[str, float] = {
    "none": 0.0,
    "notify_analyst": 0.5,
    "forensic_capture": 1.0,
}

# Discount factor on escalation action cost when triage flagged for review
TRIAGE_FLAG_DISCOUNT: float = 0.50

# Heavy penalty per excess escalation action beyond budget B
EXCESS_BUDGET_PENALTY: float = 5.0

# Team synergy bonus: true attack is both contained (posture != none) and escalated (action != none)
CONTAINED_AND_NOTIFIED_BONUS: float = 2.0

# -----------------------------------------------------------------------------
# 5. Reward Weight Settings
# -----------------------------------------------------------------------------
REWARD_SETTINGS: Dict[str, Dict[str, float]] = {
    "balanced": {
        "damage_weight": 1.0,
        "collateral_weight": 1.0,
        "cost_weight": 1.0,
        "bonus_weight": 1.0,
    },
    "security-first": {
        "damage_weight": 2.0,
        "collateral_weight": 1.0,
        "cost_weight": 1.0,
        "bonus_weight": 1.0,
    },
    "availability-first": {
        "damage_weight": 1.0,
        "collateral_weight": 2.0,
        "cost_weight": 1.0,
        "bonus_weight": 1.0,
    },
}

# -----------------------------------------------------------------------------
# 6. Observation Space Dimensions
# -----------------------------------------------------------------------------
# Observation vector per agent:
# - 8-class probabilities: 8 dims
# - Confidence: 1 dim
# - One-hot severity (Low, Medium, High, Critical): 4 dims
# - Normalized step fraction (t / T): 1 dim
# - Escalations left normalized (remaining / B): 1 dim
# - Containment posture active (0 or 1): 1 dim
# - Containment posture steps remaining normalized (remaining / K): 1 dim
# - Rolling history counts of last 5 predicted classes: 8 dims
# Subtotal feature dims = 8 + 1 + 4 + 1 + 1 + 1 + 1 + 8 = 25 dims
# Plus one-hot agent ID: 3 dims (Triage=[1,0,0], Containment=[0,1,0], Escalation=[0,0,1])
# Agent observation dim = 28 dims
ENV_FEATURE_DIM = 25
AGENT_ID_DIM = 3
AGENT_OBS_DIM = ENV_FEATURE_DIM + AGENT_ID_DIM  # 28 dims
NUM_AGENTS = 3
CENTRAL_CRITIC_OBS_DIM = NUM_AGENTS * AGENT_OBS_DIM  # 84 dims

# -----------------------------------------------------------------------------
# 7. MAPPO Hyperparameters
# -----------------------------------------------------------------------------
HIDDEN_DIM = 64
NUM_LAYERS = 2
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 4
BATCH_SIZE = 64
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5

PILOT_STEPS = 20_000
PILOT_TARGET_SECONDS = 1500  # 25 minutes target per seed
MAX_TRAIN_STEPS = 200_000
MIN_TRAIN_STEPS = 100_000
EVAL_INTERVAL_STEPS = 20_000
STAGE4_RUNTIME_LIMIT_SEC = 2700  # 45 minutes hard cap
STAGE4_EXTENDED_LIMIT_SEC = 4500  # 75 minutes extended cap

# -----------------------------------------------------------------------------
# 8. Adversarial Stress Test
# -----------------------------------------------------------------------------
STRESS_TEST_SAMPLE_SIZE = 500
STRESS_TEST_EPSILON = 0.10
STRESS_TEST_TIME_LIMIT_SEC = 600  # 10 minutes

# -----------------------------------------------------------------------------
# 9. Pre-registered Evaluation Decision Rule & Baseline Definitions
# -----------------------------------------------------------------------------
NON_LEARNED_BASELINES: List[str] = [
    "random",
    "always_none",
    "max_containment",
    "severity_rule",
    "severity_rule_budget_aware",
]

REFERENCE_LABEL_AWARE_BASELINES: List[str] = [
    "label_aware_rule",
    "label_aware_rule_budget_aware",
]

DECISION_RULE_TEXT = (
    "MAPPO is reported as better than the best non-learned baseline only if the paired-difference "
    "interval excludes zero in the positive direction for that reward setting; if it excludes zero "
    "in the negative direction it is reported as worse; otherwise it is reported as not distinguishable."
)
