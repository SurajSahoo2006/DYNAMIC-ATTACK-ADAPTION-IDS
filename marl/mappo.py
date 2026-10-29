"""
marl/mappo.py
=============
Stage 4: Multi-Agent Proximal Policy Optimization (MAPPO) Implementation.

Features:
  - 3 cooperative actor MLPs (Triage, Containment, Escalation) with 2 hidden layers (64 units).
  - Centralized Critic MLP evaluating concatenated observations of all 3 agents (84 dims).
  - Generalized Advantage Estimation (GAE), clipped PPO objective, entropy regularization,
    advantage normalization, and gradient clipping.
  - Pilot benchmark (20,000 steps) to determine empirical training budget.
  - Multi-seed training (5 seeds) across reward settings with best rl_val checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config_marl import (
    AGENT_OBS_DIM,
    BATCH_SIZE,
    CENTRAL_CRITIC_OBS_DIM,
    CLIP_EPS,
    CONTAINMENT_ACTIONS,
    ENTROPY_COEF,
    ESCALATION_ACTIONS,
    EVAL_INTERVAL_STEPS,
    GAE_LAMBDA,
    GAMMA,
    HIDDEN_DIM,
    LR_ACTOR,
    LR_CRITIC,
    MARL_MODELS_DIR,
    MAX_GRAD_NORM,
    MAX_TRAIN_STEPS,
    MIN_TRAIN_STEPS,
    PILOT_JSON_PATH,
    PILOT_STEPS,
    PILOT_TARGET_SECONDS,
    PPO_EPOCHS,
    REPORTS_DIR,
    REWARD_SETTINGS,
    STAGE4_RUNTIME_LIMIT_SEC,
    TRAIN_SEEDS,
    TRIAGE_ACTIONS,
    VALUE_LOSS_COEF,
)
from marl.env import TacticalDefenseEnv
from marl.stats import bootstrap_ci_mean, wilson_score_interval

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger("marl_mappo")


def get_peak_ram_mb() -> float:
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


# -----------------------------------------------------------------------------
# 1. Neural Network Modules
# -----------------------------------------------------------------------------
class ActorMLP(nn.Module):
    """2-layer MLP policy for discrete action selection."""

    def __init__(self, obs_dim: int = AGENT_OBS_DIM, action_dim: int = 3, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)


class CentralCriticMLP(nn.Module):
    """Centralized Critic evaluating concatenated multi-agent observations."""

    def __init__(self, joint_obs_dim: int = CENTRAL_CRITIC_OBS_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        return self.net(joint_obs).squeeze(-1)


# -----------------------------------------------------------------------------
# 2. MAPPO Multi-Agent Team
# -----------------------------------------------------------------------------
class MAPPOTeam:
    """Manages the 3 cooperative actor networks and the centralized critic."""

    def __init__(self, lr_actor: float = LR_ACTOR, lr_critic: float = LR_CRITIC):
        self.actor_triage = ActorMLP(action_dim=len(TRIAGE_ACTIONS))
        self.actor_containment = ActorMLP(action_dim=len(CONTAINMENT_ACTIONS))
        self.actor_escalation = ActorMLP(action_dim=len(ESCALATION_ACTIONS))
        self.critic = CentralCriticMLP()

        self.actor_params = (
            list(self.actor_triage.parameters())
            + list(self.actor_containment.parameters())
            + list(self.actor_escalation.parameters())
        )
        self.optimizer_actors = optim.Adam(self.actor_params, lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_actions(
        self, obs_matrix: np.ndarray, deterministic: bool = False
    ) -> Tuple[Tuple[int, int, int], Tuple[float, float, float], float]:
        """
        obs_matrix: shape (3, 28)
        Returns: joint_action (a0, a1, a2), log_probs (lp0, lp1, lp2), state_value
        """
        self.actor_triage.eval()
        self.actor_containment.eval()
        self.actor_escalation.eval()
        self.critic.eval()

        with torch.no_grad():
            t_obs = torch.from_numpy(obs_matrix).float()
            joint_obs = t_obs.view(-1)  # shape (84,)

            dist_t = self.actor_triage(t_obs[0])
            dist_c = self.actor_containment(t_obs[1])
            dist_e = self.actor_escalation(t_obs[2])

            if deterministic:
                a_t = int(torch.argmax(dist_t.logits).item())
                a_c = int(torch.argmax(dist_c.logits).item())
                a_e = int(torch.argmax(dist_e.logits).item())
            else:
                a_t = int(dist_t.sample().item())
                a_c = int(dist_c.sample().item())
                a_e = int(dist_e.sample().item())

            lp_t = float(dist_t.log_prob(torch.tensor(a_t)).item())
            lp_c = float(dist_c.log_prob(torch.tensor(a_c)).item())
            lp_e = float(dist_e.log_prob(torch.tensor(a_e)).item())

            val = float(self.critic(joint_obs).item())

        return (a_t, a_c, a_e), (lp_t, lp_c, lp_e), val

    def save_checkpoint(self, path: Path, extra_meta: Optional[Dict] = None):
        checkpoint = {
            "actor_triage": self.actor_triage.state_dict(),
            "actor_containment": self.actor_containment.state_dict(),
            "actor_escalation": self.actor_escalation.state_dict(),
            "critic": self.critic.state_dict(),
            "meta": extra_meta or {},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.actor_triage.load_state_dict(checkpoint["actor_triage"])
        self.actor_containment.load_state_dict(checkpoint["actor_containment"])
        self.actor_escalation.load_state_dict(checkpoint["actor_escalation"])
        self.critic.load_state_dict(checkpoint["critic"])
        return checkpoint.get("meta", {})


# -----------------------------------------------------------------------------
# 3. Rollout Buffer
# -----------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.obs_list: List[np.ndarray] = []
        self.joint_obs_list: List[np.ndarray] = []
        self.actions_list: List[Tuple[int, int, int]] = []
        self.logprobs_list: List[Tuple[float, float, float]] = []
        self.rewards_list: List[float] = []
        self.values_list: List[float] = []
        self.dones_list: List[bool] = []

    def add(
        self,
        obs: np.ndarray,
        joint_obs: np.ndarray,
        actions: Tuple[int, int, int],
        logprobs: Tuple[float, float, float],
        reward: float,
        value: float,
        done: bool,
    ):
        self.obs_list.append(obs)
        self.joint_obs_list.append(joint_obs)
        self.actions_list.append(actions)
        self.logprobs_list.append(logprobs)
        self.rewards_list.append(reward)
        self.values_list.append(value)
        self.dones_list.append(done)

    def is_full(self) -> bool:
        return len(self.rewards_list) >= self.capacity

    def clear(self):
        self.obs_list.clear()
        self.joint_obs_list.clear()
        self.actions_list.clear()
        self.logprobs_list.clear()
        self.rewards_list.clear()
        self.values_list.clear()
        self.dones_list.clear()


# -----------------------------------------------------------------------------
# 4. GAE & PPO Update
# -----------------------------------------------------------------------------
def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1:
            next_non_terminal = 1.0 - float(dones[t])
            next_value = last_value
        else:
            next_non_terminal = 1.0 - float(dones[t])
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def update_mappo(
    team: MAPPOTeam,
    buffer: RolloutBuffer,
    last_value: float,
    epochs: int = PPO_EPOCHS,
    batch_size: int = BATCH_SIZE,
    clip_eps: float = CLIP_EPS,
    entropy_coef: float = ENTROPY_COEF,
    value_loss_coef: float = VALUE_LOSS_COEF,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> Tuple[float, float]:
    advantages, returns = compute_gae(
        buffer.rewards_list, buffer.values_list, buffer.dones_list, last_value
    )

    # Advantage normalization
    adv_mean = np.mean(advantages)
    adv_std = np.std(advantages) + 1e-8
    norm_advantages = (advantages - adv_mean) / adv_std

    # Convert buffer to tensors
    obs_triage = torch.from_numpy(np.array([o[0] for o in buffer.obs_list])).float()
    obs_contain = torch.from_numpy(np.array([o[1] for o in buffer.obs_list])).float()
    obs_escalate = torch.from_numpy(np.array([o[2] for o in buffer.obs_list])).float()
    joint_obs_tensor = torch.from_numpy(np.array(buffer.joint_obs_list)).float()

    acts_t = torch.tensor([a[0] for a in buffer.actions_list], dtype=torch.long)
    acts_c = torch.tensor([a[1] for a in buffer.actions_list], dtype=torch.long)
    acts_e = torch.tensor([a[2] for a in buffer.actions_list], dtype=torch.long)

    old_lp_t = torch.tensor([lp[0] for lp in buffer.logprobs_list], dtype=torch.float32)
    old_lp_c = torch.tensor([lp[1] for lp in buffer.logprobs_list], dtype=torch.float32)
    old_lp_e = torch.tensor([lp[2] for lp in buffer.logprobs_list], dtype=torch.float32)

    returns_tensor = torch.from_numpy(returns).float()
    advantages_tensor = torch.from_numpy(norm_advantages).float()

    dataset_size = len(buffer.rewards_list)
    indices = np.arange(dataset_size)

    team.actor_triage.train()
    team.actor_containment.train()
    team.actor_escalation.train()
    team.critic.train()

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    updates_count = 0

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            b_idx = indices[start_idx:end_idx]

            # Batch advantages and returns
            b_adv = advantages_tensor[b_idx]
            b_ret = returns_tensor[b_idx]

            # 1. Actors Update
            dist_t = team.actor_triage(obs_triage[b_idx])
            dist_c = team.actor_containment(obs_contain[b_idx])
            dist_e = team.actor_escalation(obs_escalate[b_idx])

            new_lp_t = dist_t.log_prob(acts_t[b_idx])
            new_lp_c = dist_c.log_prob(acts_c[b_idx])
            new_lp_e = dist_e.log_prob(acts_e[b_idx])

            ratio_t = torch.exp(new_lp_t - old_lp_t[b_idx])
            ratio_c = torch.exp(new_lp_c - old_lp_c[b_idx])
            ratio_e = torch.exp(new_lp_e - old_lp_e[b_idx])

            surr1_t = ratio_t * b_adv
            surr2_t = torch.clamp(ratio_t, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            surr1_c = ratio_c * b_adv
            surr2_c = torch.clamp(ratio_c, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            surr1_e = ratio_e * b_adv
            surr2_e = torch.clamp(ratio_e, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv

            loss_actor_t = -torch.min(surr1_t, surr2_t).mean() - entropy_coef * dist_t.entropy().mean()
            loss_actor_c = -torch.min(surr1_c, surr2_c).mean() - entropy_coef * dist_c.entropy().mean()
            loss_actor_e = -torch.min(surr1_e, surr2_e).mean() - entropy_coef * dist_e.entropy().mean()

            combined_actor_loss = loss_actor_t + loss_actor_c + loss_actor_e

            team.optimizer_actors.zero_grad()
            combined_actor_loss.backward()
            nn.utils.clip_grad_norm_(team.actor_params, max_grad_norm)
            team.optimizer_actors.step()

            # 2. Central Critic Update
            pred_v = team.critic(joint_obs_tensor[b_idx])
            critic_loss = value_loss_coef * nn.functional.mse_loss(pred_v, b_ret)

            team.optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(team.critic.parameters(), max_grad_norm)
            team.optimizer_critic.step()

            total_actor_loss += float(combined_actor_loss.item())
            total_critic_loss += float(critic_loss.item())
            updates_count += 1

    mean_a_loss = total_actor_loss / max(1, updates_count)
    mean_c_loss = total_critic_loss / max(1, updates_count)
    return mean_a_loss, mean_c_loss


# -----------------------------------------------------------------------------
# 5. Fast Evaluation Helper
# -----------------------------------------------------------------------------
def evaluate_mappo_on_val(
    team: MAPPOTeam,
    reward_setting: str,
    num_episodes: int = 20,
    base_seed: int = 5000,
) -> Dict[str, float]:
    env = TacticalDefenseEnv(cache_mode="rl_val", reward_setting=reward_setting, seed=base_seed)
    returns = []
    attacks_contained = 0
    total_attacks = 0
    benigns_disrupted = 0
    total_benigns = 0

    for ep in range(num_episodes):
        obs = env.reset(seed=base_seed + ep)
        done = False
        ep_ret = 0.0

        while not done:
            actions, _, _ = team.select_actions(obs, deterministic=True)
            obs, reward, done, info = env.step(actions)
            ep_ret += reward
            if info["is_attack"]:
                total_attacks += 1
                if info["attack_contained"]:
                    attacks_contained += 1
            else:
                total_benigns += 1
                if info["benign_disrupted"]:
                    benigns_disrupted += 1

        returns.append(ep_ret)

    att_pct = (attacks_contained / max(1, total_attacks)) * 100.0
    ben_pct = (benigns_disrupted / max(1, total_benigns)) * 100.0

    return {
        "mean_return": round(float(np.mean(returns)), 4),
        "attacks_contained_pct": round(att_pct, 2),
        "benigns_disrupted_pct": round(ben_pct, 2),
    }


# -----------------------------------------------------------------------------
# 6. Pilot Benchmark
# -----------------------------------------------------------------------------
def run_pilot() -> Dict:
    log.info(f"--- Running Stage 4 Pilot ({PILOT_STEPS} steps) ---")
    start_time = time.time()
    env = TacticalDefenseEnv(cache_mode="rl_train", reward_setting="balanced", seed=42)
    team = MAPPOTeam()
    buffer = RolloutBuffer(capacity=500)

    obs = env.reset(seed=42)
    step_count = 0

    while step_count < PILOT_STEPS:
        joint_obs = obs.reshape(-1)
        actions, logprobs, value = team.select_actions(obs)
        next_obs, reward, done, info = env.step(actions)
        buffer.add(obs, joint_obs, actions, logprobs, reward, value, done)

        step_count += 1
        if buffer.is_full() or done:
            if done:
                last_v = 0.0
                obs = env.reset()
            else:
                _, _, last_v = team.select_actions(next_obs)
                obs = next_obs

            update_mappo(team, buffer, last_v)
            buffer.clear()
        else:
            obs = next_obs

    elapsed = time.time() - start_time
    steps_per_sec = round(float(step_count) / elapsed, 2)
    log.info(f"Pilot completed {step_count} steps in {elapsed:.2f}s ({steps_per_sec} steps/sec).")

    # Fixed rule: budget = min(MAX_TRAIN_STEPS, steps/sec * 1500s), >= 100,000 steps
    computed_budget = int(steps_per_sec * PILOT_TARGET_SECONDS)
    chosen_budget = min(MAX_TRAIN_STEPS, computed_budget)
    chosen_budget = max(MIN_TRAIN_STEPS, chosen_budget)

    # Estimate total runtime for all 3 settings x 5 seeds = 15 runs
    total_est_seconds = (chosen_budget / max(1.0, steps_per_sec)) * 15
    log.info(f"Chosen training budget per seed: {chosen_budget} steps.")
    log.info(f"Estimated runtime for 15 runs: {total_est_seconds:.1f}s (budget cap: {STAGE4_RUNTIME_LIMIT_SEC}s).")

    if total_est_seconds > STAGE4_RUNTIME_LIMIT_SEC:
        active_settings = ["balanced"]
        skipped_reason = (
            f"Estimated training runtime ({total_est_seconds:.1f}s) exceeds 45-minute limit "
            f"({STAGE4_RUNTIME_LIMIT_SEC}s). Reduced to 'balanced' setting only as per Stage 4 specification."
        )
        log.warning(f"Restricting to 'balanced' setting only: {skipped_reason}")
    else:
        active_settings = ["balanced", "security-first", "availability-first"]
        skipped_reason = None

    pilot_metrics = {
        "description": "Stage 4 MAPPO Pilot Benchmark and Budget Selection",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pilot_steps": step_count,
        "elapsed_seconds": round(elapsed, 2),
        "steps_per_second": steps_per_sec,
        "rule": "budget = min(MAX_TRAIN_STEPS, steps_per_sec * 1500s), >= 100,000 steps",
        "chosen_budget": chosen_budget,
        "active_settings": active_settings,
        "skipped_settings_reason": skipped_reason,
        "peak_ram_mb": round(get_peak_ram_mb(), 2),
    }

    with open(PILOT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(pilot_metrics, f, indent=2)
    log.info(f"Saved pilot results to {PILOT_JSON_PATH}")
    return pilot_metrics


# -----------------------------------------------------------------------------
# 7. MAPPO Training Loop
# -----------------------------------------------------------------------------
def train_mappo_seed(
    reward_setting: str,
    seed: int,
    total_steps: int,
    eval_interval: int = EVAL_INTERVAL_STEPS,
) -> Dict:
    log.info(f"=== Starting MAPPO Training: Setting '{reward_setting}', Seed {seed} ({total_steps} steps) ===")
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = TacticalDefenseEnv(cache_mode="rl_train", reward_setting=reward_setting, seed=seed)
    team = MAPPOTeam()
    buffer = RolloutBuffer(capacity=500)

    best_val_return = -1e9
    best_checkpoint_path = MARL_MODELS_DIR / f"mappo_{reward_setting}_seed{seed}_best.pt"
    final_checkpoint_path = MARL_MODELS_DIR / f"mappo_{reward_setting}_seed{seed}_final.pt"
    train_log_path = REPORTS_DIR / f"marl_train_{reward_setting}_seed{seed}.json"

    obs = env.reset(seed=seed)
    step_count = 0
    learning_curve = []

    while step_count < total_steps:
        joint_obs = obs.reshape(-1)
        actions, logprobs, value = team.select_actions(obs)
        next_obs, reward, done, info = env.step(actions)
        buffer.add(obs, joint_obs, actions, logprobs, reward, value, done)

        step_count += 1

        if buffer.is_full() or done:
            if done:
                last_v = 0.0
                obs = env.reset()
            else:
                _, _, last_v = team.select_actions(next_obs)
                obs = next_obs

            a_loss, c_loss = update_mappo(team, buffer, last_v)
            buffer.clear()
        else:
            obs = next_obs

        # Periodic evaluation on rl_val partition
        if step_count % eval_interval == 0 or step_count >= total_steps:
            val_res = evaluate_mappo_on_val(team, reward_setting, num_episodes=20, base_seed=5000 + seed)
            entry = {
                "step": step_count,
                "elapsed_sec": round(time.time() - start_time, 2),
                "rl_val_return": val_res["mean_return"],
                "rl_val_attacks_contained_pct": val_res["attacks_contained_pct"],
                "rl_val_benigns_disrupted_pct": val_res["benigns_disrupted_pct"],
                "actor_loss": round(a_loss, 4),
                "critic_loss": round(c_loss, 4),
            }
            learning_curve.append(entry)
            log.info(
                f"[{reward_setting} | Seed {seed}] Step {step_count:>6}/{total_steps} | "
                f"rl_val Return: {val_res['mean_return']:>8.2f} (best: {best_val_return:>8.2f}) | "
                f"Attacks: {val_res['attacks_contained_pct']:>5.1f}% | Disrupted: {val_res['benigns_disrupted_pct']:>5.1f}%"
            )

            # Checkpoint selection strictly on BEST rl_val return
            if val_res["mean_return"] > best_val_return:
                best_val_return = val_res["mean_return"]
                team.save_checkpoint(
                    best_checkpoint_path,
                    extra_meta={
                        "step": step_count,
                        "reward_setting": reward_setting,
                        "seed": seed,
                        "best_val_return": best_val_return,
                    },
                )
                log.info(f"  -> Saved new best checkpoint to {best_checkpoint_path}")

    # Save final model
    team.save_checkpoint(
        final_checkpoint_path,
        extra_meta={
            "step": step_count,
            "reward_setting": reward_setting,
            "seed": seed,
            "final_val_return": learning_curve[-1]["rl_val_return"],
        },
    )

    training_record = {
        "reward_setting": reward_setting,
        "seed": seed,
        "total_steps": step_count,
        "runtime_seconds": round(time.time() - start_time, 2),
        "peak_ram_mb": round(get_peak_ram_mb(), 2),
        "best_val_return": round(best_val_return, 4),
        "final_val_return": learning_curve[-1]["rl_val_return"],
        "learning_curve": learning_curve,
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
    }

    with open(train_log_path, "w", encoding="utf-8") as f:
        json.dump(training_record, f, indent=2)
    log.info(f"Saved training log to {train_log_path}")

    return training_record


# -----------------------------------------------------------------------------
# 8. Main CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MAPPO Pilot and Training Suite")
    parser.add_argument("--pilot", action="store_true", help="Run 20k steps pilot only")
    parser.add_argument("--setting", choices=["balanced", "security-first", "availability-first", "all"], default="all")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps per seed")
    parser.add_argument("--resume", action="store_true", help="Skip already trained seeds")
    args = parser.parse_args()

    # Always check pilot first or load existing pilot
    if args.pilot or not PILOT_JSON_PATH.exists():
        pilot_meta = run_pilot()
        if args.pilot:
            return
    else:
        with open(PILOT_JSON_PATH, "r", encoding="utf-8") as f:
            pilot_meta = json.load(f)
        log.info(f"Loaded existing pilot metadata from {PILOT_JSON_PATH} (budget: {pilot_meta['chosen_budget']} steps)")

    budget = args.steps if args.steps is not None else pilot_meta["chosen_budget"]
    target_settings = pilot_meta["active_settings"] if args.setting == "all" else [args.setting]

    for setting in target_settings:
        for seed in TRAIN_SEEDS:
            train_log_path = REPORTS_DIR / f"marl_train_{setting}_seed{seed}.json"
            best_ckpt = MARL_MODELS_DIR / f"mappo_{setting}_seed{seed}_best.pt"
            if args.resume and train_log_path.exists() and best_ckpt.exists():
                log.info(f"Seed {seed} for setting '{setting}' already trained. Skipping due to --resume.")
                continue
            # Train seed
            train_mappo_seed(reward_setting=setting, seed=seed, total_steps=budget)


if __name__ == "__main__":
    main()
