#!/usr/bin/env python3
"""Train PPO per-regime and mixed-regime, produce analysis plots.

Produces:
  plots/ppo_per_regime.png  — 3x2: learning curve + action heatmap per regime
  plots/ppo_mixed.png       — 2x3: per-regime actions + overall actions + learning curve + PnL

Usage:
  uv run python scripts/run_ppo_analysis.py
  uv run python scripts/run_ppo_analysis.py --fast
"""
import argparse

import jax
import jax.numpy as jnp

from lob_sim.config import SimConfig
from lob_sim.agents.ppo import PPOAgent, PPOConfig
from lob_sim.training.rollout import collect_rollout_batch
from lob_sim.training.trainer import compute_gae, ppo_update, create_optimizer
from lob_sim.training.eval import evaluate_agent

from agent_eval import (
    action_freq_matrix,
    collect_eval_data_batch,
    make_per_regime_figure,
    make_mixed_figure,
    REGIME_NAMES,
)

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true",
                    help="Quick smoke test with minimal iterations")
parser.add_argument("--n_iterations", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.fast:
    args.n_iterations = 10

EVAL_EVERY = 5 if args.fast else 10
N_EVAL_EPISODES = 5 if args.fast else 20

SIM_CFG = SimConfig()
PPO_CFG = PPOConfig(n_envs=16, n_steps=128)


def train_ppo(locked_regime, n_iters, seed=42):
    """Train PPO, return (agent, (iters_list, rewards_list))."""
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    agent = PPOAgent(ppo_config=PPO_CFG, key=k0)
    optimizer, opt_state = create_optimizer(PPO_CFG, agent)

    eval_iters, eval_rewards = [], []
    regime_str = REGIME_NAMES[locked_regime] if locked_regime >= 0 else "Mixed"

    for i in range(n_iters):
        key, k_roll, k_upd = jax.random.split(key, 3)
        batch = collect_rollout_batch(
            agent, SIM_CFG, k_roll,
            n_envs=PPO_CFG.n_envs, n_steps=PPO_CFG.n_steps,
            locked_regime=locked_regime,
        )
        adv, ret = compute_gae(batch, PPO_CFG.gamma, PPO_CFG.gae_lambda)
        agent, opt_state, metrics = ppo_update(
            agent, optimizer, opt_state, batch, adv, ret, PPO_CFG, k_upd,
        )

        if i % EVAL_EVERY == 0 or i == n_iters - 1:
            key, k_eval = jax.random.split(key)
            stats = evaluate_agent(agent, SIM_CFG, k_eval,
                                   n_episodes=N_EVAL_EPISODES,
                                   locked_regime=locked_regime)
            mr = float(stats["mean_reward"])
            eval_iters.append(i)
            eval_rewards.append(mr)
            ent = float(metrics["entropy"])
            print(f"  [{regime_str:>5s}] iter {i:>4d} | reward {mr:>8.2f} | entropy {ent:.3f}")

    return agent, (eval_iters, eval_rewards)


# ═══════════════════════════════════════════════════════════════
#  Part 1: Per-regime PPO
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  Part 1: Training PPO separately on each regime")
print("=" * 60)

agents_by_regime = {}
reward_histories = {}
per_regime_freqs = {}

for regime_idx in range(3):
    print(f"\n--- {REGIME_NAMES[regime_idx]} regime ---")
    agent, history = train_ppo(regime_idx, args.n_iterations, seed=args.seed)
    agents_by_regime[regime_idx] = agent
    reward_histories[regime_idx] = history

    # Collect action freq for passing to mixed figure as star overlay
    key = jax.random.PRNGKey(99 + regime_idx)
    data = collect_eval_data_batch(agent, SIM_CFG, key,
                                   n_episodes=N_EVAL_EPISODES,
                                   n_steps=SIM_CFG.max_steps,
                                   locked_regime=regime_idx)
    per_regime_freqs[regime_idx] = action_freq_matrix(data.actions)

print("\nGenerating ppo_per_regime.png...")
make_per_regime_figure(agents_by_regime, reward_histories, SIM_CFG,
                       agent_name="PPO", n_eval_episodes=N_EVAL_EPISODES)

# ═══════════════════════════════════════════════════════════════
#  Part 2: Mixed-regime PPO
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Part 2: Training PPO on mixed regimes")
print("=" * 60)

agent_mixed, history_mixed = train_ppo(-1, args.n_iterations, seed=args.seed + 100)

print("\nGenerating ppo_mixed.png...")
make_mixed_figure(agent_mixed, history_mixed, SIM_CFG,
                  agent_name="PPO", n_eval_episodes=N_EVAL_EPISODES,
                  per_regime_freqs=per_regime_freqs)

print("\nDone.")
