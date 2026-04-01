#!/usr/bin/env python3
"""PPO Diagnostics — per-regime vs mixed-regime analysis.

Produces:
  plots/ppo_per_regime.png   — PPO trained on each locked regime separately
  plots/ppo_mixed_regime.png — PPO trained on mixed regimes, evaluated per regime

Usage:
  uv run python scripts/ppo_diagnostics.py
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from lob_sim.config import SimConfig
from lob_sim.actions import ACTION_TABLE, N_ACTIONS, BID_TICKS, ASK_TICKS
from lob_sim.agents.ppo import PPOAgent, PPOConfig
from lob_sim.state import init_state
from lob_sim.obs import observe
from lob_sim.step import make_step_fn
from lob_sim.training.rollout import collect_rollout_batch
from lob_sim.training.trainer import compute_gae, ppo_update, create_optimizer
from lob_sim.training.eval import evaluate_agent

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--fast", action="store_true",
                     help="Quick smoke test with minimal iterations")
_args = _parser.parse_args()

os.makedirs("plots", exist_ok=True)

REGIME_NAMES = ["NOISE", "BULL", "BEAR"]
REGIME_COLORS = ["#2196F3", "#4CAF50", "#F44336"]

PPO_CFG = PPOConfig(
    lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
    entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
    n_epochs=4, n_minibatches=4, n_envs=16, n_steps=128,
)
SIM_CFG = SimConfig()
N_ITERS = 10 if _args.fast else 200
EVAL_EVERY = 5 if _args.fast else 10
N_EVAL_EPISODES = 5 if _args.fast else 30


def collect_eval_actions(agent, sim_config, rng_key, n_episodes, locked_regime):
    """Run evaluation episodes and collect all actions taken."""
    step_fn = make_step_fn(sim_config, locked_regime=locked_regime)
    n_steps = sim_config.max_steps

    def run_one(key):
        k_init, k_agent, k_run = jax.random.split(key, 3)
        sim_state = init_state(sim_config, k_init)
        agent_state = agent.initial_agent_state(k_agent)

        def step(carry, _):
            sim_state, agent_state, rng = carry
            rng, rng_action = jax.random.split(rng)
            obs = observe(sim_state, sim_config)
            action, new_agent_state, _ = agent.get_action(obs, agent_state, rng_action)
            new_sim_state, sim_out = step_fn(sim_state, action)
            new_agent_state = new_agent_state._replace(
                prev_reward=sim_out["reward"],
                prev_done=sim_out["done"].astype(jnp.float32),
            )
            return (new_sim_state, new_agent_state, rng), action

        _, actions = jax.lax.scan(
            step, (sim_state, agent_state, k_run), None, length=n_steps
        )
        return actions

    keys = jax.random.split(rng_key, n_episodes)
    return jax.vmap(run_one)(keys)  # (n_episodes, n_steps)


def action_freq_matrix(actions):
    """Convert flat action array to 3x3 frequency matrix (bid_ticks × ask_ticks)."""
    flat = np.array(actions).reshape(-1)
    counts = np.bincount(flat, minlength=N_ACTIONS)
    freqs = counts / counts.sum()
    return freqs.reshape(len(BID_TICKS), len(ASK_TICKS))


def plot_heatmap(ax, freq_matrix, title, vmin=0, vmax=None):
    """Plot 3x3 action frequency heatmap with annotations."""
    if vmax is None:
        vmax = max(0.4, freq_matrix.max() * 1.1)
    im = ax.imshow(freq_matrix, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(ASK_TICKS)))
    ax.set_xticklabels(ASK_TICKS)
    ax.set_yticks(range(len(BID_TICKS)))
    ax.set_yticklabels(BID_TICKS)
    ax.set_xlabel("Ask ticks")
    ax.set_ylabel("Bid ticks")
    ax.set_title(title, fontsize=11)
    for i in range(freq_matrix.shape[0]):
        for j in range(freq_matrix.shape[1]):
            pct = freq_matrix[i, j] * 100
            color = "white" if freq_matrix[i, j] > vmax * 0.55 else "black"
            ax.text(j, i, f"{pct:.1f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    return im


def train_ppo(locked_regime, n_iters, seed=42):
    """Train PPO and return (agent, eval_rewards, eval_iters, entropies)."""
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    agent = PPOAgent(ppo_config=PPO_CFG, key=k0)
    optimizer, opt_state = create_optimizer(PPO_CFG, agent)

    eval_rewards, eval_iters, entropies = [], [], []
    for i in range(n_iters):
        key, k_roll, k_upd = jax.random.split(key, 3)
        batch = collect_rollout_batch(agent, SIM_CFG, k_roll,
                                      n_envs=PPO_CFG.n_envs,
                                      n_steps=PPO_CFG.n_steps,
                                      locked_regime=locked_regime)
        adv, ret = compute_gae(batch, PPO_CFG.gamma, PPO_CFG.gae_lambda)
        agent, opt_state, metrics = ppo_update(
            agent, optimizer, opt_state, batch, adv, ret, PPO_CFG, k_upd
        )

        if i % EVAL_EVERY == 0 or i == n_iters - 1:
            key, k_eval = jax.random.split(key)
            stats = evaluate_agent(agent, SIM_CFG, k_eval,
                                    n_episodes=N_EVAL_EPISODES,
                                    locked_regime=locked_regime if locked_regime >= 0 else -1)
            eval_rewards.append(float(stats["mean_reward"]))
            eval_iters.append(i)
            entropies.append(float(metrics["entropy"]))
            regime_str = REGIME_NAMES[locked_regime] if locked_regime >= 0 else "MIXED"
            print(f"  [{regime_str}] iter {i:3d} | reward: {eval_rewards[-1]:7.3f} | "
                  f"entropy: {entropies[-1]:.3f}")

    return agent, eval_rewards, eval_iters, entropies


# ═══════════════════════════════════════════════════════════════
#  Part 1: Per-regime PPO
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  Part 1: Training PPO separately on each regime")
print("=" * 60)

per_regime_agents = {}
per_regime_freqs = {}

fig1, axes1 = plt.subplots(3, 3, figsize=(16, 12))

for regime_idx in range(3):
    name = REGIME_NAMES[regime_idx]
    color = REGIME_COLORS[regime_idx]
    print(f"\n--- {name} regime ---")
    agent, rewards, iters, entropies = train_ppo(regime_idx, N_ITERS)
    per_regime_agents[regime_idx] = agent

    # Col 0: Learning curve
    ax = axes1[regime_idx, 0]
    ax.plot(iters, rewards, color=color, linewidth=2)
    ax.set_title(f"{name} — Reward", fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Eval Reward")
    ax.grid(True, alpha=0.3)

    # Col 1: Entropy curve
    ax = axes1[regime_idx, 1]
    ax.plot(iters, entropies, color=color, linewidth=2)
    ax.set_title(f"{name} — Entropy", fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.3)

    # Col 2: Action distribution
    key_eval = jax.random.PRNGKey(99 + regime_idx)
    actions = collect_eval_actions(agent, SIM_CFG, key_eval,
                                   n_episodes=N_EVAL_EPISODES,
                                   locked_regime=regime_idx)
    freq = action_freq_matrix(actions)
    per_regime_freqs[regime_idx] = freq
    plot_heatmap(axes1[regime_idx, 2], freq, f"{name} — Actions")

fig1.suptitle("PPO Trained Per-Regime (each regime trained separately)",
              fontsize=14, fontweight="bold")
fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig("plots/ppo_per_regime.png", dpi=150, bbox_inches="tight")
print(f"\n>>> Saved plots/ppo_per_regime.png")
plt.close(fig1)


# ═══════════════════════════════════════════════════════════════
#  Part 2: Mixed-regime PPO
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Part 2: Training PPO on mixed regimes")
print("=" * 60)

agent_mixed, rewards_mixed, iters_mixed, entropies_mixed = train_ppo(-1, N_ITERS, seed=123)

# Collect action distributions per regime
mixed_freqs = {}
for regime_idx in range(3):
    key_eval = jax.random.PRNGKey(200 + regime_idx)
    actions = collect_eval_actions(agent_mixed, SIM_CFG, key_eval,
                                   n_episodes=N_EVAL_EPISODES,
                                   locked_regime=regime_idx)
    mixed_freqs[regime_idx] = action_freq_matrix(actions)

# Build figure
fig2 = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig2, hspace=0.35, wspace=0.3)

# Top-left: Learning curve
ax_curve = fig2.add_subplot(gs[0, :2])
ax_curve.plot(iters_mixed, rewards_mixed, color="black", linewidth=2)
ax_curve.set_title("Mixed-Regime PPO — Learning Curve", fontsize=11)
ax_curve.set_xlabel("Iteration")
ax_curve.set_ylabel("Mean Eval Reward")
ax_curve.grid(True, alpha=0.3)

# Top-right: Entropy curve
ax_ent = fig2.add_subplot(gs[0, 2])
ax_ent.plot(iters_mixed, entropies_mixed, color="gray", linewidth=2)
ax_ent.set_title("Entropy", fontsize=11)
ax_ent.set_xlabel("Iteration")
ax_ent.set_ylabel("Entropy")
ax_ent.grid(True, alpha=0.3)

# Bottom: Action heatmaps per regime + comparison annotation
vmax_global = max(f.max() for f in mixed_freqs.values()) * 1.1
vmax_global = max(0.4, vmax_global)

for regime_idx in range(3):
    name = REGIME_NAMES[regime_idx]
    ax = fig2.add_subplot(gs[1, regime_idx])
    plot_heatmap(ax, mixed_freqs[regime_idx], f"Eval on {name}", vmax=vmax_global)

    # Mark per-regime optimal with a star
    optimal_flat = np.argmax(per_regime_freqs[regime_idx])
    opt_row, opt_col = divmod(optimal_flat, len(ASK_TICKS))
    ax.plot(opt_col, opt_row, marker="*", markersize=18,
            color=REGIME_COLORS[regime_idx], markeredgecolor="black",
            markeredgewidth=1.0)

fig2.suptitle("PPO on Mixed Regimes — Action Distribution per Regime\n"
              "(stars = per-regime optimal from separate training)",
              fontsize=13, fontweight="bold")
fig2.savefig("plots/ppo_mixed_regime.png", dpi=150, bbox_inches="tight")
print(f"\n>>> Saved plots/ppo_mixed_regime.png")
plt.close(fig2)


# ═══════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print("\nPer-regime trained PPO — most frequent action:")
for regime_idx in range(3):
    freq = per_regime_freqs[regime_idx]
    best = np.argmax(freq)
    bid, ask = BID_TICKS[best // len(ASK_TICKS)], ASK_TICKS[best % len(ASK_TICKS)]
    print(f"  {REGIME_NAMES[regime_idx]:6s}: bid={bid}, ask={ask}  "
          f"({freq.max()*100:.1f}% of actions)")

print("\nMixed-regime PPO — most frequent action per eval regime:")
for regime_idx in range(3):
    freq = mixed_freqs[regime_idx]
    best = np.argmax(freq)
    bid, ask = BID_TICKS[best // len(ASK_TICKS)], ASK_TICKS[best % len(ASK_TICKS)]
    pct_same = freq.max() * 100
    print(f"  {REGIME_NAMES[regime_idx]:6s}: bid={bid}, ask={ask}  "
          f"({pct_same:.1f}% of actions)")

# Check if mixed-regime PPO uses same action across regimes
mixed_best = [np.argmax(mixed_freqs[r]) for r in range(3)]
if len(set(mixed_best)) == 1:
    print("\n>>> Mixed-regime PPO plays the SAME action in all regimes — compromise policy!")
    print("    This is where RL²/VariBAD can improve by adapting per-regime.")
elif len(set(mixed_best)) < 3:
    print(f"\n>>> Mixed-regime PPO partially differentiates ({len(set(mixed_best))}/3 distinct actions)")
    print("    Still suboptimal compared to per-regime specialists.")
else:
    print("\n>>> Mixed-regime PPO uses different actions per regime — surprising!")
