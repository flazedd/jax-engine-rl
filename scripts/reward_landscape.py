"""Per-inventory-level reward landscape for each action.

Runs many single-step trials at each inventory level, reports mean reward ± std
in a grid: rows = actions, columns = inventory levels.

Usage:
    uv run python scripts/reward_landscape.py                # noise, default
    uv run python scripts/reward_landscape.py --regime bull
    uv run python scripts/reward_landscape.py --regime bear
    uv run python scripts/reward_landscape.py --fast
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from lob_sim.actions import ACTION_TABLE, N_ACTIONS
from lob_sim.config import SimConfig
from lob_sim.state import init_state
from lob_sim.step import make_step_fn

from plot_style import apply_style, save_fig, REGIME_COLORS, REGIME_NAMES, OPTIMAL_EDGE

apply_style()

REGIME_MAP = {"noise": 0, "bull": 1, "bear": 2}

parser = argparse.ArgumentParser()
parser.add_argument("--regime", default="noise", choices=["noise", "bull", "bear"])
parser.add_argument("--fast", action="store_true")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--horizon", type=int, default=30,
                    help="Steps per trial (longer = less MTM noise)")
args = parser.parse_args()

N_TRIALS = 500 if args.fast else 2000
HORIZON = args.horizon
SIM_CFG = SimConfig()
MAX_INV = SIM_CFG.max_inventory
REGIME_IDX = REGIME_MAP[args.regime]

INV_RANGE = 15
inv_levels = np.arange(-INV_RANGE, INV_RANGE + 1)
n_inv = len(inv_levels)

step_fn = make_step_fn(SIM_CFG, locked_regime=REGIME_IDX)


def multi_step_reward(inv, action, rng_key):
    """Init state at given inventory, repeat same action for HORIZON steps, return total reward."""
    k_init, k_run = jax.random.split(rng_key)
    state = init_state(SIM_CFG, k_init)
    state = state._replace(inventory=jnp.float32(inv))
    actions = jnp.full((HORIZON,), action, dtype=jnp.int32)

    def step(s, a):
        new_s, out = step_fn(s, a)
        return new_s, out["reward"]

    _, rewards = jax.lax.scan(step, state, actions)
    return rewards.mean()


batched_reward = jax.jit(jax.vmap(multi_step_reward, in_axes=(None, None, 0)))

# ── Collect data ──────────────────────────────────────────────
print(f"Reward landscape: regime={args.regime}, trials={N_TRIALS}, "
      f"horizon={HORIZON}, inv=[{-INV_RANGE}..{INV_RANGE}]")

# (N_ACTIONS, n_inv)
mean_rewards = np.zeros((N_ACTIONS, n_inv))
std_rewards = np.zeros((N_ACTIONS, n_inv))

master_key = jax.random.PRNGKey(args.seed)

for i, inv in enumerate(inv_levels):
    for a in range(N_ACTIONS):
        key = jax.random.fold_in(master_key, i * N_ACTIONS + a)
        keys = jax.random.split(key, N_TRIALS)
        rewards = np.array(batched_reward(jnp.float32(inv), jnp.int32(a), keys))
        mean_rewards[a, i] = rewards.mean()
        std_rewards[a, i] = rewards.std()

    best = mean_rewards[:, i].argmax()
    bt, at = int(ACTION_TABLE[best][0]), int(ACTION_TABLE[best][1])
    print(f"  inv={inv:+3d}  best=({bt},{at})  "
          f"reward={mean_rewards[best, i]:+.4f} ± {std_rewards[best, i]:.4f}")

# ── Plot: grid heatmap with text ──────────────────────────────
action_labels = [f"({int(ACTION_TABLE[a][0])},{int(ACTION_TABLE[a][1])})"
                 for a in range(N_ACTIONS)]

fig, ax = plt.subplots(figsize=(max(16, n_inv * 0.55), N_ACTIONS * 0.7 + 2))

im = ax.imshow(mean_rewards, cmap="RdYlGn", aspect="auto", interpolation="nearest")

# Mark the best action per inventory column (exactly one per column)
best_per_inv = mean_rewards.argmax(axis=0)

from matplotlib.patches import Rectangle

for i in range(n_inv):
    for a in range(N_ACTIONS):
        m = mean_rewards[a, i]
        s = std_rewards[a, i]

        mid_val = (mean_rewards.max() + mean_rewards.min()) / 2
        color = "white" if abs(m - mid_val) > 0.55 * abs(mean_rewards.max() - mid_val) else "black"

        ax.text(i, a, f"{m:+.3f}\n±{s:.3f}", ha="center", va="center",
                fontsize=5.5, color=color)

    # Draw exactly one rectangle for the best action
    best_a = best_per_inv[i]
    rect = Rectangle((i - 0.5, best_a - 0.5), 1, 1,
                      linewidth=2.5, edgecolor=OPTIMAL_EDGE,
                      facecolor="none", zorder=5)
    ax.add_patch(rect)

ax.set_xticks(range(n_inv))
ax.set_xticklabels([f"{v:+d}" for v in inv_levels], fontsize=6, rotation=90)
ax.set_yticks(range(N_ACTIONS))
ax.set_yticklabels(action_labels, fontsize=8)
ax.set_xlabel("Inventory")
ax.set_ylabel("Action (bid, ask) ticks")

regime_name = REGIME_NAMES[REGIME_IDX]
ax.set_title(f"Reward Landscape — {regime_name} Regime\n"
             f"N={N_TRIALS} trials x {HORIZON} steps per cell  |  seed={args.seed}  |  "
             f"inv_range=±{INV_RANGE}",
             fontsize=11)

for spine in ax.spines.values():
    spine.set_visible(True)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Mean reward")

fig.tight_layout()
save_fig(fig, f"reward_landscape_{args.regime}.png", script_file=__file__)
