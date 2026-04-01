#!/usr/bin/env python3
"""Check convergence of Monte Carlo reward estimates.

Runs plot_divergence evaluation at increasing sample sizes and reports
whether optimal actions and reward estimates stabilize.

Usage:
  uv run python scripts/convergence_check.py
  uv run python scripts/convergence_check.py --fast
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from lob_sim.actions import ACTION_TABLE, N_ACTIONS, BID_TICKS, ASK_TICKS
from lob_sim.config import SimConfig
from lob_sim.regime import N_REGIMES
from lob_sim.step import run_episode

REGIME_NAMES = ["Noise", "Bull", "Bear"]

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true")
args = parser.parse_args()

if args.fast:
    T_VALUES = [200, 500]
    N_VALUES = [20, 50]
else:
    T_VALUES = [200, 500, 1000, 2000]
    N_VALUES = [50, 100, 200, 400]


def compute_reward_matrix(n_episodes, t_steps):
    config = SimConfig(max_steps=t_steps)
    master_key = jax.random.PRNGKey(42)
    mean_matrix = np.zeros((N_ACTIONS, N_REGIMES))

    jit_run = jax.jit(
        lambda k, a, r: run_episode(config, k, a, locked_regime=r),
        static_argnums=(2,),
    )

    for regime_idx in range(N_REGIMES):
        for action_idx in range(N_ACTIONS):
            key = jax.random.fold_in(master_key, regime_idx * N_ACTIONS + action_idx)
            keys = jax.random.split(key, n_episodes)
            actions = jnp.full((t_steps,), action_idx, dtype=jnp.int32)
            batched = jax.vmap(lambda k: jit_run(k, actions, regime_idx))
            _, outputs = batched(keys)
            mean_matrix[action_idx, regime_idx] = float(outputs["reward"].sum(axis=1).mean())

    return mean_matrix


# --- Vary episode length (T), fixed N ---
N_FIXED = 200 if not args.fast else 50
print(f"=== Varying episode length (T), N={N_FIXED} episodes ===")
print(f"{'T':>6s}  {'Noise optimal':>14s}  {'Bull optimal':>14s}  {'Bear optimal':>14s}  "
      f"{'Noise reward':>13s}  {'Bull reward':>13s}  {'Bear reward':>13s}")
print("-" * 100)

for t in T_VALUES:
    m = compute_reward_matrix(N_FIXED, t)
    opts = []
    rews = []
    for r in range(N_REGIMES):
        opt_idx = m[:, r].argmax()
        bid, ask = int(ACTION_TABLE[opt_idx][0]), int(ACTION_TABLE[opt_idx][1])
        opts.append(f"({bid},{ask})")
        rews.append(f"{m[opt_idx, r]:+.2f}")
    print(f"{t:>6d}  {opts[0]:>14s}  {opts[1]:>14s}  {opts[2]:>14s}  "
          f"{rews[0]:>13s}  {rews[1]:>13s}  {rews[2]:>13s}")

# --- Vary number of episodes (N), fixed T ---
T_FIXED = 1000 if not args.fast else 200
print(f"\n=== Varying number of episodes (N), T={T_FIXED} steps ===")
print(f"{'N':>6s}  {'Noise optimal':>14s}  {'Bull optimal':>14s}  {'Bear optimal':>14s}  "
      f"{'Noise reward':>13s}  {'Bull reward':>13s}  {'Bear reward':>13s}")
print("-" * 100)

for n in N_VALUES:
    m = compute_reward_matrix(n, T_FIXED)
    opts = []
    rews = []
    for r in range(N_REGIMES):
        opt_idx = m[:, r].argmax()
        bid, ask = int(ACTION_TABLE[opt_idx][0]), int(ACTION_TABLE[opt_idx][1])
        opts.append(f"({bid},{ask})")
        rews.append(f"{m[opt_idx, r]:+.2f}")
    print(f"{n:>6d}  {opts[0]:>14s}  {opts[1]:>14s}  {opts[2]:>14s}  "
          f"{rews[0]:>13s}  {rews[1]:>13s}  {rews[2]:>13s}")

print("\nIf optimal actions are stable across rows, the estimates have converged.")
