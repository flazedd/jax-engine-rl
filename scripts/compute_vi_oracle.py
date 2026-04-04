"""Compute and save VI-optimal oracle policies (isolated + mixed).

Run once (or when SimConfig/regime params change):
    uv run python scripts/compute_vi_oracle.py [--fast]

Saves:
    results/vi_isolated.json  — optimal policy per regime in isolation
    results/vi_optimal.json   — optimal policy under regime switching
"""
import argparse
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import numpy as np

from lob_sim.config import SimConfig
from lob_sim.regime import TRANSITION_MATRIX, RegimeStepParams, get_regime_params
from lob_sim.agents.oracle import (
    compute_vi_isolated_and_mixed, print_vi_policy,
    evaluate_vi_oracle,
)

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true")
args = parser.parse_args()

SIM_CFG = SimConfig()
gamma = 0.99
n_mc = 50 if args.fast else 200
n_eval = 50 if args.fast else 200
seed_vi = 999
seed_eval = 888

RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))

# ── Print parameters ──
print("=" * 60)
print("  compute_vi_oracle.py")
print("=" * 60)
print(f"  fast:           {args.fast}")
print(f"  gamma:          {gamma}")
print(f"  n_mc_episodes:  {n_mc}")
print(f"  n_eval:         {n_eval}")
print(f"  seed_vi:        {seed_vi}")
print(f"  seed_eval:      {seed_eval}")

print(f"\n  SimConfig:")
for field in SIM_CFG._fields:
    print(f"    {field:30s} = {getattr(SIM_CFG, field)}")

regime_names = ["Noise", "Bull", "Bear"]
print(f"\n  HMM transition matrix:")
for i, row in enumerate(TRANSITION_MATRIX):
    print(f"    {regime_names[i]:>5s} -> {[f'{p:.3f}' for p in row]}")

print(f"\n  Regime step params:")
regime_params = [get_regime_params(r) for r in range(3)]
header = f"    {'':>20s}" + "".join(f"{n:>10s}" for n in regime_names)
print(header)
for field in RegimeStepParams._fields:
    vals = [float(getattr(regime_params[r], field)) for r in range(3)]
    print(f"    {field:>20s}" + "".join(f"{v:>10.4f}" for v in vals))

print("=" * 60)

# ── Compute both policies (shared MC samples) ──
print("\nComputing VI policies (isolated + mixed)...")
(iso_policy, iso_values, iso_info), (mix_policy, mix_values, mix_info) = \
    compute_vi_isolated_and_mixed(SIM_CFG, jax.random.PRNGKey(seed_vi),
                                  gamma=gamma, n_mc_episodes=n_mc)

print("\nIsolated-regime VI policy:")
print_vi_policy(iso_policy, max_inv=SIM_CFG.max_inventory)

print("\nMixed-regime VI policy:")
print_vi_policy(mix_policy, max_inv=SIM_CFG.max_inventory)

# ── Convergence diagnostics ──
print("\n  Convergence diagnostics:")
for label, info in [("Isolated", iso_info), ("Mixed", mix_info)]:
    print(f"    {label}: {info['n_iterations']} iterations, "
          f"final delta={info['final_delta']:.2e}, "
          f"policy stable since iter {info['policy_stable_since']}, "
          f"max Bellman residual={info['max_bellman_residual']:.2e}")

# ── Evaluate VI oracles (with trajectories, using same seeds as compute_agent) ──
# compute_agent.py uses seed + 1000 + regime_idx for eval keys (default seed=42)
# We use the same convention so episodes are paired for fair comparison.
AGENT_EVAL_SEED = 42  # must match compute_agent.py --seed default

print(f"\nEvaluating VI policies ({n_eval} episodes each, trajectories=True)...")

# Mixed VI on mixed regimes (unlocked, with trajectories)
mix_stats = evaluate_vi_oracle(SIM_CFG, jax.random.PRNGKey(seed_eval),
                               mix_policy, n_episodes=n_eval,
                               return_trajectories=True)
mix_step_rewards = np.array(mix_stats["step_rewards"])
mix_step_mask = np.array(mix_stats["step_mask"])
mix_cum = np.cumsum(mix_step_rewards * mix_step_mask, axis=1)
print(f"  Mixed VI oracle:  mean={mix_stats['mean_reward']:.2f} "
      f"(std={mix_stats['std_reward']:.2f})")

# Per-regime: isolated VI on locked regimes (with trajectories)
per_regime_eval = {}
vi_trajectories = {}
print("\n  Per-regime evaluation (isolated VI on locked regimes):")
for r in range(3):
    # Same key as compute_agent.py uses for this regime
    eval_key = jax.random.PRNGKey(AGENT_EVAL_SEED + 1000 + r)
    iso_r_stats = evaluate_vi_oracle(SIM_CFG, eval_key,
                                     iso_policy, n_episodes=n_eval,
                                     locked_regime=r,
                                     return_trajectories=True)
    per_regime_eval[regime_names[r]] = {
        "vi_mean": float(iso_r_stats["mean_reward"]),
        "vi_std": float(iso_r_stats["std_reward"]),
    }

    # Cumulative reward trajectory
    step_rewards = np.array(iso_r_stats["step_rewards"])  # (n_eval, n_steps)
    step_mask = np.array(iso_r_stats["step_mask"])
    cum_rewards = np.cumsum(step_rewards * step_mask, axis=1)
    vi_trajectories[regime_names[r]] = {
        "cumulative_reward_mean": cum_rewards.mean(axis=0).tolist(),
        "cumulative_reward_std": cum_rewards.std(axis=0).tolist(),
        "episode_rewards": np.array(iso_r_stats["rewards"]).tolist(),
    }

    print(f"    {regime_names[r]:>5s}: mean={iso_r_stats['mean_reward']:.2f} "
          f"(std={iso_r_stats['std_reward']:.2f})")

# Mixed VI with trajectories (eval on each locked regime separately)
print("\n  Mixed VI evaluated per-regime (with trajectories):")
vi_mixed_trajectories = {}
for r in range(3):
    eval_key = jax.random.PRNGKey(AGENT_EVAL_SEED + 1000 + r)
    mix_r_stats = evaluate_vi_oracle(SIM_CFG, eval_key,
                                     mix_policy, n_episodes=n_eval,
                                     locked_regime=r,
                                     return_trajectories=True)
    step_rewards = np.array(mix_r_stats["step_rewards"])
    step_mask = np.array(mix_r_stats["step_mask"])
    cum_rewards = np.cumsum(step_rewards * step_mask, axis=1)
    vi_mixed_trajectories[regime_names[r]] = {
        "cumulative_reward_mean": cum_rewards.mean(axis=0).tolist(),
        "cumulative_reward_std": cum_rewards.std(axis=0).tolist(),
        "episode_rewards": np.array(mix_r_stats["rewards"]).tolist(),
    }
    print(f"    {regime_names[r]:>5s}: mean={mix_r_stats['mean_reward']:.2f} "
          f"(std={mix_r_stats['std_reward']:.2f})")

# ── Save everything ──
def _save(filename, data):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path}")

print("\nSaving results...")
_save("vi_isolated.json", {
    "policy": iso_policy.tolist(),
    "values": iso_values.tolist(),
    "max_inventory": SIM_CFG.max_inventory,
    "convergence": {k: v for k, v in iso_info.items() if k != "deltas"},
    "convergence_deltas": iso_info["deltas"],
})

_save("vi_optimal.json", {
    "policy": mix_policy.tolist(),
    "values": mix_values.tolist(),
    "max_inventory": SIM_CFG.max_inventory,
    "convergence": {k: v for k, v in mix_info.items() if k != "deltas"},
    "convergence_deltas": mix_info["deltas"],
})

_save("vi_eval.json", {
    "mixed_vi": {
        "mean_reward": float(mix_stats["mean_reward"]),
        "std_reward": float(mix_stats["std_reward"]),
        "cumulative_reward_mean": mix_cum.mean(axis=0).tolist(),
        "cumulative_reward_std": mix_cum.std(axis=0).tolist(),
        "episode_rewards": np.array(mix_stats["rewards"]).tolist(),
    },
    "per_regime": per_regime_eval,
    "isolated_trajectories": vi_trajectories,
    "mixed_trajectories": vi_mixed_trajectories,
    "n_eval": n_eval,
    "seed_eval": seed_eval,
    "agent_eval_seed": AGENT_EVAL_SEED,
})

print("\nDone.")
