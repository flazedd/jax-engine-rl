"""Compute and save the VI-optimal oracle policy.

Run once (or when SimConfig/regime params change):
    uv run python scripts/compute_vi_oracle.py [--fast]

Saves plots/vi_optimal.json, which train.py loads automatically.
"""
import argparse

import jax

from lob_sim.config import SimConfig
from lob_sim.regime import TRANSITION_MATRIX, RegimeStepParams, get_regime_params
from lob_sim.agents.oracle import (
    compute_vi_policy, print_vi_policy, save_vi_policy,
    evaluate_vi_oracle,
)

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true")
args = parser.parse_args()

SIM_CFG = SimConfig()
gamma = 0.99
n_mc = 50 if args.fast else 200
n_eval = 200
seed_vi = 999
seed_eval = 888

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

# ── Compute ──
print("\nComputing VI-optimal oracle (regime x inventory)...")
policy, values = compute_vi_policy(SIM_CFG, jax.random.PRNGKey(seed_vi),
                                   gamma=gamma, n_mc_episodes=n_mc)
print_vi_policy(policy, max_inv=SIM_CFG.max_inventory)
save_vi_policy(policy, values)

print(f"\nEvaluating on mixed regime ({n_eval} episodes)...")
stats = evaluate_vi_oracle(SIM_CFG, jax.random.PRNGKey(seed_eval), policy,
                           n_episodes=n_eval)
print(f"  VI oracle mean reward: {stats['mean_reward']:.2f} "
      f"(std: {stats['std_reward']:.2f})")
