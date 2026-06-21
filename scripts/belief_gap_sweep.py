"""Analytical agnostic -> belief -> oracle decomposition across a kappa sweep.

For each env config (and optional kappa override) prints:
  oracle (VI full-info), belief (Q-MDP MC), agnostic (best inv-only policy),
  and the two gap components:
    compromise cost = belief   - agnostic   (what regime-agnostic loses)
    inference  cost = oracle   - belief     (what belief loses vs full info)
    total gap       = oracle   - agnostic

Usage:
  uv run python -m scripts.belief_gap_sweep --env-config <cfg> --kappas 0.05 0.10 0.20
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from envs.market_making_v1 import MarketMakingV1
from oracles.value_iteration import (
    belief_qmdp_expected_return,
    compromise_policy_expected_returns,
    solve_value_iteration,
)
from training.config import _load_yaml_with_extends


def evaluate(env: MarketMakingV1) -> dict:
    vi = solve_value_iteration(env)
    oracle = float(vi.mixed_expected_episode_return)
    _, agnostic, _ = compromise_policy_expected_returns(env, vi)
    belief = belief_qmdp_expected_return(env, vi.Q)
    return {
        "oracle": oracle,
        "belief": float(belief),
        "agnostic": float(agnostic),
        "compromise_cost": float(belief - agnostic),
        "inference_cost": float(oracle - belief),
        "total_gap": float(oracle - agnostic),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", required=True)
    p.add_argument("--kappas", nargs="*", type=float, default=None)
    args = p.parse_args()

    env_cfg = _load_yaml_with_extends(Path(args.env_config))["env"]
    base = MarketMakingV1(**env_cfg["params"])
    kappas = args.kappas if args.kappas else [base.inventory_penalty]

    hdr = f"{'kappa':>6} | {'oracle':>7} | {'belief':>7} | {'agnostic':>8} | {'compromise':>10} | {'inference':>9} | {'total gap':>9}"
    print(hdr)
    print("-" * len(hdr))
    for k in kappas:
        env = dataclasses.replace(base, inventory_penalty=k)
        m = evaluate(env)
        print(
            f"{k:>6.2f} | {m['oracle']:>7.1f} | {m['belief']:>7.1f} | {m['agnostic']:>8.1f} | "
            f"{m['compromise_cost']:>10.1f} | {m['inference_cost']:>9.1f} | {m['total_gap']:>9.1f}"
        )
    print(f"[belief_gap_sweep] OK | env={Path(args.env_config).stem} | output=stdout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
