"""No-training probe of R1 / R3-analytical / R4-entropy for a given env.

A cheap eyeball on whether a candidate env is worth a full M2 run. Training is
still required to settle R2 (per-regime PPO convergence) and the gap-closure
half of R4, but the analytical / simulation pieces below catch the common
failure modes in seconds.

Runs:
  R1 — VI policy-disagreement fraction + wrong-regime value loss.
  R3 (analytical proxy) — oracle-VI mixed return vs compromise-policy VI return.
  R4 (entropy half)     — mean posterior entropy over time under random policy.

Usage:
  uv run python -m scripts.env_requirements_probe \
      --env-config experiments/configs/envs/e_final.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from envs.market_making_v1 import MarketMakingV1
from oracles.value_iteration import (
    compromise_policy_expected_returns,
    policy_disagreement,
    solve_value_iteration,
    wrong_regime_value_loss,
)
from oracles.verify_requirements import _entropy_over_time, _simulate_belief_trajectories
from training.config import _load_yaml_with_extends

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same thresholds as oracles/verify_requirements.py.
R1_DISAGREE_MIN = 0.80
R4_DECAY_MIN = 0.35


def probe(env_config_path: Path) -> int:
    env_cfg = _load_yaml_with_extends(env_config_path)["env"]
    env = MarketMakingV1(**env_cfg["params"])
    print(f"[probe] env={env_config_path.stem}  n_regimes={env.n_regimes}  T={env.episode_length}")

    # --- R1 -----------------------------------------------------------------
    vi = solve_value_iteration(env)
    disagree_frac, _ = policy_disagreement(vi)
    mean_loss, rel_loss, _ = wrong_regime_value_loss(env, vi)
    r1_pass = disagree_frac >= R1_DISAGREE_MIN
    print(
        f"[probe] R1: disagree_frac={disagree_frac:.3f} (>={R1_DISAGREE_MIN})  "
        f"rel_loss={rel_loss:.3f}  mean_loss={mean_loss:.3f}  "
        f"pass={r1_pass}"
    )

    # --- VI per-regime / mixed + compromise --------------------------------
    per_regime_oracle = vi.per_regime_expected_episode_return
    mixed_oracle = float(vi.mixed_expected_episode_return)
    per_regime_compr, mixed_compr, compromise_policy = compromise_policy_expected_returns(env, vi)
    total_gap_analytical = mixed_oracle - mixed_compr
    total_gap_pct = 100 * total_gap_analytical / max(mixed_compr, 1e-9)
    print(
        f"[probe] R3 (analytical): oracle_VI={mixed_oracle:.2f}  "
        f"compromise_VI={mixed_compr:.2f}  gap={total_gap_analytical:.2f} "
        f"({total_gap_pct:.1f}% of compromise)"
    )
    print(f"[probe]   per-regime oracle VI   = {np.round(per_regime_oracle, 2).tolist()}")
    print(f"[probe]   per-regime compr  VI   = {np.round(per_regime_compr, 2).tolist()}")
    print(f"[probe]   compromise policy (inv→act) = {compromise_policy.tolist()}")

    # Per-regime VI policy — quick readable dump (inventory × regime).
    # policy shape is [inventory, regime]; print transposed rows by regime.
    print("[probe]   oracle VI policy (rows=regime, cols=inventory):")
    for r in range(env.n_regimes):
        print(f"[probe]     regime {r}: {vi.policy[:, r].tolist()}")

    # --- R4 entropy half ----------------------------------------------------
    beliefs = _simulate_belief_trajectories(env, n_envs=256, seed=0)
    ent_curve = _entropy_over_time(beliefs)
    ent_t1 = float(ent_curve[0])
    ent_mid = float(ent_curve[env.episode_length // 2])
    decay_frac = float(1.0 - ent_mid / max(ent_t1, 1e-12))
    initial_ent = float(np.log(env.n_regimes))
    r4_ent_pass = decay_frac >= R4_DECAY_MIN
    print(
        f"[probe] R4 (entropy only): initial={initial_ent:.3f}  "
        f"ent(t=1)={ent_t1:.3f}  ent(t=mid)={ent_mid:.3f}  "
        f"decay={decay_frac:.3f} (>={R4_DECAY_MIN})  pass={r4_ent_pass}"
    )

    # --- Summary ------------------------------------------------------------
    print()
    print("[probe] Eyeball summary (training still needed for R2 + R4 gap-closure):")
    print(f"  R1 analytical  : {'PASS' if r1_pass else 'FAIL'}")
    print(f"  R3 analytical  : gap={total_gap_analytical:.2f} ({total_gap_pct:.1f}%)")
    print(f"  R4 entropy     : {'PASS' if r4_ent_pass else 'FAIL'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.env_requirements_probe")
    parser.add_argument("--env-config", required=True)
    args = parser.parse_args()

    path = Path(args.env_config)
    if not path.exists():
        path = REPO_ROOT / path
    return probe(path)


if __name__ == "__main__":
    raise SystemExit(main())
