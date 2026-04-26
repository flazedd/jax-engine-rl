"""Third-pass VI+Q-MDP sweep to widen compromise-cost on E_final.

The D-sweep confirmed E3 is near-Pareto-optimal on total_gap, but the
compromise policy on E3 only bleeds meaningfully in the bear regime
(bull's optimum is flat FAVOR_BID at every inventory, so the compromise
that picks FAVOR_BID most of the time is already near-optimal for bull).

This sweep pulls levers that force the compromise policy to fail on
*multiple* regimes simultaneously, widening the belief_qmdp - compromise
component (= compromise_cost):

  F0 — reference (E3)
  F1 — moderated bull (tight_ask 0.95→0.80, tight_bid 0.10→0.25)
        → bull optimal becomes V-shaped instead of flat FAVOR_BID.
  F2 — 4 regimes: noise/bull/bear + sideways (sym-fill, medium liquidity)
        → adds a fourth distinct optimal that pulls the compromise.
  F3 — asymmetric persistence [0.99 noise, 0.95 bull, 0.97 bear]
        → biases compromise toward the sticky regimes (noise+bear).
  F4 — F1 + inventory_penalty 0.01 → 0.02 (amplifies wrong-action cost)

Each candidate reports oracle_vi / belief_qmdp / compromise_vi, with the
two gap components and total_gap, plus per-regime breakdown of the
oracle VI returns so we can see which regimes bleed under compromise.

Pure analytical + Monte Carlo; seconds per candidate.
Writes: results/milestones/M3/env_gap_sweep_f.json + console table.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from envs.market_making_v1 import MarketMakingV1
from oracles.value_iteration import (
    belief_qmdp_expected_return,
    compromise_policy_expected_returns,
    policy_disagreement,
    solve_value_iteration,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "results" / "milestones" / "M3" / "env_gap_sweep_f.json"

_PERSISTENCE_098 = [
    0.98, 0.01, 0.01,
    0.01, 0.98, 0.01,
    0.01, 0.01, 0.98,
]

_E3_PARAMS = dict(
    transition_matrix=_PERSISTENCE_098,
    regime_p_tight_bid=[0.60, 0.10, 0.75],
    regime_p_wide_bid=[0.20, 0.02, 0.30],
    regime_p_tight_ask=[0.60, 0.95, 0.20],
    regime_p_wide_ask=[0.20, 0.60, 0.05],
)

_BASE_3 = dict(
    inventory_max=5,
    episode_length=128,
    inventory_penalty=0.01,
    gamma=0.99,
    reset_inventory_range=0,
    n_regimes=3,
    initial_distribution=[1 / 3, 1 / 3, 1 / 3],
)

# F2: 4 regimes (noise, bull, bear, sideways). 0.97 diag, 0.01 off-diag.
_PERSISTENCE_097_4 = [
    0.97, 0.01, 0.01, 0.01,
    0.01, 0.97, 0.01, 0.01,
    0.01, 0.01, 0.97, 0.01,
    0.01, 0.01, 0.01, 0.97,
]

_BASE_4 = dict(
    inventory_max=5,
    episode_length=128,
    inventory_penalty=0.01,
    gamma=0.99,
    reset_inventory_range=0,
    n_regimes=4,
    initial_distribution=[0.25, 0.25, 0.25, 0.25],
)


def _env_from(params: dict, base: dict) -> MarketMakingV1:
    return MarketMakingV1(**{**base, **params})


# F3: asymmetric persistence. Noise stickiest (0.99), bull moderately sticky
# (0.95), bear jittery (0.97). Off-diag rebalanced so each row sums to 1.
# Row 0 (noise → *): 0.99 / 0.005 / 0.005
# Row 1 (bull  → *): 0.025 / 0.95 / 0.025
# Row 2 (bear  → *): 0.015 / 0.015 / 0.97
_PERSISTENCE_ASYM = [
    0.99, 0.005, 0.005,
    0.025, 0.95, 0.025,
    0.015, 0.015, 0.97,
]


CANDIDATES: list[tuple[str, str, dict, dict]] = [
    ("F0_reference", "E3 baseline (current E_final)", dict(_E3_PARAMS), _BASE_3),
    (
        "F1_moderated_bull",
        "E3 with bull moderated (tight_ask 0.95→0.80, tight_bid 0.10→0.25)",
        {
            **_E3_PARAMS,
            #                   noise  bull  bear
            "regime_p_tight_bid": [0.60, 0.25, 0.75],
            "regime_p_wide_bid":  [0.20, 0.08, 0.30],
            "regime_p_tight_ask": [0.60, 0.80, 0.20],
            "regime_p_wide_ask":  [0.20, 0.40, 0.05],
        },
        _BASE_3,
    ),
    (
        "F2_four_regimes",
        "E3 + 4th sideways regime (sym-fill, medium liquidity)",
        {
            "transition_matrix": _PERSISTENCE_097_4,
            #                     noise  bull  bear  sideways
            "regime_p_tight_bid": [0.60, 0.10, 0.75, 0.50],
            "regime_p_wide_bid":  [0.20, 0.02, 0.30, 0.15],
            "regime_p_tight_ask": [0.60, 0.95, 0.20, 0.50],
            "regime_p_wide_ask":  [0.20, 0.60, 0.05, 0.15],
        },
        _BASE_4,
    ),
    (
        "F3_asym_persistence",
        "E3 fills with asymmetric persistence [0.99 noise, 0.95 bull, 0.97 bear]",
        {**_E3_PARAMS, "transition_matrix": _PERSISTENCE_ASYM},
        _BASE_3,
    ),
    (
        "F4_mod_bull_steeper",
        "F1 with inventory_penalty 0.01→0.02",
        {
            #                   noise  bull  bear
            "regime_p_tight_bid": [0.60, 0.25, 0.75],
            "regime_p_wide_bid":  [0.20, 0.08, 0.30],
            "regime_p_tight_ask": [0.60, 0.80, 0.20],
            "regime_p_wide_ask":  [0.20, 0.40, 0.05],
            "transition_matrix": _PERSISTENCE_098,
        },
        {**_BASE_3, "inventory_penalty": 0.02},
    ),
]


def _eval_candidate(name: str, description: str, params: dict, base: dict) -> dict:
    env = _env_from(params, base)
    t0 = time.perf_counter()
    vi = solve_value_iteration(env)
    _, compr_mixed, compr_policy = compromise_policy_expected_returns(env, vi)
    belief_mixed = belief_qmdp_expected_return(env, vi.Q, n_trajectories=2000, seed=0)
    divergence_frac, _ = policy_disagreement(vi)
    elapsed = time.perf_counter() - t0

    oracle_mixed = float(vi.mixed_expected_episode_return)
    inference_cost = oracle_mixed - belief_mixed
    compromise_cost = belief_mixed - compr_mixed
    total_gap = oracle_mixed - compr_mixed

    return {
        "name": name,
        "description": description,
        "n_regimes": env.n_regimes,
        "params": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in params.items()},
        "episode_length": env.episode_length,
        "inventory_penalty": env.inventory_penalty,
        "oracle_vi_mixed": oracle_mixed,
        "belief_qmdp_mixed": float(belief_mixed),
        "compromise_vi_mixed": float(compr_mixed),
        "inference_cost": float(inference_cost),
        "compromise_cost": float(compromise_cost),
        "total_gap": float(total_gap),
        "per_regime_oracle_vi": vi.per_regime_expected_episode_return.tolist(),
        "policy_divergence_fraction": float(divergence_frac),
        "compromise_policy": compr_policy.tolist(),
        "eval_seconds": elapsed,
    }


def _fmt_table(rows: list[dict]) -> str:
    header = (
        f"{'name':<24} {'R':>2} {'oracle':>8} {'belief':>8} {'compr':>8} "
        f"{'infer':>7} {'comp':>7} {'total':>7} {'div':>5}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['name']:<24} "
            f"{r['n_regimes']:>2d} "
            f"{r['oracle_vi_mixed']:>8.2f} "
            f"{r['belief_qmdp_mixed']:>8.2f} "
            f"{r['compromise_vi_mixed']:>8.2f} "
            f"{r['inference_cost']:>7.2f} "
            f"{r['compromise_cost']:>7.2f} "
            f"{r['total_gap']:>7.2f} "
            f"{r['policy_divergence_fraction']:>5.2f}"
        )
    return "\n".join(lines)


def main() -> int:
    rows = []
    for name, description, params, base in CANDIDATES:
        print(f"[f-sweep] evaluating {name}: {description}", flush=True)
        row = _eval_candidate(name, description, params, base)
        rows.append(row)
        print(
            f"[f-sweep]   oracle={row['oracle_vi_mixed']:.2f}  "
            f"belief={row['belief_qmdp_mixed']:.2f}  "
            f"compr={row['compromise_vi_mixed']:.2f}  "
            f"infer={row['inference_cost']:.2f}  "
            f"comp={row['compromise_cost']:.2f}  "
            f"total={row['total_gap']:.2f}  "
            f"per_regime={['%.1f' % x for x in row['per_regime_oracle_vi']]}  "
            f"({row['eval_seconds']:.1f}s)",
            flush=True,
        )

    print("\n" + _fmt_table(rows) + "\n")

    ref = rows[0]
    print(
        f"[f-sweep] reference (F0): total={ref['total_gap']:.2f} "
        f"infer={ref['inference_cost']:.2f} comp={ref['compromise_cost']:.2f}",
        flush=True,
    )
    # Widens compromise_cost specifically (the goal of this sweep).
    comp_winners = [r for r in rows[1:] if r["compromise_cost"] > ref["compromise_cost"]]
    if comp_winners:
        print("[f-sweep] candidates with WIDER compromise_cost than F0:")
        for r in comp_winners:
            print(
                f"[f-sweep]   {r['name']}: comp={r['compromise_cost']:.2f} "
                f"(+{r['compromise_cost'] - ref['compromise_cost']:.2f}) "
                f"infer={r['inference_cost']:.2f} total={r['total_gap']:.2f}",
                flush=True,
            )
    else:
        print("[f-sweep] no candidate widens compromise_cost vs F0.", flush=True)

    dominants = [
        r
        for r in rows[1:]
        if r["inference_cost"] >= ref["inference_cost"]
        and r["compromise_cost"] >= ref["compromise_cost"]
        and r["total_gap"] > ref["total_gap"]
    ]
    if dominants:
        print("[f-sweep] strictly Pareto-dominant over F0 on both components:")
        for r in dominants:
            print(
                f"[f-sweep]   {r['name']}: total={r['total_gap']:.2f} "
                f"(+{r['total_gap'] - ref['total_gap']:.2f})",
                flush=True,
            )

    best = max(rows, key=lambda r: r["total_gap"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {"candidates": rows, "reference_name": ref["name"], "best_name": best["name"]},
            f,
            indent=2,
        )
    print(f"[f-sweep] OK | candidates={len(rows)} | best={best['name']} | output={OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
