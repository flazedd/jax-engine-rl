"""Fourth-pass VI+Q-MDP sweep — reverse the F levers.

F-sweep surprise: every attempt to make compromise suck more (moderated
bull, 4th sideways regime, asymmetric persistence biased to noise,
steeper penalty) SHRUNK compromise_cost. The read: E3's extreme bull/bear
asymmetry is what forces the compromise to bleed; any moderation gives
compromise a single action that fits all regimes.

So this sweep pushes the opposite direction on each lever:

  G0 — E3 reference
  G1 — sharper extremes: bull near-1 ask-skew, bear near-1 bid-skew,
        with asymmetric total liquidity so each regime's optimum is
        as far as possible from the others'
  G2 — asymmetric persistence biased AWAY from noise
        [0.95 noise, 0.99 bull, 0.99 bear] → stationary mass shifts to
        bull+bear → compromise can't settle on noise's SYM
  G3 — bigger inventory space (inv_max 5→8) → more states where
        per-regime optima disagree, more accumulated compromise loss
  G4 — G1 + longer episodes (T 128→256) → compromise bleed accumulates
  G5 — G1 + G2 combined

Pure analytical + Monte Carlo; seconds per candidate.
Writes: results/milestones/M3/env_gap_sweep_g.json + console table.
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
OUT_PATH = REPO_ROOT / "results" / "milestones" / "M3" / "env_gap_sweep_g.json"

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

_BASE = dict(
    inventory_max=5,
    episode_length=128,
    inventory_penalty=0.01,
    gamma=0.99,
    reset_inventory_range=0,
    n_regimes=3,
    initial_distribution=[1 / 3, 1 / 3, 1 / 3],
)

# G1: sharper bull/bear extremes with asymmetric total liquidity.
# Bull: very low bid fill + very high ask fill (illiquid tight-ask regime).
# Bear: mirror.
_SHARP_EXTREMES_FILLS = dict(
    #                   noise  bull  bear
    regime_p_tight_bid=[0.60, 0.05, 0.99],
    regime_p_wide_bid= [0.20, 0.01, 0.70],
    regime_p_tight_ask=[0.60, 0.99, 0.05],
    regime_p_wide_ask= [0.20, 0.70, 0.01],
)

# G2: persistence biased away from noise.
# Row 0 noise (0.95): 0.95 / 0.025 / 0.025
# Row 1 bull  (0.99): 0.005 / 0.99 / 0.005
# Row 2 bear  (0.99): 0.005 / 0.005 / 0.99
_PERSISTENCE_AWAY_NOISE = [
    0.95, 0.025, 0.025,
    0.005, 0.99, 0.005,
    0.005, 0.005, 0.99,
]


def _env_from(params: dict, base_override: dict | None = None) -> MarketMakingV1:
    base = dict(_BASE)
    if base_override:
        base.update(base_override)
    return MarketMakingV1(**{**base, **params})


CANDIDATES: list[tuple[str, str, dict, dict | None]] = [
    ("G0_reference", "E3 baseline (current E_final)", dict(_E3_PARAMS), None),
    (
        "G1_sharp_extremes",
        "Bull (0.05,0.99) bear (0.99,0.05), asym total liquidity",
        {"transition_matrix": _PERSISTENCE_098, **_SHARP_EXTREMES_FILLS},
        None,
    ),
    (
        "G2_persistence_away_noise",
        "E3 fills, persistence [0.95 noise, 0.99 bull, 0.99 bear]",
        {**_E3_PARAMS, "transition_matrix": _PERSISTENCE_AWAY_NOISE},
        None,
    ),
    (
        "G3_bigger_inventory",
        "E3 fills, inventory_max 5→8",
        dict(_E3_PARAMS),
        {"inventory_max": 8},
    ),
    (
        "G4_sharp_long_episodes",
        "G1 fills, episode_length 128→256",
        {"transition_matrix": _PERSISTENCE_098, **_SHARP_EXTREMES_FILLS},
        {"episode_length": 256},
    ),
    (
        "G5_sharp_plus_persistence",
        "G1 fills + G2 persistence (sharp extremes, noise jittery)",
        {"transition_matrix": _PERSISTENCE_AWAY_NOISE, **_SHARP_EXTREMES_FILLS},
        None,
    ),
]


def _eval_candidate(
    name: str, description: str, params: dict, base_override: dict | None
) -> dict:
    env = _env_from(params, base_override)
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
        "inventory_max": env.inventory_max,
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
        f"{'name':<28} {'Q':>2} {'T':>4} {'oracle':>8} {'belief':>8} {'compr':>8} "
        f"{'infer':>7} {'comp':>7} {'total':>7} {'div':>5}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['name']:<28} "
            f"{r['inventory_max']:>2d} "
            f"{r['episode_length']:>4d} "
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
    for name, description, params, base_override in CANDIDATES:
        print(f"[g-sweep] evaluating {name}: {description}", flush=True)
        row = _eval_candidate(name, description, params, base_override)
        rows.append(row)
        print(
            f"[g-sweep]   oracle={row['oracle_vi_mixed']:.2f}  "
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
        f"[g-sweep] reference (G0): total={ref['total_gap']:.2f} "
        f"infer={ref['inference_cost']:.2f} comp={ref['compromise_cost']:.2f}",
        flush=True,
    )

    comp_winners = [r for r in rows[1:] if r["compromise_cost"] > ref["compromise_cost"]]
    if comp_winners:
        print("[g-sweep] candidates with WIDER compromise_cost than G0:")
        for r in comp_winners:
            print(
                f"[g-sweep]   {r['name']}: comp={r['compromise_cost']:.2f} "
                f"(+{r['compromise_cost'] - ref['compromise_cost']:.2f}) "
                f"infer={r['inference_cost']:.2f} total={r['total_gap']:.2f}",
                flush=True,
            )
    else:
        print("[g-sweep] no candidate widens compromise_cost vs G0.", flush=True)

    dominants = [
        r
        for r in rows[1:]
        if r["inference_cost"] >= ref["inference_cost"]
        and r["compromise_cost"] >= ref["compromise_cost"]
        and r["total_gap"] > ref["total_gap"]
    ]
    if dominants:
        print("[g-sweep] strictly Pareto-dominant over G0 on both components:")
        for r in dominants:
            print(
                f"[g-sweep]   {r['name']}: total={r['total_gap']:.2f} "
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
    print(f"[g-sweep] OK | candidates={len(rows)} | best={best['name']} | output={OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
