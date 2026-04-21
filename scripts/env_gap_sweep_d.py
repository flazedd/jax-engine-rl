"""Second-pass VI+Q-MDP sweep to widen the 3-level gap on E_final.

After the first C-sweep picked C2 (E3), this sweep pulls the remaining four
levers identified in the M2-passing E3 analysis to see if we can widen the
agnostic/belief/oracle ladder further:

  D0 — reference (current E3)
  D1 — shorter episodes (T=128 → T=64)        [widens inference cost]
  D2 — noise regime also biased                [removes compromise's safe middle]
  D3 — stronger symmetric extremes (bull AND bear near-1 skew, asym liquidity)
  D4 — steeper inventory penalty (0.01 → 0.03)

Each candidate reports:
  - oracle VI mixed return (ceiling)
  - Q-MDP belief mixed return (belief-PPO ceiling proxy)
  - compromise VI mixed return (floor)
  - inference_cost = oracle_VI - belief_QMDP
  - compromise_cost = belief_QMDP - compromise_VI
  - total_gap      = oracle_VI - compromise_VI

Pure analytical + Monte Carlo; no training. Seconds per candidate.
Writes: results/milestones/M3/env_gap_sweep_d.json + console table.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from envs.mm_reduced import MMReducedEnv
from oracles.value_iteration import (
    belief_qmdp_expected_return,
    compromise_policy_expected_returns,
    policy_disagreement,
    solve_value_iteration,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "results" / "milestones" / "M3" / "env_gap_sweep_d.json"

_PERSISTENCE_098 = [
    0.98, 0.01, 0.01,
    0.01, 0.98, 0.01,
    0.01, 0.01, 0.98,
]

# E3 baseline (C2 from the first sweep, current symlinked E_final).
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


def _env_from(params: dict) -> MMReducedEnv:
    return MMReducedEnv(**{**_BASE, **params})


CANDIDATES: list[tuple[str, str, dict]] = [
    ("D0_reference", "E3 baseline (current E_final)", dict(_E3_PARAMS)),
    (
        "D1_shorter_episode",
        "E3 with episode_length 128 → 64 (widens inference cost)",
        {**_E3_PARAMS, "episode_length": 64},
    ),
    (
        "D2_biased_noise",
        "E3 with noise regime moderately ask-heavy (no safe middle)",
        {
            **_E3_PARAMS,
            #                   noise  bull  bear
            "regime_p_tight_bid": [0.50, 0.10, 0.75],
            "regime_p_wide_bid":  [0.15, 0.02, 0.30],
            "regime_p_tight_ask": [0.70, 0.95, 0.20],
            "regime_p_wide_ask":  [0.25, 0.60, 0.05],
        },
    ),
    (
        "D3_strong_sym_extremes",
        "Bull AND bear near-1 skew, asymmetric total liquidity",
        {
            "transition_matrix": _PERSISTENCE_098,
            # Bull: low-liquidity + strong ask-heavy; bear: higher-liquidity + strong bid-heavy.
            "regime_p_tight_bid": [0.60, 0.05, 0.95],
            "regime_p_wide_bid":  [0.20, 0.01, 0.50],
            "regime_p_tight_ask": [0.60, 0.95, 0.05],
            "regime_p_wide_ask":  [0.20, 0.50, 0.01],
        },
    ),
    (
        "D4_steeper_penalty",
        "E3 with inventory_penalty 0.01 → 0.03 (amplifies wrong-regime action cost)",
        {**_E3_PARAMS, "inventory_penalty": 0.03},
    ),
]


def _eval_candidate(name: str, description: str, params: dict) -> dict:
    env = _env_from(params)
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
        f"{'name':<24} {'T':>4} {'oracle':>8} {'belief':>8} {'compr':>8} "
        f"{'infer':>7} {'comp':>7} {'total':>7} {'div':>5}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['name']:<24} "
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
    for name, description, params in CANDIDATES:
        print(f"[d-sweep] evaluating {name}: {description}", flush=True)
        row = _eval_candidate(name, description, params)
        rows.append(row)
        print(
            f"[d-sweep]   oracle={row['oracle_vi_mixed']:.2f}  "
            f"belief={row['belief_qmdp_mixed']:.2f}  "
            f"compr={row['compromise_vi_mixed']:.2f}  "
            f"infer={row['inference_cost']:.2f}  "
            f"comp={row['compromise_cost']:.2f}  "
            f"total={row['total_gap']:.2f}  "
            f"({row['eval_seconds']:.1f}s)",
            flush=True,
        )

    print("\n" + _fmt_table(rows) + "\n")

    ref = rows[0]
    # Rank candidates by total gap, highlight any Pareto-dominant over D0.
    print(
        f"[d-sweep] reference (D0): total={ref['total_gap']:.2f} "
        f"infer={ref['inference_cost']:.2f} comp={ref['compromise_cost']:.2f}",
        flush=True,
    )
    dominants = [
        r
        for r in rows[1:]
        if r["inference_cost"] >= ref["inference_cost"]
        and r["compromise_cost"] >= ref["compromise_cost"]
        and r["total_gap"] > ref["total_gap"]
    ]
    if dominants:
        print("[d-sweep] strictly Pareto-dominant over D0:")
        for r in dominants:
            print(
                f"[d-sweep]   {r['name']}: total={r['total_gap']:.2f} "
                f"(+{r['total_gap'] - ref['total_gap']:.2f}) "
                f"infer={r['inference_cost']:.2f} comp={r['compromise_cost']:.2f}",
                flush=True,
            )
    else:
        print(
            "[d-sweep] no candidate strictly Pareto-dominates D0 on both components.",
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
    print(f"[d-sweep] OK | candidates={len(rows)} | best={best['name']} | output={OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
