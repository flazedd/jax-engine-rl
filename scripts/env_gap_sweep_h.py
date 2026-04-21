"""Fifth-pass VI+Q-MDP sweep — widen the gap further on G2's direction.

G-sweep finding: G2 (persistence `[0.95 noise, 0.99 bull, 0.99 bear]` with
E3 fills) widened compromise_cost 16.4 → 23.1 (total 21.0 → 26.7) by
shifting the stationary distribution to ~[0.09, 0.45, 0.45]. Bull+bear
share ~91% of episode time → compromise can no longer settle on noise's
SYM.

T=256 ruled out (doubles training cost). This sweep pushes G2's direction
further on persistence / initial distribution / structural regime setup,
keeping T=128:

  H0 — G2 reference
  H1 — more extreme persistence [0.90 noise, 0.995 bull, 0.995 bear]
       → stationary moves further toward bull+bear
  H2 — G2 persistence + initial_distribution = stationary (~[0.09, 0.45, 0.45])
       → compromise bleeds from step 1 instead of warming up via noise
  H3 — 2-regime env (bull + bear only, no noise escape hatch)
       → removes the middle regime that helped compromise hide
  H4 — G2 persistence + inventory_penalty 0.01 → 0.02
       → amplifies wrong-action inventory accumulation
  H5 — asymmetric persistence: bull dominant
       [0.95 noise, 0.995 bull, 0.97 bear]
       → stationary biased to bull so compromise leans favor_bid, bear bleeds harder
  H6 — H1 + H2 (max-stationary-mass + stationary init)

Pure analytical + Monte Carlo; seconds per candidate.
Writes: results/milestones/M3/env_gap_sweep_h.json + console table.
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
OUT_PATH = REPO_ROOT / "results" / "milestones" / "M3" / "env_gap_sweep_h.json"

_E3_FILLS_3 = dict(
    regime_p_tight_bid=[0.60, 0.10, 0.75],
    regime_p_wide_bid=[0.20, 0.02, 0.30],
    regime_p_tight_ask=[0.60, 0.95, 0.20],
    regime_p_wide_ask=[0.20, 0.60, 0.05],
)

# G2 baseline: persistence biased away from noise (the H0 reference).
_PERSIST_G2 = [
    0.95, 0.025, 0.025,
    0.005, 0.99, 0.005,
    0.005, 0.005, 0.99,
]

# H1: more extreme.
_PERSIST_H1 = [
    0.90, 0.05, 0.05,
    0.0025, 0.995, 0.0025,
    0.0025, 0.0025, 0.995,
]

# H5: asymmetric, bull dominant.
_PERSIST_H5 = [
    0.95, 0.025, 0.025,
    0.0025, 0.995, 0.0025,
    0.015, 0.015, 0.97,
]


def _stationary(P: np.ndarray) -> np.ndarray:
    """Left eigenvector with eigenvalue 1, normalized."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    v = np.real(eigvecs[:, idx])
    v = v / v.sum()
    return v


def _stationary_init_for(persistence_flat: list[float]) -> list[float]:
    P = np.asarray(persistence_flat, dtype=np.float64).reshape(3, 3)
    return _stationary(P).tolist()


_BASE_3 = dict(
    inventory_max=5,
    episode_length=128,
    inventory_penalty=0.01,
    gamma=0.99,
    reset_inventory_range=0,
    n_regimes=3,
    initial_distribution=[1 / 3, 1 / 3, 1 / 3],
)

# H3: 2-regime env (bull + bear only). Use E3's bull and bear fill probs.
_BASE_2 = dict(
    inventory_max=5,
    episode_length=128,
    inventory_penalty=0.01,
    gamma=0.99,
    reset_inventory_range=0,
    n_regimes=2,
    initial_distribution=[0.5, 0.5],
)

_E3_FILLS_2 = dict(
    #                   bull  bear
    regime_p_tight_bid=[0.10, 0.75],
    regime_p_wide_bid= [0.02, 0.30],
    regime_p_tight_ask=[0.95, 0.20],
    regime_p_wide_ask= [0.60, 0.05],
)

_PERSIST_2 = [
    0.99, 0.01,
    0.01, 0.99,
]


def _env_from(params: dict, base: dict) -> MMReducedEnv:
    return MMReducedEnv(**{**base, **params})


CANDIDATES: list[tuple[str, str, dict, dict]] = [
    (
        "H0_reference_G2",
        "G2: persistence [0.95, 0.99, 0.99] + E3 fills",
        {"transition_matrix": _PERSIST_G2, **_E3_FILLS_3},
        _BASE_3,
    ),
    (
        "H1_persistence_extreme",
        "Persistence [0.90, 0.995, 0.995] + E3 fills",
        {"transition_matrix": _PERSIST_H1, **_E3_FILLS_3},
        _BASE_3,
    ),
    (
        "H2_stationary_init",
        "G2 persistence + initial_distribution = stationary",
        {"transition_matrix": _PERSIST_G2, **_E3_FILLS_3},
        {**_BASE_3, "initial_distribution": _stationary_init_for(_PERSIST_G2)},
    ),
    (
        "H3_two_regimes",
        "2-regime env: bull + bear only (no noise escape)",
        {"transition_matrix": _PERSIST_2, **_E3_FILLS_2},
        _BASE_2,
    ),
    (
        "H4_G2_steeper_penalty",
        "G2 + inventory_penalty 0.01 → 0.02",
        {"transition_matrix": _PERSIST_G2, **_E3_FILLS_3},
        {**_BASE_3, "inventory_penalty": 0.02},
    ),
    (
        "H5_bull_dominant",
        "Asymmetric persistence [0.95, 0.995, 0.97] (bull sticky)",
        {"transition_matrix": _PERSIST_H5, **_E3_FILLS_3},
        _BASE_3,
    ),
    (
        "H6_extreme_plus_stationary_init",
        "H1 persistence + initial_distribution = stationary",
        {"transition_matrix": _PERSIST_H1, **_E3_FILLS_3},
        {**_BASE_3, "initial_distribution": _stationary_init_for(_PERSIST_H1)},
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
        "inventory_max": env.inventory_max,
        "episode_length": env.episode_length,
        "inventory_penalty": env.inventory_penalty,
        "initial_distribution": list(env.initial_distribution),
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
        f"{'name':<34} {'R':>2} {'oracle':>8} {'belief':>8} {'compr':>8} "
        f"{'infer':>7} {'comp':>7} {'total':>7} {'div':>5}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['name']:<34} "
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
        print(f"[h-sweep] evaluating {name}: {description}", flush=True)
        row = _eval_candidate(name, description, params, base)
        rows.append(row)
        print(
            f"[h-sweep]   oracle={row['oracle_vi_mixed']:.2f}  "
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
        f"[h-sweep] reference (H0=G2): total={ref['total_gap']:.2f} "
        f"infer={ref['inference_cost']:.2f} comp={ref['compromise_cost']:.2f}",
        flush=True,
    )

    comp_winners = [r for r in rows[1:] if r["compromise_cost"] > ref["compromise_cost"]]
    if comp_winners:
        print("[h-sweep] candidates with WIDER compromise_cost than H0:")
        for r in comp_winners:
            print(
                f"[h-sweep]   {r['name']}: comp={r['compromise_cost']:.2f} "
                f"(+{r['compromise_cost'] - ref['compromise_cost']:.2f}) "
                f"infer={r['inference_cost']:.2f} total={r['total_gap']:.2f}",
                flush=True,
            )
    else:
        print("[h-sweep] no candidate widens compromise_cost vs H0.", flush=True)

    dominants = [
        r
        for r in rows[1:]
        if r["inference_cost"] >= ref["inference_cost"]
        and r["compromise_cost"] >= ref["compromise_cost"]
        and r["total_gap"] > ref["total_gap"]
    ]
    if dominants:
        print("[h-sweep] strictly Pareto-dominant over H0 on both components:")
        for r in dominants:
            print(
                f"[h-sweep]   {r['name']}: total={r['total_gap']:.2f} "
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
    print(f"[h-sweep] OK | candidates={len(rows)} | best={best['name']} | output={OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
