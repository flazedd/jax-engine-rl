"""Sixth-pass VI+Q-MDP sweep — push H6 further.

H6 locked as new reference: persistence [0.90, 0.995, 0.995] + E3 fills
+ initial_distribution=stationary. Current stats:
  oracle=76.60, belief=72.84, compr=46.09
  infer=3.76, comp=26.76, total=30.52

Levers to push compromise_cost further (keeping T=128, 3-regime structure
where possible, symmetric bull=bear persistence since H5 showed asymmetry
backfires):

  I0 — H6 reference
  I1 — more extreme persistence [0.85, 0.998, 0.998]
  I2 — very extreme persistence [0.80, 0.999, 0.999]
  I3 — bear mirrored to bull's asymmetry:
        bull (0.10, 0.95) wide (0.02, 0.60)
        bear (0.95, 0.10) wide (0.60, 0.02)
       → both regimes equally extreme, directly opposite → compromise
         must pick SYM or lean one side; either way big bleed on the other
  I4 — inventory_max 5→6 (modest) — more disagreement states
  I5 — I3 fills + I1 persistence (compound push)
  I6 — I3 fills + H6 persistence + inventory_max 5→6
  I7 — H6 + steeper wide-side extremes:
        bull wide (0.01, 0.70), bear wide (0.40, 0.03)
       → deeper wide-side asymmetry makes each regime's preferred
         FAVOR action more profitable, amplifying wrong-regime cost

Pure analytical + Monte Carlo; seconds per candidate.
Writes: results/milestones/M3/env_gap_sweep_i.json + console table.
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
OUT_PATH = REPO_ROOT / "results" / "milestones" / "M3" / "env_gap_sweep_i.json"

_E3_FILLS = dict(
    regime_p_tight_bid=[0.60, 0.10, 0.75],
    regime_p_wide_bid=[0.20, 0.02, 0.30],
    regime_p_tight_ask=[0.60, 0.95, 0.20],
    regime_p_wide_ask=[0.20, 0.60, 0.05],
)

# Bear mirrored (bull's directional intensities flipped).
_MIRROR_FILLS = dict(
    #                   noise  bull  bear
    regime_p_tight_bid=[0.60, 0.10, 0.95],
    regime_p_wide_bid= [0.20, 0.02, 0.60],
    regime_p_tight_ask=[0.60, 0.95, 0.10],
    regime_p_wide_ask= [0.20, 0.60, 0.02],
)

# I7: H6 with deeper wide-side extremes, preserving E3 tight fills.
_DEEP_WIDE_FILLS = dict(
    regime_p_tight_bid=[0.60, 0.10, 0.75],
    regime_p_wide_bid= [0.20, 0.01, 0.40],
    regime_p_tight_ask=[0.60, 0.95, 0.20],
    regime_p_wide_ask= [0.20, 0.70, 0.03],
)

_PERSIST_H6 = [
    0.90, 0.05, 0.05,
    0.0025, 0.995, 0.0025,
    0.0025, 0.0025, 0.995,
]
_PERSIST_I1 = [
    0.85, 0.075, 0.075,
    0.001, 0.998, 0.001,
    0.001, 0.001, 0.998,
]
_PERSIST_I2 = [
    0.80, 0.10, 0.10,
    0.0005, 0.999, 0.0005,
    0.0005, 0.0005, 0.999,
]


def _stationary(P: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    v = np.real(eigvecs[:, idx])
    v = v / v.sum()
    return v


def _stationary_init_for(persistence_flat: list[float]) -> list[float]:
    P = np.asarray(persistence_flat, dtype=np.float64).reshape(3, 3)
    return _stationary(P).tolist()


_BASE = dict(
    inventory_max=5,
    episode_length=128,
    inventory_penalty=0.01,
    gamma=0.99,
    reset_inventory_range=0,
    n_regimes=3,
)


def _base_with_stationary(persistence_flat: list[float], **overrides) -> dict:
    return {
        **_BASE,
        "initial_distribution": _stationary_init_for(persistence_flat),
        **overrides,
    }


CANDIDATES: list[tuple[str, str, dict, dict]] = [
    (
        "I0_reference_H6",
        "H6: persistence [0.90, 0.995, 0.995] + E3 fills + stationary init",
        {"transition_matrix": _PERSIST_H6, **_E3_FILLS},
        _base_with_stationary(_PERSIST_H6),
    ),
    (
        "I1_persist_extreme",
        "Persistence [0.85, 0.998, 0.998] + E3 fills + stationary init",
        {"transition_matrix": _PERSIST_I1, **_E3_FILLS},
        _base_with_stationary(_PERSIST_I1),
    ),
    (
        "I2_persist_very_extreme",
        "Persistence [0.80, 0.999, 0.999] + E3 fills + stationary init",
        {"transition_matrix": _PERSIST_I2, **_E3_FILLS},
        _base_with_stationary(_PERSIST_I2),
    ),
    (
        "I3_bear_mirrored",
        "H6 persistence + mirrored bear fills (bear (0.95, 0.10))",
        {"transition_matrix": _PERSIST_H6, **_MIRROR_FILLS},
        _base_with_stationary(_PERSIST_H6),
    ),
    (
        "I4_inv6",
        "H6 + inventory_max 5→6",
        {"transition_matrix": _PERSIST_H6, **_E3_FILLS},
        _base_with_stationary(_PERSIST_H6, inventory_max=6),
    ),
    (
        "I5_mirror_plus_extreme",
        "I3 fills + I1 persistence (mirror + [0.85, 0.998, 0.998])",
        {"transition_matrix": _PERSIST_I1, **_MIRROR_FILLS},
        _base_with_stationary(_PERSIST_I1),
    ),
    (
        "I6_mirror_plus_inv6",
        "I3 fills + H6 persistence + inventory_max 5→6",
        {"transition_matrix": _PERSIST_H6, **_MIRROR_FILLS},
        _base_with_stationary(_PERSIST_H6, inventory_max=6),
    ),
    (
        "I7_deep_wide",
        "H6 + wide-side extremes (bull (0.01,0.70), bear (0.40,0.03))",
        {"transition_matrix": _PERSIST_H6, **_DEEP_WIDE_FILLS},
        _base_with_stationary(_PERSIST_H6),
    ),
]


def _env_from(params: dict, base: dict) -> MarketMakingV1:
    return MarketMakingV1(**{**base, **params})


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
        f"{'name':<28} {'Q':>2} {'oracle':>8} {'belief':>8} {'compr':>8} "
        f"{'infer':>7} {'comp':>7} {'total':>7} {'div':>5}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['name']:<28} "
            f"{r['inventory_max']:>2d} "
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
        print(f"[i-sweep] evaluating {name}: {description}", flush=True)
        row = _eval_candidate(name, description, params, base)
        rows.append(row)
        print(
            f"[i-sweep]   oracle={row['oracle_vi_mixed']:.2f}  "
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
        f"[i-sweep] reference (I0=H6): total={ref['total_gap']:.2f} "
        f"infer={ref['inference_cost']:.2f} comp={ref['compromise_cost']:.2f}",
        flush=True,
    )

    comp_winners = [r for r in rows[1:] if r["compromise_cost"] > ref["compromise_cost"]]
    if comp_winners:
        print("[i-sweep] candidates with WIDER compromise_cost than I0:")
        for r in comp_winners:
            print(
                f"[i-sweep]   {r['name']}: comp={r['compromise_cost']:.2f} "
                f"(+{r['compromise_cost'] - ref['compromise_cost']:.2f}) "
                f"infer={r['inference_cost']:.2f} total={r['total_gap']:.2f}",
                flush=True,
            )
    else:
        print("[i-sweep] no candidate widens compromise_cost vs I0.", flush=True)

    dominants = [
        r
        for r in rows[1:]
        if r["inference_cost"] >= ref["inference_cost"]
        and r["compromise_cost"] >= ref["compromise_cost"]
        and r["total_gap"] > ref["total_gap"]
    ]
    if dominants:
        print("[i-sweep] strictly Pareto-dominant over I0 on both components:")
        for r in dominants:
            print(
                f"[i-sweep]   {r['name']}: total={r['total_gap']:.2f} "
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
    print(f"[i-sweep] OK | candidates={len(rows)} | best={best['name']} | output={OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
