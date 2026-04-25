"""E5 sweep — widen belief−agnostic specifically, ranked by compromise/inference ratio.

Motivation. M3 on E_final=E3 measured belief=97.6, regime_agnostic=84.7
(belief−agnostic ≈ 12.9, 15% of agnostic mean), narrow enough that RQ2's
ladder may not separate methods statistically across 5 seeds. Prior sweeps
F–I explored persistence + bull/bear-mirror but were ranked by total_gap
(oracle − compromise) and ignored the *split* between inference cost and
compromise cost. Some sweep-I winners (e.g. I7_deep_wide) had bull regime
structurally unprofitable (37 vs 152 noise), failing inferability spirit.

This sweep targets the **compromise-policy slice** directly:

    compromise_cost = belief_qmdp_mixed − compromise_vi_mixed
    inference_cost  = oracle_vi_mixed   − belief_qmdp_mixed
    ratio           = compromise_cost / inference_cost  (want > 2)

Belief is approximated by Q-MDP Monte Carlo, the same proxy used in the F–I
sweeps. A high ratio means: knowing the regime *exactly* (oracle) helps only
marginally beyond a tracked posterior (belief), but committing to a single
regime-agnostic policy hurts a lot — exactly the regime where Belief-PPO
should beat Stacked-PPO and RL² should approach Belief-PPO.

Untested axes (vs F–I):
  - inventory_penalty κ (E3-baseline 0.01); higher κ punishes wrong-side
    inventory build-up under a regime-agnostic action.
  - inventory_max 5 → 7 (F–I tested 5 and 6); more disagreement states.
  - n_regimes = 2 (pure bull/bear, no neutral), forcing the compromise into
    a strict 50/50 trade-off.
  - 4-regime (noise + mild/strong-bull + strong-bear); more action diversity.

Constraints checked per candidate:
  - belief_qmdp_mixed >= compromise_vi_mixed (sanity: belief beats compromise)
  - per_regime_oracle_vi[r] > 0 for all r (no regime is structurally a loss
    pit — soft check, sweep-I's I7 failed this in spirit with 37/152 ratio)
  - policy_divergence_fraction in [0.5, 1.0] (R1 spirit)

Writes: results/milestones/M3/env_gap_sweep_e5.json + console table sorted
by `ratio` descending.

Pure analytical + Monte Carlo; seconds per candidate.
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
OUT_PATH = REPO_ROOT / "results" / "milestones" / "M3" / "env_gap_sweep_e5.json"


# ---- fill-prob templates --------------------------------------------------

# Current E_final (E3): asymmetric (bull extreme, bear moderate).
_E3_FILLS = dict(
    n_regimes=3,
    regime_p_tight_bid=[0.60, 0.10, 0.75],
    regime_p_wide_bid= [0.20, 0.02, 0.30],
    regime_p_tight_ask=[0.60, 0.95, 0.20],
    regime_p_wide_ask= [0.20, 0.60, 0.05],
    initial_distribution=[1 / 3, 1 / 3, 1 / 3],
)

# Full mirror: bull and bear equally extreme, mirror images of each other.
_MIRROR3_FILLS = dict(
    n_regimes=3,
    #                   noise  bull  bear
    regime_p_tight_bid=[0.60, 0.10, 0.95],
    regime_p_wide_bid= [0.20, 0.02, 0.60],
    regime_p_tight_ask=[0.60, 0.95, 0.10],
    regime_p_wide_ask= [0.20, 0.60, 0.02],
    initial_distribution=[1 / 3, 1 / 3, 1 / 3],
)

# 2-regime pure mirror: no neutral.
_MIRROR2_FILLS = dict(
    n_regimes=2,
    #                   bull  bear
    regime_p_tight_bid=[0.10, 0.95],
    regime_p_wide_bid= [0.02, 0.60],
    regime_p_tight_ask=[0.95, 0.10],
    regime_p_wide_ask= [0.60, 0.02],
    initial_distribution=[0.5, 0.5],
)

# 4-regime: noise + mild-bull + strong-bull + strong-bear. Asymmetric counts
# force the compromise to choose which regime to favor.
_4REG_FILLS = dict(
    n_regimes=4,
    #                   noise  mild-bull  strong-bull  strong-bear
    regime_p_tight_bid=[0.60,  0.30,      0.10,        0.95],
    regime_p_wide_bid= [0.20,  0.10,      0.02,        0.60],
    regime_p_tight_ask=[0.60,  0.80,      0.95,        0.10],
    regime_p_wide_ask= [0.20,  0.45,      0.60,        0.02],
    initial_distribution=[0.25, 0.25, 0.25, 0.25],
)


# ---- transition templates --------------------------------------------------

def _symmetric_persistence(n: int, p_diag: float) -> list[float]:
    """Symmetric n×n transition matrix with `p_diag` on the diagonal."""
    off = (1.0 - p_diag) / (n - 1)
    M = np.full((n, n), off)
    np.fill_diagonal(M, p_diag)
    return M.flatten().tolist()


# ---- candidate definitions -------------------------------------------------

_BASE_DEFAULTS = dict(
    inventory_max=5,
    episode_length=128,
    inventory_penalty=0.01,
    gamma=0.99,
    reset_inventory_range=0,
)


def _make(*, fills: dict, persist: float, **overrides) -> dict:
    """Build a full env params dict."""
    n = fills["n_regimes"]
    return {
        **_BASE_DEFAULTS,
        **fills,
        "transition_matrix": _symmetric_persistence(n, persist),
        **overrides,
    }


CANDIDATES: list[tuple[str, str, dict]] = [
    # --- references ---------------------------------------------------------
    (
        "R0_E3_reference",
        "Current E_final (E3 asymmetric, κ=0.01, p=0.98)",
        _make(fills=_E3_FILLS, persist=0.98),
    ),

    # --- κ axis on E3 fills (isolate inventory-penalty effect) --------------
    (
        "K1_E3_kappa02",
        "E3 fills + κ=0.02",
        _make(fills=_E3_FILLS, persist=0.98, inventory_penalty=0.02),
    ),
    (
        "K2_E3_kappa03",
        "E3 fills + κ=0.03",
        _make(fills=_E3_FILLS, persist=0.98, inventory_penalty=0.03),
    ),
    (
        "K3_E3_kappa05",
        "E3 fills + κ=0.05",
        _make(fills=_E3_FILLS, persist=0.98, inventory_penalty=0.05),
    ),

    # --- mirror axis (3 regimes), κ axis ------------------------------------
    (
        "M1_mirror3_kappa01",
        "Bull/bear mirror (3 reg) + κ=0.01 + p=0.98",
        _make(fills=_MIRROR3_FILLS, persist=0.98),
    ),
    (
        "M2_mirror3_kappa02",
        "Bull/bear mirror (3 reg) + κ=0.02 + p=0.98",
        _make(fills=_MIRROR3_FILLS, persist=0.98, inventory_penalty=0.02),
    ),
    (
        "M3_mirror3_kappa03",
        "Bull/bear mirror (3 reg) + κ=0.03 + p=0.98 ← E5 default",
        _make(fills=_MIRROR3_FILLS, persist=0.98, inventory_penalty=0.03),
    ),
    (
        "M4_mirror3_kappa05",
        "Bull/bear mirror (3 reg) + κ=0.05 + p=0.98",
        _make(fills=_MIRROR3_FILLS, persist=0.98, inventory_penalty=0.05),
    ),

    # --- mirror + persistence axis ------------------------------------------
    (
        "P1_mirror3_p95_kappa03",
        "Mirror (3 reg) + p=0.95 + κ=0.03",
        _make(fills=_MIRROR3_FILLS, persist=0.95, inventory_penalty=0.03),
    ),

    # --- 2-regime pure mirror -----------------------------------------------
    (
        "T1_mirror2_kappa01",
        "Pure 2-regime mirror (no neutral) + κ=0.01 + p=0.97",
        _make(fills=_MIRROR2_FILLS, persist=0.97),
    ),
    (
        "T2_mirror2_kappa03",
        "Pure 2-regime mirror (no neutral) + κ=0.03 + p=0.97",
        _make(fills=_MIRROR2_FILLS, persist=0.97, inventory_penalty=0.03),
    ),

    # --- inventory_max axis on the strongest 3-regime mirror ----------------
    (
        "I1_mirror3_inv7_kappa03",
        "Mirror (3 reg) + inv_max=7 + κ=0.03",
        _make(
            fills=_MIRROR3_FILLS,
            persist=0.98,
            inventory_penalty=0.03,
            inventory_max=7,
        ),
    ),

    # --- 4-regime ------------------------------------------------------------
    (
        "F1_4reg_kappa01",
        "4 regimes (noise/mild-bull/strong-bull/strong-bear) + κ=0.01",
        _make(fills=_4REG_FILLS, persist=0.98),
    ),
    (
        "F2_4reg_kappa03",
        "4 regimes + κ=0.03",
        _make(fills=_4REG_FILLS, persist=0.98, inventory_penalty=0.03),
    ),
]


# ---- evaluation ------------------------------------------------------------


def _env_from(params: dict) -> MMReducedEnv:
    return MMReducedEnv(**params)


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
    ratio = (
        compromise_cost / inference_cost
        if inference_cost > 1e-6
        else float("inf")
    )

    per_regime = vi.per_regime_expected_episode_return.tolist()
    per_regime_min = float(min(per_regime))
    per_regime_max = float(max(per_regime))

    # Sanity flags (soft — recorded, not enforced).
    belief_beats_compromise = belief_mixed > compr_mixed + 1e-3
    all_regimes_profitable = per_regime_min > 0.0
    divergence_ok = 0.5 <= divergence_frac <= 1.0
    sanity_ok = bool(
        belief_beats_compromise and all_regimes_profitable and divergence_ok
    )

    return {
        "name": name,
        "description": description,
        "n_regimes": env.n_regimes,
        "inventory_max": env.inventory_max,
        "episode_length": env.episode_length,
        "inventory_penalty": env.inventory_penalty,
        "transition_matrix": list(env.transition_matrix),
        "regime_p_tight_bid": list(env.regime_p_tight_bid),
        "regime_p_wide_bid": list(env.regime_p_wide_bid),
        "regime_p_tight_ask": list(env.regime_p_tight_ask),
        "regime_p_wide_ask": list(env.regime_p_wide_ask),
        "initial_distribution": list(env.initial_distribution),
        "oracle_vi_mixed": oracle_mixed,
        "belief_qmdp_mixed": float(belief_mixed),
        "compromise_vi_mixed": float(compr_mixed),
        "inference_cost": float(inference_cost),
        "compromise_cost": float(compromise_cost),
        "ratio_compromise_over_inference": float(ratio),
        "total_gap": float(total_gap),
        "per_regime_oracle_vi": per_regime,
        "per_regime_min": per_regime_min,
        "per_regime_max": per_regime_max,
        "policy_divergence_fraction": float(divergence_frac),
        "compromise_policy": compr_policy.tolist(),
        "sanity_belief_beats_compromise": belief_beats_compromise,
        "sanity_all_regimes_profitable": all_regimes_profitable,
        "sanity_divergence_in_range": divergence_ok,
        "sanity_ok": sanity_ok,
        "eval_seconds": elapsed,
    }


def _fmt_table(rows: list[dict]) -> str:
    header = (
        f"{'name':<28} {'nR':>2} {'κ':>5} {'oracle':>7} {'belief':>7} "
        f"{'compr':>7} {'infer':>6} {'comp':>6} {'ratio':>6} "
        f"{'div':>5} {'min_r':>6} {'OK':>3}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        ratio_str = (
            f"{r['ratio_compromise_over_inference']:>6.2f}"
            if np.isfinite(r["ratio_compromise_over_inference"])
            else "   inf"
        )
        lines.append(
            f"{r['name']:<28} "
            f"{r['n_regimes']:>2d} "
            f"{r['inventory_penalty']:>5.2f} "
            f"{r['oracle_vi_mixed']:>7.2f} "
            f"{r['belief_qmdp_mixed']:>7.2f} "
            f"{r['compromise_vi_mixed']:>7.2f} "
            f"{r['inference_cost']:>6.2f} "
            f"{r['compromise_cost']:>6.2f} "
            f"{ratio_str} "
            f"{r['policy_divergence_fraction']:>5.2f} "
            f"{r['per_regime_min']:>6.1f} "
            f"{'Y' if r['sanity_ok'] else 'n':>3}"
        )
    return "\n".join(lines)


def main() -> int:
    rows = []
    for name, description, params in CANDIDATES:
        print(f"[e5-sweep] evaluating {name}: {description}", flush=True)
        row = _eval_candidate(name, description, params)
        rows.append(row)
        print(
            f"[e5-sweep]   oracle={row['oracle_vi_mixed']:.2f}  "
            f"belief={row['belief_qmdp_mixed']:.2f}  "
            f"compr={row['compromise_vi_mixed']:.2f}  "
            f"infer={row['inference_cost']:.2f}  "
            f"comp={row['compromise_cost']:.2f}  "
            f"ratio={row['ratio_compromise_over_inference']:.2f}  "
            f"min_r={row['per_regime_min']:.1f}  "
            f"sanity={'OK' if row['sanity_ok'] else 'FAIL'}  "
            f"({row['eval_seconds']:.1f}s)",
            flush=True,
        )

    # Print original-order table.
    print("\n[e5-sweep] all candidates (original order):")
    print(_fmt_table(rows))

    # Reference is R0 (current E_final = E3).
    ref = next(r for r in rows if r["name"] == "R0_E3_reference")

    # Ranking 1: by compromise_cost / inference_cost ratio (the headline metric).
    ranked_by_ratio = sorted(
        rows,
        key=lambda r: (
            r["sanity_ok"],
            r["ratio_compromise_over_inference"]
            if np.isfinite(r["ratio_compromise_over_inference"])
            else 1e9,
        ),
        reverse=True,
    )
    print("\n[e5-sweep] sanity-passing candidates ranked by ratio (compromise/inference):")
    print(_fmt_table([r for r in ranked_by_ratio if r["sanity_ok"]]))

    # Ranking 2: by absolute compromise_cost among sanity-passers.
    ranked_by_compromise = sorted(
        [r for r in rows if r["sanity_ok"]],
        key=lambda r: r["compromise_cost"],
        reverse=True,
    )
    print("\n[e5-sweep] sanity-passing candidates ranked by compromise_cost:")
    print(_fmt_table(ranked_by_compromise))

    # Headline comparison vs reference.
    print(
        f"\n[e5-sweep] reference (R0_E3): compromise_cost={ref['compromise_cost']:.2f} "
        f"inference_cost={ref['inference_cost']:.2f} "
        f"ratio={ref['ratio_compromise_over_inference']:.2f}",
        flush=True,
    )

    winners = [
        r
        for r in rows
        if r["sanity_ok"]
        and r["compromise_cost"] > ref["compromise_cost"]
        and r["ratio_compromise_over_inference"]
        > ref["ratio_compromise_over_inference"]
    ]
    if winners:
        print(
            "[e5-sweep] candidates dominating R0 on BOTH ratio and absolute "
            "compromise_cost (sanity-passing):"
        )
        for r in winners:
            print(
                f"[e5-sweep]   {r['name']}: "
                f"compromise_cost={r['compromise_cost']:.2f} "
                f"(+{r['compromise_cost'] - ref['compromise_cost']:.2f}), "
                f"ratio={r['ratio_compromise_over_inference']:.2f} "
                f"(+{r['ratio_compromise_over_inference'] - ref['ratio_compromise_over_inference']:.2f})",
                flush=True,
            )
    else:
        print("[e5-sweep] no candidate strictly dominates R0 on both metrics.", flush=True)

    # Pick best by ratio first, then break ties on compromise_cost.
    sanity_passers = [r for r in rows if r["sanity_ok"]]
    if sanity_passers:
        best = max(
            sanity_passers,
            key=lambda r: (
                r["ratio_compromise_over_inference"]
                if np.isfinite(r["ratio_compromise_over_inference"])
                else 1e9,
                r["compromise_cost"],
            ),
        )
        best_name = best["name"]
    else:
        best_name = "NONE_PASSED_SANITY"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {
                "candidates": rows,
                "reference_name": ref["name"],
                "best_name": best_name,
                "ranking_by_ratio": [r["name"] for r in ranked_by_ratio],
                "ranking_by_compromise_cost": [r["name"] for r in ranked_by_compromise],
            },
            f,
            indent=2,
        )
    print(
        f"[e5-sweep] OK | candidates={len(rows)} | best={best_name} | output={OUT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
