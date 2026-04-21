"""VI-only sweep to widen the analytical oracle-vs-compromise gap on E_final.

Runs VI on a curated list of env parameter candidates. For each candidate,
reports the analytical gap `oracle_vi_return - compromise_vi_return` plus
per-regime optima and policy divergence. No training — pure numpy VI, seconds
per config.

Candidates are ordered most→least promising:
  C0 — baseline (current E_final) for reference
  C1 — break bull/bear mirror symmetry (mild)
  C2 — stronger asymmetry
  C3 — C2 + asymmetric noise regime (slight upward drift in neutral)
  C4 — C2 + persistence 0.98 → 0.95
  C5 — C2 + persistence 0.98 → 0.90
  C6 — C2 + heavier inventory penalty (raises value of regime info)

Writes: results/milestones/M3/env_gap_sweep.json + console table.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from envs.mm_reduced import MMReducedEnv
from oracles.value_iteration import (
    compromise_policy_expected_returns,
    policy_disagreement,
    solve_value_iteration,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "results" / "milestones" / "M3" / "env_gap_sweep.json"


_PERSISTENCE_098 = [
    0.98, 0.01, 0.01,
    0.01, 0.98, 0.01,
    0.01, 0.01, 0.98,
]
_PERSISTENCE_095 = [
    0.95, 0.025, 0.025,
    0.025, 0.95, 0.025,
    0.025, 0.025, 0.95,
]
_PERSISTENCE_090 = [
    0.90, 0.05, 0.05,
    0.05, 0.90, 0.05,
    0.05, 0.05, 0.90,
]

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
    (
        "C0_baseline",
        "Current E_final (symmetric bull/bear mirror, p=0.98)",
        dict(
            transition_matrix=_PERSISTENCE_098,
            regime_p_tight_bid=[0.6, 0.2, 0.9],
            regime_p_wide_bid=[0.2, 0.05, 0.4],
            regime_p_tight_ask=[0.6, 0.9, 0.2],
            regime_p_wide_ask=[0.2, 0.4, 0.05],
        ),
    ),
    (
        "C1_asym_mild",
        "Break mirror: bull strongly ask-heavy, bear moderately bid-heavy",
        dict(
            transition_matrix=_PERSISTENCE_098,
            #                   noise  bull  bear
            regime_p_tight_bid=[0.60, 0.15, 0.70],
            regime_p_wide_bid=[0.20, 0.03, 0.30],
            regime_p_tight_ask=[0.60, 0.85, 0.30],
            regime_p_wide_ask=[0.20, 0.50, 0.10],
        ),
    ),
    (
        "C2_asym_strong",
        "Stronger asym: bull extreme ask-heavy, bear moderate bid + quieter",
        dict(
            transition_matrix=_PERSISTENCE_098,
            regime_p_tight_bid=[0.60, 0.10, 0.75],
            regime_p_wide_bid=[0.20, 0.02, 0.30],
            regime_p_tight_ask=[0.60, 0.95, 0.20],
            regime_p_wide_ask=[0.20, 0.60, 0.05],
        ),
    ),
    (
        "C3_noise_biased",
        "C2 + noise biased upward (asymmetric even in neutral regime)",
        dict(
            transition_matrix=_PERSISTENCE_098,
            regime_p_tight_bid=[0.50, 0.10, 0.75],
            regime_p_wide_bid=[0.15, 0.02, 0.30],
            regime_p_tight_ask=[0.70, 0.95, 0.20],
            regime_p_wide_ask=[0.25, 0.60, 0.05],
        ),
    ),
    (
        "C4_asym_p95",
        "C2 with persistence dropped 0.98 → 0.95 (widens inference cost)",
        dict(
            transition_matrix=_PERSISTENCE_095,
            regime_p_tight_bid=[0.60, 0.10, 0.75],
            regime_p_wide_bid=[0.20, 0.02, 0.30],
            regime_p_tight_ask=[0.60, 0.95, 0.20],
            regime_p_wide_ask=[0.20, 0.60, 0.05],
        ),
    ),
    (
        "C5_asym_p90",
        "C2 with persistence dropped 0.98 → 0.90",
        dict(
            transition_matrix=_PERSISTENCE_090,
            regime_p_tight_bid=[0.60, 0.10, 0.75],
            regime_p_wide_bid=[0.20, 0.02, 0.30],
            regime_p_tight_ask=[0.60, 0.95, 0.20],
            regime_p_wide_ask=[0.20, 0.60, 0.05],
        ),
    ),
    (
        "C6_asym_penalty05",
        "C2 with inventory_penalty 0.01 → 0.05 (makes regime info more valuable)",
        dict(
            transition_matrix=_PERSISTENCE_098,
            regime_p_tight_bid=[0.60, 0.10, 0.75],
            regime_p_wide_bid=[0.20, 0.02, 0.30],
            regime_p_tight_ask=[0.60, 0.95, 0.20],
            regime_p_wide_ask=[0.20, 0.60, 0.05],
            inventory_penalty=0.05,
        ),
    ),
]


def _eval_candidate(name: str, description: str, params: dict) -> dict:
    env = _env_from(params)
    t0 = time.perf_counter()
    vi = solve_value_iteration(env)
    per_regime_compromise_ep, compromise_mixed_ep, compromise_policy = (
        compromise_policy_expected_returns(env, vi)
    )
    divergence_frac, _ = policy_disagreement(vi)
    elapsed = time.perf_counter() - t0

    oracle_mixed = float(vi.mixed_expected_episode_return)
    total_gap = oracle_mixed - compromise_mixed_ep
    per_regime_oracle = vi.per_regime_expected_episode_return.tolist()

    return {
        "name": name,
        "description": description,
        "params": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in params.items()},
        "oracle_mixed_vi_return": oracle_mixed,
        "compromise_mixed_vi_return": float(compromise_mixed_ep),
        "analytical_total_gap": float(total_gap),
        "per_regime_oracle_vi_return": per_regime_oracle,
        "per_regime_compromise_vi_return": per_regime_compromise_ep.tolist(),
        "policy_divergence_fraction": float(divergence_frac),
        "oracle_policy_per_regime": vi.policy.tolist(),
        "compromise_policy": compromise_policy.tolist(),
        "vi_seconds": elapsed,
    }


def _fmt_table(rows: list[dict]) -> str:
    header = (
        f"{'name':<20} {'oracle':>9} {'compr':>9} {'gap':>7} "
        f"{'gap%':>6} {'div':>5} {'per-regime oracle':<28}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        gap_pct = (
            r["analytical_total_gap"] / r["compromise_mixed_vi_return"] * 100
            if r["compromise_mixed_vi_return"] > 0
            else 0.0
        )
        per_reg = ", ".join(f"{v:.1f}" for v in r["per_regime_oracle_vi_return"])
        lines.append(
            f"{r['name']:<20} "
            f"{r['oracle_mixed_vi_return']:>9.2f} "
            f"{r['compromise_mixed_vi_return']:>9.2f} "
            f"{r['analytical_total_gap']:>7.2f} "
            f"{gap_pct:>5.1f}% "
            f"{r['policy_divergence_fraction']:>5.2f} "
            f"[{per_reg}]"
        )
    return "\n".join(lines)


def main() -> int:
    rows = []
    for name, description, params in CANDIDATES:
        print(f"[sweep] evaluating {name}: {description}", flush=True)
        row = _eval_candidate(name, description, params)
        rows.append(row)
        print(
            f"[sweep]   oracle={row['oracle_mixed_vi_return']:.2f}  "
            f"compromise={row['compromise_mixed_vi_return']:.2f}  "
            f"gap={row['analytical_total_gap']:.2f}  "
            f"divergence={row['policy_divergence_fraction']:.2f}  "
            f"({row['vi_seconds']:.2f}s)",
            flush=True,
        )

    print("\n" + _fmt_table(rows) + "\n")

    baseline_gap = rows[0]["analytical_total_gap"]
    best = max(rows, key=lambda r: r["analytical_total_gap"])
    print(
        f"[sweep] baseline gap (C0) = {baseline_gap:.2f}  |  "
        f"best = {best['name']} gap={best['analytical_total_gap']:.2f} "
        f"({best['analytical_total_gap'] / baseline_gap:.1f}× baseline)",
        flush=True,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {"candidates": rows, "baseline_name": rows[0]["name"], "best_name": best["name"]},
            f,
            indent=2,
        )
    print(f"[sweep] OK | candidates={len(rows)} | best={best['name']} | output={OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
