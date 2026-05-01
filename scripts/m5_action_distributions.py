"""M5 — per-(method, regime) action-distribution analysis on E_final.

For each of the 7 trained methods (3 references + 4 meta-RL), roll out
the trained policy on MarketMakingV1 E_final and record `P(action |
true_regime)`. Aggregates per-method across all available seeds.

The expected pattern, based on M2 / M3 findings:
  - regime_agnostic: flat across regimes (no regime info → no per-regime
    differentiation possible).
  - belief_ppo / oracle_ppo: clean per-regime preferences — r0 likes
    skewed quotes (favor_X), r1 / r2 like sym (per E_final's fill
    structure: r0 has high p_wide, r1 / r2 have high p_tight).
  - rl2_hypernet / varibad_hypernet: should resemble belief / oracle.
  - rl2_concat / varibad_concat: should be partially blurred (concat
    sits below floor → its action distribution likely fails to
    differentiate by regime).

Outputs:
  - results/milestones/M5/stats_M5_action_distributions.json — for each
    method, per-regime action-probability vectors aggregated over all
    seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from evaluation.action_distribution import (
    collect_action_regime_rollouts, compute_action_given_regime,
)
from evaluation.posterior_probe import load_experiment
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

METHODS: list[tuple[str, str]] = [
    ("regime_agnostic_ppo", "m3_regime_agnostic"),
    ("belief_ppo",          "m3_belief"),
    ("oracle_ppo",          "m3_oracle"),
    ("rl2_concat",          "m5_step4_rl2_concat"),
    ("rl2_hypernet",        "m5_step4_rl2_hypernet"),
    ("varibad_concat",      "m5_step4_varibad_concat"),
    ("varibad_hypernet",    "m5_step4_varibad_hypernet"),
]

ACTION_NAMES = ("sym", "favor_ask", "favor_bid")
N_ACTIONS = 3
N_REGIMES = 3


def _probe_one_method(
    label: str, experiment_name: str, n_rollouts: int, rollout_length: int,
) -> dict[str, Any]:
    exp_dir = RESULTS_ROOT / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"missing experiment dir: {exp_dir}")
    checkpoints = sorted(exp_dir.glob("checkpoint_seed_*.pkl"))
    seeds = [int(p.stem.split("_")[-1]) for p in checkpoints]
    if not seeds:
        raise FileNotFoundError(f"no checkpoints in {exp_dir}")

    per_seed_dist: list[np.ndarray] = []
    per_seed_returns: list[float] = []
    for seed in seeds:
        bundle = load_experiment(exp_dir, seed)
        key = jax.random.PRNGKey(seed)
        data = collect_action_regime_rollouts(
            bundle.env, bundle.agent, bundle.agent_state,
            n_rollouts=n_rollouts, rollout_length=rollout_length, key=key,
        )
        dist = compute_action_given_regime(
            data["action"], data["regime"], N_ACTIONS, N_REGIMES,
        )
        per_seed_dist.append(dist)
        per_seed_returns.append(float(data["reward"].sum(axis=0).mean()))
    arr = np.stack(per_seed_dist)  # [n_seeds, n_regimes, n_actions]
    return {
        "experiment_name": experiment_name,
        "seeds": seeds,
        "per_seed_action_given_regime": arr.tolist(),  # [seeds, R, A]
        "mean_action_given_regime": arr.mean(axis=0).tolist(),  # [R, A]
        "std_action_given_regime": arr.std(axis=0, ddof=1).tolist() if len(seeds) > 1 else
            np.zeros_like(arr[0]).tolist(),
        "per_seed_episode_return": per_seed_returns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5_action_distributions")
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=128)
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / "milestones" / "M5"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_M5_action_distributions.json"
    summary_path = out_dir / "stats_M5_action_distributions_run.json"

    run = ScriptRun(script="m5_action_distributions")

    t0 = time.perf_counter()
    by_method: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for label, experiment_name in METHODS:
        try:
            print(f"[m5_action] probing {experiment_name} ({label})", flush=True)
            result = _probe_one_method(
                label, experiment_name,
                n_rollouts=args.n_rollouts, rollout_length=args.rollout_length,
            )
            by_method[label] = result
            mean = np.asarray(result["mean_action_given_regime"])
            print(
                f"[m5_action] {label:>22s} | "
                f"r0={[f'{x:.2f}' for x in mean[0]]}  "
                f"r1={[f'{x:.2f}' for x in mean[1]]}  "
                f"r2={[f'{x:.2f}' for x in mean[2]]}",
                flush=True,
            )
        except Exception as e:
            print(f"[m5_action] {experiment_name} FAILED: {e}", flush=True)
            failed.append(experiment_name)

    elapsed = (time.perf_counter() - t0) / 60
    if failed:
        run.fail(
            reason=f"{len(failed)} methods failed: " + ", ".join(failed),
            summary_path=summary_path,
        )
        return 1

    payload = {
        "env": "market_making_v1 (E_final)",
        "n_rollouts": args.n_rollouts,
        "rollout_length": args.rollout_length,
        "action_names": list(ACTION_NAMES),
        "n_regimes": N_REGIMES,
        "by_method": by_method,
    }
    with open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    run.add_output(str(stats_path))
    run.ok(
        key_stats={
            "elapsed_min": round(elapsed, 2),
            "n_methods": len(METHODS),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
