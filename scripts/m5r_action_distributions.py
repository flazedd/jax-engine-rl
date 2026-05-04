"""M5R — per-(method, regime) action-distribution analysis on E_final.

Re-runs the M5 action-distribution probe on the four matched-tuning meta-RL
cells, plus the three references reused from M3. Writes per-method
P(action | true_regime) tables that supersede the M5 numbers in the thesis.

Outputs:
  - results/M5R/final/m5r_action_distributions.json
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
    ("rl2_concat",          "m5r_final_rl2_concat_e_final"),
    ("rl2_hypernet",        "m5r_final_rl2_hypernet_e_final"),
    ("varibad_concat",      "m5r_final_varibad_concat_e_final"),
    ("varibad_hypernet",    "m5r_final_varibad_hypernet_e_final"),
]

N_ACTIONS = 3
N_REGIMES = 3


def _probe_one(experiment_name: str, n_rollouts: int, rollout_length: int) -> dict[str, Any]:
    exp_dir = RESULTS_ROOT / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"missing experiment dir: {exp_dir}")
    checkpoints = sorted(exp_dir.glob("checkpoint_seed_*.pkl"))
    seeds = [int(p.stem.split("_")[-1]) for p in checkpoints]
    if not seeds:
        raise FileNotFoundError(f"no checkpoints in {exp_dir}")
    per_seed_dist = []
    per_seed_returns = []
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
    arr = np.stack(per_seed_dist)
    return {
        "experiment_name": experiment_name,
        "seeds": seeds,
        "per_seed_action_given_regime": arr.tolist(),
        "mean_action_given_regime": arr.mean(axis=0).tolist(),
        "std_action_given_regime": arr.std(axis=0, ddof=1).tolist() if len(seeds) > 1
            else np.zeros_like(arr[0]).tolist(),
        "per_seed_episode_return": per_seed_returns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_action_distributions")
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=128)
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / "M5R" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "m5r_action_distributions.json"
    summary_path = out_dir / "m5r_action_distributions_run.json"

    run = ScriptRun(script="m5r_action_distributions")
    t0 = time.perf_counter()
    by_method: dict[str, Any] = {}
    failed = []
    for label, experiment_name in METHODS:
        try:
            print(f"[m5r_action] probing {experiment_name} ({label})", flush=True)
            result = _probe_one(experiment_name, args.n_rollouts, args.rollout_length)
            by_method[label] = result
            mean = np.asarray(result["mean_action_given_regime"])
            print(
                f"[m5r_action] {label:>22s} | "
                f"sym fractions r0={mean[0,0]:.2f}  r1={mean[1,0]:.2f}  r2={mean[2,0]:.2f}",
                flush=True,
            )
        except Exception as e:
            print(f"[m5r_action] {experiment_name} FAILED: {e}", flush=True)
            failed.append(experiment_name)

    elapsed = (time.perf_counter() - t0) / 60
    if failed:
        run.fail(
            reason=f"{len(failed)} methods failed: " + ", ".join(failed),
            summary_path=summary_path,
        )
        return 1
    payload = {
        "env": "market_making_v1 (E_final), matched-tuning",
        "n_rollouts": args.n_rollouts,
        "rollout_length": args.rollout_length,
        "action_names": ["sym", "favor_ask", "favor_bid"],
        "n_regimes": N_REGIMES,
        "by_method": by_method,
    }
    with open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    run.add_output(str(stats_path))
    run.ok(
        key_stats={"elapsed_min": round(elapsed, 2), "n_methods": len(METHODS)},
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
