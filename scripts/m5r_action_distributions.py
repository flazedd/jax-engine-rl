"""M5R — per-(method, regime) action-distribution analysis on E_final.

Runs the action-distribution probe on the four matched meta-RL variants and
on the references, and writes per-method P(action | true_regime) tables.

Every method read here comes from the matched-fairness family, so the
reference rows and the variant rows sit on the same per-step tuple, optimiser
settings, budget and capacity. The pre-matched `m3_*` runs must not be
substituted: they train on the unaugmented observation and would put a
different input on the reference rows of the same table.

Rollouts are collected with the env's regime locked, one batch per regime,
so the tables measure what a policy does while a regime holds rather than
mixing in the post-switch steps where the belief still trails the regime.

Outputs:
  - results/M5R/final/m5r_action_distributions.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from utils.paths import analysis_dir, experiment_dir
from typing import Any

import jax
import numpy as np

from evaluation.action_distribution import (
    collect_action_regime_rollouts,
    compute_action_given_regime,
    compute_action_given_regime_inventory,
)
from evaluation.posterior_probe import load_experiment
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

METHODS: list[tuple[str, str]] = [
    ("regime_agnostic_ppo", "m5r_ref_regime_agnostic_e9"),
    ("belief_ppo",          "m5r_ref_belief_e9"),
    ("oracle_ppo",          "m5r_ref_oracle_e9"),
    ("stacked_obs_ppo",     "m5r_ref_stacked_obs_e9"),
    ("rl2_concat",          "m5r_final_rl2_concat_e9"),
    ("rl2_hypernet",        "m5r_final_rl2_hypernet_e9"),
    ("varibad_concat",      "m5r_final_varibad_concat_e9"),
    ("varibad_hypernet",    "m5r_final_varibad_hypernet_e9"),
]

N_ACTIONS = 3
N_REGIMES = 3
INV_MAX = 5  # MarketMakingV1 default; inventory ∈ [-INV_MAX, +INV_MAX]


def _probe_one(experiment_name: str, n_rollouts: int, rollout_length: int) -> dict[str, Any]:
    exp_dir = experiment_dir(experiment_name)
    if not exp_dir.exists():
        raise FileNotFoundError(f"missing experiment dir: {exp_dir}")
    checkpoints = sorted(exp_dir.glob("checkpoint_seed_*.pkl"))
    seeds = [int(p.stem.split("_")[-1]) for p in checkpoints]
    if not seeds:
        raise FileNotFoundError(f"no checkpoints in {exp_dir}")
    per_seed_dist = []
    per_seed_dist_inv = []
    per_seed_counts_inv = []
    per_seed_returns = []
    for seed in seeds:
        # One rollout batch per regime, with the env's regime locked for the
        # whole episode. The diagnostic asks what a policy does *in* a regime;
        # under free switching the steps just after a switch are attributed to
        # the new regime while the belief still reflects the old one, which
        # attenuates the measurement for any method that infers slowly.
        dist = np.zeros((N_REGIMES, N_ACTIONS), dtype=np.float64)
        dist_inv = np.zeros((N_REGIMES, 2 * INV_MAX + 1, N_ACTIONS), dtype=np.float64)
        counts_inv = np.zeros((N_REGIMES, 2 * INV_MAX + 1), dtype=np.int64)
        returns = []
        for regime in range(N_REGIMES):
            bundle = load_experiment(
                exp_dir, seed, env_overrides={"lock_regime": regime},
            )
            key = jax.random.PRNGKey(seed * N_REGIMES + regime)
            data = collect_action_regime_rollouts(
                bundle.env, bundle.agent, bundle.agent_state,
                n_rollouts=n_rollouts, rollout_length=rollout_length, key=key,
            )
            observed = np.asarray(data["regime"])
            if not (observed == regime).all():
                raise RuntimeError(
                    f"lock_regime={regime} did not hold for {experiment_name} "
                    f"seed {seed}: saw regimes {sorted(set(observed.reshape(-1).tolist()))}"
                )
            dist[regime] = compute_action_given_regime(
                data["action"], observed, N_ACTIONS, N_REGIMES,
            )[regime]
            if "inventory" in data:
                inv = np.asarray(data["inventory"])
                dist_inv[regime] = compute_action_given_regime_inventory(
                    data["action"], observed, inv,
                    N_ACTIONS, N_REGIMES, INV_MAX,
                )[regime]
                counts_inv[regime] = np.bincount(
                    (inv.reshape(-1) + INV_MAX), minlength=2 * INV_MAX + 1,
                )
            returns.append(float(data["reward"].sum(axis=0).mean()))
        per_seed_dist.append(dist)
        per_seed_dist_inv.append(dist_inv)
        per_seed_counts_inv.append(counts_inv)
        per_seed_returns.append(float(np.mean(returns)))
    arr = np.stack(per_seed_dist)
    out = {
        "experiment_name": experiment_name,
        "seeds": seeds,
        "per_seed_action_given_regime": arr.tolist(),
        "mean_action_given_regime": arr.mean(axis=0).tolist(),
        "std_action_given_regime": arr.std(axis=0, ddof=1).tolist() if len(seeds) > 1
            else np.zeros_like(arr[0]).tolist(),
        "per_seed_episode_return": per_seed_returns,
    }
    if per_seed_dist_inv:
        arr_inv = np.stack(per_seed_dist_inv)
        out["per_seed_action_given_regime_inventory"] = arr_inv.tolist()
        out["mean_action_given_regime_inventory"] = arr_inv.mean(axis=0).tolist()
        out["std_action_given_regime_inventory"] = (
            arr_inv.std(axis=0, ddof=1).tolist() if len(seeds) > 1
            else np.zeros_like(arr_inv[0]).tolist()
        )
        # Per-(regime, inventory) visitation, so downstream summaries can weight
        # inventory levels by how often the policy actually occupies them
        # instead of treating every level alike.
        counts = np.stack(per_seed_counts_inv)
        out["per_seed_inventory_counts"] = counts.tolist()
        out["inventory_counts"] = counts.sum(axis=0).tolist()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_action_distributions")
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=128)
    args = parser.parse_args()

    out_dir = analysis_dir()
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
        "regime_locked": True,
        "n_rollouts": args.n_rollouts,
        "rollout_length": args.rollout_length,
        "action_names": ["sym", "favor_ask", "favor_bid"],
        "n_regimes": N_REGIMES,
        "inv_max": INV_MAX,
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
