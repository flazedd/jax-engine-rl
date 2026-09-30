"""Evaluate the frozen final RSMM policies on fresh episodes.

This script does not train or update any policy.  It loads every final
checkpoint from the matched medium-difficulty experiment, runs fresh episodes
from the same RSMM distribution, and writes one mean evaluation return per
trained run.  Those run-level means are the observations used for the final
performance figure and tests.

Usage:
    uv run python -m scripts.m5r_post_training_evaluation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import numpy as np

from evaluation.action_distribution import collect_action_regime_rollouts
from evaluation.posterior_probe import load_experiment
from utils.paths import analysis_dir, experiment_dir
from utils.script_output import ScriptRun


METHODS: list[tuple[str, str]] = [
    ("regime_agnostic_ppo", "m5r_ref_regime_agnostic_e9"),
    ("belief_ppo", "m5r_ref_belief_e9"),
    ("oracle_ppo", "m5r_ref_oracle_e9"),
    ("stacked_obs_ppo", "m5r_ref_stacked_obs_e9"),
    ("rl2_concat", "m5r_final_rl2_concat_e9"),
    ("rl2_hypernet", "m5r_final_rl2_hypernet_e9"),
    ("varibad_concat", "m5r_final_varibad_concat_e9"),
    ("varibad_hypernet", "m5r_final_varibad_hypernet_e9"),
]
N_EPISODES = 512
EPISODE_LENGTH = 128
N_BOOT = 10_000


def _ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    indices = rng.integers(0, values.size, size=(N_BOOT, values.size))
    means = values[indices].mean(axis=1)
    return tuple(map(float, np.percentile(means, [2.5, 97.5])))


def _evaluate_method(method_index: int, name: str, experiment: str) -> dict:
    directory = experiment_dir(experiment)
    checkpoints = sorted(directory.glob("checkpoint_seed_*.pkl"))
    seeds = [int(path.stem.rsplit("_", 1)[1]) for path in checkpoints]
    if len(seeds) != 20:
        raise RuntimeError(f"{experiment}: expected 20 checkpoints, found {len(seeds)}")

    returns: list[float] = []
    for seed in seeds:
        bundle = load_experiment(directory, seed)
        # This key is reserved for post-training evaluation.  It is distinct
        # from the training and diagnostic keys and no gradients are computed.
        key = jax.random.PRNGKey(1_000_000 + 10_000 * method_index + seed)
        rollout = collect_action_regime_rollouts(
            bundle.env,
            bundle.agent,
            bundle.agent_state,
            n_rollouts=N_EPISODES,
            rollout_length=EPISODE_LENGTH,
            key=key,
        )
        episode_returns = np.asarray(rollout["reward"], dtype=float).sum(axis=0)
        returns.append(float(episode_returns.mean()))

    values = np.asarray(returns)
    lo, hi = _ci(values)
    print(f"[post_training_eval] {name:22s} mean={values.mean():.3f} "
          f"CI=[{lo:.3f}, {hi:.3f}]", flush=True)
    return {
        "experiment": experiment,
        "n_trained_runs": len(seeds),
        "seeds": seeds,
        "evaluation_episodes_per_run": N_EPISODES,
        "episode_length": EPISODE_LENGTH,
        "per_seed_evaluation_return": returns,
        "evaluation_return_mean": float(values.mean()),
        "evaluation_return_ci95": [lo, hi],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=[name for name, _ in METHODS])
    parser.add_argument("--combine", action="store_true",
                        help="combine previously written per-method results without reevaluating")
    args = parser.parse_args()
    if args.combine and args.method:
        parser.error("--combine cannot be used with --method")
    output_dir = analysis_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.method}" if args.method else ""
    output_path = output_dir / f"m5r_post_training_evaluation{suffix}.json"
    summary_path = output_dir / f"m5r_post_training_evaluation{suffix}_run.json"
    run = ScriptRun(script="m5r_post_training_evaluation", run_mode="full")
    if args.combine:
        methods = {}
        for name, _experiment in METHODS:
            part_path = output_dir / f"m5r_post_training_evaluation_{name}.json"
            if not part_path.exists():
                run.fail(reason=f"missing {part_path}", summary_path=summary_path)
                return 1
            with open(part_path) as handle:
                methods.update(json.load(handle)["methods"])
        payload = {
            "description": "Frozen final policies evaluated on fresh RSMM episodes; no PPO updates.",
            "methods": methods,
        }
        with open(output_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        run.add_output(output_path)
        run.ok(key_stats={"methods": len(methods), "episodes_per_run": N_EPISODES},
               summary_path=summary_path)
        return 0

    selected = [(name, experiment) for name, experiment in METHODS if args.method in (None, name)]
    try:
        methods = {
            name: _evaluate_method(METHODS.index((name, experiment)), name, experiment)
            for name, experiment in selected
        }
    except Exception as error:
        run.fail(reason=str(error), summary_path=summary_path)
        return 1

    payload = {
        "description": "Frozen final policies evaluated on fresh RSMM episodes; no PPO updates.",
        "methods": methods,
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    run.add_output(output_path)
    run.ok(
        key_stats={"methods": len(methods), "episodes_per_run": N_EPISODES},
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
