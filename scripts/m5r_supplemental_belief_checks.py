"""Exploratory direct-posterior and full VariBAD-input checks on saved policies.

This does not replace the canonical 500-rollout probe or alter its test families.
The same held-out episodes are used for the mean-only and full-input readouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from evaluation.posterior_probe import collect_probe_rollouts, load_experiment, train_probe
from utils.paths import analysis_dir, experiment_dir


METHODS = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")


def run_one(method: str, seed: int, n_rollouts: int) -> dict:
    bundle = load_experiment(experiment_dir(f"m5r_final_{method}_e9"), seed)
    vari = method.startswith("varibad")
    data = collect_probe_rollouts(
        bundle.env, bundle.agent, bundle.agent_state,
        n_rollouts=n_rollouts, rollout_length=128, key=jax.random.PRNGKey(seed),
        extra_keys=("log_var",) if vari else (),
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rollouts)
    n_test = max(1, round(n_rollouts * 0.2))
    test, train = perm[:n_test], perm[n_test:]
    omega = data["analytical_belief"]
    true = data["regime"]
    direct = np.argmax(omega[:, test], axis=-1) == true[:, test]
    row = {
        "method": method,
        "seed": seed,
        "n_train_episodes": len(train),
        "n_test_episodes": len(test),
        "direct_posterior_accuracy": float(direct.mean()),
        "direct_posterior_accuracy_by_step": direct.mean(axis=1).tolist(),
    }
    if vari:
        sigma = np.exp(0.5 * data["log_var"])
        full = np.concatenate((data["belief"], sigma), axis=-1)
        for key, beliefs in (("posterior_mean", data["belief"]), ("full_policy_input", full)):
            result = train_probe(
                beliefs, true, train, test, classifier="logistic", seed=seed,
                omega_TND=omega,
            )
            row[key] = {
                "accuracy": result["test_acc"],
                "kl_to_analytical": result["test_kl_to_omega"],
                "log_loss": result["test_log_loss"],
                "brier": result["test_brier"],
            }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rollouts", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path,
                        default=analysis_dir() / "m5r_supplemental_belief_checks.json")
    args = parser.parse_args()
    if args.n_rollouts < 10 or args.seeds < 1 or args.seeds > 20:
        parser.error("use at least 10 rollouts and between 1 and 20 saved seeds")
    rows = []
    for method in METHODS:
        for seed in range(args.seeds):
            row = run_one(method, seed, args.n_rollouts)
            rows.append(row)
            print(method, seed, "posterior", round(row["direct_posterior_accuracy"], 4), flush=True)
    result = {
        "status": "exploratory post-review diagnostic; not a planned comparison",
        "rollout_length": 128,
        "n_rollouts_per_seed": args.n_rollouts,
        "split": "whole episodes; 20 percent test; random permutation seeded by checkpoint seed",
        "posterior_timing": "before the action, after evidence through the previous step",
        "full_policy_input": "VariBAD posterior mean concatenated with exp(0.5 * clipped log variance)",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
