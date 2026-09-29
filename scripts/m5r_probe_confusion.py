"""Recompute the existing linear probes and retain held-out regime confusion counts.

Uses the same 500 rollouts, 400/100 trajectory split, seed keys, and frozen
checkpoints as ``scripts.m5r_posterior_probe``. The archived probe summary
contains aggregate accuracy but not per-class predictions, so those counts
must be recovered by repeating the probe evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.posterior_probe import load_experiment, probe_one_seed
from utils.paths import analysis_dir, experiment_dir


METHODS = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")


def main() -> None:
    prior = json.loads((analysis_dir() / "m5r_posterior_vs_performance.json").read_text())
    if prior["n_rollouts"] != 500 or prior["rollout_length"] != 128:
        raise ValueError("Archived probe uses a different sampling protocol")

    by_method = {}
    for method in METHODS:
        exp_dir = experiment_dir(f"m5r_final_{method}_e9")
        seeds = sorted(int(p.stem.rsplit("_", 1)[1])
                       for p in exp_dir.glob("checkpoint_seed_*.pkl"))
        if seeds != list(range(20)):
            raise ValueError(f"{method}: expected seeds 0–19, found {seeds}")
        matrices = []
        accuracies = []
        for seed in seeds:
            bundle = load_experiment(exp_dir, seed)
            result = probe_one_seed(bundle, n_rollouts=500, rollout_length=128,
                                    classifier="logistic", rng_key=seed)
            counts = np.asarray(result["method"]["confusion_counts"], dtype=np.int64)
            if counts.shape != (3, 3) or counts.sum() != 12_800:
                raise ValueError(f"{method} seed {seed}: unexpected test count")
            matrices.append(counts)
            accuracies.append(result["method"]["test_acc"])
            print(f"{method} seed {seed:02d}: accuracy={accuracies[-1]:.4f}", flush=True)

        prior_acc = prior["per_method_per_env"]["e9"][method]["method_test_acc_per_seed"]
        lexical_seed_order = [int(p.stem.rsplit("_", 1)[1])
                              for p in sorted(exp_dir.glob("checkpoint_seed_*.pkl"))]
        prior_by_seed = dict(zip(lexical_seed_order, prior_acc))
        if not np.allclose(accuracies, [prior_by_seed[s] for s in seeds],
                           atol=1e-10, rtol=0):
            raise ValueError(f"{method}: recomputed accuracy differs from archived probe")
        total = np.sum(matrices, axis=0)
        errors = int(total.sum() - np.trace(total))
        confusion_1_2 = int(total[1, 2] + total[2, 1])
        by_method[method] = {
            "seeds": seeds,
            "per_seed_confusion_counts": [m.tolist() for m in matrices],
            "confusion_counts": total.tolist(),
            "row_normalized": (total / total.sum(axis=1, keepdims=True)).tolist(),
            "n_test_steps": int(total.sum()),
            "n_errors": errors,
            "n_regime_1_2_confusions": confusion_1_2,
            "fraction_of_errors_from_regime_1_2_confusion": confusion_1_2 / errors,
            "mean_test_accuracy": float(np.mean(accuracies)),
        }

    output = analysis_dir() / "m5r_probe_confusion.json"
    output.write_text(json.dumps({
        "protocol": "linear probe; 500 rollouts per seed; 400 train, 100 test; 128 steps",
        "by_method": by_method,
    }, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
