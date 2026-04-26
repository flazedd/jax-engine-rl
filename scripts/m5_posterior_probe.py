"""M5 Step-5 — Posterior-quality probe orchestrator.

For each (experiment_name, seed) pair, load the saved checkpoint, run
evaluation rollouts, train a regime classifier on the agent's belief
representation, and aggregate per-seed probe accuracies into a stats
JSON. Also probes the analytical posterior on the same rollouts as an
upper-bound reference.

Pass criterion: a thesis-grade probe has p_seed and 95% CI for each
method's logistic-probe test accuracy and per-timestep accuracy curve.

Spec lives at `results/milestones/M5/STEP5_PROBE_DESIGN.md`.

Usage:
  uv run python -m scripts.m5_posterior_probe \\
    --experiment m5_step3_rl2_hypernet \\
    --experiment m5_step3_varibad_hypernet \\
    [--n-rollouts 200] [--rollout-length 128] [--classifier logistic|mlp]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.posterior_probe import load_experiment, probe_one_seed
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


def _bootstrap_ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _probe_one_method(
    experiment_name: str,
    n_rollouts: int,
    rollout_length: int,
    classifier: str,
) -> dict[str, Any]:
    exp_dir = RESULTS_ROOT / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"experiment dir missing: {exp_dir}")

    # Find all checkpoint_seed_*.pkl files.
    checkpoints = sorted(exp_dir.glob("checkpoint_seed_*.pkl"))
    seeds = [int(p.stem.split("_")[-1]) for p in checkpoints]
    if not seeds:
        raise FileNotFoundError(
            f"no checkpoint_seed_*.pkl found in {exp_dir}; "
            "did training save checkpoints?"
        )

    print(
        f"[probe] {experiment_name}: {len(seeds)} seeds, n_rollouts={n_rollouts}, "
        f"T={rollout_length}, classifier={classifier}",
        flush=True,
    )
    per_seed: list[dict[str, Any]] = []
    t_method = time.perf_counter()
    for s_idx, seed in enumerate(seeds):
        t0 = time.perf_counter()
        bundle = load_experiment(exp_dir, seed)
        result = probe_one_seed(
            bundle,
            n_rollouts=n_rollouts,
            rollout_length=rollout_length,
            classifier=classifier,
            rng_key=seed,
        )
        result["seed"] = seed
        per_seed.append(result)
        elapsed = time.perf_counter() - t0
        print(
            f"[probe]   seed {seed}: method_test_acc={result['method']['test_acc']:.3f} "
            f"analytical_test_acc={result['analytical']['test_acc']:.3f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    # Aggregate across seeds.
    method_test_accs = [r["method"]["test_acc"] for r in per_seed]
    analytical_test_accs = [r["analytical"]["test_acc"] for r in per_seed]
    method_per_t = np.array([r["method"]["per_t_test_acc"] for r in per_seed])  # [n_seeds, T]
    analytical_per_t = np.array([r["analytical"]["per_t_test_acc"] for r in per_seed])

    method_ci = _bootstrap_ci(method_test_accs)
    analytical_ci = _bootstrap_ci(analytical_test_accs)

    elapsed_method = time.perf_counter() - t_method
    print(
        f"[probe] {experiment_name}: mean method test_acc={np.mean(method_test_accs):.3f} "
        f"({method_ci[0]:.3f}-{method_ci[1]:.3f}) | "
        f"analytical={np.mean(analytical_test_accs):.3f} | "
        f"elapsed={elapsed_method/60:.1f} min",
        flush=True,
    )

    # Read env name from the first checkpoint's config (consistent across
    # seeds for a given experiment).
    sample_bundle = load_experiment(exp_dir, seeds[0])
    env_name = sample_bundle.config.get("env", {}).get("name", "?")
    return {
        "experiment_name": experiment_name,
        "env_name": env_name,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "method_test_acc_per_seed": method_test_accs,
        "method_test_acc_mean": float(np.mean(method_test_accs)),
        "method_test_acc_ci95": list(method_ci),
        "method_per_t_test_acc_mean": method_per_t.mean(axis=0).tolist(),
        "method_per_t_test_acc_per_seed": method_per_t.tolist(),
        "analytical_test_acc_per_seed": analytical_test_accs,
        "analytical_test_acc_mean": float(np.mean(analytical_test_accs)),
        "analytical_test_acc_ci95": list(analytical_ci),
        "analytical_per_t_test_acc_mean": analytical_per_t.mean(axis=0).tolist(),
        "belief_dim": per_seed[0]["belief_dim"],
        "n_classes_observed": per_seed[0]["n_regimes_observed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5_posterior_probe")
    parser.add_argument(
        "--experiment", action="append", required=True,
        help="experiment_name to probe (repeatable)",
    )
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument(
        "--classifier", choices=["logistic", "mlp"], default="logistic",
    )
    parser.add_argument(
        "--output", default=None,
        help="output JSON path (default: results/milestones/M5/stats_M5_posterior_probe.json)",
    )
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / "milestones" / "M5"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = (
        Path(args.output) if args.output
        else out_dir / "stats_M5_posterior_probe.json"
    )
    summary_path = out_dir / "stats_M5_posterior_probe_run.json"

    run = ScriptRun(script="m5_posterior_probe")

    t_start = time.perf_counter()
    by_method: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for exp_name in args.experiment:
        try:
            by_method[exp_name] = _probe_one_method(
                exp_name,
                n_rollouts=args.n_rollouts,
                rollout_length=args.rollout_length,
                classifier=args.classifier,
            )
        except Exception as e:
            print(f"[probe] {exp_name} FAILED: {e}", flush=True)
            failed.append(exp_name)

    if failed:
        run.fail(
            reason=f"{len(failed)}/{len(args.experiment)} experiments failed: " + ", ".join(failed),
            summary_path=summary_path,
        )
        return 1

    # Probe currently assumes MarketMakingV1 (the only env with a `regime`
    # field exposed via info dict). Map env class names to canonical
    # project labels (CLAUDE.md → "E_final = e6e_symmetric_kappa05").
    # The legacy `mm_reduced*` keys are kept for backward compatibility
    # with checkpoints saved before the env was renamed.
    env_names = {by_method[k].get("env_name", "?") for k in by_method}
    _DISPLAY = {
        "market_making_v1":                  "MarketMakingV1",
        "market_making_v1_oracle":           "MarketMakingV1 (oracle obs)",
        "market_making_v1_belief":           "MarketMakingV1 (analytical-belief obs)",
        "market_making_v1_belief_constant":  "MarketMakingV1 (constant-belief obs)",
        "market_making_v1_stacked":          "MarketMakingV1 (stacked obs)",
        # legacy names from pre-rename checkpoints
        "mm_reduced":                  "MarketMakingV1",
        "mm_reduced_oracle":           "MarketMakingV1 (oracle obs)",
        "mm_reduced_belief":           "MarketMakingV1 (analytical-belief obs)",
        "mm_reduced_belief_constant":  "MarketMakingV1 (constant-belief obs)",
        "mm_reduced_stacked":          "MarketMakingV1 (stacked obs)",
    }
    env_label = (
        _DISPLAY.get(next(iter(env_names)), next(iter(env_names)))
        if len(env_names) == 1 else "MM E_final"
    )
    stats = {
        "n_rollouts": args.n_rollouts,
        "rollout_length": args.rollout_length,
        "classifier": args.classifier,
        "env_label": env_label,
        "methods": by_method,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_min = (time.perf_counter() - t_start) / 60
    run.ok(
        key_stats={
            "n_methods": len(by_method),
            "elapsed_min": round(total_min, 2),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )

    print("[probe] === summary ===", flush=True)
    for name, m in by_method.items():
        print(
            f"[probe] {name:>40s} | method_test_acc={m['method_test_acc_mean']:.3f} "
            f"({m['method_test_acc_ci95'][0]:.3f}-{m['method_test_acc_ci95'][1]:.3f}) | "
            f"analytical={m['analytical_test_acc_mean']:.3f} | "
            f"belief_dim={m['belief_dim']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
