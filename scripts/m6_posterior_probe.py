"""M6 — posterior-vs-performance probe orchestrator.

Iterates over every meta-RL cell in the M6 sweep and runs the same
linear-probe analysis used for M5 Step 5: load each per-seed checkpoint,
roll out the policy, train a logistic regression on the belief
representation to predict the latent regime, score test accuracy.
Aggregates per-seed (posterior_error, gap_closed) into a scatter that
tests RQ3's posterior-vs-performance relationship across difficulty.

`posterior_error` is defined as `analytical_test_acc - method_test_acc`
on the same rollouts, so the units are "probe-accuracy gap below the
analytical upper bound." A low posterior_error means the method's
belief is nearly as decodable as the analytical HMM posterior.

Reference-level cells (regime-agnostic / belief / oracle PPO) are NOT
probed: regime-agnostic has no regime-conditioned representation, belief
ingests the analytical posterior directly (probe is trivially the
analytical reference), and oracle sees the true regime (probe is
trivially perfect). Only the 4 meta-RL cells × 6 (axis × level) = 24
probe tasks are executed.

Outputs:
  - results/milestones/M6/stats_M6_posterior_vs_performance.json
    (per-seed scatter_points, per-method/overall correlations,
     decoupling diagnostic, signal_pattern flag)

Usage:
  uv run python -m scripts.m6_posterior_probe
  uv run python -m scripts.m6_posterior_probe --n-rollouts 50  # quick

The script reads `stats_M6_sweep.json` for the cell list, so it requires
the sweep aggregation step to have run first.
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

# Only meta-RL cells get probed — see module docstring for why the
# reference levels are skipped.
PROBED_METHODS = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")


def _gap_closed_per_seed(
    per_seed_return: list[float], floor_mean: float, oracle_mean: float,
) -> list[float]:
    if np.isnan(floor_mean) or np.isnan(oracle_mean) or oracle_mean == floor_mean:
        return [float("nan")] * len(per_seed_return)
    arr = np.asarray(per_seed_return, dtype=float)
    return ((arr - floor_mean) / (oracle_mean - floor_mean)).tolist()


def _probe_cell(
    experiment_name: str, n_rollouts: int, rollout_length: int, classifier: str,
) -> dict[str, Any]:
    """Probe one cell — return per-seed `method_test_acc` and `analytical_test_acc`."""
    exp_dir = RESULTS_ROOT / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"experiment dir missing: {exp_dir}")
    checkpoints = sorted(exp_dir.glob("checkpoint_seed_*.pkl"))
    seeds = [int(p.stem.split("_")[-1]) for p in checkpoints]
    if not seeds:
        raise FileNotFoundError(f"no checkpoints in {exp_dir}")

    method_accs: list[float] = []
    analytical_accs: list[float] = []
    for seed in seeds:
        bundle = load_experiment(exp_dir, seed)
        result = probe_one_seed(
            bundle, n_rollouts=n_rollouts, rollout_length=rollout_length,
            classifier=classifier, rng_key=seed,
        )
        method_accs.append(float(result["method"]["test_acc"]))
        analytical_accs.append(float(result["analytical"]["test_acc"]))
    return {
        "seeds": seeds,
        "method_test_acc_per_seed": method_accs,
        "analytical_test_acc_per_seed": analytical_accs,
    }


def _decoupling_diagnostic(
    scatter_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect methods where rank-by-posterior differs sharply from rank-by-return.

    For each (axis, level) where multiple methods are present, rank the
    methods by mean `1 - posterior_error` and by mean `gap_closed`. A
    method is flagged as "decoupled" if the two rank lists put it in the
    top half by one metric and bottom half by the other.
    """
    by_cell: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for p in scatter_points:
        by_cell.setdefault((p["axis"], p["level"]), {}).setdefault(p["method"], []).append(p)

    flagged: dict[str, list[str]] = {}
    for (axis, level), methods in by_cell.items():
        if len(methods) < 2:
            continue
        post_means = {m: float(np.mean([1.0 - p["posterior_error"] for p in pts])) for m, pts in methods.items()}
        gc_means = {m: float(np.mean([p["gap_closed"] for p in pts])) for m, pts in methods.items()}
        post_order = sorted(post_means, key=post_means.get, reverse=True)
        gc_order = sorted(gc_means, key=gc_means.get, reverse=True)
        n = len(methods)
        half = n // 2 if n % 2 == 0 else n // 2 + 1
        for method in methods:
            in_top_post = post_order.index(method) < half
            in_top_gc = gc_order.index(method) < half
            if in_top_post != in_top_gc:
                flagged.setdefault(method, []).append(f"{axis}/{level}")

    return {
        "methods_with_decoupling": sorted(flagged),
        "per_method_cells": flagged,
        "description": (
            "Method is flagged as decoupled if it is top-half by posterior "
            "decodability but bottom-half by gap_closed (or vice-versa) "
            "in some (axis, level) cell."
        ),
    }


def _signal_pattern(
    correlation_overall: float, decoupling: dict[str, Any],
) -> str:
    if abs(correlation_overall) >= 0.5:
        return "tight_correlation"
    if decoupling["methods_with_decoupling"]:
        return "decoupling"
    return "noise"


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m6_posterior_probe")
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument(
        "--classifier", choices=["logistic", "mlp"], default="logistic",
    )
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / "milestones" / "M6"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_M6_posterior_vs_performance.json"
    summary_path = out_dir / "stats_M6_posterior_vs_performance_run.json"
    sweep_stats_path = out_dir / "stats_M6_sweep.json"

    run = ScriptRun(script="m6_posterior_probe")
    if not sweep_stats_path.exists():
        run.fail(
            reason=f"missing {sweep_stats_path}; run scripts.m6_difficulty_sweep first",
            summary_path=summary_path,
        )
        return 1

    with open(sweep_stats_path) as f:
        sweep = json.load(f)

    t_start = time.perf_counter()
    scatter_points: list[dict[str, Any]] = []
    failed: list[str] = []
    n_total = 0
    for axis, by_level in sweep.get("results", {}).items():
        for level, cells in by_level.items():
            floor = cells.get("regime_agnostic_ppo", {}).get("final_return_mean", float("nan"))
            oracle = cells.get("oracle_ppo", {}).get("final_return_mean", float("nan"))
            for method in PROBED_METHODS:
                cell = cells.get(method)
                if cell is None:
                    continue
                n_total += 1
                experiment_name = cell["experiment_name"]
                try:
                    probe = _probe_cell(
                        experiment_name,
                        n_rollouts=args.n_rollouts,
                        rollout_length=args.rollout_length,
                        classifier=args.classifier,
                    )
                except Exception as e:
                    print(f"[probe] {experiment_name} FAILED: {e}", flush=True)
                    failed.append(experiment_name)
                    continue
                # Per-seed scatter points.
                gc_per_seed = _gap_closed_per_seed(
                    cell["per_seed_final_return"], floor, oracle,
                )
                for seed, m_acc, a_acc, gc in zip(
                    probe["seeds"], probe["method_test_acc_per_seed"],
                    probe["analytical_test_acc_per_seed"], gc_per_seed,
                ):
                    scatter_points.append({
                        "method": method,
                        "axis": axis,
                        "level": level,
                        "experiment_name": experiment_name,
                        "seed": seed,
                        "method_test_acc": m_acc,
                        "analytical_test_acc": a_acc,
                        "posterior_error": a_acc - m_acc,
                        "gap_closed": gc,
                    })
                m_mean = float(np.mean(probe["method_test_acc_per_seed"]))
                a_mean = float(np.mean(probe["analytical_test_acc_per_seed"]))
                print(
                    f"[probe] {experiment_name:>50s} | method={m_mean:.3f} "
                    f"analytical={a_mean:.3f} posterior_error={a_mean - m_mean:+.3f}",
                    flush=True,
                )

    if not scatter_points:
        run.fail(
            reason="no scatter points collected (every probe failed)",
            summary_path=summary_path,
        )
        return 1

    # ---- Correlations -----------------------------------------------------
    arr_pe = np.asarray([p["posterior_error"] for p in scatter_points])
    arr_gc = np.asarray([p["gap_closed"] for p in scatter_points])
    valid = ~np.isnan(arr_pe) & ~np.isnan(arr_gc)
    if valid.sum() >= 2 and np.std(arr_pe[valid]) > 0 and np.std(arr_gc[valid]) > 0:
        correlation_overall = float(np.corrcoef(arr_pe[valid], arr_gc[valid])[0, 1])
    else:
        correlation_overall = float("nan")

    by_method: dict[str, list[dict[str, Any]]] = {}
    for p in scatter_points:
        by_method.setdefault(p["method"], []).append(p)
    correlation_per_method: dict[str, float] = {}
    for method, pts in by_method.items():
        pe = np.asarray([p["posterior_error"] for p in pts])
        gc = np.asarray([p["gap_closed"] for p in pts])
        v = ~np.isnan(pe) & ~np.isnan(gc)
        if v.sum() >= 2 and np.std(pe[v]) > 0 and np.std(gc[v]) > 0:
            correlation_per_method[method] = float(np.corrcoef(pe[v], gc[v])[0, 1])
        else:
            correlation_per_method[method] = float("nan")

    decoupling = _decoupling_diagnostic(scatter_points)
    signal_pattern = _signal_pattern(correlation_overall, decoupling)
    scatter_interpretable = signal_pattern in ("tight_correlation", "decoupling")

    stats = {
        "n_rollouts": args.n_rollouts,
        "rollout_length": args.rollout_length,
        "classifier": args.classifier,
        "n_scatter_points": len(scatter_points),
        "n_cells_attempted": n_total,
        "n_cells_failed": len(failed),
        "scatter_points": scatter_points,
        "correlation_overall": correlation_overall,
        "correlation_per_method": correlation_per_method,
        "decoupling_detected": decoupling,
        "signal_pattern": signal_pattern,
        "scatter_interpretable": scatter_interpretable,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_min = (time.perf_counter() - t_start) / 60
    if failed:
        run.fail(
            reason=f"{len(failed)}/{n_total} cells failed: " + ", ".join(failed[:5]),
            summary_path=summary_path,
        )
        return 1
    run.ok(
        key_stats={
            "n_scatter_points": len(scatter_points),
            "correlation_overall": correlation_overall,
            "signal_pattern": signal_pattern,
            "scatter_interpretable": scatter_interpretable,
            "elapsed_min": round(total_min, 2),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )

    print("[probe] === summary ===", flush=True)
    print(
        f"[probe] correlation_overall={correlation_overall:+.3f} "
        f"signal_pattern={signal_pattern} interpretable={scatter_interpretable}",
        flush=True,
    )
    for method, r in correlation_per_method.items():
        print(f"[probe]   {method:>22s} | r={r:+.3f}", flush=True)
    if decoupling["methods_with_decoupling"]:
        print(
            f"[probe] decoupling: {decoupling['methods_with_decoupling']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
