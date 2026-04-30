"""Cartpole — posterior-vs-performance probe orchestrator.

For each of the 4 meta-RL cells × 8 seeds = 32 (cell, seed) points:
  - Load the trained checkpoint
  - Roll out the policy
  - Train logistic regression on the policy's internal belief → regime
    (test_acc on held-out rollouts = method probe accuracy)
  - Train the same classifier on the analytical posterior → regime
    (analytical probe accuracy = upper bound under the same trajectories)
  - Record (posterior_error = analytical_acc − method_acc, gap_closed)

Aggregates into a scatter that tests the same RQ3 question as M5/M6:
*does posterior decoding accuracy predict task return?* Decoupling means
two methods can have similar posterior_error but very different
gap_closed, i.e. the integration mechanism dominates.

Outputs:
  - results/milestones/cartpole/stats_cartpole_posterior_vs_performance.json

Usage:
  uv run python -m scripts.cartpole_posterior_probe
  uv run python -m scripts.cartpole_posterior_probe --n-rollouts 50  # quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.posterior_probe import load_experiment
from evaluation.posterior_probe_cartpole import probe_one_seed_cartpole
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

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
        result = probe_one_seed_cartpole(
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


def _decoupling_diagnostic(scatter_points: list[dict[str, Any]]) -> dict[str, Any]:
    """Same heuristic as M6: a method is decoupled if it lands in the top
    half by posterior decodability but bottom half by gap_closed (or
    vice versa) — i.e. its rank by the two metrics disagree.
    """
    by_method: dict[str, list[dict[str, Any]]] = {}
    for p in scatter_points:
        by_method.setdefault(p["method"], []).append(p)
    if len(by_method) < 2:
        return {"methods_with_decoupling": [], "per_method_cells": {}}
    post_means = {m: float(np.mean([1.0 - p["posterior_error"] for p in pts])) for m, pts in by_method.items()}
    gc_means = {m: float(np.mean([p["gap_closed"] for p in pts])) for m, pts in by_method.items()}
    post_order = sorted(post_means, key=post_means.get, reverse=True)
    gc_order = sorted(gc_means, key=gc_means.get, reverse=True)
    n = len(by_method)
    half = n // 2 if n % 2 == 0 else n // 2 + 1
    flagged: dict[str, list[str]] = {}
    for method in by_method:
        in_top_post = post_order.index(method) < half
        in_top_gc = gc_order.index(method) < half
        if in_top_post != in_top_gc:
            flagged.setdefault(method, []).append("medium")
    return {
        "methods_with_decoupling": sorted(flagged),
        "per_method_cells": flagged,
        "post_means": post_means,
        "gc_means": gc_means,
    }


def _signal_pattern(
    correlation_overall: float, decoupling: dict[str, Any],
) -> str:
    if abs(correlation_overall) >= 0.5:
        return "tight_correlation"
    if decoupling["methods_with_decoupling"]:
        return "decoupling"
    return "noise"


def _load_per_seed_returns(experiment_name: str) -> list[float]:
    p = RESULTS_ROOT / experiment_name / "metrics.json"
    with open(p) as f:
        m = json.load(f)
    return list(map(float, m["per_seed_final_return"]))


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.cartpole_posterior_probe")
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument(
        "--classifier", choices=["logistic", "mlp"], default="logistic",
    )
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / "milestones" / "cartpole"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_cartpole_posterior_vs_performance.json"
    summary_path = out_dir / "stats_cartpole_posterior_vs_performance_run.json"

    run = ScriptRun(script="cartpole_posterior_probe")

    # Reference returns for gap_closed normalisation.
    floor_returns = _load_per_seed_returns("m_cartpole_regime_agnostic")
    oracle_returns = _load_per_seed_returns("m_cartpole_oracle")
    floor_mean = float(np.mean(floor_returns))
    oracle_mean = float(np.mean(oracle_returns))

    t_start = time.perf_counter()
    scatter_points: list[dict[str, Any]] = []
    failed: list[str] = []
    for method in PROBED_METHODS:
        experiment_name = f"m_cartpole_{method}"
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

        per_seed_return = _load_per_seed_returns(experiment_name)
        gc_per_seed = _gap_closed_per_seed(per_seed_return, floor_mean, oracle_mean)
        for seed, m_acc, a_acc, gc in zip(
            probe["seeds"], probe["method_test_acc_per_seed"],
            probe["analytical_test_acc_per_seed"], gc_per_seed,
        ):
            scatter_points.append({
                "method": method,
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
            f"[probe] {experiment_name:>40s} | method={m_mean:.3f} "
            f"analytical={a_mean:.3f} posterior_error={a_mean - m_mean:+.3f}",
            flush=True,
        )

    if not scatter_points:
        run.fail(reason="no scatter points collected", summary_path=summary_path)
        return 1

    # ---- correlations ---------------------------------------------------
    arr_pe = np.asarray([p["posterior_error"] for p in scatter_points])
    arr_gc = np.asarray([p["gap_closed"] for p in scatter_points])
    valid = ~np.isnan(arr_pe) & ~np.isnan(arr_gc)
    if valid.sum() >= 2 and np.std(arr_pe[valid]) > 0 and np.std(arr_gc[valid]) > 0:
        correlation_overall = float(np.corrcoef(arr_pe[valid], arr_gc[valid])[0, 1])
    else:
        correlation_overall = float("nan")

    # Bootstrap CI on overall Pearson r
    rng = np.random.default_rng(0)
    pe_v = arr_pe[valid]
    gc_v = arr_gc[valid]
    if pe_v.size >= 2:
        boot_rs = []
        for _ in range(10_000):
            idx = rng.integers(0, pe_v.size, size=pe_v.size)
            x, y = pe_v[idx], gc_v[idx]
            if np.std(x) > 0 and np.std(y) > 0:
                boot_rs.append(np.corrcoef(x, y)[0, 1])
        boot_rs = np.asarray(boot_rs)
        ci_low = float(np.percentile(boot_rs, 2.5))
        ci_high = float(np.percentile(boot_rs, 97.5))
    else:
        ci_low = ci_high = float("nan")

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
    decoupling_supported = abs(correlation_overall) < 0.30

    stats = {
        "env": "cartpole_regime_v1",
        "n_rollouts": args.n_rollouts,
        "rollout_length": args.rollout_length,
        "classifier": args.classifier,
        "n_scatter_points": len(scatter_points),
        "floor_mean": floor_mean,
        "oracle_mean": oracle_mean,
        "scatter_points": scatter_points,
        "correlation_overall": correlation_overall,
        "correlation_overall_ci95": [ci_low, ci_high],
        "correlation_per_method": correlation_per_method,
        "decoupling_detected": decoupling,
        "signal_pattern": signal_pattern,
        "scatter_interpretable": scatter_interpretable,
        "decoupling_supported": decoupling_supported,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_min = (time.perf_counter() - t_start) / 60
    if failed:
        run.fail(
            reason=f"{len(failed)} cells failed: " + ", ".join(failed),
            summary_path=summary_path,
        )
        return 1
    run.ok(
        key_stats={
            "n_scatter_points": len(scatter_points),
            "correlation_overall": correlation_overall,
            "correlation_overall_ci95": [ci_low, ci_high],
            "signal_pattern": signal_pattern,
            "scatter_interpretable": scatter_interpretable,
            "decoupling_supported": decoupling_supported,
            "elapsed_min": round(total_min, 2),
        },
        summary_path=summary_path,
    )

    print("[probe] === summary ===", flush=True)
    print(
        f"[probe] correlation_overall={correlation_overall:+.3f} "
        f"CI=[{ci_low:+.3f}, {ci_high:+.3f}] | signal={signal_pattern} | "
        f"decoupling_supported={decoupling_supported}",
        flush=True,
    )
    for method, r in correlation_per_method.items():
        print(f"[probe]   {method:>22s} | r={r:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
