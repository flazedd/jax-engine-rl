"""M5R Stage D — posterior-vs-performance probe across the M5R final cells.

Mirrors `scripts/m6_posterior_probe.py` but reads cells from M5R's
`results/M5R/final/per_cell_env.json` instead of the M6 axis-level structure.

For each of the 4 meta-RL cells × 5 evaluation environments = 20 cells, load
each per-seed checkpoint, roll out the policy, train a probe (logistic by
default; `--classifier mlp` for the robustness check) on the belief
representation to predict the latent regime, and score test accuracy.
Aggregates per-seed (posterior_error, gap_closed) into a scatter that feeds
Family C of the M5R hypothesis tests.

References (regime-agnostic / belief / oracle PPO) are NOT probed for the
same reasons as in M6: floor has no regime representation, belief ingests
the analytical posterior directly, and oracle sees the true regime.

Outputs:
  - results/M5R/final/m5r_posterior_vs_performance{_classifier}.json

Usage:
  uv run python -m scripts.m5r_posterior_probe                    # logistic
  uv run python -m scripts.m5r_posterior_probe --classifier mlp   # MLP probe
  uv run python -m scripts.m5r_posterior_probe --n-rollouts 50    # quick
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

PROBED_METHODS = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
PER_CELL_ENV_PATH = RESULTS_ROOT / "M5R" / "final" / "per_cell_env.json"


def _gap_closed_per_seed(
    per_seed_return: list[float], floor_mean: float | None, oracle_mean: float | None,
) -> list[float]:
    if floor_mean is None or oracle_mean is None:
        return [float("nan")] * len(per_seed_return)
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


def _decoupling_diagnostic(scatter_points: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for p in scatter_points:
        by_cell.setdefault(p["env_label"], {}).setdefault(p["method"], []).append(p)
    flagged: dict[str, list[str]] = {}
    for env_label, methods in by_cell.items():
        if len(methods) < 2:
            continue
        post_means = {
            m: float(np.mean([1.0 - p["posterior_error"] for p in pts]))
            for m, pts in methods.items()
        }
        gc_means = {
            m: float(np.mean([p["gap_closed"] for p in pts]))
            for m, pts in methods.items()
        }
        post_order = sorted(post_means, key=post_means.get, reverse=True)
        gc_order = sorted(gc_means, key=gc_means.get, reverse=True)
        n = len(methods)
        half = n // 2 if n % 2 == 0 else n // 2 + 1
        for method in methods:
            in_top_post = post_order.index(method) < half
            in_top_gc = gc_order.index(method) < half
            if in_top_post != in_top_gc:
                flagged.setdefault(method, []).append(env_label)
    return {
        "methods_with_decoupling": sorted(flagged),
        "per_method_envs": flagged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_posterior_probe")
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument(
        "--classifier", choices=["logistic", "mlp"], default="logistic",
    )
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / "M5R" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.classifier == "logistic" else f"_{args.classifier}"
    stats_path = out_dir / f"m5r_posterior_vs_performance{suffix}.json"
    summary_path = out_dir / f"m5r_posterior_vs_performance{suffix}_run.json"

    run = ScriptRun(script="m5r_posterior_probe")
    if not PER_CELL_ENV_PATH.exists():
        run.fail(
            reason=f"missing {PER_CELL_ENV_PATH}; run scripts.m5r_final_eval first",
            summary_path=summary_path,
        )
        return 1
    with open(PER_CELL_ENV_PATH) as f:
        per_cell_env = json.load(f)

    t_start = time.perf_counter()
    scatter_points: list[dict[str, Any]] = []
    failed: list[str] = []
    n_total = 0
    for env_label, env_block in per_cell_env.get("per_env", {}).items():
        refs = env_block.get("refs", {})
        floor = refs.get("regime_agnostic_ppo")
        oracle = refs.get("oracle_ppo")
        cells = env_block.get("cells", {})
        for method in PROBED_METHODS:
            cell = cells.get(method)
            if cell is None:
                continue
            n_total += 1
            experiment_name = f"m5r_final_{method}_{env_label}"
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
            gc_per_seed = _gap_closed_per_seed(
                cell["per_seed_final_return"], floor, oracle,
            )
            for seed, m_acc, a_acc, gc in zip(
                probe["seeds"], probe["method_test_acc_per_seed"],
                probe["analytical_test_acc_per_seed"], gc_per_seed,
            ):
                scatter_points.append({
                    "method": method,
                    "env_label": env_label,
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
                f"analytical={a_mean:.3f} pe={a_mean - m_mean:+.3f}",
                flush=True,
            )

    if not scatter_points:
        run.fail(
            reason="no scatter points collected",
            summary_path=summary_path,
        )
        return 1

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
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_min = (time.perf_counter() - t_start) / 60
    if failed:
        run.fail(
            reason=f"{len(failed)}/{n_total} cells failed: " + ", ".join(failed[:3]),
            summary_path=summary_path,
        )
        return 1

    run.ok(
        key_stats={
            "n_scatter_points": len(scatter_points),
            "correlation_overall": correlation_overall,
            "elapsed_min": round(total_min, 2),
            "classifier": args.classifier,
        },
        summary_path=summary_path,
    )

    print("[probe] === summary ===", flush=True)
    print(f"[probe] r_overall={correlation_overall:+.3f}", flush=True)
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
