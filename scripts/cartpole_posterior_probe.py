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

from utils.paths import cartpole_dir, experiment_dir
from typing import Any

import numpy as np

from evaluation import protocol as P
from evaluation.posterior_probe import load_experiment
from evaluation.posterior_probe_cartpole import probe_one_seed_cartpole
from scripts.cartpole_names import cartpole_experiment_name
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

PROBED_METHODS = ("rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet")
LEVELS = ("easy", "medium", "hard")
AXES = ("asymmetry", "persistence")


def _experiment_name(method: str, level: str, axis: str = "asymmetry") -> str:
    return cartpole_experiment_name(method, axis, level)


def _reference_name(method: str, level: str, axis: str = "asymmetry") -> str:
    return _experiment_name(method, level, axis)


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
    exp_dir = experiment_dir(experiment_name)
    if not exp_dir.exists():
        raise FileNotFoundError(f"experiment dir missing: {exp_dir}")
    checkpoints = sorted(exp_dir.glob("checkpoint_seed_*.pkl"))
    seeds = [int(p.stem.split("_")[-1]) for p in checkpoints]
    if not seeds:
        raise FileNotFoundError(f"no checkpoints in {exp_dir}")

    method_accs: list[float] = []
    analytical_accs: list[float] = []
    method_kls: list[float] = []
    analytical_kls: list[float] = []
    method_lls: list[float] = []
    method_briers: list[float] = []
    for seed in seeds:
        bundle = load_experiment(exp_dir, seed)
        result = probe_one_seed_cartpole(
            bundle, n_rollouts=n_rollouts, rollout_length=rollout_length,
            classifier=classifier, rng_key=seed,
        )
        method_accs.append(float(result["method"]["test_acc"]))
        analytical_accs.append(float(result["analytical"]["test_acc"]))
        method_kls.append(float(result["method"]["test_kl_to_omega"]))
        analytical_kls.append(float(result["analytical"]["test_kl_to_omega"]))
        method_lls.append(float(result["method"]["test_log_loss"]))
        method_briers.append(float(result["method"]["test_brier"]))
    return {
        "seeds": seeds,
        "method_test_acc_per_seed": method_accs,
        "analytical_test_acc_per_seed": analytical_accs,
        "method_kl_per_seed": method_kls,
        "analytical_kl_per_seed": analytical_kls,
        "method_log_loss_per_seed": method_lls,
        "method_brier_per_seed": method_briers,
    }


def _decoupling_diagnostic(
    scatter_points: list[dict[str, Any]], error_key: str = "posterior_error",
) -> dict[str, Any]:
    """A method is decoupled if it lands in the top half by belief quality
    but bottom half by gap_closed (or vice versa). ``error_key`` selects the
    belief-error measure (higher = worse); pass ``belief_error_kl`` for the
    KL-based (primary) diagnostic.
    """
    by_method: dict[str, list[dict[str, Any]]] = {}
    for p in scatter_points:
        by_method.setdefault(p["method"], []).append(p)
    if len(by_method) < 2:
        return {"methods_with_decoupling": [], "per_method_cells": {}}
    post_means = {m: float(np.mean([-p[error_key] for p in pts])) for m, pts in by_method.items()}
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
    p = experiment_dir(experiment_name) / "metrics.json"
    with open(p) as f:
        m = json.load(f)
    return list(map(float, m["per_seed_final_return"]))


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.cartpole_posterior_probe")
    # The protocol fixes the probe buffer at PROBE_TIMESTEPS, so the default
    # is derived rather than typed: a bare run and the programme run then
    # produce the same numbers, which a hard-coded 200 did not.
    parser.add_argument("--n-rollouts", type=int,
                        default=P.PROBE_TIMESTEPS // P.ROLLOUT_LENGTH)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument(
        "--classifier", choices=["logistic", "mlp"], default="logistic",
    )
    parser.add_argument(
        "--levels", nargs="+", choices=LEVELS, default=["medium"],
        help="Which difficulty levels to probe. Default: medium only "
             "(historical Phase-4 behaviour). Pass `--levels easy medium "
             "hard` for the full sweep probe.",
    )
    parser.add_argument(
        "--axis", choices=AXES, default="asymmetry",
        help="Which difficulty axis the level names refer to. Default "
             "'asymmetry' (legacy single-axis behaviour). 'persistence' "
             "looks up persistence-axis experiment dirs.",
    )
    args = parser.parse_args()

    out_dir = cartpole_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Classifier-, sweep-, and axis-suffixed filename so all variants
    # coexist on disk.
    is_sweep = set(args.levels) != {"medium"}
    sweep_suffix = "_sweep" if is_sweep else ""
    axis_suffix = "" if args.axis == "asymmetry" else f"_{args.axis}"
    cls_suffix = "" if args.classifier == "logistic" else f"_{args.classifier}"
    base = f"stats_cartpole_posterior_vs_performance{sweep_suffix}{axis_suffix}{cls_suffix}"
    stats_path = out_dir / f"{base}.json"
    summary_path = out_dir / f"{base}_run.json"

    run = ScriptRun(script="cartpole_posterior_probe")

    t_start = time.perf_counter()
    scatter_points: list[dict[str, Any]] = []
    failed: list[str] = []
    # Reference means cached per level — different levels have different
    # floor / oracle, and gap_closed must be normalised within-level.
    ref_cache: dict[str, tuple[float, float]] = {}

    for level in args.levels:
        floor_exp = _reference_name("regime_agnostic", level, args.axis)
        oracle_exp = _reference_name("oracle", level, args.axis)
        floor_returns = _load_per_seed_returns(floor_exp)
        oracle_returns = _load_per_seed_returns(oracle_exp)
        floor_mean = float(np.mean(floor_returns))
        oracle_mean = float(np.mean(oracle_returns))
        ref_cache[level] = (floor_mean, oracle_mean)

        for method in PROBED_METHODS:
            experiment_name = _experiment_name(method, level, args.axis)
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
            gc_per_seed = _gap_closed_per_seed(
                per_seed_return, floor_mean, oracle_mean,
            )
            for seed, m_acc, a_acc, m_kl, a_kl, m_ll, m_br, gc in zip(
                probe["seeds"], probe["method_test_acc_per_seed"],
                probe["analytical_test_acc_per_seed"],
                probe["method_kl_per_seed"], probe["analytical_kl_per_seed"],
                probe["method_log_loss_per_seed"], probe["method_brier_per_seed"],
                gc_per_seed,
            ):
                scatter_points.append({
                    "axis": args.axis,
                    "level": level,
                    "method": method,
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "method_test_acc": m_acc,
                    "analytical_test_acc": a_acc,
                    "posterior_error": a_acc - m_acc,
                    "method_kl_to_omega": m_kl,
                    "analytical_kl_to_omega": a_kl,
                    "method_log_loss": m_ll,
                    "method_brier": m_br,
                    "belief_error_kl": m_kl - a_kl,
                    "gap_closed": gc,
                })

            m_mean = float(np.mean(probe["method_test_acc_per_seed"]))
            a_mean = float(np.mean(probe["analytical_test_acc_per_seed"]))
            m_kl_mean = float(np.mean(probe["method_kl_per_seed"]))
            print(
                f"[probe] {experiment_name:>45s} | acc={m_mean:.3f} "
                f"(anal={a_mean:.3f}) KL={m_kl_mean:.3f}",
                flush=True,
            )

    # For backwards compat with the single-level path, expose floor/oracle
    # at the medium cell as before. Multi-level runs include all levels
    # via the per-point `level` field.
    if "medium" in ref_cache:
        floor_mean, oracle_mean = ref_cache["medium"]
    else:
        floor_mean, oracle_mean = ref_cache[args.levels[0]]

    if not scatter_points:
        run.fail(reason="no scatter points collected", summary_path=summary_path)
        return 1

    # ---- correlations ---------------------------------------------------
    def _corr_ci(points, xkey, ykey="gap_closed", n_boot=10_000, seed=0):
        x = np.asarray([p[xkey] for p in points], dtype=float)
        y = np.asarray([p[ykey] for p in points], dtype=float)
        v = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[v], y[v]
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan"), float("nan"), float("nan")
        r = float(np.corrcoef(x, y)[0, 1])
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, x.size, size=x.size)
            xx, yy = x[idx], y[idx]
            if np.std(xx) > 0 and np.std(yy) > 0:
                boot.append(np.corrcoef(xx, yy)[0, 1])
        boot = np.asarray(boot)
        return r, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    def _corr(points, xkey, ykey="gap_closed"):
        x = np.asarray([p[xkey] for p in points], dtype=float)
        y = np.asarray([p[ykey] for p in points], dtype=float)
        v = ~np.isnan(x) & ~np.isnan(y)
        if v.sum() >= 2 and np.std(x[v]) > 0 and np.std(y[v]) > 0:
            return float(np.corrcoef(x[v], y[v])[0, 1])
        return float("nan")

    # posterior_error is accuracy-based (kept); belief_error_kl is primary.
    # Both are error metrics (higher = worse belief), so both keep the same
    # sign against gap_closed; no sign flip between them.
    correlation_overall, ci_low, ci_high = _corr_ci(scatter_points, "posterior_error")
    correlation_overall_kl, ci_low_kl, ci_high_kl = _corr_ci(
        scatter_points, "belief_error_kl"
    )

    by_method: dict[str, list[dict[str, Any]]] = {}
    for p in scatter_points:
        by_method.setdefault(p["method"], []).append(p)
    correlation_per_method = {
        m: _corr(pts, "posterior_error") for m, pts in by_method.items()
    }
    correlation_per_method_kl = {
        m: _corr(pts, "belief_error_kl") for m, pts in by_method.items()
    }

    decoupling = _decoupling_diagnostic(scatter_points, error_key="posterior_error")
    decoupling_kl = _decoupling_diagnostic(scatter_points, error_key="belief_error_kl")
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
        "correlation_overall_kl": correlation_overall_kl,
        "correlation_overall_kl_ci95": [ci_low_kl, ci_high_kl],
        "correlation_per_method_kl": correlation_per_method_kl,
        "decoupling_detected": decoupling,
        "decoupling_detected_kl": decoupling_kl,
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
        f"[probe] r(acc)={correlation_overall:+.3f} CI=[{ci_low:+.3f},{ci_high:+.3f}] "
        f"| r(KL)={correlation_overall_kl:+.3f} CI=[{ci_low_kl:+.3f},{ci_high_kl:+.3f}] "
        f"| signal={signal_pattern}",
        flush=True,
    )
    for method in correlation_per_method:
        print(
            f"[probe]   {method:>22s} | r_acc={correlation_per_method[method]:+.3f} "
            f"r_kl={correlation_per_method_kl[method]:+.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
