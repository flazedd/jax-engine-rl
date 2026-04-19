"""Orchestrate a milestone end-to-end.

    uv run python -m scripts.make_milestone M0

For M0: runs the dummy pipeline in super-fast / fast / full, runs the
run-mode test, regenerates figures, and writes stats_M0_pipeline.json with
the pass-criteria fields.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

from envs.mm_reduced import ACTION_FAVOR_ASK, ACTION_FAVOR_BID, MMReducedEnv
from oracles.analytical_as import solve_analytical_as
from plotting.regenerate_figures import _HANDLERS as FIGURE_HANDLERS
from training.config import apply_run_mode, load_config
from training.train import train
from utils.script_output import (
    ScriptRun,
    validate_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"


def _run_training(config_path: Path, mode: str) -> tuple[float, bool, bool]:
    cfg = load_config(config_path)
    apply_run_mode(cfg, mode)
    t0 = time.perf_counter()
    exit_zero = True
    try:
        train(cfg)
    except SystemExit as e:
        exit_zero = (e.code == 0)
    except Exception:
        exit_zero = False
        raise
    elapsed = time.perf_counter() - t0

    summary_path = RESULTS_ROOT / cfg.experiment_name / "summary.json"
    wrote_json = summary_path.exists()
    schema_ok = False
    if wrote_json:
        with open(summary_path) as f:
            summary = json.load(f)
        schema_ok, _ = validate_summary(summary)
    return elapsed, (exit_zero and wrote_json), schema_ok


def _regenerate_figures(milestone: str) -> bool:
    handler = FIGURE_HANDLERS.get(milestone)
    if handler is None:
        return False
    inner_run = ScriptRun(script="regenerate_figures")
    try:
        handler(inner_run)
        summary_path = RESULTS_ROOT / "milestones" / milestone / f"plot_summary_{milestone}.json"
        inner_run.ok(
            key_stats={"milestone": milestone, "figures": len(inner_run.outputs)},
            summary_path=summary_path,
        )
        return True
    except Exception as e:
        summary_path = RESULTS_ROOT / "milestones" / milestone / f"plot_summary_{milestone}.json"
        inner_run.fail(reason=str(e), summary_path=summary_path)
        return False


def make_m0() -> dict:
    config = CONFIG_ROOT / "m0_dummy.yaml"
    durations: dict[str, float] = {}
    exit_zero_all = True
    wrote_json_all = True
    schema_ok_all = True

    for mode in ("super_fast", "fast", "full"):
        elapsed, ok, schema_ok = _run_training(config, mode)
        durations[mode] = elapsed
        exit_zero_all &= ok
        wrote_json_all &= ok  # wrote_json folded into ok above
        schema_ok_all &= schema_ok

    figures_ok = _regenerate_figures("M0")

    # Shared-schema validation of every summary.json in the last full run.
    # (Schema is the contract for Claude Code; do a full sweep.)
    all_jsons_ok = True
    for path in RESULTS_ROOT.rglob("summary.json"):
        with open(path) as f:
            s = json.load(f)
        ok, _ = validate_summary(s)
        all_jsons_ok &= ok

    key_stats = {
        "pipeline_version": "0.1",
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "run_modes_tested": ["super_fast", "fast", "full"],
        "super_fast_duration_seconds": round(durations["super_fast"], 3),
        "fast_duration_seconds": round(durations["fast"], 3),
        "full_duration_seconds": round(durations["full"], 3),
        "all_scripts_exit_zero": bool(exit_zero_all),
        "all_scripts_wrote_expected_json": bool(wrote_json_all),
        "schema_validates": bool(schema_ok_all and all_jsons_ok),
        "figures_regenerated": bool(figures_ok),
        "make_milestone_script_succeeded": bool(
            exit_zero_all
            and wrote_json_all
            and schema_ok_all
            and all_jsons_ok
            and figures_ok
            and durations["super_fast"] < 30.0
            and durations["fast"] < 300.0
        ),
    }
    return key_stats


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    if denom == 0.0:
        return 0.0
    return float((xc * yc).sum() / denom)


def make_m1() -> dict:
    config = CONFIG_ROOT / "m1_ppo_as.yaml"

    print(
        "[make_milestone] M1 plan: super_fast → fast → full, then figures + diagnostics.\n"
        "[make_milestone] rough wall-clock: super_fast ~10s, fast ~30s, full ~4-6min (depends on epochs/seeds).",
        flush=True,
    )

    durations: dict[str, float] = {}
    exit_zero_all = True
    schema_ok_all = True
    for mode in ("super_fast", "fast", "full"):
        print(f"[make_milestone] running mode={mode}", flush=True)
        elapsed, ok, schema_ok = _run_training(config, mode)
        print(f"[make_milestone] mode={mode} done in {elapsed:.1f}s", flush=True)
        durations[mode] = elapsed
        exit_zero_all &= ok
        schema_ok_all &= schema_ok

    cfg_for_env = load_config(config)
    env = MMReducedEnv(**cfg_for_env.env.params)
    sol = solve_analytical_as(env)
    as_return = float(sol.expected_episode_return)

    metrics_path = RESULTS_ROOT / "m1_ppo_as" / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    final_per_seed = [float(x) for x in metrics["per_seed_final_return"]]
    final_mean = float(metrics["final_return_mean"])
    final_ci = [float(x) for x in metrics["final_return_ci95"]]
    return_ratio = final_mean / as_return if as_return > 0 else 0.0

    mean_curve = np.asarray(metrics["mean_return_per_iter"], dtype=float)
    iters = np.arange(mean_curve.size)
    last20 = mean_curve[-20:]
    if last20.size >= 2:
        slope = float(np.polyfit(np.arange(last20.size), last20, 1)[0])
    else:
        slope = 0.0
    plateau_threshold = 0.99 * final_mean
    above = np.where(mean_curve >= plateau_threshold)[0]
    plateau_reached_at = int(above[0]) if above.size > 0 else int(mean_curve.size)
    diffs = np.diff(mean_curve)
    monotonic_increase_fraction = float((diffs > 0).mean()) if diffs.size > 0 else 0.0
    converged = bool(abs(slope) < 0.5 and plateau_reached_at < 90)

    per_seed_probs = np.asarray(metrics["per_seed_final_action_probs"], dtype=float)
    probs_mean = per_seed_probs.mean(axis=0)
    ppo_skew = probs_mean[:, ACTION_FAVOR_ASK] - probs_mean[:, ACTION_FAVOR_BID]
    as_skew = np.asarray(sol.skew, dtype=float)
    q_margin = np.asarray(sol.q_margin, dtype=float)

    # Margin filter: states where AS has a clear preference. At small margin
    # (incl. boundary ties and near-tied states where skew ≈ sym in Q-value)
    # the AS argmax is arbitrary and shouldn't dominate the comparison.
    # Threshold 0.05 chosen so the filter keeps states with >~5% per-step Q gap.
    margin_threshold = 0.05
    meaningful = q_margin > margin_threshold

    corr_full = _pearson(ppo_skew, as_skew)
    if meaningful.sum() >= 2:
        corr_meaningful = _pearson(ppo_skew[meaningful], as_skew[meaningful])
    else:
        corr_meaningful = float("nan")

    nonzero = as_skew != 0
    if nonzero.any():
        agreement_full = float((np.sign(ppo_skew[nonzero]) == np.sign(as_skew[nonzero])).mean())
    else:
        agreement_full = 1.0
    if meaningful.any():
        agreement_meaningful = float(
            (np.sign(ppo_skew[meaningful]) == np.sign(as_skew[meaningful])).mean()
        )
    else:
        agreement_meaningful = 1.0

    # Primary pass criterion: shape matches on states where AS has a real
    # preference. Full-vector stats kept as diagnostics.
    policy_shape_matches = bool(
        (not np.isnan(corr_meaningful)) and corr_meaningful > 0.9 and agreement_meaningful > 0.95
    )

    figures_ok = _regenerate_figures("M1")

    wall_time = float(sum(durations.values()))

    pass_criteria = bool(
        return_ratio >= 0.95 and converged and policy_shape_matches
    )

    stats = {
        "method": "ppo",
        "env": "as_e0",
        "seeds": list(range(cfg_for_env.num_seeds)),
        "final_return_per_seed": final_per_seed,
        "final_return_mean": final_mean,
        "final_return_ci": final_ci,
        "as_analytical_return": as_return,
        "return_ratio": return_ratio,
        "iterations": int(metrics["iterations"]),
        "parallel_envs": int(metrics["parallel_envs"]),
        "wall_time_seconds": wall_time,
        "slope_last_20_iterations": slope,
        "plateau_reached_at_iteration": plateau_reached_at,
        "monotonic_increase_fraction": monotonic_increase_fraction,
        "converged": converged,
        "q_margin_threshold": margin_threshold,
        "n_meaningful_states": int(meaningful.sum()),
        "policy_skew_correlation": corr_meaningful,
        "policy_skew_direction_agreement": agreement_meaningful,
        "policy_skew_correlation_full": corr_full,
        "policy_skew_direction_agreement_full": agreement_full,
        "policy_shape_matches": policy_shape_matches,
        "figures_regenerated": bool(figures_ok),
        "all_scripts_exit_zero": bool(exit_zero_all),
        "schema_validates": bool(schema_ok_all),
        "make_milestone_script_succeeded": bool(
            pass_criteria and figures_ok and exit_zero_all and schema_ok_all
        ),
    }
    return stats


_MILESTONE_HANDLERS = {
    "M0": make_m0,
    "M1": make_m1,
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.make_milestone")
    parser.add_argument("milestone", help="milestone id, e.g. M0")
    args = parser.parse_args()

    handler = _MILESTONE_HANDLERS.get(args.milestone)
    run = ScriptRun(script="make_milestone")
    summary_dir = RESULTS_ROOT / "milestones" / args.milestone
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"stats_{args.milestone}_pipeline.json"

    if handler is None:
        run.fail(reason=f"no handler for {args.milestone}", summary_path=summary_path)
        return 1

    try:
        stats = handler()
    except Exception as e:
        run.fail(reason=f"{type(e).__name__}: {e}", summary_path=summary_path)
        raise

    # Write stats_{milestone}_pipeline.json (the milestone-level artifact)
    stats_path = summary_dir / f"stats_{args.milestone}_pipeline.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    # Our own script summary (shared schema) — separate file so both schemas coexist.
    run_summary_path = summary_dir / f"summary_make_{args.milestone}.json"
    print_stats = {
        "milestone": args.milestone,
        "pass": stats["make_milestone_script_succeeded"],
    }
    for k in (
        "super_fast_duration_seconds",
        "fast_duration_seconds",
        "full_duration_seconds",
        "return_ratio",
        "converged",
        "policy_shape_matches",
    ):
        if k in stats:
            print_stats[k] = stats[k]
    run.ok(
        key_stats=stats,
        summary_path=run_summary_path,
        print_stats=print_stats,
    )
    return 0 if stats["make_milestone_script_succeeded"] else 1


if __name__ == "__main__":
    sys.exit(main())
