"""M6 — difficulty sweep orchestrator (skeleton).

Sequentially trains the reduced 5-method ladder at each (axis, level) cell
on MarketMakingV1, where axis ∈ {persistence, distinguishability} and
level ∈ {easy, medium, hard}. Reuses M5 Step-4 / M3 reference results when
the env config matches (medium-medium = E_final).

For each (axis, level, method) cell, writes a per-experiment metrics.json
under `results/m6_<method>_<axis>_<level>/`. After all cells finish,
aggregates per-method gap_closed curves and ranking stability into
`results/milestones/M6/stats_M6_sweep.json`.

Usage:
  uv run python -m scripts.m6_difficulty_sweep                # full budget, both axes
  uv run python -m scripts.m6_difficulty_sweep --super-fast   # 2 iter × 16 envs × 1 seed × persistence-only
  uv run python -m scripts.m6_difficulty_sweep --fast         # 20 iter × 128 envs × 1 seed
  uv run python -m scripts.m6_difficulty_sweep --axis persistence
  uv run python -m scripts.m6_difficulty_sweep --axis distinguishability
  uv run python -m scripts.m6_difficulty_sweep --axis both    # default

This is a *skeleton*. Pass criteria computation (monotonicity, ranking
stability per `docs/milestones/m6.md`) is wired in but not yet
unit-tested against real M6 data.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.config import apply_run_mode, load_config
from training.train import train_or_sweep
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"

# (method_label, base_yaml_filename). Base YAMLs already exist from M3 / M5
# Step-4 — we mutate the env.params in-place per difficulty level.
METHODS: list[tuple[str, str]] = [
    ("regime_agnostic_ppo", "m3_regime_agnostic.yaml"),
    ("belief_ppo",          "m3_belief.yaml"),
    ("oracle_ppo",          "m3_oracle.yaml"),
    ("rl2_concat",          "m5_step4_rl2_concat.yaml"),
    ("varibad_concat",      "m5_step4_varibad_concat.yaml"),
]

LEVELS = ("easy", "medium", "hard")
AXES = ("persistence", "distinguishability")


def _load_env_params(axis: str, level: str) -> dict[str, Any]:
    env_yaml = CONFIG_ROOT / "envs" / f"m6_{axis}_{level}.yaml"
    if not env_yaml.exists():
        raise FileNotFoundError(f"missing M6 env config: {env_yaml}")
    with open(env_yaml) as f:
        d = yaml.safe_load(f)
    return d["env"]["params"]


def _build_cell_cfg(method: str, base_yaml: str, axis: str, level: str, mode: str):
    """Load the method's base YAML, override env.params with the M6 level
    values, and rename the experiment so each cell writes to its own dir."""
    cfg = load_config(CONFIG_ROOT / base_yaml)
    apply_run_mode(cfg, mode)
    cfg.env.params = copy.deepcopy(_load_env_params(axis, level))
    cfg.experiment_name = f"m6_{method}_{axis}_{level}"
    return cfg


def _read_metrics(experiment_name: str) -> dict[str, Any] | None:
    p = RESULTS_ROOT / experiment_name / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _bootstrap_ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        m = float(arr.mean()) if arr.size == 1 else float("nan")
        return m, m
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _gap_closed(cell_mean: float, floor: float, ceiling: float) -> float | None:
    if np.isnan(floor) or np.isnan(ceiling) or ceiling == floor:
        return None
    return (cell_mean - floor) / (ceiling - floor)


def _per_axis_summary(axis: str, by_level: dict[str, dict[str, dict]]) -> dict[str, Any]:
    """Compute per-method monotonicity + ranking stability across levels."""
    method_curves: dict[str, list[float]] = {}
    rankings_per_level: list[list[str]] = []
    for level in LEVELS:
        cells = by_level.get(level, {})
        floor = cells.get("regime_agnostic_ppo", {}).get("final_return_mean", float("nan"))
        oracle = cells.get("oracle_ppo", {}).get("final_return_mean", float("nan"))
        ordered = sorted(
            (m for m in cells if m not in ("regime_agnostic_ppo", "oracle_ppo")),
            key=lambda m: cells[m]["final_return_mean"], reverse=True,
        )
        rankings_per_level.append(ordered)
        for method, c in cells.items():
            gc = _gap_closed(c["final_return_mean"], floor, oracle)
            if gc is not None:
                method_curves.setdefault(method, []).append(gc)

    per_method_monotonicity: dict[str, dict[str, Any]] = {}
    all_monotonic = True
    for method, gcs in method_curves.items():
        if len(gcs) < 2:
            per_method_monotonicity[method] = {"monotonic": None, "slope_sign": "n/a"}
            continue
        diffs = np.diff(gcs)
        non_increasing = bool(np.all(diffs <= 1e-9))
        non_decreasing = bool(np.all(diffs >= -1e-9))
        slope = "negative" if non_increasing and not non_decreasing else (
            "positive" if non_decreasing and not non_increasing else "flat" if non_increasing and non_decreasing else "non-monotonic"
        )
        is_monotonic = non_increasing  # gap_closed should not increase as difficulty increases
        per_method_monotonicity[method] = {
            "monotonic": is_monotonic,
            "gap_closed_slope_sign": slope,
            "gap_closed_curve": gcs,
        }
        if not is_monotonic:
            all_monotonic = False

    ranking_stable = (
        len({tuple(r) for r in rankings_per_level}) == 1
        if rankings_per_level else False
    )

    return {
        "levels": list(LEVELS),
        "per_method_monotonicity": per_method_monotonicity,
        "rankings_per_level": rankings_per_level,
        "all_methods_monotonic": all_monotonic,
        "ranking_stable": ranking_stable,
    }


def _train_cell(
    method: str, base_yaml: str, axis: str, level: str, mode: str,
    skip_existing: bool,
) -> dict[str, Any] | None:
    """Train (or skip) one cell and return its summary dict."""
    cfg = _build_cell_cfg(method, base_yaml, axis, level, mode)
    if skip_existing:
        existing = _read_metrics(cfg.experiment_name)
        if existing and existing.get("iterations", 0) >= cfg.iterations:
            print(
                f"[m6] skip {cfg.experiment_name}: existing metrics with "
                f"iters>={existing.get('iterations')}",
                flush=True,
            )
            m = existing
            return {
                "experiment_name": cfg.experiment_name,
                "final_return_mean": float(m["final_return_mean"]),
                "per_seed_final_return": list(map(float, m["per_seed_final_return"])),
                "iterations": int(m["iterations"]),
                "num_seeds": int(m["num_seeds"]),
                "skipped_existing": True,
            }

    print(
        f"[m6] launching {cfg.experiment_name} | iters={cfg.iterations} "
        f"envs={cfg.parallel_envs} seeds={cfg.num_seeds}",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        train_or_sweep(cfg)
    except SystemExit as e:
        if e.code != 0:
            print(f"[m6] {cfg.experiment_name} FAILED (exit {e.code})", flush=True)
            return None
    elapsed = time.perf_counter() - t0
    m = _read_metrics(cfg.experiment_name)
    if m is None:
        return None
    print(
        f"[m6] {cfg.experiment_name} done in {elapsed/60:.1f} min | "
        f"mean={float(m['final_return_mean']):.2f}",
        flush=True,
    )
    return {
        "experiment_name": cfg.experiment_name,
        "final_return_mean": float(m["final_return_mean"]),
        "per_seed_final_return": list(map(float, m["per_seed_final_return"])),
        "iterations": int(m["iterations"]),
        "num_seeds": int(m["num_seeds"]),
        "skipped_existing": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m6_difficulty_sweep")
    parser.add_argument("--super-fast", action="store_true",
                        help="2 iter × 16 envs × 1 seed (smoke test)")
    parser.add_argument("--fast", action="store_true",
                        help="20 iter × 128 envs × 1 seed")
    parser.add_argument("--mid", action="store_true",
                        help="100 iter × 256 envs × 1 seed")
    parser.add_argument("--axis", choices=("persistence", "distinguishability", "both"),
                        default="both")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="ignore on-disk metrics and re-train every cell")
    args = parser.parse_args()
    if sum([args.super_fast, args.fast, args.mid]) > 1:
        raise SystemExit("--super-fast / --fast / --mid are mutually exclusive")
    mode = (
        "super_fast" if args.super_fast
        else "fast" if args.fast
        else "mid" if args.mid
        else "full"
    )
    skip_existing = not args.no_skip_existing
    axes_to_run = AXES if args.axis == "both" else (args.axis,)

    run = ScriptRun(script="m6_difficulty_sweep", run_mode=mode)
    out_dir = RESULTS_ROOT / "milestones" / "M6"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_M6_sweep.json"
    summary_path = out_dir / "stats_M6_sweep_run.json"

    t_start = time.perf_counter()
    print(
        f"[m6] start: axes={axes_to_run}, methods={[m for m,_ in METHODS]}, "
        f"levels={LEVELS}, mode={mode}, skip_existing={skip_existing}",
        flush=True,
    )

    results: dict[str, dict[str, dict[str, dict]]] = {a: {l: {} for l in LEVELS} for a in axes_to_run}
    failed: list[str] = []
    for axis in axes_to_run:
        for level in LEVELS:
            for method, base_yaml in METHODS:
                cell = _train_cell(method, base_yaml, axis, level, mode, skip_existing)
                if cell is None:
                    failed.append(f"{method}/{axis}/{level}")
                    continue
                results[axis][level][method] = cell

    if failed:
        run.fail(
            reason=f"{len(failed)} cells failed: " + ", ".join(failed[:5]),
            summary_path=summary_path,
        )
        return 1

    # ----- Aggregate per-axis summaries -------------------------------------
    sweep_block: dict[str, Any] = {}
    interpretable_overall = True
    for axis in axes_to_run:
        s = _per_axis_summary(axis, results[axis])
        # interpretable: ranking stable OR every method monotonic.
        interp = s["ranking_stable"] or s["all_methods_monotonic"]
        s["interpretable_outcome"] = interp
        if not interp:
            interpretable_overall = False
        sweep_block[f"{axis}_sweep"] = s

    stats = {
        "env_version_medium": "e_final",
        "axes": list(axes_to_run),
        "levels": list(LEVELS),
        "methods": [m for m, _ in METHODS],
        "run_mode": mode,
        "results": results,  # nested {axis: {level: {method: cell_summary}}}
        **sweep_block,
        "interpretable_overall": interpretable_overall,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_min = (time.perf_counter() - t_start) / 60
    run.ok(
        key_stats={
            "elapsed_min": round(total_min, 2),
            "n_axes": len(axes_to_run),
            "n_cells": sum(len(results[a][l]) for a in axes_to_run for l in LEVELS),
            "interpretable_overall": bool(interpretable_overall),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )

    print("[m6] === sweep summary ===", flush=True)
    for axis in axes_to_run:
        s = sweep_block[f"{axis}_sweep"]
        print(
            f"[m6] {axis}: monotonic_all={s['all_methods_monotonic']} "
            f"ranking_stable={s['ranking_stable']} interpretable={s['interpretable_outcome']}",
            flush=True,
        )
        for method, mono in s["per_method_monotonicity"].items():
            curve = mono.get("gap_closed_curve", [])
            curve_s = ", ".join(f"{c:.2f}" for c in curve) if curve else "n/a"
            print(
                f"[m6]   {method:>22s} | gc={curve_s} | "
                f"slope={mono.get('gap_closed_slope_sign')}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
