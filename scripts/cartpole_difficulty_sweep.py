"""Cartpole — difficulty sweep orchestrator.

Mirrors `scripts/m6_difficulty_sweep.py` but for the cartpole second-
POMDP env. Sweeps the within-regime action-success asymmetry: easy
(strong, range 0.78), medium (current, range 0.65), hard (mild, range
0.20). Tests whether the inverted-decoupling pattern from the medium
cell generalises across cartpole difficulties — same role M6 played
for M5 on MarketMakingV1.

For each (level, method) cell, writes per-experiment metrics to
`results/m_cartpole_<method>_<level>/metrics.json` (medium reuses
existing `m_cartpole_<method>` dirs from the original Phase-4 run).
After all cells finish, aggregates into
`results/milestones/cartpole/stats_cartpole_sweep.json`.

Usage:
  uv run python -m scripts.cartpole_difficulty_sweep                # full budget
  uv run python -m scripts.cartpole_difficulty_sweep --refs-only --mid
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

# Same 7-method ladder as the medium-cell Phase-4 run: 3 references at
# n=5, 4 meta-RL at n=8. Configs reused verbatim — only the env params
# get overridden per level.
METHODS: list[tuple[str, str]] = [
    ("regime_agnostic", "m_cartpole_regime_agnostic.yaml"),
    ("belief",          "m_cartpole_belief.yaml"),
    ("oracle",          "m_cartpole_oracle.yaml"),
    ("rl2_concat",      "m_cartpole_rl2_concat.yaml"),
    ("rl2_hypernet",    "m_cartpole_rl2_hypernet.yaml"),
    ("varibad_concat",  "m_cartpole_varibad_concat.yaml"),
    ("varibad_hypernet","m_cartpole_varibad_hypernet.yaml"),
]
REFERENCE_METHODS = {"regime_agnostic", "belief", "oracle"}
LEVELS = ("easy", "medium", "hard")
AXES = ("asymmetry", "persistence")


def _load_env_params(axis: str, level: str) -> dict[str, Any]:
    """Medium is shared across both axes (same env config, same trained
    cells). Easy / hard read from axis-specific YAMLs.

    Asymmetry axis (legacy naming): `e_cartpole_v1_<level>.yaml`.
    Persistence axis: `e_cartpole_v1_persistence_<level>.yaml`.
    """
    if level == "medium":
        fname = "e_cartpole_v1.yaml"
    elif axis == "asymmetry":
        fname = f"e_cartpole_v1_{level}.yaml"
    else:
        fname = f"e_cartpole_v1_{axis}_{level}.yaml"
    env_yaml = CONFIG_ROOT / "envs" / fname
    if not env_yaml.exists():
        raise FileNotFoundError(f"missing cartpole env config: {env_yaml}")
    with open(env_yaml) as f:
        d = yaml.safe_load(f)
    return d["env"]["params"]


def _experiment_name(method: str, axis: str, level: str) -> str:
    """Medium cells reuse the historical `m_cartpole_<method>` paths
    (shared across axes — the env at medium is identical for both).

    Asymmetry easy/hard reuse the legacy `m_cartpole_<method>_<level>`
    naming from the original asymmetry sweep so we do not retrain.

    Persistence easy/hard get axis-prefixed names so they coexist with
    the asymmetry runs in `results/`.
    """
    if level == "medium":
        return f"m_cartpole_{method}"
    if axis == "asymmetry":
        return f"m_cartpole_{method}_{level}"
    return f"m_cartpole_{method}_{axis}_{level}"


def _build_cell_cfg(method: str, base_yaml: str, axis: str, level: str, mode: str):
    cfg = load_config(CONFIG_ROOT / base_yaml)
    apply_run_mode(cfg, mode)
    cfg.env.params = copy.deepcopy(_load_env_params(axis, level))
    cfg.experiment_name = _experiment_name(method, axis, level)
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


def _per_level_summary(by_level: dict[str, dict[str, dict]]) -> dict[str, Any]:
    method_curves: dict[str, list[float]] = {}
    rankings_per_level: list[list[str]] = []
    for level in LEVELS:
        cells = by_level.get(level, {})
        floor = cells.get("regime_agnostic", {}).get("final_return_mean", float("nan"))
        oracle = cells.get("oracle", {}).get("final_return_mean", float("nan"))
        ordered = sorted(
            (m for m in cells if m not in ("regime_agnostic", "oracle")),
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
            "positive" if non_decreasing and not non_increasing
            else "flat" if non_increasing and non_decreasing else "non-monotonic"
        )
        per_method_monotonicity[method] = {
            "monotonic": non_increasing,
            "gap_closed_slope_sign": slope,
            "gap_closed_curve": gcs,
        }
        if not non_increasing:
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
    method: str, base_yaml: str, axis: str, level: str, mode: str, skip_existing: bool,
) -> dict[str, Any] | None:
    cfg = _build_cell_cfg(method, base_yaml, axis, level, mode)
    if skip_existing:
        existing = _read_metrics(cfg.experiment_name)
        if existing and existing.get("iterations", 0) >= cfg.iterations:
            print(
                f"[cp_sweep] skip {cfg.experiment_name}: existing metrics with "
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
        f"[cp_sweep] launching {cfg.experiment_name} | iters={cfg.iterations} "
        f"envs={cfg.parallel_envs} seeds={cfg.num_seeds}",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        train_or_sweep(cfg)
    except SystemExit as e:
        if e.code != 0:
            print(f"[cp_sweep] {cfg.experiment_name} FAILED (exit {e.code})", flush=True)
            return None
    elapsed = time.perf_counter() - t0
    m = _read_metrics(cfg.experiment_name)
    if m is None:
        return None
    print(
        f"[cp_sweep] {cfg.experiment_name} done in {elapsed/60:.1f} min | "
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
    parser = argparse.ArgumentParser(prog="scripts.cartpole_difficulty_sweep")
    parser.add_argument("--super-fast", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--mid", action="store_true")
    parser.add_argument("--refs-only", action="store_true",
                        help="train only the 3 reference methods (gate before full)")
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--axis", choices=AXES, default="asymmetry",
                        help="Which difficulty axis to sweep. Default 'asymmetry' "
                             "(legacy single-axis behaviour). 'persistence' sweeps "
                             "the HMM transition diagonal, mirroring M6's persistence "
                             "axis on MM.")
    parser.add_argument("--no-skip-existing", action="store_true")
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
    methods_to_run = [
        (m, y) for m, y in METHODS if (m in REFERENCE_METHODS) or not args.refs_only
    ]

    run = ScriptRun(script="cartpole_difficulty_sweep", run_mode=mode)
    out_dir = RESULTS_ROOT / "milestones" / "cartpole"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-axis stats filename so the asymmetry and persistence sweeps
    # coexist on disk (default 'asymmetry' keeps the legacy filename).
    axis_suffix = "" if args.axis == "asymmetry" else f"_{args.axis}"
    stats_path = out_dir / f"stats_cartpole_sweep{axis_suffix}.json"
    summary_path = out_dir / f"stats_cartpole_sweep{axis_suffix}_run.json"

    t_start = time.perf_counter()
    print(
        f"[cp_sweep] start: axis={args.axis}, methods={[m for m,_ in methods_to_run]}, "
        f"levels={args.levels}, mode={mode}, skip_existing={skip_existing}",
        flush=True,
    )

    results: dict[str, dict[str, dict]] = {l: {} for l in args.levels}
    failed: list[str] = []
    for level in args.levels:
        for method, base_yaml in methods_to_run:
            cell = _train_cell(method, base_yaml, args.axis, level, mode, skip_existing)
            if cell is None:
                failed.append(f"{method}/{level}")
                continue
            results[level][method] = cell

    if failed:
        run.fail(
            reason=f"{len(failed)} cells failed: " + ", ".join(failed[:5]),
            summary_path=summary_path,
        )
        return 1

    summary = _per_level_summary(results)
    stats = {
        "env_version_medium": "e_cartpole_v1",
        "axis": args.axis,
        "levels": list(args.levels),
        "methods": [m for m, _ in methods_to_run],
        "run_mode": mode,
        "results": results,
        "summary": summary,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_min = (time.perf_counter() - t_start) / 60
    run.ok(
        key_stats={
            "elapsed_min": round(total_min, 2),
            "n_cells": sum(len(results[l]) for l in args.levels),
            "all_methods_monotonic": summary["all_methods_monotonic"],
            "ranking_stable": summary["ranking_stable"],
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )
    print("[cp_sweep] === per-level reference triple ===", flush=True)
    for level in args.levels:
        cells = results[level]
        f_ = cells.get("regime_agnostic", {}).get("final_return_mean", float("nan"))
        b_ = cells.get("belief", {}).get("final_return_mean", float("nan"))
        o_ = cells.get("oracle", {}).get("final_return_mean", float("nan"))
        ok = (f_ < b_ < o_) if all(np.isfinite([f_, b_, o_])) else False
        print(
            f"[cp_sweep] {level:>8s} | floor={f_:.2f}  belief={b_:.2f}  oracle={o_:.2f}  "
            f"| gap={o_ - f_:+.2f}  | R1={ok}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
