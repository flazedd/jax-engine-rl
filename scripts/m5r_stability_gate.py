"""M5R Stage B — selection-stability gate.

For each cell, take the winning config from `results/M5R/tuning/{cell}/best.json`
and re-run it at full budget (200 iter × 512 envs) at n=3 seeds on the same
held-out tuning environment. Pass criterion is that the 3-seed plateau (mean
of last 50 iters across 3 seeds) is within 10% of the tuning trial's selection
metric — catching the case where the tuning trial got a single-seed lottery.

Outputs (per cell):
  experiments/configs/m5r_gate/{cell}.yaml          # materialised full-budget config
  experiments/configs/m5r_locked/{cell}.yaml        # written iff gate passes
  results/m5r_gate_{cell}/                          # train output (n=3 seeds)
  results/M5R/gate/{cell}/result.json               # per-cell pass/fail + metrics
  results/M5R/gate/m5r_gate_run.json                # script summary

Resumability: cells whose `m5r_gate_{cell}/metrics.json` already exists are
skipped (analysis is re-computed from cached metrics).

Usage:
  uv run python -m scripts.m5r_stability_gate           # all 4 cells
  uv run python -m scripts.m5r_stability_gate --cell rl2_concat
"""
from __future__ import annotations

import argparse
import json
import shutil
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
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
RESULTS_ROOT = REPO_ROOT / "results"
TUNING_RESULTS = RESULTS_ROOT / "M5R" / "tuning"
GATE_RESULTS = RESULTS_ROOT / "M5R" / "gate"
GATE_CONFIG_DIR = CONFIG_ROOT / "m5r_gate"
LOCKED_CONFIG_DIR = CONFIG_ROOT / "m5r_locked"

CELLS = ["rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet"]
CELL_BASE = {
    "rl2_concat":      "m5r_base/rl2_concat.yaml",
    "rl2_hypernet":    "m5r_base/rl2_hypernet.yaml",
    "varibad_concat":  "m5r_base/varibad_concat.yaml",
    "varibad_hypernet":"m5r_base/varibad_hypernet.yaml",
}

# Pass criterion: 3-seed plateau >= TOL_FRAC × tuning selection metric.
# Allows for a small drop in case longer full-budget training over-trains
# slightly past the mid-mode plateau.
TOL_FRAC = 0.90
SEL_WINDOW = 50  # last N iters used for plateau computation


# ---------------------------------------------------------------------------
# Config materialisation
# ---------------------------------------------------------------------------


def _set_dotted(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _read_best(cell: str) -> dict[str, Any]:
    p = TUNING_RESULTS / cell / "best.json"
    if not p.exists():
        raise FileNotFoundError(f"missing tuning best.json for {cell}: {p}")
    with open(p) as f:
        cell_summary = json.load(f)
    if not cell_summary.get("best"):
        raise RuntimeError(f"{cell} has no best trial — check tuning output")
    return cell_summary


def _materialise_gate_config(cell: str, best_knobs: dict[str, Any]) -> Path:
    """Write a full-budget gate YAML extending the cell's base config."""
    GATE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = GATE_CONFIG_DIR / f"{cell}.yaml"

    overrides: dict[str, Any] = {}
    for dotted_key, value in best_knobs.items():
        _set_dotted(overrides, dotted_key, value)

    doc: dict[str, Any] = {
        "extends": [CELL_BASE[cell]],
        "experiment_name": f"m5r_gate_{cell}",
        # Full budget — overrides the n=1 in the base config.
        "iterations": 200,
        "parallel_envs": 512,
        "rollout_length": 128,
        "num_seeds": 3,
        "seed_base": 0,
    }
    for k, v in overrides.items():
        doc[k] = v
    with open(yaml_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return yaml_path


def _materialise_locked_config(cell: str, best_knobs: dict[str, Any]) -> Path:
    """Write the locked YAML used by Stage C (final eval).

    The locked file pins everything — it's what gets extended by per-eval-env
    overlays in Stage C, so it must define `agent.params` fully and not rely
    on the n=1 base. We extend the cell's base (PPO + MM env) and override
    the env via Stage C's overlays.
    """
    LOCKED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = LOCKED_CONFIG_DIR / f"{cell}.yaml"

    overrides: dict[str, Any] = {}
    for dotted_key, value in best_knobs.items():
        _set_dotted(overrides, dotted_key, value)

    doc: dict[str, Any] = {
        "extends": [CELL_BASE[cell]],
        "experiment_name": f"m5r_locked_{cell}_PLACEHOLDER",
        # Stage C will override env, num_seeds, etc. via per-env overlays.
        "iterations": 200,
        "parallel_envs": 512,
        "rollout_length": 128,
        "num_seeds": 8,
        "seed_base": 0,
    }
    for k, v in overrides.items():
        doc[k] = v
    with open(yaml_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return yaml_path


# ---------------------------------------------------------------------------
# Gate execution and analysis
# ---------------------------------------------------------------------------


def _read_metrics(experiment_name: str) -> dict[str, Any] | None:
    p = RESULTS_ROOT / experiment_name / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _plateau_stats(metrics: dict[str, Any]) -> dict[str, float]:
    """Mean / median / per-seed plateau over the last SEL_WINDOW iters.

    Computed from per_seed_mean_return_per_iter ([n_seeds, n_iters]) by
    averaging the last SEL_WINDOW iters per seed, then aggregating across
    seeds for mean / median.
    """
    per_seed = np.asarray(
        metrics.get("per_seed_mean_return_per_iter", []), dtype=float
    )
    if per_seed.ndim != 2 or per_seed.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "per_seed_plateau": []}
    window = min(SEL_WINDOW, per_seed.shape[1])
    seed_plateaus = per_seed[:, -window:].mean(axis=1)
    return {
        "mean": float(seed_plateaus.mean()),
        "median": float(np.median(seed_plateaus)),
        "per_seed_plateau": seed_plateaus.round(3).tolist(),
    }


def _bootstrap_ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _gate_one(cell: str) -> dict[str, Any]:
    cell_dir = GATE_RESULTS / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    result_path = cell_dir / "result.json"

    summary = _read_best(cell)
    best = summary["best"]
    best_knobs = best["sampled_knobs"]
    tuning_sel = float(best["selection_metric_mean"])

    yaml_path = _materialise_gate_config(cell, best_knobs)
    cfg = load_config(yaml_path)
    apply_run_mode(cfg, "full")

    existing = _read_metrics(cfg.experiment_name)
    skipped = existing is not None
    if skipped:
        metrics = existing
        elapsed = float("nan")
    else:
        t0 = time.perf_counter()
        train_or_sweep(cfg)
        elapsed = time.perf_counter() - t0
        metrics = _read_metrics(cfg.experiment_name)
        if metrics is None:
            raise RuntimeError(
                f"{cell}: metrics.json missing after train_or_sweep"
            )

    plateau = _plateau_stats(metrics)
    per_seed_finals = list(map(float, metrics.get("per_seed_final_return", [])))
    ci = _bootstrap_ci(per_seed_finals) if per_seed_finals else (float("nan"), float("nan"))

    pass_threshold = TOL_FRAC * tuning_sel
    plateau_passes = plateau["mean"] >= pass_threshold
    nan_free = all(np.isfinite(v) for v in per_seed_finals)
    pass_ = bool(plateau_passes and nan_free)

    locked_path: str | None = None
    if pass_:
        lp = _materialise_locked_config(cell, best_knobs)
        locked_path = str(lp.relative_to(REPO_ROOT))

    result = {
        "cell": cell,
        "skipped_train": skipped,
        "elapsed_sec": elapsed,
        "tuning_selection_metric": tuning_sel,
        "tuning_final_return": float(best["final_return_mean"]),
        "best_knobs": best_knobs,
        "gate_iterations": int(metrics.get("iterations", 0)),
        "gate_num_seeds": int(metrics.get("num_seeds", 0)),
        "gate_per_seed_final_return": per_seed_finals,
        "gate_final_return_mean": float(metrics.get("final_return_mean", float("nan"))),
        "gate_final_return_ci95": list(ci),
        "gate_plateau_mean": plateau["mean"],
        "gate_plateau_median": plateau["median"],
        "gate_per_seed_plateau": plateau["per_seed_plateau"],
        "pass_threshold": pass_threshold,
        "plateau_passes": bool(plateau_passes),
        "nan_free": bool(nan_free),
        "pass": pass_,
        "locked_config_path": locked_path,
        "gate_config_path": str(yaml_path.relative_to(REPO_ROOT)),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_stability_gate")
    parser.add_argument("--cell", type=str, default=None,
                        help="run only this cell (default: all 4)")
    args = parser.parse_args()

    cells = [args.cell] if args.cell else CELLS
    if args.cell and args.cell not in CELLS:
        raise SystemExit(f"unknown cell: {args.cell!r}; valid: {CELLS}")

    GATE_RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = GATE_RESULTS / "m5r_gate_run.json"
    run = ScriptRun(script="m5r_stability_gate", run_mode="full")

    print(f"[m5r_gate] start: cells={cells} TOL_FRAC={TOL_FRAC}", flush=True)
    t_start = time.perf_counter()

    results: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for cell in cells:
        print(f"[m5r_gate] cell={cell} starting", flush=True)
        t0 = time.perf_counter()
        try:
            r = _gate_one(cell)
        except Exception as e:
            print(f"[m5r_gate] {cell} FAILED: {e}", flush=True)
            failed.append(cell)
            results[cell] = {"cell": cell, "error": str(e), "pass": False}
            continue
        results[cell] = r
        print(
            f"[m5r_gate] {cell} done in {(time.perf_counter()-t0)/60:.1f} min "
            f"| tuning_sel={r['tuning_selection_metric']:.2f} "
            f"| gate_plateau_mean={r['gate_plateau_mean']:.2f} "
            f"| per_seed={r['gate_per_seed_plateau']} "
            f"| pass={r['pass']}",
            flush=True,
        )

    total_min = (time.perf_counter() - t_start) / 60
    n_pass = sum(1 for r in results.values() if r.get("pass"))
    n_total = len(results)

    out = {
        "tol_frac": TOL_FRAC,
        "selection_window": SEL_WINDOW,
        "total_min": total_min,
        "n_pass": n_pass,
        "n_total": n_total,
        "cells": results,
    }
    results_path = GATE_RESULTS / "m5r_gate_results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    run.add_output(results_path)

    if failed:
        run.fail(reason=f"{len(failed)}/{n_total} cells errored: {failed}",
                 summary_path=summary_path)
        return 1

    run.ok(
        key_stats={
            "n_pass": n_pass,
            "n_total": n_total,
            "total_min": round(total_min, 2),
        },
        summary_path=summary_path,
    )
    return 0 if n_pass == n_total else 2


if __name__ == "__main__":
    sys.exit(main())
