"""M5R fair-tune sweep — random search over architectural-care knobs per cell.

For each of the four meta-RL cells (RL² × {concat, hypernet}, VariBAD ×
{concat, hypernet}), draw `trials_per_cell` random configurations from the
declarative search space, run each at --mid budget on the held-out tuning
environment, and pick the configuration with the best mean return over the
last `selection_window` iterations.

Pre-registration: `experiments/configs/m5r_search_space.yaml` is committed
to git before this script runs. The `search_seed` field in that file
deterministically reproduces the trial sequence; per-cell RNG is derived
from `search_seed` + a stable cell-name hash.

Outputs (per cell):
  experiments/configs/m5r_trials/{cell}/trial_NNN.yaml   # materialised YAML
  results/m5r_tune_{cell}_trial_NNN/metrics.json         # train output
  results/M5R/tuning/{cell}/trial_NNN.json               # per-trial summary
  results/M5R/tuning/{cell}/best.json                    # selected winner
  results/M5R/tuning/m5r_tune_run.json                   # script summary

Resumability: trials whose `metrics.json` already exists are skipped, so
you can kill and resume the sweep without redoing work.

Usage:
  uv run python -m scripts.m5r_tune                # full sweep, all cells
  uv run python -m scripts.m5r_tune --smoke        # 1 trial per cell (Stage A gate)
  uv run python -m scripts.m5r_tune --cell rl2_concat  # one cell only
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
RESULTS_ROOT = REPO_ROOT / "results"
SEARCH_SPACE = CONFIG_ROOT / "m5r_search_space.yaml"
TRIAL_CONFIG_DIR = CONFIG_ROOT / "m5r_trials"
TUNING_RESULTS = RESULTS_ROOT / "M5R" / "tuning"


# ----------------------------------------------------------------------------
# Search-space loading and trial sampling
# ----------------------------------------------------------------------------


def _load_search_space() -> dict[str, Any]:
    with open(SEARCH_SPACE) as f:
        return yaml.safe_load(f)


def _cell_rng(search_seed: int, cell_name: str) -> np.random.Generator:
    """Deterministic per-cell RNG: same search_seed + cell_name → same trials."""
    h = hashlib.md5(cell_name.encode()).digest()
    cell_offset = int.from_bytes(h[:4], "little")
    return np.random.default_rng((int(search_seed) + cell_offset) & 0xFFFFFFFF)


def _sample_value(spec: dict[str, Any], rng: np.random.Generator) -> Any:
    dist = spec["dist"]
    if dist == "log_uniform":
        lo, hi = float(spec["low"]), float(spec["high"])
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if dist == "uniform":
        lo, hi = float(spec["low"]), float(spec["high"])
        return float(rng.uniform(lo, hi))
    if dist == "choice":
        values = list(spec["values"])
        return values[int(rng.integers(0, len(values)))]
    raise ValueError(f"unknown distribution: {dist!r}")


def _sample_trial(
    knobs: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    return {key: _sample_value(spec, rng) for key, spec in knobs.items()}


def _set_dotted(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set d['a']['b'] = value given dotted_key = 'a.b'."""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


# ----------------------------------------------------------------------------
# Trial materialisation
# ----------------------------------------------------------------------------


def _trial_yaml_path(cell: str, idx: int) -> Path:
    return TRIAL_CONFIG_DIR / cell / f"trial_{idx:03d}.yaml"


def _trial_experiment_name(cell: str, idx: int) -> str:
    return f"m5r_tune_{cell}_trial_{idx:03d}"


def _materialise_trial(
    cell: str, idx: int, base_config_rel: str, sampled: dict[str, Any]
) -> Path:
    """Write a per-trial YAML that extends the cell's base config with sampled knobs."""
    yaml_path = _trial_yaml_path(cell, idx)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, Any] = {}
    for dotted_key, value in sampled.items():
        _set_dotted(overrides, dotted_key, value)

    doc: dict[str, Any] = {
        "extends": [base_config_rel],
        "experiment_name": _trial_experiment_name(cell, idx),
    }
    # Merge overrides into the doc.
    for k, v in overrides.items():
        doc[k] = v

    with open(yaml_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return yaml_path


# ----------------------------------------------------------------------------
# Trial execution and selection
# ----------------------------------------------------------------------------


def _read_metrics(experiment_name: str) -> dict[str, Any] | None:
    p = RESULTS_ROOT / experiment_name / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _selection_metric(metrics: dict[str, Any], window: int) -> tuple[float, float]:
    """Return (mean, var) of mean_return_per_iter over the last `window` iters."""
    series = np.asarray(metrics.get("mean_return_per_iter", []), dtype=float)
    if series.size == 0:
        return float("nan"), float("nan")
    tail = series[-min(window, series.size):]
    return float(tail.mean()), float(tail.var())


def _run_trial(cell: str, idx: int, base_config_rel: str, sampled: dict[str, Any],
               run_mode: str) -> dict[str, Any]:
    """Run one trial. Resumable: skips if metrics.json already exists."""
    yaml_path = _materialise_trial(cell, idx, base_config_rel, sampled)
    cfg = load_config(yaml_path)
    apply_run_mode(cfg, run_mode)

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
                f"{cell} trial {idx:03d}: metrics.json missing after train_or_sweep"
            )

    selection_window = int(metrics.get("iterations", cfg.iterations))
    metric_mean, metric_var = _selection_metric(
        metrics, window=min(50, selection_window)
    )

    return {
        "cell": cell,
        "trial_idx": idx,
        "experiment_name": cfg.experiment_name,
        "skipped": skipped,
        "elapsed_sec": elapsed,
        "sampled_knobs": sampled,
        "selection_metric_mean": metric_mean,
        "selection_metric_var": metric_var,
        "final_return_mean": float(metrics.get("final_return_mean", float("nan"))),
        "iterations": int(metrics.get("iterations", 0)),
        "yaml_path": str(yaml_path.relative_to(REPO_ROOT)),
    }


def _select_best(trial_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the trial with the highest mean over the selection window.
    Ties broken by lower variance (more stable plateau).
    """
    valid = [t for t in trial_summaries if not np.isnan(t["selection_metric_mean"])]
    if not valid:
        raise RuntimeError("no valid trials to select from")
    valid.sort(key=lambda t: (-t["selection_metric_mean"], t["selection_metric_var"]))
    return valid[0]


# ----------------------------------------------------------------------------
# Cell-level driver
# ----------------------------------------------------------------------------


def _run_cell(cell: str, cell_spec: dict[str, Any], search_seed: int,
              trials_per_cell: int, run_mode: str, smoke: bool) -> dict[str, Any]:
    rng = _cell_rng(search_seed, cell)
    n_trials = 1 if smoke else trials_per_cell
    knobs = cell_spec["knobs"]
    base_config_rel = cell_spec["base_config"]

    cell_dir = TUNING_RESULTS / cell
    cell_dir.mkdir(parents=True, exist_ok=True)

    print(f"[m5r_tune] cell={cell} trials={n_trials} run_mode={run_mode}", flush=True)
    trial_summaries: list[dict[str, Any]] = []
    t_cell_start = time.perf_counter()

    for idx in range(n_trials):
        sampled = _sample_trial(knobs, rng)
        t_trial = time.perf_counter()
        try:
            summary = _run_trial(cell, idx, base_config_rel, sampled, run_mode)
        except Exception as e:
            print(f"[m5r_tune] {cell} trial {idx:03d} FAILED: {e}", flush=True)
            summary = {
                "cell": cell,
                "trial_idx": idx,
                "skipped": False,
                "elapsed_sec": time.perf_counter() - t_trial,
                "sampled_knobs": sampled,
                "selection_metric_mean": float("nan"),
                "selection_metric_var": float("nan"),
                "error": str(e),
            }

        trial_summaries.append(summary)
        with open(cell_dir / f"trial_{idx:03d}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        elapsed_total = (time.perf_counter() - t_cell_start) / 60
        sm = summary.get("selection_metric_mean", float("nan"))
        msg_skip = " (cached)" if summary.get("skipped") else ""
        print(
            f"[m5r_tune] {cell} {idx + 1:03d}/{n_trials} "
            f"sel={sm:.2f} cell_t={elapsed_total:.1f}m{msg_skip}",
            flush=True,
        )

    # Selection
    cell_summary: dict[str, Any] = {
        "cell": cell,
        "search_seed": search_seed,
        "trials": trial_summaries,
        "n_trials": n_trials,
        "run_mode": run_mode,
        "cell_elapsed_min": (time.perf_counter() - t_cell_start) / 60,
    }
    try:
        best = _select_best(trial_summaries)
        cell_summary["best"] = best
        print(
            f"[m5r_tune] cell={cell} best trial={best['trial_idx']:03d} "
            f"sel={best['selection_metric_mean']:.2f} "
            f"final={best['final_return_mean']:.2f}",
            flush=True,
        )
    except Exception as e:
        cell_summary["best"] = None
        cell_summary["error"] = str(e)
        print(f"[m5r_tune] cell={cell} selection failed: {e}", flush=True)

    with open(cell_dir / "best.json", "w") as f:
        json.dump(cell_summary, f, indent=2, default=str)
    return cell_summary


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_tune")
    parser.add_argument("--smoke", action="store_true",
                        help="1 trial per cell (Stage A gate)")
    parser.add_argument("--cell", type=str, default=None,
                        help="run only this cell (default: all 4)")
    parser.add_argument("--mode", type=str, default=None,
                        help="override trial_run_mode from search space")
    args = parser.parse_args()

    space = _load_search_space()
    search_seed = int(space["search_seed"])
    trials_per_cell = int(space["trials_per_cell"])
    run_mode = args.mode or space.get("trial_run_mode", "mid")

    cells_block = space["cells"]
    if args.cell:
        if args.cell not in cells_block:
            raise SystemExit(f"unknown cell: {args.cell!r}; valid: {list(cells_block)}")
        cells_block = {args.cell: cells_block[args.cell]}

    TUNING_RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = TUNING_RESULTS / "m5r_tune_run.json"
    # The summary-schema run_mode allowlist is {super_fast, fast, mid, full}.
    # We expose smoke status via the per-cell trial count, not the script run_mode.
    run = ScriptRun(script="m5r_tune", run_mode=run_mode)

    print(
        f"[m5r_tune] start: cells={list(cells_block)} "
        f"search_seed={search_seed} trials_per_cell={1 if args.smoke else trials_per_cell} "
        f"run_mode={run_mode}",
        flush=True,
    )
    t_start = time.perf_counter()

    cell_summaries: dict[str, Any] = {}
    for cell, cell_spec in cells_block.items():
        cell_summaries[cell] = _run_cell(
            cell, cell_spec, search_seed,
            trials_per_cell, run_mode, args.smoke,
        )

    total_min = (time.perf_counter() - t_start) / 60

    out = {
        "search_seed": search_seed,
        "trials_per_cell": 1 if args.smoke else trials_per_cell,
        "run_mode": run_mode,
        "smoke": args.smoke,
        "total_min": total_min,
        "cells": {
            cell: {
                "best_trial_idx": cs.get("best", {}).get("trial_idx") if cs.get("best") else None,
                "best_selection_metric": cs.get("best", {}).get("selection_metric_mean") if cs.get("best") else None,
                "best_final_return": cs.get("best", {}).get("final_return_mean") if cs.get("best") else None,
                "best_knobs": cs.get("best", {}).get("sampled_knobs") if cs.get("best") else None,
                "n_trials": cs["n_trials"],
                "cell_elapsed_min": cs["cell_elapsed_min"],
            }
            for cell, cs in cell_summaries.items()
        },
    }

    results_path = TUNING_RESULTS / "m5r_tune_results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    run.add_output(results_path)

    run.ok(
        key_stats={
            "cells": list(cells_block),
            "trials_per_cell": 1 if args.smoke else trials_per_cell,
            "total_min": round(total_min, 2),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
