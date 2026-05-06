"""M5R small-budget probe: train each meta-RL cell at ~5k parameters on
the medium-difficulty environment.

This is the under-capacity sanity check: if methods cannot learn at this
budget, the matched-180k design's choice of budget is justified
empirically rather than by intuition. Each cell is trained with the same
PPO procedure and seed schedule as the matched-180k Stage C eval.

Outputs:
  results/m5r_smallbudget_{cell}/                  per-cell train output
  results/M5R/smallbudget/per_cell.json            aggregated stats

Usage:
  uv run python -m scripts.m5r_smallbudget
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from training.config import apply_run_mode, load_config
from training.train import train_or_sweep
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs" / "m5r_smallbudget"
RESULTS_ROOT = REPO_ROOT / "results"
OUT_DIR = RESULTS_ROOT / "M5R" / "smallbudget"

CELLS = [
    ("rl2_concat",       "rl2_concat.yaml"),
    ("rl2_hypernet",     "rl2_hypernet.yaml"),
    ("varibad_concat",   "varibad_concat.yaml"),
    ("varibad_hypernet", "varibad_hypernet.yaml"),
]


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


def _run_one(cell: str, cfg_name: str) -> dict[str, Any]:
    cfg_path = CONFIG_ROOT / cfg_name
    cfg = load_config(cfg_path)
    apply_run_mode(cfg, "full")
    existing = _read_metrics(cfg.experiment_name)
    if existing is not None:
        metrics = existing
    else:
        train_or_sweep(cfg)
        metrics = _read_metrics(cfg.experiment_name)
        if metrics is None:
            raise RuntimeError(f"{cell}: metrics missing after training")
    finals = list(map(float, metrics.get("per_seed_final_return", [])))
    mean = float(np.mean(finals)) if finals else float("nan")
    ci_lo, ci_hi = _bootstrap_ci(finals) if finals else (float("nan"), float("nan"))
    return {
        "cell": cell,
        "experiment_name": cfg.experiment_name,
        "per_seed_final_return": finals,
        "final_return_mean": mean,
        "final_return_ci95": [ci_lo, ci_hi],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "m5r_smallbudget_run.json"
    run = ScriptRun(script="m5r_smallbudget", run_mode="full")

    rows = []
    failed = []
    t_start = time.perf_counter()
    for cell, cfg_name in CELLS:
        print(f"[m5r_smallbudget] {cell} starting", flush=True)
        try:
            r = _run_one(cell, cfg_name)
        except Exception as e:
            print(f"[m5r_smallbudget] {cell} FAILED: {e}", flush=True)
            failed.append(cell)
            continue
        rows.append(r)
        ci = r["final_return_ci95"]
        print(f"[m5r_smallbudget] {cell} done | mean={r['final_return_mean']:.2f} "
              f"ci=[{ci[0]:.2f}, {ci[1]:.2f}]", flush=True)

    total_min = (time.perf_counter() - t_start) / 60
    payload = {"rows": rows, "n_failed": len(failed), "failed": failed,
               "total_min": round(total_min, 2)}
    out_path = OUT_DIR / "per_cell.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    run.add_output(out_path)

    print("\n[m5r_smallbudget] === summary ===")
    for r in rows:
        ci = r["final_return_ci95"]
        print(f"  {r['cell']:22s} mean={r['final_return_mean']:.2f} "
              f"ci=[{ci[0]:.2f}, {ci[1]:.2f}]")

    if failed:
        run.fail(reason=f"{len(failed)} cells failed: {failed}",
                 summary_path=summary_path)
        return 1
    run.ok(
        key_stats={"n_cells": len(rows), "total_min": round(total_min, 2)},
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
