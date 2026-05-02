"""M5R Stage C — final evaluation on the 5 thesis evaluation environments.

For each of the four locked meta-RL configs (RL² × {concat, hypernet},
VariBAD × {concat, hypernet}), train at full budget × n=8 seeds on each of:
  - E_final (= persistence-medium = distinguishability-medium)
  - persistence_easy   (P_ii = 0.99)
  - persistence_hard   (P_ii = 0.95)
  - distinguishability_easy
  - distinguishability_hard

References (regime_agnostic / belief / oracle PPO) and stacked-obs PPO are
NOT re-trained — their existing M3 / M6 numbers stay valid because the
fairness fix only touches the meta-RL cells. Their values are read from
the M3 + M6 stats JSONs and used to compute gap-closed fractions.

Outputs:
  experiments/configs/m5r_final/{cell}_{env_label}.yaml  # per-cell-env config
  results/m5r_final_{cell}_{env_label}/                  # train output
  results/M5R/final/per_cell_env.json                    # aggregated stats
  results/M5R/final/m5r_final_run.json                   # script summary

Resumability: cells whose `m5r_final_{cell}_{env_label}/metrics.json` exists
are skipped — kill and resume freely.

Usage:
  uv run python -m scripts.m5r_final_eval                      # full sweep
  uv run python -m scripts.m5r_final_eval --env e_final        # one env
  uv run python -m scripts.m5r_final_eval --cell rl2_hypernet  # one cell
"""
from __future__ import annotations

import argparse
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
FINAL_CONFIG_DIR = CONFIG_ROOT / "m5r_final"
FINAL_RESULTS = RESULTS_ROOT / "M5R" / "final"

CELLS = ["rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet"]

# (env_label, env_yaml_relative_to_CONFIG_ROOT). Order is env-major: outer
# loop runs all 4 cells on one env before moving to the next, so partial
# results give a complete cross-section per env.
EVAL_ENVS: list[tuple[str, str]] = [
    ("e_final",                "envs/e_final.yaml"),
    ("persistence_easy",       "envs/m6_persistence_easy.yaml"),
    ("persistence_hard",       "envs/m6_persistence_hard.yaml"),
    ("distinguishability_easy","envs/m6_distinguishability_easy.yaml"),
    ("distinguishability_hard","envs/m6_distinguishability_hard.yaml"),
]

LOCKED_CONFIG_DIR = CONFIG_ROOT / "m5r_locked"

M3_REF_PATH = RESULTS_ROOT / "milestones" / "M3" / "stats_M3_reference_levels.json"
M6_SWEEP_PATH = RESULTS_ROOT / "milestones" / "M6" / "stats_M6_sweep.json"


# ---------------------------------------------------------------------------
# Reference loading (no retraining; reuse M3 + M6)
# ---------------------------------------------------------------------------


def _load_e_final_refs() -> dict[str, float]:
    if not M3_REF_PATH.exists():
        return {}
    with open(M3_REF_PATH) as f:
        d = json.load(f)
    rl = d.get("reference_levels", {})
    return {
        "regime_agnostic_ppo": float(rl.get("regime_agnostic_ppo", {}).get("mean", float("nan"))),
        "belief_ppo":          float(rl.get("belief_ppo", {}).get("mean", float("nan"))),
        "oracle_ppo":          float(rl.get("oracle_ppo", {}).get("mean", float("nan"))),
    }


def _load_m6_refs(env_label: str) -> dict[str, float]:
    """Pull floor / belief / oracle for an M6 sweep env from stats_M6_sweep.json.

    The stats file structure is `results.{axis}.{level}.{method}.final_return_mean`.
    `env_label` is split on the FIRST underscore: 'persistence_easy' →
    axis='persistence', level='easy'.
    """
    if not M6_SWEEP_PATH.exists():
        return {}
    if "_" not in env_label:
        return {}
    axis, level = env_label.split("_", 1)
    with open(M6_SWEEP_PATH) as f:
        d = json.load(f)
    cells = d.get("results", {}).get(axis, {}).get(level, {})
    out: dict[str, float] = {}
    for k in ("regime_agnostic_ppo", "belief_ppo", "oracle_ppo"):
        v = cells.get(k, {})
        if isinstance(v, dict) and "final_return_mean" in v:
            out[k] = float(v["final_return_mean"])
    return out


def _load_refs(env_label: str) -> dict[str, float]:
    if env_label == "e_final":
        return _load_e_final_refs()
    return _load_m6_refs(env_label)


# ---------------------------------------------------------------------------
# Per-cell, per-env config materialisation
# ---------------------------------------------------------------------------


def _materialise_final_config(cell: str, env_label: str, env_yaml: str) -> Path:
    """Compose locked config + env override into a per-cell-env YAML."""
    FINAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = FINAL_CONFIG_DIR / f"{cell}_{env_label}.yaml"
    locked_rel = f"m5r_locked/{cell}.yaml"
    doc: dict[str, Any] = {
        "extends": [locked_rel, env_yaml],
        "experiment_name": f"m5r_final_{cell}_{env_label}",
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return yaml_path


# ---------------------------------------------------------------------------
# Trial execution and stats
# ---------------------------------------------------------------------------


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


def _gap_closed(
    cell_mean: float, floor: float | None, ceiling: float | None
) -> float | None:
    if floor is None or ceiling is None:
        return None
    if not np.isfinite(floor) or not np.isfinite(ceiling) or ceiling == floor:
        return None
    return (cell_mean - floor) / (ceiling - floor)


def _run_one(cell: str, env_label: str, env_yaml: str) -> dict[str, Any]:
    yaml_path = _materialise_final_config(cell, env_label, env_yaml)
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
                f"{cell}/{env_label}: metrics.json missing after train_or_sweep"
            )

    finals = list(map(float, metrics.get("per_seed_final_return", [])))
    mean = float(np.mean(finals)) if finals else float("nan")
    ci_lo, ci_hi = _bootstrap_ci(finals) if finals else (float("nan"), float("nan"))

    return {
        "cell": cell,
        "env_label": env_label,
        "experiment_name": cfg.experiment_name,
        "skipped_train": skipped,
        "elapsed_sec": elapsed,
        "iterations": int(metrics.get("iterations", 0)),
        "num_seeds": int(metrics.get("num_seeds", 0)),
        "per_seed_final_return": finals,
        "final_return_mean": mean,
        "final_return_ci95": [ci_lo, ci_hi],
        "config_path": str(yaml_path.relative_to(REPO_ROOT)),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(per_cell_env: dict[str, dict[str, dict]],
               envs: list[tuple[str, str]]) -> dict[str, Any]:
    """Cross-cell summary per env: gap-closed fractions vs reused refs."""
    out: dict[str, Any] = {"per_env": {}}
    for env_label, _ in envs:
        refs = _load_refs(env_label)
        floor = refs.get("regime_agnostic_ppo")
        oracle = refs.get("oracle_ppo")
        belief = refs.get("belief_ppo")

        env_block: dict[str, Any] = {
            "refs": refs,
            "cells": {},
        }
        for cell in CELLS:
            r = per_cell_env.get(cell, {}).get(env_label)
            if r is None:
                continue
            cm = r["final_return_mean"]
            env_block["cells"][cell] = {
                "final_return_mean": cm,
                "final_return_ci95": r["final_return_ci95"],
                "delta_vs_floor": (cm - floor) if floor is not None else None,
                "gap_closed_vs_oracle": _gap_closed(cm, floor, oracle),
                "gap_closed_vs_belief": _gap_closed(cm, floor, belief),
                "per_seed_final_return": r["per_seed_final_return"],
            }
        out["per_env"][env_label] = env_block
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5r_final_eval")
    parser.add_argument("--cell", type=str, default=None,
                        help="run only this cell (default: all 4)")
    parser.add_argument("--env", type=str, default=None,
                        help="run only this env_label (default: all 5)")
    args = parser.parse_args()

    if args.cell and args.cell not in CELLS:
        raise SystemExit(f"unknown cell: {args.cell!r}; valid: {CELLS}")
    valid_env_labels = [el for el, _ in EVAL_ENVS]
    if args.env and args.env not in valid_env_labels:
        raise SystemExit(f"unknown env: {args.env!r}; valid: {valid_env_labels}")

    cells = [args.cell] if args.cell else CELLS
    envs = [(el, ep) for el, ep in EVAL_ENVS if (args.env is None or el == args.env)]

    # Pre-flight: confirm all locked configs exist before launching anything.
    missing = [c for c in cells if not (LOCKED_CONFIG_DIR / f"{c}.yaml").exists()]
    if missing:
        raise SystemExit(f"missing locked configs (run Stage B first): {missing}")

    FINAL_RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = FINAL_RESULTS / "m5r_final_run.json"
    run = ScriptRun(script="m5r_final_eval", run_mode="full")

    n_total = len(cells) * len(envs)
    print(
        f"[m5r_final] start: {len(cells)} cells × {len(envs)} envs = {n_total} runs",
        flush=True,
    )
    t_start = time.perf_counter()

    per_cell_env: dict[str, dict[str, dict[str, Any]]] = {c: {} for c in cells}
    failed: list[str] = []
    done = 0

    # Env-major: complete one env across all cells before moving on.
    for env_label, env_yaml in envs:
        for cell in cells:
            done += 1
            label = f"{cell}/{env_label}"
            print(f"[m5r_final] ({done}/{n_total}) {label} starting", flush=True)
            t0 = time.perf_counter()
            try:
                r = _run_one(cell, env_label, env_yaml)
            except Exception as e:
                print(f"[m5r_final] {label} FAILED: {e}", flush=True)
                failed.append(label)
                continue
            per_cell_env[cell][env_label] = r
            t_cell = (time.perf_counter() - t0) / 60
            t_total = (time.perf_counter() - t_start) / 60
            print(
                f"[m5r_final] ({done}/{n_total}) {label} done in {t_cell:.1f} min "
                f"| total {t_total:.1f} min "
                f"| mean={r['final_return_mean']:.2f} "
                f"ci=[{r['final_return_ci95'][0]:.2f}, {r['final_return_ci95'][1]:.2f}]",
                flush=True,
            )

    total_min = (time.perf_counter() - t_start) / 60

    aggregated = _aggregate(per_cell_env, envs)
    aggregated["total_min"] = total_min
    aggregated["n_runs"] = n_total
    aggregated["n_failed"] = len(failed)
    aggregated["failed"] = failed

    per_env_path = FINAL_RESULTS / "per_cell_env.json"
    with open(per_env_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    run.add_output(per_env_path)

    if failed:
        run.fail(reason=f"{len(failed)}/{n_total} runs failed: {failed}",
                 summary_path=summary_path)
        return 1

    run.ok(
        key_stats={
            "n_runs": n_total,
            "n_cells": len(cells),
            "n_envs": len(envs),
            "total_min": round(total_min, 2),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
