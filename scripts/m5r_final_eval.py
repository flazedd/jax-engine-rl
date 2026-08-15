"""Final evaluation of the four conditioning variants on the medium env.

Trains RL² × {concat, hypernet} and VariBAD × {concat, hypernet} from the
matched-fairness configs on the medium-difficulty environment, at the budget
those configs declare. The difficulty levels are owned by
`scripts.sweep_redesign_n20`, which trains their references and cells from the
same configs, so each environment label has exactly one writer.

References (regime-agnostic / Belief-PPO / Oracle-PPO / stacked-obs) are not
re-trained here: the programme driver trains them from the matched configs and
this script reads their metrics to form the gap-closed denominators. Because
they come from the same matched family, the denominators sit on the same
inputs, optimiser settings, budget and capacity as the cells they normalise.

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

from utils.paths import analysis_dir, experiment_dir
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
FINAL_RESULTS = analysis_dir()

CELLS = ["rl2_concat", "rl2_hypernet", "varibad_concat", "varibad_hypernet"]

# (env_label, env_yaml_relative_to_CONFIG_ROOT). Order is env-major: outer
# loop runs all 4 cells on one env before moving to the next, so partial
# results give a complete cross-section per env.
# The medium-difficulty environment only. The difficulty levels are owned by
# `scripts.sweep_redesign_n20`, which trains their references and cells from
# the same matched configs and merges them into the same per_cell_env.json.
# Splitting them this way keeps one writer per environment label.
EVAL_ENVS: list[tuple[str, str]] = [
    ("e9", "envs/e9_rare_fills.yaml"),
]

# Matched-fairness configs: identical inputs, optimiser settings, budget and
# capacity across methods, enforced by `scripts.config_fairness_audit`.
MATCHED_CONFIG_DIR = CONFIG_ROOT / "m5r_e9"

# Reference experiments trained from the matched configs by the programme
# driver. Reading their metrics directly, rather than a cached milestone stats
# file, keeps the denominators on the same inputs, optimiser settings, budget
# and capacity as the cells they normalise.
MATCHED_REF_EXPERIMENTS = {
    "regime_agnostic_ppo": "m5r_ref_regime_agnostic_e9",
    "belief_ppo": "m5r_ref_belief_e9",
    "oracle_ppo": "m5r_ref_oracle_e9",
    "stacked_obs_ppo": "m5r_ref_stacked_obs_e9",
}


# ---------------------------------------------------------------------------
# Reference loading (no retraining; reads the matched reference runs)
# ---------------------------------------------------------------------------


def _load_refs(env_label: str) -> dict[str, float]:
    """Mean final return of each reference on the medium-difficulty env."""
    del env_label  # only the medium instance is evaluated; see EVAL_ENVS.
    out: dict[str, float] = {}
    for key, experiment in MATCHED_REF_EXPERIMENTS.items():
        m = _read_metrics(experiment)
        if m and m.get("per_seed_final_return"):
            out[key] = float(np.mean(m["per_seed_final_return"]))
    return out


# ---------------------------------------------------------------------------
# Per-cell, per-env config materialisation
# ---------------------------------------------------------------------------


def _materialise_final_config(cell: str, env_label: str, env_yaml: str) -> Path:
    """Compose matched config + env override into a per-cell-env YAML.

    The matched directory is read from MATCHED_CONFIG_DIR rather than written
    out here. It used to be hardcoded to `m5r_matched`, so repointing that
    constant at a new config set had no effect: this composed the new
    environment with the *old* budget and retrained over completed runs at the
    wrong iteration count.
    """
    FINAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = FINAL_CONFIG_DIR / f"{cell}_{env_label}.yaml"
    matched_rel = f"{MATCHED_CONFIG_DIR.name}/{cell}.yaml"
    doc: dict[str, Any] = {
        "extends": [matched_rel, env_yaml],
        "experiment_name": f"m5r_final_{cell}_{env_label}",
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    return yaml_path


# ---------------------------------------------------------------------------
# Trial execution and stats
# ---------------------------------------------------------------------------


def _read_metrics(experiment_name: str) -> dict[str, Any] | None:
    p = experiment_dir(experiment_name) / "metrics.json"
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


_ALLOW_TRAIN: bool = False
_ITER_OVERRIDE: int | None = None
_SEED_OVERRIDE: int | None = None


def _run_one(cell: str, env_label: str, env_yaml: str) -> dict[str, Any]:
    yaml_path = _materialise_final_config(cell, env_label, env_yaml)
    cfg = load_config(yaml_path)
    apply_run_mode(cfg, "full")
    if _ITER_OVERRIDE is not None:
        cfg.iterations = _ITER_OVERRIDE
    if _SEED_OVERRIDE is not None:
        cfg.num_seeds = _SEED_OVERRIDE

    existing = _read_metrics(cfg.experiment_name)
    # Budget-aware skip: only resume a cell already trained at the *target*
    # budget, so an older lower-budget metrics.json is correctly retrained.
    skipped = (
        existing is not None
        and int(existing.get("num_seeds", 0)) == cfg.num_seeds
        and int(existing.get("iterations", 0)) == cfg.iterations
    )
    if skipped:
        metrics = existing
        elapsed = float("nan")
    else:
        # Retraining is opt-in. The default used to be to train silently on any
        # mismatch, which overwrote a finished 1500-iteration run with a
        # 600-iteration one because the budget came from the wrong config set.
        # An analysis stage must not be able to destroy training data by
        # default; a mismatch is now a loud failure unless --train is passed.
        if not _ALLOW_TRAIN:
            have = (f"{existing.get('num_seeds')} seeds x "
                    f"{existing.get('iterations')} iters" if existing else "nothing")
            raise SystemExit(
                f"[m5r_final_eval] FAIL | reason=budget_mismatch | cell={cell} "
                f"| want={cfg.num_seeds} seeds x {cfg.iterations} iters | have={have} "
                f"| refusing to retrain over {cfg.experiment_name}; pass --train to allow"
            )
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
    parser.add_argument("--train", action="store_true",
                        help="allow retraining a cell whose budget does not match "
                             "(off by default: an analysis must not overwrite runs)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="override training iterations for every cell")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="override number of seeds for every cell")
    args = parser.parse_args()
    global _ITER_OVERRIDE, _SEED_OVERRIDE, _ALLOW_TRAIN
    _ALLOW_TRAIN = args.train
    _ITER_OVERRIDE = args.iterations
    _SEED_OVERRIDE = args.num_seeds

    if args.cell and args.cell not in CELLS:
        raise SystemExit(f"unknown cell: {args.cell!r}; valid: {CELLS}")
    valid_env_labels = [el for el, _ in EVAL_ENVS]
    if args.env and args.env not in valid_env_labels:
        raise SystemExit(f"unknown env: {args.env!r}; valid: {valid_env_labels}")

    cells = [args.cell] if args.cell else CELLS
    envs = [(el, ep) for el, ep in EVAL_ENVS if (args.env is None or el == args.env)]

    # Pre-flight: confirm all matched configs exist before launching anything.
    missing = [c for c in cells if not (MATCHED_CONFIG_DIR / f"{c}.yaml").exists()]
    if missing:
        raise SystemExit(f"missing matched configs: {missing}")

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
