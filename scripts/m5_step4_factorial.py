"""M5 Step-4 — full-budget 4-cell factorial on MM E_final.

Pipeline:
  1. Sequentially train 4 cells (rl2/varibad × concat/hypernet) at
     full M5 budget (200 iter × 512 envs × 128 rollout × 8 seeds).
  2. After each cell, read metrics.json and per-seed final returns.
  3. Compute primary hypothesis tests (Holm-corrected) and reference
     comparisons against M3 floor / belief / oracle.
  4. Write `results/milestones/M5/stats_M5_step4_ladder.json` with
     per-cell finals, per-cell gap-closed-vs-belief, hypothesis tests,
     leave-one-out sensitivity, and ranking stability.

The 4 cells use the same h64 hyperparameters proved viable in Step 3
(RL²: ent_coef=0.01; VariBAD: kl=0.1, gaussian decoder). Only the
`integration` field varies between the concat/hypernet variants.

Usage:
  uv run python -m scripts.m5_step4_factorial            # full budget (default)
  uv run python -m scripts.m5_step4_factorial --fast     # quick sanity
  uv run python -m scripts.m5_step4_factorial --super-fast  # smoke test
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

from evaluation.metrics import (
    leave_one_out_sensitivity,
    primary_hypothesis_test,
    ranking_stable_across_seeds,
)
from training.config import apply_run_mode, load_config
from training.train import train_or_sweep
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"

# (cell_label, config_yaml_filename). Each cell has a dedicated YAML
# config in experiments/configs/m5_step4_*.yaml, all using h64 / M4-frozen
# hyperparameters with only `integration` varying between concat and hypernet.
CELLS: list[tuple[str, str]] = [
    ("rl2_concat",       "m5_step4_rl2_concat.yaml"),
    ("rl2_hypernet",     "m5_step4_rl2_hypernet.yaml"),
    ("varibad_concat",   "m5_step4_varibad_concat.yaml"),
    ("varibad_hypernet", "m5_step4_varibad_hypernet.yaml"),
]

# Pre-registered primary hypotheses for the 4-cell factorial:
#   (hyp_name, method_cell, baseline_cell)
# `regime_agnostic_ppo` is read from M3 reference for the absolute floor
# tests; the integration main effects are tested as paired hypothesis tests
# between cells with the same belief source.
PRIMARY_HYPOTHESES: list[tuple[str, str, str]] = [
    # Integration main effects (the new factorial axis):
    ("rl2_hypernet_beats_concat",     "rl2_hypernet",     "rl2_concat"),
    ("varibad_hypernet_beats_concat", "varibad_hypernet", "varibad_concat"),
]
# Family size for Holm correction. We'd extend this to include
# rl2/varibad_beats_agnostic + rl2/varibad_beats_stacked_ppo if those
# baselines were re-run at the new budget; for now we keep family_size=4
# (2 in this script + 2 in the broader ladder family per methodology.md).
FAMILY_SIZE = 4
ALPHA = 0.05

M3_REFERENCE_PATH = RESULTS_ROOT / "milestones" / "M3" / "stats_M3_reference_levels.json"


def _bootstrap_ci(values: list[float], n_boot: int = 10_000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _read_metrics(experiment_name: str) -> dict[str, Any]:
    with open(RESULTS_ROOT / experiment_name / "metrics.json") as f:
        return json.load(f)


def _load_m3_refs() -> dict[str, float]:
    if not M3_REFERENCE_PATH.exists():
        return {}
    with open(M3_REFERENCE_PATH) as f:
        d = json.load(f)
    refs = d.get("reference_levels", {})
    return {
        "agnostic": float(refs.get("regime_agnostic_ppo", {}).get("mean", float("nan"))),
        "belief":   float(refs.get("belief_ppo", {}).get("mean", float("nan"))),
        "oracle":   float(refs.get("oracle_ppo", {}).get("mean", float("nan"))),
    }


def _build_cell_cfg(yaml_name: str, mode: str):
    """Load the cell's dedicated config YAML and apply the run-mode flag."""
    cfg = load_config(CONFIG_ROOT / yaml_name)
    apply_run_mode(cfg, mode)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.m5_step4_factorial")
    parser.add_argument("--super-fast", action="store_true",
                        help="2 iter × 16 envs × 1 seed (smoke)")
    parser.add_argument("--fast", action="store_true",
                        help="20 iter × 128 envs × 1 seed")
    parser.add_argument("--mid", action="store_true",
                        help="100 iter × 256 envs × 1 seed")
    args = parser.parse_args()
    if sum([args.super_fast, args.fast, args.mid]) > 1:
        raise SystemExit("--super-fast / --fast / --mid are mutually exclusive")
    mode = (
        "super_fast" if args.super_fast
        else "fast" if args.fast
        else "mid" if args.mid
        else "full"
    )

    run = ScriptRun(script="m5_step4_factorial", run_mode=mode)
    out_dir = RESULTS_ROOT / "milestones" / "M5"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_M5_step4_ladder.json"
    summary_path = out_dir / "stats_M5_step4_ladder_run.json"

    refs = _load_m3_refs()
    print(
        f"[m5_step4] start: {len(CELLS)} cells, mode={mode}, "
        f"floor={refs.get('agnostic', float('nan')):.2f}, "
        f"belief={refs.get('belief', float('nan')):.2f}, "
        f"oracle={refs.get('oracle', float('nan')):.2f}",
        flush=True,
    )
    t_start = time.perf_counter()

    cell_finals: dict[str, list[float]] = {}
    cell_curves: dict[str, list[list[float]]] = {}
    cell_meta: dict[str, dict[str, Any]] = {}
    failed: list[str] = []

    for i, (label, yaml_name) in enumerate(CELLS, start=1):
        cfg = _build_cell_cfg(yaml_name, mode)
        integration = cfg.agent.params.get("integration", "concat")
        print(
            f"[m5_step4] ({i}/{len(CELLS)}) launching {label} "
            f"iters={cfg.iterations} envs={cfg.parallel_envs} seeds={cfg.num_seeds} "
            f"integration={integration}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            train_or_sweep(cfg)
        except SystemExit as e:
            if e.code != 0:
                print(f"[m5_step4] ({i}/{len(CELLS)}) {label} FAILED (exit {e.code})", flush=True)
                failed.append(label)
                continue
        elapsed = time.perf_counter() - t0
        total_min = (time.perf_counter() - t_start) / 60

        m = _read_metrics(cfg.experiment_name)
        cell_finals[label] = list(map(float, m["per_seed_final_return"]))
        cell_curves[label] = list(m["per_seed_mean_return_per_iter"])
        cell_meta[label] = {
            "experiment_name": cfg.experiment_name,
            "iterations": int(m["iterations"]),
            "num_seeds": int(m["num_seeds"]),
            "parallel_envs": int(m["parallel_envs"]),
            "rollout_length": int(m["rollout_length"]),
            "agent": m["agent"],
            "env": m["env"],
            "integration": integration,
        }
        print(
            f"[m5_step4] ({i}/{len(CELLS)}) {label} done in {elapsed/60:.1f} min "
            f"| total {total_min:.1f} min | mean={float(np.mean(cell_finals[label])):.2f}",
            flush=True,
        )

    if failed:
        run.fail(reason=f"{len(failed)}/{len(CELLS)} failed: " + ", ".join(failed),
                 summary_path=summary_path)
        return 1

    # ----- Per-cell summary --------------------------------------------------
    cells_block: dict[str, Any] = {}
    floor_mean = refs.get("agnostic", float("nan"))
    belief_mean = refs.get("belief", float("nan"))
    oracle_mean = refs.get("oracle", float("nan"))

    def _gap_closed(cell_mean: float, floor: float, ceiling: float) -> float | None:
        if np.isnan(floor) or np.isnan(ceiling) or ceiling == floor:
            return None
        return (cell_mean - floor) / (ceiling - floor)

    for label, finals in cell_finals.items():
        mean = float(np.mean(finals))
        ci_lo, ci_hi = _bootstrap_ci(finals)
        cells_block[label] = {
            "experiment_name": cell_meta[label]["experiment_name"],
            "integration": cell_meta[label]["integration"],
            "final_return": {
                "mean": mean,
                "ci": [ci_lo, ci_hi],
                "seed_returns": list(map(float, finals)),
            },
            "delta_vs_floor": (mean - floor_mean) if not np.isnan(floor_mean) else None,
            "gap_closed_vs_oracle": _gap_closed(mean, floor_mean, oracle_mean),
            "gap_closed_vs_belief": _gap_closed(mean, floor_mean, belief_mean),
        }

    ranking_by_return = sorted(
        cell_finals.keys(),
        key=lambda c: cells_block[c]["final_return"]["mean"],
        reverse=True,
    )

    # ----- Primary hypotheses ------------------------------------------------
    primary: dict[str, Any] = {}
    for hyp_name, method, baseline in PRIMARY_HYPOTHESES:
        if method not in cell_finals or baseline not in cell_finals:
            primary[hyp_name] = {"error": "missing cell"}
            continue
        primary[hyp_name] = primary_hypothesis_test(
            method=cell_finals[method],
            baseline=cell_finals[baseline],
            family_size=FAMILY_SIZE,
            alpha=ALPHA,
            n_boot=10_000,
            alternative="greater",
        )

    # ----- Leave-one-out sensitivity -----------------------------------------
    loo: dict[str, Any] = {}
    for hyp_name, method, baseline in PRIMARY_HYPOTHESES:
        if method not in cell_finals or baseline not in cell_finals:
            loo[hyp_name] = {"error": "missing cell"}
            continue
        loo[hyp_name] = leave_one_out_sensitivity(
            method=cell_finals[method],
            baseline=cell_finals[baseline],
            alpha=ALPHA,
            n_corrections=FAMILY_SIZE,
            alternative="greater",
        )

    # ----- Ranking stability across seeds ------------------------------------
    ranking_stability = ranking_stable_across_seeds(cell_finals)

    # ----- Pass summary ------------------------------------------------------
    all_supported = all(
        primary.get(h, {}).get("supported", False) for h, _, _ in PRIMARY_HYPOTHESES
    )
    all_loo_robust = all(
        loo.get(h, {}).get("robust_to_loo", False) for h, _, _ in PRIMARY_HYPOTHESES
    )
    pass_summary = {
        "rl2_hypernet_beats_concat":     primary.get("rl2_hypernet_beats_concat", {}).get("supported"),
        "varibad_hypernet_beats_concat": primary.get("varibad_hypernet_beats_concat", {}).get("supported"),
        "all_primary_supported":         all_supported,
        "all_loo_robust":                all_loo_robust,
        "ranking_stable":                ranking_stability["stable"],
    }

    stats = {
        "env_version": "e_final",
        "alpha": ALPHA,
        "family_size": FAMILY_SIZE,
        "run_mode": mode,
        "reference_levels": {
            "regime_agnostic_ppo_mean": floor_mean,
            "belief_ppo_mean":           belief_mean,
            "oracle_ppo_mean":           oracle_mean,
            "source": str(M3_REFERENCE_PATH),
        },
        "cells": cells_block,
        "ranking_by_return": ranking_by_return,
        "primary_hypotheses": primary,
        "leave_one_out_sensitivity": loo,
        "ranking_stable_across_seeds": ranking_stability,
        "pass_summary": pass_summary,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    total_min = (time.perf_counter() - t_start) / 60
    run.ok(
        key_stats={
            "n_cells": len(cell_finals),
            "elapsed_min": round(total_min, 2),
            "ranking": ranking_by_return,
            "all_primary_supported": bool(all_supported),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )

    # ----- Human-readable summary --------------------------------------------
    print("[m5_step4] === ladder summary ===", flush=True)
    for cell in ranking_by_return:
        info = cells_block[cell]
        gb = info.get("gap_closed_vs_belief")
        gb_str = f"{gb:.2f}" if gb is not None else "n/a"
        print(
            f"[m5_step4] {cell:>20s} | mean={info['final_return']['mean']:7.2f} "
            f"({info['final_return']['ci'][0]:.2f}–{info['final_return']['ci'][1]:.2f}) | "
            f"Δfloor={info['delta_vs_floor']:+.2f} | gap_closed_belief={gb_str}",
            flush=True,
        )
    print(f"[m5_step4] === primary hypotheses (Holm m={FAMILY_SIZE}) ===", flush=True)
    for hyp_name, _, _ in PRIMARY_HYPOTHESES:
        r = primary[hyp_name]
        sup = r.get("supported")
        p = r.get("holm_corrected_p")
        d = r.get("median_paired_delta")
        ci = r.get("delta_ci")
        ci_str = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "n/a"
        d_str = f"{d:.2f}" if d is not None else "n/a"
        print(
            f"[m5_step4] {hyp_name:>32s} | supported={sup} | p_corr={p} "
            f"| delta_median={d_str} | ci={ci_str}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
