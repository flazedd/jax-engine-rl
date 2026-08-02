"""Overnight runner for the 3-simplex-belief ablation (RQ2 mechanism probe).

One command trains the four m5r_simplex cells (RL2 / VariBAD x concat, with a
learned 3-simplex belief head, supervised and pure-RL), then probes each cell's
belief (KL to the analytical posterior, accuracy, log-loss) and reports return
against the regime-agnostic floor and the Belief-PPO ceiling on the medium env.

Tests whether PPO+concat drives a clean, low-dimensional LEARNED belief to the
ceiling: if the supervised (clean, low-KL) cell reaches the ceiling, the concat
failure was the representation; if it still stalls, the cause is deeper.

Usage:
  uv run python -m scripts.m5r_simplex_run                      # full overnight run
  uv run python -m scripts.m5r_simplex_run --run-mode super_fast  # fast pipeline check
  uv run python -m scripts.m5r_simplex_run --skip-train         # probe already-trained cells

Output: results/M5R/final/m5r_simplex_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from training.config import apply_run_mode, load_config
from training.train import train_or_sweep
from evaluation.posterior_probe import load_experiment, probe_one_seed
from utils.script_output import ScriptRun

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CFG_DIR = REPO / "experiments" / "configs" / "m5r_simplex"
PER_CELL_ENV = RESULTS / "M5R" / "final" / "per_cell_env.json"

# (config stem, saved experiment_name)
CELLS = [
    ("rl2_concat_simplex_sup", "m5r_simplex_rl2_concat_simplex_sup"),
    ("rl2_concat_simplex_rl", "m5r_simplex_rl2_concat_simplex_rl"),
    ("varibad_concat_simplex_sup", "m5r_simplex_varibad_concat_simplex_sup"),
    ("varibad_concat_simplex_rl", "m5r_simplex_varibad_concat_simplex_rl"),
]


def _refs() -> tuple[float, float, float]:
    d = json.load(open(PER_CELL_ENV))
    r = d["per_env"]["e_final"]["refs"]
    return r["regime_agnostic_ppo"], r["belief_ppo"], r["oracle_ppo"]


def _probe_cell(exp_name: str, n_rollouts: int, rollout_length: int) -> dict[str, Any]:
    exp_dir = RESULTS / exp_name
    seeds = [int(p.stem.split("_")[-1]) for p in sorted(exp_dir.glob("checkpoint_seed_*.pkl"))]
    kl, acc, ll = [], [], []
    for seed in seeds:
        bundle = load_experiment(exp_dir, seed)
        r = probe_one_seed(
            bundle, n_rollouts=n_rollouts, rollout_length=rollout_length,
            classifier="logistic", rng_key=seed,
        )
        kl.append(float(r["method"]["test_kl_to_omega"]))
        acc.append(float(r["method"]["test_acc"]))
        ll.append(float(r["method"]["test_log_loss"]))
    return {"seeds": seeds, "kl": kl, "acc": acc, "log_loss": ll}


def main() -> int:
    p = argparse.ArgumentParser(prog="scripts.m5r_simplex_run")
    p.add_argument("--skip-train", action="store_true", help="probe already-trained cells only")
    p.add_argument("--n-rollouts", type=int, default=200)
    p.add_argument("--rollout-length", type=int, default=128)
    p.add_argument("--run-mode", default="full",
                   help="full | mid | fast | super_fast (super_fast = pipeline check)")
    args = p.parse_args()

    run = ScriptRun(script="m5r_simplex_run")
    out = RESULTS / "M5R" / "final" / "m5r_simplex_ablation.json"
    summary_path = out.with_name(out.stem + "_run.json")
    t0 = time.perf_counter()

    # 1. Train the four cells (each trains all its seeds and saves checkpoints).
    if not args.skip_train:
        for cfg_name, exp_name in CELLS:
            cfg = apply_run_mode(load_config(str(CFG_DIR / f"{cfg_name}.yaml")), args.run_mode)
            print(
                f"[simplex] TRAIN {exp_name} | seeds={cfg.num_seeds} iters={cfg.iterations}",
                flush=True,
            )
            train_or_sweep(cfg)

    # 2. References and per-cell probe.
    floor, ceiling, oracle = _refs()
    cells: dict[str, Any] = {}
    for cfg_name, exp_name in CELLS:
        mpath = RESULTS / exp_name / "metrics.json"
        if not mpath.exists():
            print(f"[simplex] {exp_name}: no metrics.json, skipping", flush=True)
            continue
        rets = json.load(open(mpath))["per_seed_final_return"]
        ret_mean = float(np.mean(rets))
        gap_closed = (ret_mean - floor) / (oracle - floor) if oracle != floor else float("nan")
        probe = _probe_cell(exp_name, args.n_rollouts, args.rollout_length)
        kl_mean = float(np.mean(probe["kl"])) if probe["kl"] else float("nan")
        acc_mean = float(np.mean(probe["acc"])) if probe["acc"] else float("nan")
        reaches_ceiling = ret_mean >= ceiling - 3.0
        cells[cfg_name] = {
            "experiment_name": exp_name,
            "return_mean": ret_mean,
            "return_per_seed": rets,
            "gap_closed": gap_closed,
            "reaches_ceiling": bool(reaches_ceiling),
            "belief_kl_mean": kl_mean,
            "belief_acc_mean": acc_mean,
            "belief_log_loss_mean": float(np.mean(probe["log_loss"])) if probe["log_loss"] else float("nan"),
            "n_seeds": len(probe["seeds"]),
        }
        print(
            f"[simplex] {cfg_name:34s} return={ret_mean:7.1f} gap_closed={gap_closed:+.2f} "
            f"KL={kl_mean:.3f} acc={acc_mean:.3f} -> "
            f"{'CEILING' if reaches_ceiling else 'below ceiling'}",
            flush=True,
        )

    summary = {
        "refs": {"floor": floor, "belief_ceiling": ceiling, "oracle": oracle},
        "n_rollouts": args.n_rollouts,
        "rollout_length": args.rollout_length,
        "run_mode": args.run_mode,
        "cells": cells,
        "elapsed_min": round((time.perf_counter() - t0) / 60, 1),
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    run.add_output(str(out))

    if not cells:
        run.fail(reason="no cells produced results", summary_path=summary_path)
        return 1

    print("[simplex] === interpretation ===", flush=True)
    for name, c in cells.items():
        clean = c["belief_kl_mean"] < 0.15  # near the analytical residual
        verdict = (
            "representation was the cause" if (clean and c["reaches_ceiling"])
            else "cause is deeper than representation" if (clean and not c["reaches_ceiling"])
            else "belief did not self-calibrate" if not clean
            else ""
        )
        print(f"[simplex]   {name}: {'clean' if clean else 'dirty'} belief, "
              f"{'ceiling' if c['reaches_ceiling'] else 'stall'} -> {verdict}", flush=True)

    run.ok(
        key_stats={"n_cells": len(cells), "elapsed_min": summary["elapsed_min"]},
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
