"""Cartpole — where the learning curves flatten.

The matched budget carries over from market making, and a budget that suits one
environment need not suit another: at 600 iterations every cartpole reference
arm was still gaining. Reporting a reference level that has not converged
understates every arm by an unknown and possibly unequal amount, so the budget
is measured rather than assumed.

Trains each arm once at a long budget and reports, at a series of cut points,
how far below its converged level each arm still sits: the windowed mean return
at the cut against the windowed mean at the end of the probe. The recommended
budget is the earliest cut from which every arm stays within `--tol` of its
converged level.

Shortfall rather than trailing drift decides it. Drift is a difference of two
windows and at one seed it oscillates by more than the tolerance long after the
curve is flat, which makes any rule over it pick the last cut on offer.
Shortfall accumulates those oscillations instead of amplifying them.

Runs at one seed by default: the question is the shape of the curve, not the
seed spread, and the 20-seed run comes after.

Usage:
  uv run python -m scripts.cartpole_plateau_probe
  uv run python -m scripts.cartpole_plateau_probe --iterations 4000 --tol 0.15
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from training.config import load_config
from training.train import train_or_sweep
from utils.paths import cartpole_dir, experiment_dir
from utils.script_output import ScriptRun, print_table

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"

# The reference triple plus the memory baseline. The meta-RL arms are probed
# separately: they carry a recurrent or variational component whose curve need
# not flatten where a feedforward policy's does.
ARMS: dict[str, str] = {
    "regime_agnostic": "m_cartpole_matched/regime_agnostic.yaml",
    "stacked_obs": "m_cartpole_matched/stacked_obs.yaml",
    "belief": "m_cartpole_matched/belief_ppo.yaml",
    "oracle": "m_cartpole_matched/oracle_ppo.yaml",
}

# Trailing-drift window, in iterations. One window against the one before it.
WINDOW = 100


def _train(arm: str, iterations: int, seeds: int, reuse: bool) -> np.ndarray | None:
    cfg = load_config(CONFIG_ROOT / ARMS[arm])
    cfg.iterations = iterations
    cfg.num_seeds = seeds
    # A separate name so the probe never writes into the directories the
    # reported runs use; those must hold the production budget only.
    cfg.experiment_name = f"m_cartpole_plateau_{arm}"
    if reuse:
        cached = experiment_dir(cfg.experiment_name) / "metrics.json"
        if cached.exists():
            with open(cached) as f:
                m = json.load(f)
            if m.get("iterations", 0) >= iterations and m.get("num_seeds", 0) >= seeds:
                print(f"[cp_plateau] reuse {cfg.experiment_name}", flush=True)
                return np.asarray(m["mean_return_per_iter"], dtype=float)
    print(
        f"[cp_plateau] launching {cfg.experiment_name} | iters={cfg.iterations} "
        f"envs={cfg.parallel_envs} seeds={cfg.num_seeds}",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        train_or_sweep(cfg)
    except SystemExit as e:
        if e.code != 0:
            print(f"[cp_plateau] {cfg.experiment_name} FAILED (exit {e.code})", flush=True)
            return None
    metrics_path = experiment_dir(cfg.experiment_name) / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        m = json.load(f)
    print(
        f"[cp_plateau] {cfg.experiment_name} done in "
        f"{(time.perf_counter() - t0) / 60:.1f} min | "
        f"final={float(m['final_return_mean']):.2f}",
        flush=True,
    )
    return np.asarray(m["mean_return_per_iter"], dtype=float)


def _level_at(curve: np.ndarray, cut: int) -> float:
    """Mean return over the WINDOW iterations ending at `cut`. A window rather
    than the single value at `cut`, which is one noisy iteration."""
    return float(curve[cut - WINDOW:cut].mean())


def main() -> int:
    parser = argparse.ArgumentParser(prog="scripts.cartpole_plateau_probe")
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--tol", type=float, default=0.20,
                        help="return below the converged level counted as flat")
    parser.add_argument("--reuse", action="store_true",
                        help="read an existing probe run instead of retraining")
    args = parser.parse_args()

    run = ScriptRun(script="cartpole_plateau_probe", run_mode="full")
    out_dir = cartpole_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats_cartpole_plateau_probe.json"
    summary_path = out_dir / "stats_cartpole_plateau_probe_run.json"

    curves: dict[str, np.ndarray] = {}
    failed: list[str] = []
    for arm in args.arms:
        c = _train(arm, args.iterations, args.seeds, args.reuse)
        if c is None:
            failed.append(arm)
            continue
        curves[arm] = c

    if failed:
        run.fail(reason="arms failed: " + ", ".join(failed), summary_path=summary_path)
        return 1

    cuts = list(range(WINDOW, args.iterations + 1, WINDOW))
    level = {arm: {cut: _level_at(c, cut) for cut in cuts} for arm, c in curves.items()}
    # Shortfall against the arm's own converged level, the last cut probed.
    shortfall = {
        arm: {cut: level[arm][cut] - level[arm][cuts[-1]] for cut in cuts}
        for arm in curves
    }

    # Earliest cut from which no arm sits more than tol *below* its converged
    # level. One-sided on purpose: a window that lands above the converged level
    # is plateau noise, not a reason to train longer, and penalising it would
    # push the recommendation to the last cut probed however flat the curve is.
    # `tol` must therefore exceed the plateau's own oscillation, which
    # `plateau_noise` reports so the choice can be checked rather than trusted.
    recommended = None
    for i, cut in enumerate(cuts):
        if all(
            all(shortfall[arm][later] >= -args.tol for later in cuts[i:])
            for arm in curves
        ):
            recommended = cut
            break

    stats = {
        "instance": "wide (envs/e_cartpole_v1_wide.yaml)",
        "iterations_probed": args.iterations,
        "seeds": args.seeds,
        "window": WINDOW,
        "tol": args.tol,
        "recommended_iterations": recommended,
        "final_return": {arm: float(c[-1]) for arm, c in curves.items()},
        "windowed_level": {
            arm: {str(cut): level[arm][cut] for cut in cuts} for arm in curves
        },
        "shortfall_vs_converged": {
            arm: {str(cut): shortfall[arm][cut] for cut in cuts} for arm in curves
        },
        # Oscillation of the windowed level over the last third of the probe,
        # the floor below which no tolerance is meaningful at this seed count.
        "plateau_noise": {
            arm: float(np.std([level[arm][c] for c in cuts[-len(cuts) // 3:]]))
            for arm in curves
        },
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    run.add_output(str(stats_path))

    shown = [c for c in cuts if c % 400 == 0 or c == cuts[-1]]
    print_table(
        ["arm"] + [str(c) for c in shown] + ["noise SD"],
        [
            [arm] + [f"{shortfall[arm][c]:+.2f}" for c in shown]
            + [f"{stats['plateau_noise'][arm]:.2f}"]
            for arm in curves
        ],
        title="Return below the converged level, by iteration budget",
        note=f"tolerance {args.tol:+.2f}; recommended budget "
             f"{recommended if recommended else 'not reached'}. A budget is "
             f"only flat where the shortfall clears the arm's own noise SD.",
    )

    run.ok(
        key_stats={
            "recommended_iterations": recommended,
            "tol": args.tol,
            "iterations_probed": args.iterations,
            "arms": list(curves),
            "stats_path": str(stats_path),
        },
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
