#!/usr/bin/env python3
"""Analytical Foundation — Per-Regime Optimal Policies.

Solves the locked-regime MDP for each regime via value iteration,
verifies that optimal policies are distinct across regimes, and
plots per-regime optimal policy heatmaps.

Outputs:
  results/analytical_foundation.json  — policy data + gate status
  plots/figure1_optimal_policies.png  — per-regime policy heatmaps

Usage:
    uv run python scripts/analytical_foundation.py [--fast]
"""
import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lob_sim.jax_env import EnvParams
from lob_sim.analytical_mdp import (
    N_REGIMES, N_ACTIONS, REGIME_NAMES,
    build_mdp_tables, solve_all_locked,
    compute_q_max, precondition_1,
)

RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")

ACTION_LABELS = ["sym(1,1)", "ask(1,3)", "bid(3,1)"]


# ---------------------------------------------------------------------------
# Figure — Per-Regime Optimal Policy Heatmaps
# ---------------------------------------------------------------------------

def plot_optimal_policies(locked_solutions, q_max, params, pc1):
    """3 heatmaps (one per regime): x=inventory, y=action, cell=optimal %."""
    inv_max = int(params.inventory_max)
    n_inv = 2 * inv_max + 1
    inv_grid = np.arange(n_inv) - inv_max

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5), sharey=True)

    for r in range(N_REGIMES):
        ax = axes[r]
        policy = np.array(locked_solutions[r].policy)

        grid = np.zeros((N_ACTIONS, n_inv))
        for qi in range(n_inv):
            grid[int(policy[qi]), qi] = 1.0

        ax.imshow(grid, cmap="Blues", vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
        ax.set_xticks(range(n_inv))
        ax.set_xticklabels(inv_grid, fontsize=8)
        ax.set_xlabel("Inventory q")
        ax.set_title(REGIME_NAMES[r], fontsize=12, fontweight="bold")

        if r == 0:
            ax.set_yticks(range(N_ACTIONS))
            ax.set_yticklabels(ACTION_LABELS, fontsize=10)
            ax.set_ylabel("Action")
        else:
            ax.set_yticks(range(N_ACTIONS))

        for a in range(N_ACTIONS):
            for qi in range(n_inv):
                val = grid[a, qi]
                color = "white" if val > 0.5 else "lightgray"
                ax.text(qi, a, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)

    pairs = [(0, 1), (0, 2), (1, 2)]
    disag = [f"{REGIME_NAMES[r1]}–{REGIME_NAMES[r2]}: "
             f"{pc1['pairwise_disagreement'][(r1, r2)]:.0f}%"
             for r1, r2 in pairs]

    status = "PASS" if pc1["passed"] else "FAIL"
    scolor = "green" if pc1["passed"] else "red"
    fig.suptitle(
        f"Per-Regime Optimal Policies   [{status}]"
        f"      disagreement: {' | '.join(disag)}",
        fontsize=11, fontweight="bold", color=scolor, y=1.04)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "figure1_optimal_policies.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(pc1, q_max, locked_solutions):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def jsonify(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

    def to_list(x):
        return x.tolist() if hasattr(x, "tolist") else x

    result = {
        "q_max": jsonify(q_max),
        "precondition": {
            "gap_per_regime": {
                REGIME_NAMES[r]: {
                    "mean": jsonify(pc1["gap_per_regime"][r]["mean"]),
                    "min": jsonify(pc1["gap_per_regime"][r]["min"]),
                    "values": to_list(pc1["gap_per_regime"][r]["values"]),
                }
                for r in range(N_REGIMES)
            },
            "pairwise_disagreement": {
                f"{REGIME_NAMES[r1]}_vs_{REGIME_NAMES[r2]}": jsonify(
                    pc1["pairwise_disagreement"][(r1, r2)])
                for r1, r2 in pc1["pairwise_disagreement"]
            },
            "pass_disagreement": bool(pc1["pass_disagreement"]),
            "pass_gap": bool(pc1["pass_gap"]),
            "passed": bool(pc1["passed"]),
        },
        "locked_policies": {
            REGIME_NAMES[r]: to_list(np.array(locked_solutions[r].policy))
            for r in range(N_REGIMES)
        },
        "gate": {"passed": bool(pc1["passed"])},
    }

    path = os.path.join(RESULTS_DIR, "analytical_foundation.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  -> {os.path.relpath(path, ROOT)}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="(Accepted for compatibility, no effect)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Analytical Foundation — Per-Regime Optimal Policies")
    print("=" * 60)

    params = EnvParams.default()

    print("\n  Building MDP tables ...")
    t0 = time.time()
    tables = build_mdp_tables(params)
    print(f"  {(time.time()-t0)*1000:.0f} ms")

    print("  Solving locked-regime VI ...")
    t0 = time.time()
    locked = solve_all_locked(tables, params)
    q_max = compute_q_max(locked)
    print(f"  {(time.time()-t0)*1000:.0f} ms  Q_max={q_max:.4f}")

    print("\n  Gate check — policies distinct and decisive ...")
    pc1 = precondition_1(locked, q_max)
    for r in range(N_REGIMES):
        g = pc1["gap_per_regime"][r]
        print(f"    {REGIME_NAMES[r]:>5s}: mean gap={g['mean']:.1f}%, min={g['min']:.1f}%")
    for (r1, r2), d in pc1["pairwise_disagreement"].items():
        print(f"    {REGIME_NAMES[r1]} vs {REGIME_NAMES[r2]}: {d:.1f}% disagree")

    status = "PASS" if pc1["passed"] else "FAIL"
    print(f"\n  GATE: {status}")
    if not pc1["passed"]:
        print("  WARNING: Gate failed — retune parameters before proceeding!")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n  Saving results ...")
    save_results(pc1, q_max, locked)

    print("  Plotting ...")
    plot_optimal_policies(locked, q_max, params, pc1)

    print("\n  Done.")


if __name__ == "__main__":
    main()
