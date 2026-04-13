#!/usr/bin/env python3
"""RL² (Meta-RL) — Plotting.

Reads results/rl2.json + results/analytical_foundation.json and produces:
  plots/figure9_rl2_policies.png  — optimal vs RL² action distributions
  plots/figure10_rl2_curves.png   — total episode reward learning curves

Usage:
    uv run python scripts/plot_rl2.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")
RL2_JSON = os.path.join(RESULTS_DIR, "rl2.json")
FOUNDATION_JSON = os.path.join(RESULTS_DIR, "analytical_foundation.json")

REGIME_ORDER = ["noise", "bull", "bear", "mixed"]
REGIME_TITLES = {"noise": "Noise", "bull": "Bull", "bear": "Bear", "mixed": "Mixed"}
ACTION_LABELS = ["sym(1,1)", "ask(1,3)", "bid(3,1)"]
N_ACTIONS = 3
INV_MAX = 5
N_INV = 2 * INV_MAX + 1
INV_GRID = np.arange(N_INV) - INV_MAX


# ---------------------------------------------------------------------------
# Load optimal policies from analytical foundation
# ---------------------------------------------------------------------------

def load_optimal_fracs():
    """Returns optimal action fracs: dict name -> (N_ACTIONS, N_INV) array."""
    with open(FOUNDATION_JSON) as f:
        foundation = json.load(f)

    locked = foundation["locked_policies"]
    optimal = {}
    per_regime = {}
    for name in ["noise", "bull", "bear"]:
        policy = np.array(locked[name.capitalize()])
        grid = np.zeros((N_ACTIONS, N_INV))
        for qi in range(N_INV):
            grid[int(policy[qi]), qi] = 1.0
        optimal[name] = grid
        per_regime[name] = grid

    optimal["mixed"] = np.mean(
        [per_regime["noise"], per_regime["bull"], per_regime["bear"]], axis=0)
    return optimal


# ---------------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------------

def draw_heatmap(ax, grid, title, show_ylabel=False):
    """Draw a single action-distribution heatmap on the given axes."""
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    ax.set_xticks(range(N_INV))
    ax.set_xticklabels(INV_GRID, fontsize=7)
    ax.set_xlabel("Inventory q", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_yticks(range(N_ACTIONS))
    if show_ylabel:
        ax.set_yticklabels(ACTION_LABELS, fontsize=9)
        ax.set_ylabel("Action", fontsize=9)
    else:
        ax.set_yticklabels([], fontsize=9)

    for a in range(N_ACTIONS):
        for qi in range(N_INV):
            val = grid[a, qi]
            color = "white" if val > 0.5 else "black"
            ax.text(qi, a, f"{val:.0%}", ha="center", va="center",
                    fontsize=6, fontweight="bold", color=color)


# ---------------------------------------------------------------------------
# Policies figure — Optimal vs RL² (3 rows × 3 cols)
# ---------------------------------------------------------------------------

def plot_rl2_policies(rl2_data, optimal):
    """3×3 grid: optimal | RL² isolated | RL² mixed (split by regime)."""
    regime_cols = ["noise", "bull", "bear"]
    regime_ids = {"noise": 0, "bull": 1, "bear": 2}
    n_cols = len(regime_cols)

    fig, axes = plt.subplots(3, n_cols, figsize=(4.0 * n_cols, 8.5), sharey=True)

    row_labels = ["Optimal", "RL² Isolated", "RL² Mixed"]

    for i, name in enumerate(regime_cols):
        rid = regime_ids[name]

        # Row 0: optimal policy
        draw_heatmap(axes[0, i], optimal[name],
                     f"Optimal — {REGIME_TITLES[name]}", show_ylabel=(i == 0))

        # Row 1: RL² isolated
        if name in rl2_data:
            iso_fracs = np.array(rl2_data[name]["action_fracs"])
            iso_grid = iso_fracs[rid].T
        else:
            iso_grid = np.zeros((N_ACTIONS, N_INV))
        draw_heatmap(axes[1, i], iso_grid,
                     f"RL² Isolated — {REGIME_TITLES[name]}",
                     show_ylabel=(i == 0))

        # Row 2: RL² mixed, split by regime
        if "mixed" in rl2_data:
            mix_fracs = np.array(rl2_data["mixed"]["action_fracs"])
            mix_grid = mix_fracs[rid].T
        else:
            mix_grid = np.zeros((N_ACTIONS, N_INV))
        draw_heatmap(axes[2, i], mix_grid,
                     f"RL² Mixed — {REGIME_TITLES[name]}",
                     show_ylabel=(i == 0))

    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label, xy=(-0.45, 0.5), xycoords="axes fraction",
            fontsize=11, fontweight="bold", ha="right", va="center", rotation=90)

    fig.suptitle(
        "Per-Regime Action Distributions: Optimal vs RL² Isolated vs RL² Mixed",
        fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "figure9_rl2_policies.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Learning curves figure
# ---------------------------------------------------------------------------

def load_oracle_bounds():
    """Load oracle mean episode returns from analytical_foundation.json."""
    with open(FOUNDATION_JSON) as f:
        foundation = json.load(f)
    return foundation.get("oracle_bounds", {})


def _draw_learning_curve(ax, runs, regime_name, oracle_bounds,
                         label_prefix="RL²"):
    """Draw agent learning curve + oracle band on a single axes."""
    all_iters = []
    all_returns = []
    for run in runs:
        all_iters.append(np.array(run["iters"]))
        prr = run.get("per_regime_returns", {})
        if regime_name in prr:
            all_returns.append(np.array(prr[regime_name]))
        else:
            all_returns.append(np.array(run["mean_returns"]))

    common_iters = all_iters[0]
    matrix = np.array(all_returns)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)

    ax.fill_between(common_iters, mean - std, mean + std,
                    alpha=0.2, color="C0")
    ax.plot(common_iters, mean, color="C0", linewidth=1.5,
            label=f"{label_prefix} ({len(runs)} seeds)")

    if regime_name in oracle_bounds:
        ob = oracle_bounds[regime_name]
        oracle_mean = ob["mean"]
        oracle_std = ob["std"]
        ax.axhline(oracle_mean, color="red", linestyle="--", linewidth=1.5)
        ax.axhspan(oracle_mean - oracle_std, oracle_mean + oracle_std,
                   color="red", alpha=0.08)
        ax.text(common_iters[-1], oracle_mean, f" {oracle_mean:.1f}",
                va="bottom", ha="right", fontsize=7, color="red",
                fontweight="bold")

    ax.set_xlabel("Iteration")
    ax.set_title(REGIME_TITLES[regime_name], fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")


def plot_rl2_curves(rl2_data, oracle_bounds):
    """2×3 grid: top row = RL² isolated per regime, bottom = RL² mixed."""
    regime_cols = ["noise", "bull", "bear"]
    has_isolated = any(r in rl2_data for r in regime_cols)
    has_mixed = "mixed" in rl2_data
    n_rows = int(has_isolated) + int(has_mixed)

    if n_rows == 0:
        print("  (no data to plot for RL² curves)")
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.5 * n_rows), squeeze=False)
    row = 0

    if has_isolated:
        for i, name in enumerate(regime_cols):
            ax = axes[row, i]
            if name in rl2_data:
                _draw_learning_curve(ax, rl2_data[name]["runs"], name,
                                     oracle_bounds,
                                     label_prefix="RL² Isolated")
            else:
                ax.set_visible(False)
            if i == 0:
                ax.set_ylabel("Mean episode return")
        axes[row, 0].annotate(
            "Isolated", xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=11, fontweight="bold", ha="right", va="center", rotation=90)
        row += 1

    if has_mixed:
        mixed_runs = rl2_data["mixed"]["runs"]
        for i, name in enumerate(regime_cols):
            ax = axes[row, i]
            _draw_learning_curve(ax, mixed_runs, name,
                                 oracle_bounds, label_prefix="RL² Mixed")
            if i == 0:
                ax.set_ylabel("Mean episode return")
        axes[row, 0].annotate(
            "Mixed", xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=11, fontweight="bold", ha="right", va="center", rotation=90)

    for r in range(n_rows):
        for c in range(3):
            h, l = axes[r, c].get_legend_handles_labels()
            if h:
                axes[r, c].legend(fontsize=7, loc="lower right")

    fig.suptitle("RL² — Learning Curves (per regime)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "figure10_rl2_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  RL² (Meta-RL) — Plotting")
    print("=" * 60)

    for path, label in [(RL2_JSON, "rl2.json"),
                        (FOUNDATION_JSON, "analytical_foundation.json")]:
        if not os.path.exists(path):
            print(f"\n  ERROR: {label} not found.")
            print(f"  Run the corresponding training script first.")
            sys.exit(1)

    with open(RL2_JSON) as f:
        rl2_data = json.load(f)

    optimal = load_optimal_fracs()
    oracle_bounds = load_oracle_bounds()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n  Plotting RL² policies (optimal vs RL² action distributions) ...")
    plot_rl2_policies(rl2_data, optimal)

    print("  Plotting RL² learning curves ...")
    plot_rl2_curves(rl2_data, oracle_bounds)

    print("\n  Done.")


if __name__ == "__main__":
    main()
