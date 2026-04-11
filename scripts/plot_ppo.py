#!/usr/bin/env python3
"""PPO MLP Baseline — Plotting.

Reads results/ppo_baseline.json + results/analytical_foundation.json and produces:
  plots/figure2_ppo_policies.png  — optimal vs PPO action distributions (2×4)
  plots/figure3_ppo_curves.png    — total episode reward learning curves

Usage:
    uv run python scripts/plot_ppo.py
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
PPO_JSON = os.path.join(RESULTS_DIR, "ppo_baseline.json")
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
    # Keys are capitalised: "Noise", "Bull", "Bear"
    optimal = {}
    per_regime = {}
    for name in ["noise", "bull", "bear"]:
        policy = np.array(locked[name.capitalize()])  # (N_INV,) optimal action indices
        grid = np.zeros((N_ACTIONS, N_INV))
        for qi in range(N_INV):
            grid[int(policy[qi]), qi] = 1.0
        optimal[name] = grid
        per_regime[name] = grid

    # Mixed optimal: oracle knows the true regime → average of 3 optimal policies
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
# Figure 2 — Optimal vs PPO Action Distributions (2 rows × 4 cols)
# ---------------------------------------------------------------------------

def plot_figure2(ppo_data, optimal):
    """Top row: optimal policies. Bottom row: PPO learned policies."""
    present = [r for r in REGIME_ORDER if r in ppo_data]
    n_cols = len(present)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.0 * n_cols, 6.0), sharey=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for i, name in enumerate(present):
        # Top row: optimal
        draw_heatmap(axes[0, i], optimal[name],
                     f"Optimal — {REGIME_TITLES[name]}", show_ylabel=(i == 0))

        # Bottom row: PPO
        fracs = np.array(ppo_data[name]["action_fracs"])  # (3, N_INV, N_ACTIONS)
        rid = ppo_data[name]["locked_regime"]
        if rid == -1:
            ppo_grid = np.mean(fracs, axis=0).T  # (N_ACTIONS, N_INV)
        else:
            ppo_grid = fracs[rid].T
        draw_heatmap(axes[1, i], ppo_grid,
                     f"PPO — {REGIME_TITLES[name]}", show_ylabel=(i == 0))

    fig.suptitle("Optimal vs PPO MLP — Action Distributions",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "figure2_ppo_policies.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Figure 3 — Learning Curves (total episode reward)
# ---------------------------------------------------------------------------

def plot_figure3(ppo_data):
    """Total episode reward vs iteration, oracle ceiling as red dotted line."""
    t_ep = ppo_data.get("t_episode", 200)

    present = [r for r in REGIME_ORDER if r in ppo_data]
    n_panels = len(present)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]

    for i, name in enumerate(present):
        ax = axes[i]
        entry = ppo_data[name]
        runs = entry["runs"]
        oracle_total = entry["oracle_rps"] * t_ep

        # Collect per-seed curves (already total episode reward in mean_returns)
        all_iters = []
        all_returns = []
        for run in runs:
            all_iters.append(np.array(run["iters"]))
            all_returns.append(np.array(run["mean_returns"]))

        common_iters = all_iters[0]
        ret_matrix = np.array(all_returns)  # (n_seeds, n_checkpoints)
        mean_ret = np.mean(ret_matrix, axis=0)
        std_ret = np.std(ret_matrix, axis=0)

        ax.fill_between(common_iters, mean_ret - std_ret, mean_ret + std_ret,
                        alpha=0.2, color="C0")
        ax.plot(common_iters, mean_ret, color="C0", linewidth=1.5,
                label=f"PPO ({len(runs)} seeds)")
        ax.axhline(oracle_total, color="red", linestyle="--", linewidth=1.5,
                   label=f"Oracle ({oracle_total:.1f})")

        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total episode reward")
        ax.set_title(REGIME_TITLES[name], fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("PPO MLP — Learning Curves",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "figure3_ppo_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  PPO MLP Baseline — Plotting")
    print("=" * 60)

    for path, label in [(PPO_JSON, "ppo_baseline.json"),
                        (FOUNDATION_JSON, "analytical_foundation.json")]:
        if not os.path.exists(path):
            print(f"\n  ERROR: {label} not found.")
            print(f"  Run the corresponding training script first.")
            sys.exit(1)

    with open(PPO_JSON) as f:
        ppo_data = json.load(f)

    optimal = load_optimal_fracs()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n  Plotting figure 2 (optimal vs PPO action distributions) ...")
    plot_figure2(ppo_data, optimal)

    print("  Plotting figure 3 (learning curves) ...")
    plot_figure3(ppo_data)

    print("\n  Done.")


if __name__ == "__main__":
    main()
