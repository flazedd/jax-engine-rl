#!/usr/bin/env python3
"""Plot Inventory Bandit with Regime-Dependent Risk results.

Four panels:
  1. Training curves (rollout reward/step)
  2. Within-episode reward (eval, greedy)
  3. Abstain rate over episode steps
  4. Mean |inventory| over episode steps (position management)

Usage:
    uv run python scripts/plot_bandit_inventory.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_JSON = os.path.join(ROOT, "results", "bandit_inventory.json")
PLOTS_DIR = os.path.join(ROOT, "plots")

AGENTS = [
    ("ppo_mlp",       "PPO MLP",       "C0"),
    ("rl2",           "RL²",           "C1"),
    ("rl2_hn",        "RL²+HN",        "C2"),
    ("varibad",       "VariBAD",       "C3"),
    ("v2_mu",         "V2:μOnly",      "C4"),
    ("v10_musigma",   "V10:μ+σ",       "C5"),
]


def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"  ERROR: {os.path.relpath(RESULTS_JSON, ROOT)} not found.")
        print("  Run: uv run python scripts/run_bandit_inventory.py --agent all")
        sys.exit(1)

    with open(RESULTS_JSON) as f:
        data = json.load(f)

    random_rps = data["random"]["reward_per_step"]
    abstain_rps = data["always_abstain"]["reward_per_step"]
    bo_rps = data["bayes_optimal"]["reward_per_step"]
    bo_reward = data["bayes_optimal"]["per_step_curve"]
    bo_abstain = data["bayes_optimal"]["abstain_curve"]
    t_episode = len(bo_reward)
    steps = np.arange(1, t_episode + 1)

    # Check if we have inventory curves
    has_inv = "inventory_curve" in data["bayes_optimal"]
    n_panels = 4 if has_inv else 3

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))
    if n_panels == 3:
        ax1, ax2, ax3 = axes
    else:
        ax1, ax2, ax3, ax4 = axes

    # --- Panel 1: Training curves ---
    ax1.set_title("Training Curves (rollout reward/step)", fontweight="bold")
    for key, label, color in AGENTS:
        if key not in data:
            continue
        hist = data[key]["train_history"]
        iters = np.arange(len(hist))
        window = max(1, len(hist) // 40)
        smoothed = np.convolve(hist, np.ones(window) / window, mode="valid")
        x_smooth = np.arange(window - 1, len(hist))
        ax1.plot(x_smooth, smoothed, color=color, linewidth=1.5, label=label)
        ax1.plot(iters, hist, color=color, alpha=0.15, linewidth=0.5)

    ax1.axhline(random_rps, color="gray", linestyle=":",
                linewidth=1, label=f"Random ({random_rps:.3f})")
    ax1.axhline(abstain_rps, color="brown", linestyle=":",
                linewidth=1, label=f"Always Abstain ({abstain_rps:.3f})")
    ax1.axhline(bo_rps, color="red", linestyle="--",
                linewidth=1, label=f"Bayes-Optimal ({bo_rps:.3f})")
    ax1.set_xlabel("Training iteration")
    ax1.set_ylabel("Mean reward / step")
    ax1.legend(fontsize=6, loc="lower right", ncol=2)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Per-step reward ---
    ax2.set_title("Within-Episode Reward (eval, greedy)", fontweight="bold")
    ax2.plot(steps, data["random"]["per_step_curve"], color="gray",
             linestyle=":", linewidth=1, label="Random")
    ax2.axhline(abstain_rps, color="brown", linestyle=":",
                linewidth=1, label="Always Abstain")
    ax2.plot(steps, bo_reward, color="red", linestyle="--",
             linewidth=1.5, label="Bayes-Optimal")
    for key, label, color in AGENTS:
        if key not in data:
            continue
        ax2.plot(steps, data[key]["per_step_curve"], color=color,
                 linewidth=1.5, label=label, alpha=0.8)
    ax2.set_xlabel("Step within episode")
    ax2.set_ylabel("Mean reward")
    ax2.legend(fontsize=6, loc="lower right", ncol=2)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Abstain rate ---
    ax3.set_title("Abstain Rate", fontweight="bold")
    ax3.plot(steps, bo_abstain, color="red", linestyle="--",
             linewidth=1.5, label="Bayes-Optimal")
    random_abstain = data["random"]["abstain_curve"][0]
    ax3.axhline(random_abstain, color="gray", linestyle=":",
                linewidth=1, label=f"Random ({random_abstain:.2f})")
    for key, label, color in AGENTS:
        if key not in data or "abstain_curve" not in data[key]:
            continue
        ax3.plot(steps, data[key]["abstain_curve"], color=color,
                 linewidth=1.5, label=label, alpha=0.8)
    ax3.set_xlabel("Step within episode")
    ax3.set_ylabel("P(abstain)")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=6, loc="upper right", ncol=2)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Inventory curve ---
    if has_inv:
        ax4.set_title("Mean |Inventory| (position size)", fontweight="bold")
        bo_inv = data["bayes_optimal"]["inventory_curve"]
        ax4.plot(steps, bo_inv, color="red", linestyle="--",
                 linewidth=1.5, label="Bayes-Optimal")
        ax4.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        ax4.set_xlabel("Step within episode")
        ax4.set_ylabel("Mean |inventory|")
        ax4.legend(fontsize=6, loc="upper left")
        ax4.grid(True, alpha=0.3)

    fig.suptitle("Inventory Bandit: Regime-Dependent Risk + Active Sensing\n"
                 "(5 tasks: 2 low-risk directional, 2 high-risk directional, "
                 "1 high-risk flat)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "bandit_inventory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
