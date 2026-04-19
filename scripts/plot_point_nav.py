#!/usr/bin/env python3
"""Plot 2D Point Navigation results.

Three panels:
  1. Training curves (rollout reward/step)
  2. Within-episode reward (eval, greedy)
  3. Sorted bar chart of eval reward/step

Usage:
    uv run python scripts/plot_point_nav.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_JSON = os.path.join(ROOT, "results", "point_nav.json")
PLOTS_DIR = os.path.join(ROOT, "plots")

AGENTS = [
    ("ppo_mlp",   "PPO MLP",   "#888888"),
    ("rl2_hn",    "RL²+HN",    "#1f77b4"),
    ("varibad",   "VariBAD",   "#2ca02c"),
    ("v2_mu",     "V2:μOnly",  "#d62728"),
    ("amago",     "AMAGO",     "#ff7f0e"),
]


def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"  ERROR: {os.path.relpath(RESULTS_JSON, ROOT)} not found.")
        print("  Run: uv run python scripts/run_point_nav.py --agent all")
        sys.exit(1)

    with open(RESULTS_JSON) as f:
        data = json.load(f)

    random_rps = data["random"]["reward_per_step"]
    random_curve = data["random"]["per_step_curve"]
    t_episode = len(random_curve)
    steps = np.arange(1, t_episode + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: Training curves ---
    ax1.set_title("Training Curves (rollout reward/step)", fontweight="bold")
    for key, label, color in AGENTS:
        if key not in data:
            continue
        hist = data[key]["train_history"]
        iters = np.arange(len(hist))
        window = max(1, len(hist) // 20)
        smoothed = np.convolve(hist, np.ones(window) / window, mode="valid")
        x_smooth = np.arange(window - 1, len(hist))
        ax1.plot(x_smooth, smoothed, color=color, linewidth=1.5, label=label)
        ax1.plot(iters, hist, color=color, alpha=0.15, linewidth=0.5)
    ax1.axhline(random_rps, color="gray", linestyle=":", linewidth=1,
                label=f"Random ({random_rps:.3f})")
    ax1.set_xlabel("Training iteration")
    ax1.set_ylabel("Mean reward / step")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Per-step eval curve ---
    ax2.set_title("Within-Episode Reward (eval, greedy)", fontweight="bold")
    ax2.plot(steps, random_curve, color="gray", linestyle=":",
             linewidth=1, label="Random")
    for key, label, color in AGENTS:
        if key not in data:
            continue
        curve = data[key]["per_step_curve"]
        if len(curve) != t_episode:
            print(f"  skipping {key}: curve length {len(curve)} != {t_episode}")
            continue
        ax2.plot(steps, curve, color=color, linewidth=1.5,
                 label=label, alpha=0.85)
    ax2.set_xlabel("Step within episode")
    ax2.set_ylabel("Mean reward")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Sorted bar chart ---
    ax3.set_title("Eval Reward/Step (sorted)", fontweight="bold")
    bar_items = []
    for key, label, color in AGENTS:
        if key not in data:
            continue
        bar_items.append((label, data[key]["reward_per_step"], color))
    bar_items.sort(key=lambda x: x[1])
    labels_bar = [b[0] for b in bar_items]
    values_bar = [b[1] for b in bar_items]
    colors_bar = [b[2] for b in bar_items]
    y_pos = np.arange(len(bar_items))
    ax3.barh(y_pos, values_bar, color=colors_bar, edgecolor="white",
             linewidth=0.5, height=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels_bar, fontsize=8)
    ax3.axvline(random_rps, color="gray", linestyle=":", linewidth=1,
                label=f"Random ({random_rps:.3f})")
    for i, v in enumerate(values_bar):
        ax3.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=7)
    ax3.set_xlabel("Mean reward / step")
    ax3.legend(fontsize=7, loc="lower right")
    ax3.grid(True, alpha=0.3, axis="x")

    fig.suptitle("2D Point Navigation — Hidden Goal on Upper Semicircle "
                 "(VariBAD task, discretized)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "point_nav.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
