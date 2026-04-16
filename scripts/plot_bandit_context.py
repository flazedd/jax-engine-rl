#!/usr/bin/env python3
"""Plot Contextual Bandit learning curves from results/bandit_context_validation.json.

Produces:
  plots/bandit_context_learning_curves.png  — training curves + per-step reward

Usage:
    uv run python scripts/plot_bandit_context.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_JSON = os.path.join(ROOT, "results", "bandit_context_validation.json")
PLOTS_DIR = os.path.join(ROOT, "plots")

AGENTS = [
    ("ppo_mlp",      "PPO MLP",       "C0"),
    ("rl2",          "RL²",           "C1"),
    ("rl2_hn",       "RL²+HN",        "C2"),
    ("varibad",      "VariBAD",       "C3"),
    ("v2_mu",        "V2:μOnly",      "C4"),
]


def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"  ERROR: {os.path.relpath(RESULTS_JSON, ROOT)} not found.")
        print("  Run: uv run python scripts/run_bandit_context_long.py")
        sys.exit(1)

    with open(RESULTS_JSON) as f:
        data = json.load(f)

    random_rps = data["random"]["reward_per_step"]
    ts_rps = data["thompson"]["reward_per_step"]
    ts_curve = data["thompson"]["per_step_curve"]
    bs_rps = data["bayes_side"]["reward_per_step"]
    bs_curve = data["bayes_side"]["per_step_curve"]
    t_episode = len(ts_curve)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Left panel: training curves ---
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
                linewidth=1, label=f"Random ({random_rps:.2f})")
    ax1.axhline(ts_rps, color="orange", linestyle="--",
                linewidth=1, label=f"Thompson ({ts_rps:.3f})")
    ax1.axhline(bs_rps, color="red", linestyle="--",
                linewidth=1, label=f"Bayes-Side ({bs_rps:.3f})")
    ax1.set_xlabel("Training iteration")
    ax1.set_ylabel("Mean reward / step")
    ax1.legend(fontsize=6, loc="lower right", ncol=2)
    ax1.grid(True, alpha=0.3)

    # --- Right panel: per-step reward curves ---
    ax2.set_title("Within-Episode Reward (eval, greedy)", fontweight="bold")
    steps = np.arange(1, t_episode + 1)

    ax2.plot(steps, data["random"]["per_step_curve"], color="gray",
             linestyle=":", linewidth=1, label="Random")
    ax2.plot(steps, ts_curve, color="orange", linestyle="--",
             linewidth=1.5, label="Thompson")
    ax2.plot(steps, bs_curve, color="red", linestyle="--",
             linewidth=1.5, label="Bayes-Side")
    for key, label, color in AGENTS:
        if key not in data:
            continue
        ax2.plot(steps, data[key]["per_step_curve"], color=color,
                 linewidth=1.5, label=label, alpha=0.8)

    ax2.set_xlabel("Step within episode")
    ax2.set_ylabel("Mean reward")
    ax2.legend(fontsize=6, loc="lower right", ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(steps)

    fig.suptitle("Contextual Bandit with Side Information — Meta-RL",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "bandit_context_learning_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
