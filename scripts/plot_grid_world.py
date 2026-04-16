#!/usr/bin/env python3
"""Plot Grid World (GridNavi) results from results/grid_world.json.

Produces:
  plots/grid_world.png  — training curve + per-episode + per-step reward

Usage:
    uv run python scripts/plot_grid_world.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_JSON = os.path.join(ROOT, "results", "grid_world.json")
PLOTS_DIR = os.path.join(ROOT, "plots")

AGENTS = [
    ("rl2_hn",     "RL²+HN",     "C2"),
    ("varibad",    "VariBAD",    "C3"),
    ("varibad_hn", "VariBAD+HN", "C4"),
]


def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"  ERROR: {os.path.relpath(RESULTS_JSON, ROOT)} not found.")
        print("  Run: uv run python scripts/run_grid_world.py")
        sys.exit(1)

    with open(RESULTS_JSON) as f:
        data = json.load(f)

    random_rps = data["random"]["reward_per_step"]
    oracle_rps = data["oracle"]["reward_per_step"]
    rand_ep = np.array(data["random"]["per_episode_curve"])
    oracle_ep = np.array(data["oracle"]["per_episode_curve"])
    n_episodes = len(oracle_ep)

    # Collect available agents
    agents = [(k, label, color) for k, label, color in AGENTS if k in data]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Training curves ---
    ax1.set_title("Training Curves", fontweight="bold")
    for key, label, color in agents:
        hist = data[key]["train_history"]
        iters = np.arange(len(hist))
        window = max(1, len(hist) // 40)
        smoothed = np.convolve(hist, np.ones(window) / window, mode="valid")
        x_smooth = np.arange(window - 1, len(hist))
        ax1.plot(x_smooth, smoothed, color=color, linewidth=1.5, label=label)
        ax1.plot(iters, hist, color=color, alpha=0.15, linewidth=0.5)

    ax1.axhline(random_rps, color="gray", linestyle=":",
                linewidth=1, label=f"Random ({random_rps:.3f})")
    ax1.axhline(oracle_rps, color="red", linestyle="--",
                linewidth=1, label=f"Oracle ({oracle_rps:.3f})")
    ax1.set_xlabel("Training iteration")
    ax1.set_ylabel("Mean reward / step")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Per-episode reward (key result) ---
    ax2.set_title("Per-Episode Reward in Trial", fontweight="bold")
    eps = np.arange(1, n_episodes + 1)
    n_bars = 2 + len(agents)   # random + agents + oracle
    width = 0.8 / n_bars
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    ax2.bar(eps + offsets[0], rand_ep, width, color="gray", alpha=0.6,
            label="Random")
    for i, (key, label, color) in enumerate(agents):
        ep_curve = np.array(data[key]["per_episode_curve"])
        ax2.bar(eps + offsets[1 + i], ep_curve, width, color=color, label=label)
    ax2.bar(eps + offsets[-1], oracle_ep, width, color="red", alpha=0.6,
            label="Oracle")

    ax2.set_xlabel("Episode within trial")
    ax2.set_ylabel("Mean reward / step")
    ax2.set_xticks(eps)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: Per-step reward across full trial ---
    ax3.set_title("Per-Step Reward Across Trial", fontweight="bold")
    rand_steps = np.array(data["random"]["per_step_curve"])
    oracle_steps = np.array(data["oracle"]["per_step_curve"])
    n_steps = len(oracle_steps)
    t_ep = n_steps // n_episodes
    steps = np.arange(1, n_steps + 1)

    ax3.plot(steps, rand_steps, color="gray", linestyle=":",
             linewidth=1, label="Random")
    ax3.plot(steps, oracle_steps, color="red", linestyle="--",
             linewidth=1.5, label="Oracle")
    for key, label, color in agents:
        agent_steps = np.array(data[key]["per_step_curve"])
        ax3.plot(steps, agent_steps, color=color, linewidth=1.5,
                 label=label, alpha=0.8)

    # Mark episode boundaries
    for ep_boundary in range(t_ep, n_steps, t_ep):
        ax3.axvline(ep_boundary + 0.5, color="black", linestyle=":",
                    linewidth=0.5, alpha=0.4)
    # Episode labels at top
    ymax = oracle_steps.max() * 1.05
    for ep in range(n_episodes):
        mid = ep * t_ep + t_ep / 2
        ax3.text(mid, ymax, f"Ep {ep + 1}", ha="center", va="bottom",
                 fontsize=7, color="black", alpha=0.5)

    ax3.set_xlabel("Step within trial")
    ax3.set_ylabel("Mean reward")
    ax3.legend(fontsize=7, loc="lower right")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Grid World (GridNavi) — RL²+HN, VariBAD & VariBAD+HN",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "grid_world.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
