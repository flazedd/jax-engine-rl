#!/usr/bin/env python3
"""Plot mean per-step reward baselines from bellman_solution.json.

Usage:
    uv run python scripts/plot_baselines.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "bellman_solution.json")
PLOTS_DIR = os.path.join(ROOT, "plots")


def main():
    with open(RESULTS) as f:
        data = json.load(f)

    sim = data["simulation"]
    psr = sim["per_step_reward"]

    labels = ["Regime-blind\n(always a0)", "POMDP\n(regime hidden)", "Full-info\n(regime observed)"]
    values = [psr["regime_blind"], psr["pomdp"], psr["full_info"]]
    colors = ["#C44E52", "#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=12, fontweight="bold")

    # RL target annotation
    rl_gap = psr["pomdp"] - psr["regime_blind"]
    mid = (psr["regime_blind"] + psr["pomdp"]) / 2
    ax.annotate("", xy=(psr["pomdp"], 1.35), xytext=(psr["regime_blind"], 1.35),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
    ax.text(mid, 1.42, f"RL target: +{rl_gap:.2f}/step",
            ha="center", va="bottom", fontsize=10, color="#333333")

    ax.set_xlabel("Mean Reward per Step", fontsize=12)
    ax.set_title("Baseline Performance Levels", fontsize=14)
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "baselines.png")
    plt.savefig(path, dpi=150)
    print(f"  → {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
