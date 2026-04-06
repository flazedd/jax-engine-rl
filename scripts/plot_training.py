"""Plot PPO training curves with analytical baselines.

Reads from results/ppo_metrics_*.json and results/bellman_solution.json.
Each regime gets its own subplot with optimal/blind baselines.

Usage:
    uv run python scripts/plot_training.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

REGIME_COLORS = {
    "noise": "#3b82f6",
    "bull": "#22c55e",
    "bear": "#ef4444",
    "mixed": "#8b5cf6",
}


def main():
    files = sorted(RESULTS_DIR.glob("ppo_metrics_*.json"))
    if not files:
        print(f"No ppo_metrics_*.json found in {RESULTS_DIR} — run train_ppo.py first")
        return

    # Load all metrics
    all_metrics = {}
    for f in files:
        with open(f) as fh:
            m = json.load(fh)
        regime = m.get("regime", "unknown")
        all_metrics[regime] = m

    # Load bellman baselines
    bellman_path = RESULTS_DIR / "bellman_solution.json"
    bellman = {}
    if bellman_path.exists():
        with open(bellman_path) as fh:
            bellman = json.load(fh)

    per_regime = bellman.get("per_regime", {})
    mixed_sim = bellman.get("simulation", {}).get("per_step_reward", {})

    # Determine subplot layout
    regimes = list(all_metrics.keys())
    n = len(regimes)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, regime in enumerate(regimes):
        ax = axes[idx // ncols][idx % ncols]
        m = all_metrics[regime]
        iters = np.array(m["iters"])
        rewards = np.array(m["eval_rewards"])
        color = REGIME_COLORS.get(regime, "#6b7280")

        ax.plot(iters, rewards, "o-", color=color, markersize=2,
                linewidth=1.5, label=f"PPO ({regime})", zorder=3)

        # Per-regime baselines
        if regime in per_regime:
            bl = per_regime[regime]
            ax.axhline(bl["optimal_per_step"], color="#10b981", linestyle="--",
                       linewidth=1.5, label=f"Optimal: {bl['optimal_per_step']:.2f}")
            ax.axhline(bl["blind_per_step"], color="#9ca3af", linestyle="--",
                       linewidth=1.5, label=f"Blind: {bl['blind_per_step']:.2f}")
        elif regime == "mixed" and mixed_sim:
            for key, lbl, clr in [
                ("full_info", "Full-info", "#10b981"),
                ("pomdp", "POMDP", "#f59e0b"),
                ("regime_blind", "Blind", "#9ca3af"),
            ]:
                val = mixed_sim.get(key)
                if val is not None:
                    ax.axhline(val, color=clr, linestyle="--", linewidth=1.5,
                               label=f"{lbl}: {val:.2f}")

        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Reward/step")
        ax.set_title(f"{regime.capitalize()} regime")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / "ppo_training.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path.relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
