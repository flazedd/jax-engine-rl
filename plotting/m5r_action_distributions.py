"""M5R action-distribution figure.

Reads the matched-tuning action-distribution JSON and writes a 7-method
× 3-regime grid of stacked bars to the thesis figure directory.

Usage:
  uv run python -m plotting.m5r_action_distributions
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import apply_style

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
THESIS_FIG_DIR = (
    REPO_ROOT.parent / "master_thesis_reinier_schep_final" / "figures"
)

METHOD_ORDER = (
    ("oracle_ppo",          "Oracle-PPO"),
    ("belief_ppo",          "Belief-PPO"),
    ("regime_agnostic_ppo", "Regime-agnostic"),
    ("rl2_hypernet",        "RL² Hypernet"),
    ("varibad_hypernet",    "VariBAD Hypernet"),
    ("rl2_concat",          "RL² Concat"),
    ("varibad_concat",      "VariBAD Concat"),
)
ACTION_COLORS = {
    "sym":       "#2ca02c",
    "favor_ask": "#ff7f0e",
    "favor_bid": "#1f77b4",
}


def main() -> int:
    stats_path = RESULTS_ROOT / "M5R" / "final" / "m5r_action_distributions.json"
    if not stats_path.exists():
        print(f"[m5r_action_plots] missing {stats_path}", flush=True)
        return 1
    with open(stats_path) as f:
        stats = json.load(f)
    action_names = stats["action_names"]
    n_regimes = stats["n_regimes"]

    apply_style()
    fig, axes = plt.subplots(
        len(METHOD_ORDER), n_regimes, figsize=(8.0, 12.0),
        sharex=True, sharey=True,
    )
    for row, (key, label) in enumerate(METHOD_ORDER):
        m_data = stats["by_method"].get(key)
        if m_data is None:
            continue
        mean = np.asarray(m_data["mean_action_given_regime"])
        std = np.asarray(m_data["std_action_given_regime"])
        sym_range = float(mean[:, 0].max() - mean[:, 0].min())
        for col in range(n_regimes):
            ax = axes[row, col]
            xs = np.arange(len(action_names))
            heights = mean[col]
            errs = std[col]
            colors = [ACTION_COLORS[n] for n in action_names]
            ax.bar(
                xs, heights, color=colors, alpha=0.85,
                edgecolor="black", linewidth=0.5,
                yerr=errs, capsize=3,
                error_kw=dict(ecolor="black", lw=0.7),
            )
            for x, h in zip(xs, heights):
                ax.text(x, h + 0.03, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(xs)
            ax.set_xticklabels(action_names, fontsize=8, rotation=15)
            if col == 0:
                ax.set_ylabel(f"{label}\nP(action | regime)", fontsize=9)
            if row == 0:
                ax.set_title(f"r{col}", fontsize=10)
        axes[row, -1].text(
            1.02, 0.5, f"sym Δ\n{sym_range:.2f}",
            transform=axes[row, -1].transAxes,
            ha="left", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )
    fig.suptitle(
        "MarketMakingV1 $E_{\\mathrm{med}}$, matched-tuning protocol\n"
        "P(action | true regime) per method",
        fontsize=11, y=0.995,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ACTION_COLORS[n], edgecolor="black", linewidth=0.5)
        for n in action_names
    ]
    fig.legend(
        handles, action_names,
        loc="lower center", ncol=len(action_names), fontsize=9,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = THESIS_FIG_DIR / "m5r_action_given_regime.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[m5r_action_plots] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
