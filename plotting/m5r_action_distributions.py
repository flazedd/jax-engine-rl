"""M5R action-distribution figure.

Reads the matched-tuning action-distribution JSON and writes a per-method
grid of grouped bar charts to the thesis figure directory. Each row is a
single wide panel for one method, with the three regimes side by side and
one bar per action within each regime.

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
PROJECT_FIG_DIR = REPO_ROOT / "figures" / "milestones" / "M5R"
THESIS_FIG_DIR = (
    REPO_ROOT.parent / "master_thesis_reinier_schep_final" / "figures"
)

METHOD_ORDER = (
    ("oracle_ppo",          "Oracle-PPO"),
    ("belief_ppo",          "Belief-PPO"),
    ("regime_agnostic_ppo", "Regime-agnostic"),
    ("stacked_obs_ppo",     "Stacked-obs PPO"),
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
    n_actions = len(action_names)

    apply_style()
    method_order = tuple(
        (key, label) for key, label in METHOD_ORDER
        if stats["by_method"].get(key) is not None
    )
    fig, axes = plt.subplots(
        len(method_order), 1, figsize=(11.0, 1.7 * len(method_order)),
        sharex=True, sharey=True,
    )
    if len(method_order) == 1:
        axes = [axes]

    bar_w = 0.25
    regime_centres = np.arange(n_regimes)  # one slot per regime
    action_offsets = (np.arange(n_actions) - (n_actions - 1) / 2) * bar_w

    for row, (key, label) in enumerate(method_order):
        ax = axes[row]
        m_data = stats["by_method"].get(key)
        if m_data is None:
            continue
        mean = np.asarray(m_data["mean_action_given_regime"])  # [R, A]
        std = np.asarray(m_data["std_action_given_regime"])    # [R, A]
        sym_range = float(mean[:, 0].max() - mean[:, 0].min())
        for a_idx, a_name in enumerate(action_names):
            xs = regime_centres + action_offsets[a_idx]
            heights = mean[:, a_idx]
            errs = std[:, a_idx]
            ax.bar(
                xs, heights, width=bar_w,
                color=ACTION_COLORS[a_name], alpha=0.85,
                edgecolor="black", linewidth=0.5,
                yerr=errs, capsize=3,
                error_kw=dict(ecolor="black", lw=0.7),
            )
            for x, h, e in zip(xs, heights, errs):
                ax.text(x, h + e + 0.02, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0.0, 1.18)
        ax.set_xticks(regime_centres)
        ax.set_xticklabels([f"r{r}" for r in range(n_regimes)], fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylabel("P(action | regime)", fontsize=10)
        ax.set_title(label, fontsize=12, loc="left")
        ax.text(
            1.005, 0.5, f"sym Δ = {sym_range:.2f}",
            transform=ax.transAxes,
            ha="left", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )
        ax.grid(axis="y", alpha=0.3, linestyle=":")

    axes[-1].set_xlabel("Regime", fontsize=11)
    fig.suptitle(
        "Action distribution conditional on the true regime, MarketMakingV1, "
        "medium difficulty",
        fontsize=13, y=0.998,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ACTION_COLORS[n],
                      edgecolor="black", linewidth=0.5)
        for n in action_names
    ]
    fig.legend(
        handles, action_names,
        loc="lower center", ncol=len(action_names), fontsize=12,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.025, 0.92, 0.985))
    PROJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for target_dir in (PROJECT_FIG_DIR, THESIS_FIG_DIR):
        out_path = target_dir / "m5r_action_given_regime.png"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[m5r_action_plots] wrote {out_path}", flush=True)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
