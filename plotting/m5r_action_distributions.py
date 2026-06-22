"""M5R action-distribution figure (regime-marginal), brand-styled.

Per-method grid of grouped bars: P(action | true regime), for Belief-PPO and
the four meta-RL cells, on the medium-difficulty environment. The reference
(Belief-PPO) and the hypernet cells differentiate their action mix across
regimes; the concat cells stay nearly flat.

Usage:
  uv run python -m plotting.m5r_action_distributions
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import PALETTE, apply_style, polish

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
PROJECT_FIG_DIR = REPO_ROOT / "figures" / "milestones" / "M5R"
THESIS_FIG_DIR = (
    REPO_ROOT.parent / "master_thesis_reinier_schep_final" / "figures"
)

# Belief-PPO as the regime-aware reference at the top, then the four cells.
METHOD_ORDER = (
    ("belief_ppo",       "Belief-PPO"),
    ("rl2_concat",       "RL² Concat"),
    ("rl2_hypernet",     "RL² Hypernet"),
    ("varibad_concat",   "VariBAD Concat"),
    ("varibad_hypernet", "VariBAD Hypernet"),
)
ACTION_COLORS = {
    "sym":       PALETTE["hyper"],       # teal
    "favor_ask": PALETTE["analytical"],  # amber
    "favor_bid": PALETTE["oracle"],      # dark slate
}
ACTION_LABELS = {
    "sym": "symmetric",
    "favor_ask": "favour ask",
    "favor_bid": "favour bid",
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
        len(method_order), 1, figsize=(9.5, 1.5 * len(method_order)),
        sharex=True, sharey=True,
    )
    if len(method_order) == 1:
        axes = [axes]

    bar_w = 0.24
    regime_centres = np.arange(n_regimes)
    action_offsets = (np.arange(n_actions) - (n_actions - 1) / 2) * bar_w

    for row, (key, label) in enumerate(method_order):
        ax = axes[row]
        m_data = stats["by_method"].get(key)
        mean = np.asarray(m_data["mean_action_given_regime"])  # [R, A]
        std = np.asarray(m_data["std_action_given_regime"])    # [R, A]
        # Differentiation hook: how much P(symmetric) varies across regimes.
        sym_range = float(mean[:, 0].max() - mean[:, 0].min())
        for a_idx, a_name in enumerate(action_names):
            xs = regime_centres + action_offsets[a_idx]
            ax.bar(
                xs, mean[:, a_idx], width=bar_w,
                color=ACTION_COLORS[a_name], edgecolor="white", linewidth=1.0,
                yerr=std[:, a_idx], capsize=2.5, zorder=3,
                error_kw=dict(ecolor="#3a3a3a", lw=0.8),
            )
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(regime_centres)
        ax.set_xticklabels([f"Regime {r}" for r in range(n_regimes)])
        ax.set_ylabel("P(a $\\mid$ z)", fontsize=9)
        ax.set_title(label, fontsize=11, loc="left",
                     color=PALETTE["oracle"], fontweight="bold")
        ax.text(
            1.012, 0.5, f"sym. spread\n$\\Delta = {sym_range:.2f}$",
            transform=ax.transAxes, ha="left", va="center", fontsize=8,
            color="#555555",
        )
        polish(ax)

    axes[-1].set_xlabel("True regime")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ACTION_COLORS[n], edgecolor="white")
        for n in action_names
    ]
    fig.legend(
        handles, [ACTION_LABELS[n] for n in action_names],
        loc="lower center", ncol=n_actions, fontsize=10,
        frameon=False, bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.03, 0.9, 1.0))
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
