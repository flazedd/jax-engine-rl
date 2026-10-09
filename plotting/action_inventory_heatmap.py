"""Action × inventory × regime heatmap grid.

Reads the matched-tuning action-distribution JSON written by
`scripts.action_distributions` and renders a per-method, per-regime
heatmap of P(action | inventory, regime).

Each row is one method (Belief-PPO + the four meta-RL cells). Each column
is one true regime. Within a panel the x-axis is signed inventory level
and the y-axis is action class; cell colour is the empirical conditional
probability of that action given the inventory and regime.

Usage:
  uv run python -m plotting.action_inventory_heatmap
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.action_distribution import regime_separation_per_seed
from plotting.style import apply_style

from utils.paths import analysis_dir, fig_targets, project_fig_dir, resolve_data, results_root

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = results_root()

# Fixed row order. The regime-agnostic agent sits at the top as the zero
# anchor: it cannot observe the regime, so its three regime panels are
# identical, which calibrates how much separation counts as conditioning.
# Belief-PPO follows as the regime-aware reference, then the four meta-RL
# variants in the same order used elsewhere in the thesis.
METHOD_ORDER = (
    ("regime_agnostic_ppo", "Regime-agnostic PPO"),
    ("belief_ppo",          "Belief-PPO"),
    ("rl2_concat",          "RL² Concat"),
    ("rl2_hypernet",        "RL² Hypernet"),
    ("varibad_concat",      "VariBAD Concat"),
    ("varibad_hypernet",    "VariBAD Hypernet"),
)


def main() -> int:
    stats_path = analysis_dir() / "m5r_action_distributions.json"
    if not resolve_data(stats_path).exists():
        print(f"[action_inv_heatmap] missing {stats_path}", flush=True)
        return 1
    with open(resolve_data(stats_path)) as f:
        stats = json.load(f)
    action_names = [
        {"sym": "bid 1 / ask 1", "favor_ask": "bid 3 / ask 1",
         "favor_bid": "bid 1 / ask 3"}.get(name, name)
        for name in stats["action_names"]
    ]
    n_regimes = stats["n_regimes"]
    inv_max = stats.get("inv_max", 5)
    n_inv = 2 * inv_max + 1
    n_actions = len(action_names)

    apply_style()

    method_order = tuple(
        (key, label) for key, label in METHOD_ORDER
        if stats["by_method"].get(key) is not None
        and "mean_action_given_regime_inventory" in stats["by_method"][key]
    )
    if not method_order:
        print("[action_inv_heatmap] no methods carry inventory tables; "
              "rerun scripts.action_distributions first", flush=True)
        return 1

    n_rows = len(method_order)
    n_cols = n_regimes
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.9 * n_cols + 1.9, 1.5 * n_rows + 0.6),
        sharex=True, sharey=True, squeeze=False,
    )

    inv_ticks = np.arange(n_inv)
    inv_labels = [str(q) for q in range(-inv_max, inv_max + 1)]
    # Brand-teal sequential map (near-white at p=0 to deep teal at p=1) so the
    # probability heatmap matches the thesis palette.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "brand_teal", ["#f4f9f8", "#9bd0c8", "#2a9d8f", "#1d6e63", "#143f39"])

    for row, (key, method_label) in enumerate(method_order):
        m_data = stats["by_method"][key]
        arr = np.asarray(m_data["mean_action_given_regime_inventory"])
        # The printed separation is the seed mean of the same per-seed
        # statistic the hypothesis tests consume, so the figure and the
        # reported p-values describe one quantity rather than two.
        separation = float(np.nanmean(regime_separation_per_seed(
            np.asarray(m_data["per_seed_action_given_regime_inventory"]),
            np.asarray(m_data["per_seed_inventory_counts"], dtype=float),
        )))
        # arr has shape [n_regimes, n_inv, n_actions]; transpose per panel
        # to put actions on the y-axis and inventory on the x-axis.
        for col in range(n_cols):
            ax = axes[row, col]
            panel = arr[col].T  # [n_actions, n_inv]
            im = ax.imshow(
                panel, aspect="auto", origin="upper",
                vmin=0.0, vmax=1.0, cmap=cmap,
            )
            ax.set_xticks(inv_ticks)
            ax.set_xticklabels(inv_labels, fontsize=7)
            ax.set_yticks(np.arange(n_actions))
            ax.set_yticklabels(action_names, fontsize=8)
            ax.tick_params(axis="both", which="both", length=0)
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax.set_title(f"Regime {col}", fontsize=10)
            if col == 0:
                ax.set_ylabel(method_label, fontsize=9, rotation=90,
                              labelpad=8)
            if col == n_cols - 1:
                ax.text(
                    1.035, 0.5, f"regime\nseparation\n{separation:.2f}",
                    transform=ax.transAxes, fontsize=7.5, ha="left",
                    va="center", color="#555555", linespacing=1.35,
                )
            for a_idx in range(n_actions):
                for q_idx in range(n_inv):
                    val = float(panel[a_idx, q_idx])
                    # Label every cell, including near-zero ones. Pick black/white
                    # text from the cell's actual luminance (vmin=0, vmax=1) so
                    # the number stays legible on any shade.
                    r, g, b = cmap(val)[:3]
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ax.text(
                        q_idx, a_idx, f"{val:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if lum < 0.55 else "black",
                    )

    for col in range(n_cols):
        axes[-1, col].set_xlabel("Inventory $q$", fontsize=9)

    fig.tight_layout(rect=(0, 0.02, 0.905, 0.99))
    cbar_ax = fig.add_axes([0.955, 0.18, 0.012, 0.66])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("P(action | regime, inventory)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    for out_path in fig_targets("action_given_regime_inventory.png"):
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[action_inv_heatmap] wrote {out_path}", flush=True)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
