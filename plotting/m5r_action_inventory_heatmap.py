"""M5R action × inventory × regime heatmap grid.

Reads the matched-tuning action-distribution JSON written by
`scripts.m5r_action_distributions` and renders a per-method, per-regime
heatmap of P(action | inventory, regime).

Each row is one method (Belief-PPO + the four meta-RL cells). Each column
is one true regime. Within a panel the x-axis is signed inventory level
and the y-axis is action class; cell colour is the empirical conditional
probability of that action given the inventory and regime.

Usage:
  uv run python -m plotting.m5r_action_inventory_heatmap
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

# Fixed row order. Belief-PPO sits at the top as the regime-aware reference;
# the four meta-RL cells follow in the same order used elsewhere in the thesis.
METHOD_ORDER = (
    ("belief_ppo",          "Belief-PPO"),
    ("rl2_concat",          "RL² Concat"),
    ("rl2_hypernet",        "RL² Hypernet"),
    ("varibad_concat",      "VariBAD Concat"),
    ("varibad_hypernet",    "VariBAD Hypernet"),
)


def main() -> int:
    stats_path = RESULTS_ROOT / "M5R" / "final" / "m5r_action_distributions.json"
    if not stats_path.exists():
        print(f"[m5r_action_inv_heatmap] missing {stats_path}", flush=True)
        return 1
    with open(stats_path) as f:
        stats = json.load(f)
    action_names = stats["action_names"]
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
        print("[m5r_action_inv_heatmap] no methods carry inventory tables; "
              "rerun scripts.m5r_action_distributions first", flush=True)
        return 1

    n_rows = len(method_order)
    n_cols = n_regimes
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols + 1.0, 1.5 * n_rows + 0.6),
        sharex=True, sharey=True, squeeze=False,
    )

    inv_ticks = np.arange(n_inv)
    inv_labels = [str(q) for q in range(-inv_max, inv_max + 1)]
    cmap = "viridis"

    for row, (key, method_label) in enumerate(method_order):
        m_data = stats["by_method"][key]
        arr = np.asarray(m_data["mean_action_given_regime_inventory"])
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
            if row == 0:
                ax.set_title(f"Regime {col}", fontsize=10)
            if col == 0:
                ax.set_ylabel(method_label, fontsize=9, rotation=90,
                              labelpad=8)
            for a_idx in range(n_actions):
                for q_idx in range(n_inv):
                    val = panel[a_idx, q_idx]
                    if val >= 0.005:
                        ax.text(
                            q_idx, a_idx, f"{val:.2f}",
                            ha="center", va="center",
                            fontsize=6,
                            color="white" if val < 0.55 else "black",
                        )

    for col in range(n_cols):
        axes[-1, col].set_xlabel("Inventory $q$", fontsize=9)

    fig.suptitle(
        "Action distribution conditional on regime and inventory, "
        "MarketMakingV1, medium difficulty",
        fontsize=12, y=0.995,
    )

    fig.tight_layout(rect=(0, 0.02, 0.95, 0.985))
    cbar_ax = fig.add_axes([0.96, 0.18, 0.012, 0.66])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("P(action | regime, inventory)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    PROJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for target_dir in (PROJECT_FIG_DIR, THESIS_FIG_DIR):
        out_path = target_dir / "m5r_action_given_regime_inventory.png"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[m5r_action_inv_heatmap] wrote {out_path}", flush=True)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
