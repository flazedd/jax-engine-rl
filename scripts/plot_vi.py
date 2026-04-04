"""Plot VI oracle results from precomputed JSON files.

Usage:
    uv run python scripts/plot_vi.py

Reads:
    plots/vi_isolated.json
    plots/vi_optimal.json
    plots/vi_eval.json

Saves:
    plots/vi_comparison.png  — 6-panel policy grid (isolated vs mixed)
    plots/vi_convergence.png — convergence diagnostics
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lob_sim.actions import ACTION_TABLE, N_ACTIONS
from plot_style import apply_style, save_fig, REGIME_NAMES, REGIME_COLORS

PLOTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "plots"))


def _load(filename):
    with open(os.path.join(PLOTS_DIR, filename)) as f:
        return json.load(f)


iso_data = _load("vi_isolated.json")
mix_data = _load("vi_optimal.json")
eval_data = _load("vi_eval.json")

iso_policy = np.array(iso_data["policy"])
mix_policy = np.array(mix_data["policy"])
max_inv = iso_data["max_inventory"]
n_inv = 2 * max_inv + 1
inv_grid = np.arange(n_inv) - max_inv

at = np.array(ACTION_TABLE)
action_labels = [f"({at[a][0]},{at[a][1]})" for a in range(N_ACTIONS)]

apply_style()

# ── Plot 1: Policy comparison (6-panel grid) ──
# Display fewer levels than computed to avoid boundary artifacts
display_range = max_inv - 5
lo = max_inv - display_range
hi = max_inv + display_range + 1
n_disp = hi - lo
disp_inv = inv_grid[lo:hi]

fig, axes = plt.subplots(3, 2, figsize=(18, 10))
fig.suptitle("VI Optimal Policy: Isolated vs Mixed Regimes",
             fontsize=14, fontweight="bold")

column_titles = ["Isolated Regime", "Mixed Regime"]
cmap = ListedColormap(["white", "#4CAF50"])

for col, (policy, title) in enumerate([(iso_policy, column_titles[0]),
                                        (mix_policy, column_titles[1])]):
    for r in range(3):
        ax = axes[r, col]

        grid = np.zeros((N_ACTIONS, n_disp))
        for i in range(n_disp):
            grid[policy[r, lo + i], i] = 1.0

        ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest", origin="lower")

        ax.set_xticks(np.arange(-0.5, n_disp, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N_ACTIONS, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.3, alpha=0.5)
        ax.tick_params(which="minor", length=0)

        major_x = np.arange(0, n_disp, 5)
        ax.set_xticks(major_x)
        ax.set_xticklabels([str(disp_inv[i]) for i in major_x], fontsize=7)
        ax.set_yticks(range(N_ACTIONS))
        ax.set_yticklabels(action_labels, fontsize=8)

        ax.set_ylabel(f"{REGIME_NAMES[r]}\n(bid, ask) ticks", fontsize=10,
                      fontweight="bold", color=REGIME_COLORS[r])

        if r == 0:
            ax.set_title(title, fontsize=12)
        if r == 2:
            ax.set_xlabel("Inventory")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

plt.tight_layout()
save_fig(fig, "vi_comparison.png", script_file=__file__)

# ── Plot 2: Convergence ──
fig2, (ax_iso, ax_mix) = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle("Value Iteration Convergence", fontsize=14, fontweight="bold")

for ax, data, label in [(ax_iso, iso_data, "Isolated"),
                          (ax_mix, mix_data, "Mixed")]:
    deltas = data["convergence_deltas"]
    conv = data["convergence"]
    ax.semilogy(deltas, linewidth=1.5)
    ax.axhline(1e-6, color="red", linestyle="--", alpha=0.5, label="threshold")
    ax.axvline(conv["policy_stable_since"], color="green", linestyle="--",
               alpha=0.5, label=f"policy stable (iter {conv['policy_stable_since']})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("max |V_new - V|")
    ax.set_title(f"{label} Regimes")
    ax.legend(fontsize=8)

plt.tight_layout()
save_fig(fig2, "vi_convergence.png", script_file=__file__)

# ── Print eval summary ──
print("\nEvaluation summary (from vi_eval.json):")
print(f"  Mixed VI oracle:  {eval_data['mixed_vi']['mean_reward']:.2f} "
      f"(std={eval_data['mixed_vi']['std_reward']:.2f})")
print(f"  Myopic oracle:    {eval_data['myopic']['mean_reward']:.2f} "
      f"(std={eval_data['myopic']['std_reward']:.2f})")
for name, stats in eval_data["per_regime"].items():
    print(f"  {name:>5s} locked: VI={stats['vi_mean']:.2f}, "
          f"Myopic={stats['myopic_mean']:.2f}")

print("\nDone.")
