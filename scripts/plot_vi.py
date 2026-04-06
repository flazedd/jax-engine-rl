"""Plot VI oracle results from precomputed JSON files.

Usage:
    uv run python scripts/plot_vi.py

Reads:
    results/vi_isolated.json
    results/vi_optimal.json
    results/vi_eval.json

Saves:
    plots/vi_isolated_noise.png — Q-value heatmap for isolated noise regime
    plots/vi_isolated_bull.png  — Q-value heatmap for isolated bull regime
    plots/vi_isolated_bear.png  — Q-value heatmap for isolated bear regime
    plots/vi_mixed.png          — Q-value heatmaps for mixed (3 regimes)
    plots/vi_convergence.png    — convergence diagnostics
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from lob_sim.actions import ACTION_TABLE, N_ACTIONS
from plot_style import apply_style, save_fig, REGIME_NAMES, REGIME_COLORS

RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def _load(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
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

iso_Q = np.array(iso_data["Q"])  # (N_REGIMES, n_inv, N_ACTIONS)
mix_Q = np.array(mix_data["Q"])  # (N_REGIMES, n_inv, N_ACTIONS)

# Display range (trim boundary artifacts)
display_range = max_inv - 5
lo = max_inv - display_range
hi = max_inv + display_range + 1
n_disp = hi - lo
disp_inv = inv_grid[lo:hi]


def _plot_q_heatmap(ax, Q_slice, title, color):
    """Plot a single Q-value heatmap on the given axis.

    Q_slice: (n_disp, N_ACTIONS) — already sliced to display range.
    Transpose to (N_ACTIONS, n_disp) for display: y=actions, x=inventory.
    """
    q = Q_slice.T  # (N_ACTIONS, n_disp)

    im = ax.imshow(q, cmap="RdYlGn", aspect="auto", interpolation="nearest")

    best_per_inv = q.argmax(axis=0)

    from matplotlib.patches import Rectangle
    for i in range(n_disp):
        for a in range(N_ACTIONS):
            v = q[a, i]
            mid_val = (q.max() + q.min()) / 2
            txt_color = ("white" if abs(v - mid_val) > 0.55 * abs(q.max() - mid_val)
                         else "black")
            ax.text(i, a, f"{v:.1f}", ha="center", va="center",
                    fontsize=5.5, color=txt_color)

        best_a = best_per_inv[i]
        rect = Rectangle((i - 0.5, best_a - 0.5), 1, 1,
                          linewidth=2.0, edgecolor="#1565C0",
                          facecolor="none", zorder=5)
        ax.add_patch(rect)

    ax.set_xticks(np.arange(0, n_disp, 5))
    ax.set_xticklabels([str(disp_inv[i]) for i in np.arange(0, n_disp, 5)],
                       fontsize=7)
    ax.set_yticks(range(N_ACTIONS))
    ax.set_yticklabels(action_labels, fontsize=8)
    ax.set_xlabel("Inventory")
    ax.set_ylabel("Action (bid, ask) ticks")
    ax.set_title(title, fontsize=12, fontweight="bold", color=color)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    return im


# ── Plot 1-3: Isolated regime Q-value heatmaps (one figure each) ──
for r in range(3):
    fig, ax = plt.subplots(figsize=(max(16, n_disp * 0.55), N_ACTIONS * 0.7 + 2))
    q_slice = iso_Q[r, lo:hi, :]  # (n_disp, N_ACTIONS)
    im = _plot_q_heatmap(ax, q_slice,
                         f"Isolated {REGIME_NAMES[r]} — VI Q-Values & Optimal Policy",
                         REGIME_COLORS[r])
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Q-value")
    plt.tight_layout()
    save_fig(fig, f"vi_isolated_{REGIME_NAMES[r].lower()}.png", script_file=__file__)

# ── Plot 4: Mixed regime — optimal action per (regime, inventory) ──
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Action colormap: one color per action
action_cmap = ListedColormap(["#4CAF50", "#2196F3", "#F44336"])  # green, blue, red

# Build policy grid: (3 regimes, n_disp) with action index as value
mix_policy_disp = np.array(mix_data["policy"])[:, lo:hi]  # (3, n_disp)

fig, ax = plt.subplots(figsize=(max(14, n_disp * 0.45), 3.5))
im = ax.imshow(mix_policy_disp, cmap=action_cmap, aspect="auto",
               interpolation="nearest", vmin=0, vmax=N_ACTIONS - 1)

# Annotate each cell with Q-values for all actions
for r in range(3):
    for i in range(n_disp):
        best_a = mix_policy_disp[r, i]
        q_best = mix_Q[r, lo + i, best_a]
        ax.text(i, r, f"{action_labels[best_a]}\nQ={q_best:.1f}",
                ha="center", va="center", fontsize=6, fontweight="bold",
                color="white")

ax.set_xticks(np.arange(0, n_disp, 5))
ax.set_xticklabels([str(disp_inv[i]) for i in np.arange(0, n_disp, 5)],
                   fontsize=8)
ax.set_yticks(range(3))
ax.set_yticklabels([REGIME_NAMES[r] for r in range(3)], fontsize=11,
                   fontweight="bold")
ax.set_xlabel("Inventory", fontsize=10)
ax.set_title("Mixed Regime — Optimal Action per (Regime, Inventory)\n"
             "HMM transition matrix active",
             fontsize=13, fontweight="bold")

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)

legend_patches = [Patch(facecolor=action_cmap(a), label=action_labels[a])
                  for a in range(N_ACTIONS)]
ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
          title="Action", title_fontsize=10)

plt.tight_layout()
save_fig(fig, "vi_mixed.png", script_file=__file__)

# ── Plot 5: Convergence ──
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle("Value Iteration Convergence", fontsize=14, fontweight="bold")

# Isolated: plot deltas (using stored deltas from first regime / noise)
ax = axes2[0]
deltas = iso_data["convergence_deltas"]
conv = iso_data["convergence"]
ax.semilogy(deltas, linewidth=1.5)
ax.axhline(1e-8, color="red", linestyle="--", alpha=0.5, label="threshold")
ax.axvline(conv["policy_stable_since"], color="green", linestyle="--",
           alpha=0.5, label=f"policy stable (iter {conv['policy_stable_since']})")
ax.set_xlabel("Iteration")
ax.set_ylabel("max |V_new - V|")
ax.set_title("Isolated (Noise)")
ax.legend(fontsize=8)

# Mixed
ax = axes2[1]
deltas = mix_data["convergence_deltas"]
conv = mix_data["convergence"]
ax.semilogy(deltas, linewidth=1.5)
ax.axhline(1e-8, color="red", linestyle="--", alpha=0.5, label="threshold")
ax.axvline(conv["policy_stable_since"], color="green", linestyle="--",
           alpha=0.5, label=f"policy stable (iter {conv['policy_stable_since']})")
ax.set_xlabel("Iteration")
ax.set_ylabel("max |V_new - V|")
ax.set_title("Mixed Regimes")
ax.legend(fontsize=8)

plt.tight_layout()
save_fig(fig2, "vi_convergence.png", script_file=__file__)

# ── Print eval summary ──
mix_vi_traj = eval_data["mixed_vi"]
print("\nEvaluation summary (from vi_eval.json):")
print(f"  Mixed VI oracle:  {mix_vi_traj['mean_reward']:.2f} "
      f"(std={mix_vi_traj['std_reward']:.2f})")
for name, stats in eval_data["per_regime"].items():
    print(f"  {name:>5s} locked: VI={stats['vi_mean']:.2f} "
          f"(std={stats['vi_std']:.2f})")

print("\nDone.")
