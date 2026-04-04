"""Plot VI oracle results from precomputed JSON files.

Usage:
    uv run python scripts/plot_vi.py

Reads:
    results/vi_isolated.json
    results/vi_optimal.json
    results/vi_eval.json

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

# ── Plot 3: Cumulative reward trajectories ──
iso_traj = eval_data["isolated_trajectories"]
mix_traj = eval_data["mixed_trajectories"]
mix_vi_traj = eval_data["mixed_vi"]
n_eval = eval_data["n_eval"]

fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5))
fig3.suptitle("VI Oracle — Cumulative Reward Trajectories",
              fontsize=14, fontweight="bold")

# Panels 1-3: per locked regime (isolated VI)
for r, name in enumerate(["Noise", "Bull", "Bear"]):
    ax = axes3[r]
    color = REGIME_COLORS[r]

    iso = iso_traj[name]
    iso_mean = np.array(iso["cumulative_reward_mean"])
    iso_sem = np.array(iso["cumulative_reward_std"]) / np.sqrt(n_eval)
    steps = np.arange(len(iso_mean))

    ax.plot(steps, iso_mean, color=color, linewidth=1.5, label="Isolated VI")
    ax.fill_between(steps, iso_mean - iso_sem, iso_mean + iso_sem,
                    color=color, alpha=0.2)

    ax.set_title(f"{name} (locked)")
    ax.set_xlabel("Step")
    if r == 0:
        ax.set_ylabel("Cumulative reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Panel 4: mixed VI on unlocked (switching) regimes
ax = axes3[3]
if "cumulative_reward_mean" in mix_vi_traj:
    mv_mean = np.array(mix_vi_traj["cumulative_reward_mean"])
    mv_sem = np.array(mix_vi_traj["cumulative_reward_std"]) / np.sqrt(n_eval)
    steps = np.arange(len(mv_mean))
    ax.plot(steps, mv_mean, color="#7B1FA2", linewidth=1.5, label="Mixed VI")
    ax.fill_between(steps, mv_mean - mv_sem, mv_mean + mv_sem,
                    color="#7B1FA2", alpha=0.2)
    ax.set_title("Mixed (switching)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
else:
    ax.set_visible(False)

plt.tight_layout()
save_fig(fig3, "vi_trajectories.png", script_file=__file__)

# ── Print eval summary ──
print("\nEvaluation summary (from vi_eval.json):")
print(f"  Mixed VI oracle:  {mix_vi_traj['mean_reward']:.2f} "
      f"(std={mix_vi_traj['std_reward']:.2f})")
for name, stats in eval_data["per_regime"].items():
    print(f"  {name:>5s} locked: VI={stats['vi_mean']:.2f} "
          f"(std={stats['vi_std']:.2f})")

print("\nDone.")
