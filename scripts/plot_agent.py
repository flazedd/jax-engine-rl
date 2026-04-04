"""Plot agent evaluation results vs VI oracle.

Usage:
    uv run python scripts/plot_agent.py                          # ppo, all regimes
    uv run python scripts/plot_agent.py --agent ppo --regime bull
    uv run python scripts/plot_agent.py --agent rl2

Reads:
    results/{agent}_{regime}.json  — from compute_agent.py
    results/vi_isolated.json       — from compute_vi_oracle.py
    results/vi_optimal.json        — from compute_vi_oracle.py
    results/vi_eval.json           — from compute_vi_oracle.py

Saves:
    plots/{agent}_{regime}_vs_vi.png — training curve + action heatmaps
"""
import argparse
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch

from lob_sim.actions import ACTION_TABLE, N_ACTIONS
from plot_style import (
    apply_style, save_fig,
    REGIME_NAMES, REGIME_COLORS, MIXED_COLOR,
)

RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))

VI_COLOR = "#E53935"  # red for VI oracle lines
NO_DATA_COLOR = "#D3D3D3"  # light grey for unvisited inventory levels

parser = argparse.ArgumentParser()
parser.add_argument("--agent", default="ppo")
parser.add_argument("--regime", default=None,
                    choices=["noise", "bull", "bear", "mixed"])
args = parser.parse_args()


def _load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        return json.load(f)


# ── Load VI data ──
vi_iso = _load("vi_isolated.json")
vi_mix = _load("vi_optimal.json")
vi_eval = _load("vi_eval.json")

iso_policy = np.array(vi_iso["policy"])
mix_policy = np.array(vi_mix["policy"])
max_inv = vi_iso["max_inventory"]
n_inv = 2 * max_inv + 1

at = np.array(ACTION_TABLE)
action_labels = [f"({at[a][0]},{at[a][1]})" for a in range(N_ACTIONS)]

# Display range (crop boundary artifacts)
display_range = max_inv - 5
lo = max_inv - display_range
hi = max_inv + display_range + 1
n_disp = hi - lo
disp_inv = np.arange(n_inv)[lo:hi] - max_inv

apply_style()

# ── Determine which files to plot ──
regime_names_lower = ["noise", "bull", "bear"]
regimes_to_plot = [args.regime] if args.regime else regime_names_lower + ["mixed"]

for regime_name in regimes_to_plot:
    filename = f"{args.agent}_{regime_name}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  Skipping {regime_name} — {filename} not found")
        continue

    agent_data = _load(filename)
    agent_upper = args.agent.upper()

    eval_regimes = list(agent_data["eval"].keys())
    is_mixed = regime_name == "mixed"

    vi_policy = mix_policy if is_mixed else iso_policy

    # VI trajectory data (same seeds as agent eval)
    vi_traj_key = "mixed_trajectories" if is_mixed else "isolated_trajectories"
    vi_trajectories = vi_eval.get(vi_traj_key, {})

    # VI eval reward for training curve reference line
    if is_mixed:
        vi_reward = vi_eval["mixed_vi"]["mean_reward"]
        vi_reward_std = vi_eval["mixed_vi"]["std_reward"]
    else:
        vi_reward = vi_eval["per_regime"][regime_name.capitalize()]["vi_mean"]
        vi_reward_std = vi_eval["per_regime"][regime_name.capitalize()]["vi_std"]

    n_eval_regimes = len(eval_regimes)

    # ── Figure layout: 1 row ──
    # Col 0: training curve
    # Cols 1+: stacked bar per eval regime (agent action dist + VI optimal markers)
    n_cols = 1 + n_eval_regimes
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(6 * n_cols, 5),
                             squeeze=False)

    regime_title = regime_name.capitalize() if not is_mixed else "Mixed"
    fig.suptitle(f"{agent_upper} trained on {regime_title} regime vs VI Oracle",
                 fontsize=14, fontweight="bold")

    # ── Col 0: Training curve ──
    ax_train = axes[0, 0]
    training = agent_data["training"]
    ax_train.plot(training["iters"], training["rewards"],
                  color=MIXED_COLOR, linewidth=2, label=f"{agent_upper}")
    ax_train.axhline(vi_reward, color=VI_COLOR, linestyle="--", alpha=0.7,
                     label=f"VI oracle ({vi_reward:.1f})")
    iters = training["iters"]
    ax_train.fill_between(iters,
                          vi_reward - vi_reward_std, vi_reward + vi_reward_std,
                          color=VI_COLOR, alpha=0.12)
    ax_train.set_xlabel("Training iteration")
    ax_train.set_ylabel("Mean eval reward")
    ax_train.set_title("Learning Curve")
    ax_train.legend(fontsize=9)
    ax_train.grid(True, alpha=0.3)

    # ── Remaining cols: agent heatmap with VI overlay per eval regime ──
    cmap_freq = plt.cm.YlOrRd

    for c, eval_name in enumerate(eval_regimes):
        eval_d = agent_data["eval"][eval_name]
        r_idx = regime_names_lower.index(eval_name)
        eval_name_cap = eval_name.capitalize()
        vi_label = "Isolated VI" if not is_mixed else "Mixed VI"

        ax = axes[0, 1 + c]
        freq = np.array(eval_d["action_inventory_freq"])
        freq_disp = freq[:, lo:hi]
        col_visited = freq_disp.sum(axis=0) > 0

        im = ax.imshow(freq_disp, aspect="auto", cmap=cmap_freq,
                        vmin=0, vmax=max(0.5, freq_disp.max()),
                        interpolation="nearest", origin="lower")

        # Grey out unvisited columns
        has_no_data = False
        for i in range(n_disp):
            if not col_visited[i]:
                ax.axvspan(i - 0.5, i + 0.5, color=NO_DATA_COLOR, zorder=2,
                           label="No data" if not has_no_data else None)
                has_no_data = True

        # Overlay VI optimal action as green markers
        for i in range(n_disp):
            vi_a = vi_policy[r_idx, lo + i]
            ax.plot(i, vi_a, marker="s", color="#4CAF50", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5,
                    label=f"{vi_label} optimal" if i == 0 else None)

        ax.set_xticks(np.arange(-0.5, n_disp, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N_ACTIONS, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.3, alpha=0.5)
        ax.tick_params(which="minor", length=0)

        major_x = np.arange(0, n_disp, 5)
        ax.set_xticks(major_x)
        ax.set_xticklabels([str(disp_inv[i]) for i in major_x], fontsize=7)
        ax.set_yticks(range(N_ACTIONS))
        ax.set_yticklabels(action_labels, fontsize=8)
        ax.set_title(f"{eval_name_cap}")
        ax.set_ylabel("(bid, ask) ticks")
        ax.set_xlabel("Inventory")
        # Add a legend entry for the heatmap
        heat_patch = Patch(facecolor=cmap_freq(0.6), edgecolor="none",
                           label=f"{agent_upper} P(action|inv)")
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0, heat_patch)
        ax.legend(handles=handles, fontsize=6, loc="upper right")

        if c == n_eval_regimes - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    plt.tight_layout()
    out_name = f"{args.agent}_{regime_name}_vs_vi.png"
    save_fig(fig, out_name, script_file=__file__)
    plt.close(fig)

    # Print summary
    for eval_name in eval_regimes:
        ed = agent_data["eval"][eval_name]
        eval_name_cap = eval_name.capitalize()
        vi_mean_str = ""
        if eval_name_cap in vi_trajectories:
            vi_ep = vi_trajectories[eval_name_cap]["episode_rewards"]
            vi_mean_str = f", VI={np.mean(vi_ep):.2f}"
        print(f"  {eval_name_cap:>5s}: agent={ed['mean_reward']:.2f} "
              f"(std={ed['std_reward']:.2f}){vi_mean_str}")

print("\nDone.")
