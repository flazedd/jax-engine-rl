"""Plot agent evaluation results vs VI oracle.

Usage:
    uv run python scripts/plot_agent.py                          # ppo, all regimes
    uv run python scripts/plot_agent.py --agent ppo --regime bull
    uv run python scripts/plot_agent.py --agent rl2

Reads:
    plots/{agent}_{regime}.json  — from compute_agent.py
    plots/vi_isolated.json       — from compute_vi_oracle.py
    plots/vi_optimal.json        — from compute_vi_oracle.py
    plots/vi_eval.json           — from compute_vi_oracle.py

Saves:
    plots/{agent}_{regime}_vs_vi.png — action-inventory grid + cumulative reward
"""
import argparse
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lob_sim.actions import ACTION_TABLE, N_ACTIONS
from plot_style import (
    apply_style, save_fig,
    REGIME_NAMES, REGIME_COLORS, MIXED_COLOR,
)

PLOTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "plots"))

VI_COLOR = "#E53935"  # red for VI oracle lines

parser = argparse.ArgumentParser()
parser.add_argument("--agent", default="ppo")
parser.add_argument("--regime", default=None,
                    choices=["noise", "bull", "bear", "mixed"])
args = parser.parse_args()


def _load(filename):
    path = os.path.join(PLOTS_DIR, filename)
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
    filepath = os.path.join(PLOTS_DIR, filename)
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
    else:
        vi_reward = vi_eval["per_regime"][regime_name.capitalize()]["vi_mean"]

    n_eval_regimes = len(eval_regimes)

    # ── Figure layout ──
    # Row 0: training curve (spans all columns)
    # Row 1: cumulative reward per eval regime (agent + VI overlay)
    # Row 2: action-inventory heatmap (agent)
    # Row 3: action-inventory grid (VI optimal)
    fig, axes = plt.subplots(4, n_eval_regimes,
                             figsize=(6 * n_eval_regimes, 18),
                             squeeze=False)

    regime_title = regime_name.capitalize() if not is_mixed else "Mixed"
    fig.suptitle(f"{agent_upper} trained on {regime_title} regime vs VI Oracle",
                 fontsize=14, fontweight="bold")

    # ── Row 0: Training curve (merge all columns) ──
    ax_train = fig.add_subplot(4, 1, 1)
    for c in range(n_eval_regimes):
        axes[0, c].set_visible(False)

    training = agent_data["training"]
    ax_train.plot(training["iters"], training["rewards"],
                  color=MIXED_COLOR, linewidth=2, label=f"{agent_upper}")
    ax_train.axhline(vi_reward, color=VI_COLOR, linestyle="--", alpha=0.7,
                     label=f"VI oracle ({vi_reward:.1f})")
    ax_train.set_xlabel("Training iteration")
    ax_train.set_ylabel("Mean eval reward")
    ax_train.set_title("Learning Curve")
    ax_train.legend(fontsize=9)
    ax_train.grid(True, alpha=0.3)

    # ── Rows 1-3: per eval regime ──
    cmap_freq = plt.cm.YlOrRd

    for c, eval_name in enumerate(eval_regimes):
        eval_d = agent_data["eval"][eval_name]
        r_idx = regime_names_lower.index(eval_name)
        color = REGIME_COLORS[r_idx]

        # Row 1: cumulative reward trajectory (agent + VI on same seeds)
        ax_cum = axes[1, c]
        cum_mean = np.array(eval_d["cumulative_reward_mean"])
        cum_std = np.array(eval_d["cumulative_reward_std"])
        steps = np.arange(len(cum_mean))

        ax_cum.plot(steps, cum_mean, color=color, linewidth=1.5,
                    label=agent_upper)
        ax_cum.fill_between(steps, cum_mean - cum_std, cum_mean + cum_std,
                            color=color, alpha=0.1)

        # Overlay VI trajectory if available (same seeds = paired)
        eval_name_cap = eval_name.capitalize()
        if eval_name_cap in vi_trajectories:
            vi_cum = np.array(vi_trajectories[eval_name_cap]["cumulative_reward_mean"])
            vi_cum_std = np.array(vi_trajectories[eval_name_cap]["cumulative_reward_std"])
            ax_cum.plot(steps[:len(vi_cum)], vi_cum, color=VI_COLOR,
                        linewidth=1.5, linestyle="--", label="VI oracle")
            ax_cum.fill_between(steps[:len(vi_cum)],
                                vi_cum - vi_cum_std, vi_cum + vi_cum_std,
                                color=VI_COLOR, alpha=0.1)

        ax_cum.set_title(f"{eval_name_cap} — Cumulative Reward")
        ax_cum.set_xlabel("Step")
        ax_cum.set_ylabel("Cumulative reward")
        ax_cum.legend(fontsize=8)
        ax_cum.grid(True, alpha=0.3)

        # Row 2: agent action-inventory heatmap
        ax_agent = axes[2, c]
        freq = np.array(eval_d["action_inventory_freq"])
        freq_disp = freq[:, lo:hi]

        im = ax_agent.imshow(freq_disp, aspect="auto", cmap=cmap_freq,
                             vmin=0, vmax=max(0.5, freq_disp.max()),
                             interpolation="nearest", origin="lower")

        ax_agent.set_xticks(np.arange(-0.5, n_disp, 1), minor=True)
        ax_agent.set_yticks(np.arange(-0.5, N_ACTIONS, 1), minor=True)
        ax_agent.grid(which="minor", color="grey", linewidth=0.3, alpha=0.5)
        ax_agent.tick_params(which="minor", length=0)

        major_x = np.arange(0, n_disp, 5)
        ax_agent.set_xticks(major_x)
        ax_agent.set_xticklabels([str(disp_inv[i]) for i in major_x], fontsize=7)
        ax_agent.set_yticks(range(N_ACTIONS))
        ax_agent.set_yticklabels(action_labels, fontsize=8)
        ax_agent.set_title(f"{eval_name_cap} — {agent_upper} P(action|inv)")
        ax_agent.set_ylabel("(bid, ask) ticks")
        if c == n_eval_regimes - 1:
            plt.colorbar(im, ax=ax_agent, fraction=0.046, pad=0.04)

        for spine in ax_agent.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        # Row 3: VI optimal (green grid)
        ax_vi = axes[3, c]
        vi_grid = np.zeros((N_ACTIONS, n_disp))
        for i in range(n_disp):
            vi_grid[vi_policy[r_idx, lo + i], i] = 1.0

        cmap_vi = ListedColormap(["white", "#4CAF50"])
        ax_vi.imshow(vi_grid, aspect="auto", cmap=cmap_vi, vmin=0, vmax=1,
                     interpolation="nearest", origin="lower")

        ax_vi.set_xticks(np.arange(-0.5, n_disp, 1), minor=True)
        ax_vi.set_yticks(np.arange(-0.5, N_ACTIONS, 1), minor=True)
        ax_vi.grid(which="minor", color="grey", linewidth=0.3, alpha=0.5)
        ax_vi.tick_params(which="minor", length=0)

        ax_vi.set_xticks(major_x)
        ax_vi.set_xticklabels([str(disp_inv[i]) for i in major_x], fontsize=7)
        ax_vi.set_yticks(range(N_ACTIONS))
        ax_vi.set_yticklabels(action_labels, fontsize=8)
        vi_label = "Isolated VI" if not is_mixed else "Mixed VI"
        ax_vi.set_title(f"{eval_name_cap} — {vi_label} Optimal")
        ax_vi.set_ylabel("(bid, ask) ticks")
        ax_vi.set_xlabel("Inventory")

        for spine in ax_vi.spines.values():
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
