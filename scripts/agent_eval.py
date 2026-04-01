"""Reusable evaluation and plotting utilities for any agent.

Provides:
- collect_eval_data: run eval episodes collecting actions, rewards, regimes
- action_freq_matrix: convert action indices to BID_TICKS x ASK_TICKS frequency grid
- plot_action_heatmap: annotated heatmap on given axes
- plot_learning_curve: reward vs iteration on given axes
- plot_pnl_with_regimes: cumulative PnL with colored regime background bands
- make_per_regime_figure: 3x2 figure (learning curve + action dist per regime)
- make_mixed_figure: 2x3 figure (per-regime actions + overall actions + learning curve + PnL)
"""
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lob_sim.actions import ACTION_TABLE, N_ACTIONS, BID_TICKS, ASK_TICKS
from lob_sim.config import SimConfig
from lob_sim.obs import observe
from lob_sim.state import init_state
from lob_sim.step import make_step_fn

REGIME_NAMES = ["Noise", "Bull", "Bear"]
REGIME_COLORS = ["#2196F3", "#4CAF50", "#F44336"]
REGIME_COLORS_LIGHT = ["#BBDEFB", "#C8E6C9", "#FFCDD2"]

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


class EvalData(NamedTuple):
    """Per-step data from an evaluation episode."""
    actions: jnp.ndarray    # (n_steps,) int32
    rewards: jnp.ndarray    # (n_steps,)
    regimes: jnp.ndarray    # (n_steps,) int32
    inventory: jnp.ndarray  # (n_steps,)
    mid_price: jnp.ndarray  # (n_steps,)


def collect_eval_data(agent, sim_config, rng_key, n_steps, locked_regime=-1):
    """Run one evaluation episode, returning per-step actions, rewards, regimes, inventory.

    Works with any agent that follows the Agent protocol.
    """
    step_fn = make_step_fn(sim_config, locked_regime=locked_regime)

    def scan_body(carry, _):
        sim_state, agent_state, rng = carry
        rng, rng_action = jax.random.split(rng)

        obs = observe(sim_state, sim_config)
        action, new_agent_state, _ = agent.get_action(obs, agent_state, rng_action)
        new_sim_state, sim_out = step_fn(sim_state, action)

        new_agent_state = new_agent_state._replace(
            prev_reward=sim_out["reward"],
            prev_done=sim_out["done"].astype(jnp.float32),
        )

        step_data = EvalData(
            actions=action,
            rewards=sim_out["reward"],
            regimes=new_sim_state.regime,
            inventory=new_sim_state.inventory,
            mid_price=new_sim_state.mid_price,
        )
        return (new_sim_state, new_agent_state, rng), step_data

    k_init, k_agent, k_run = jax.random.split(rng_key, 3)
    sim_state = init_state(sim_config, k_init)
    agent_state = agent.initial_agent_state(k_agent)

    _, data = jax.lax.scan(scan_body, (sim_state, agent_state, k_run), None, length=n_steps)
    return data


def collect_eval_data_batch(agent, sim_config, rng_key, n_episodes, n_steps, locked_regime=-1):
    """vmap collect_eval_data over n_episodes. Returns EvalData with leading dim n_episodes."""
    keys = jax.random.split(rng_key, n_episodes)
    return jax.vmap(
        lambda k: collect_eval_data(agent, sim_config, k, n_steps, locked_regime)
    )(keys)


def action_freq_matrix(actions):
    """Convert flat action indices to BID_TICKS x ASK_TICKS frequency matrix."""
    flat = np.asarray(actions).reshape(-1)
    counts = np.bincount(flat, minlength=N_ACTIONS)
    freqs = counts / counts.sum()
    return freqs.reshape(len(BID_TICKS), len(ASK_TICKS))


def plot_action_heatmap(ax, freq_matrix, title, vmin=0, vmax=None):
    """Plot action frequency heatmap with percentage annotations."""
    if vmax is None:
        vmax = max(0.4, freq_matrix.max() * 1.1)
    im = ax.imshow(freq_matrix, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                   aspect="equal", origin="lower")
    ax.set_xticks(range(len(ASK_TICKS)))
    ax.set_xticklabels(ASK_TICKS)
    ax.set_yticks(range(len(BID_TICKS)))
    ax.set_yticklabels(BID_TICKS)
    ax.set_xlabel("Ask ticks")
    ax.set_ylabel("Bid ticks")
    ax.set_title(title, fontsize=11)
    for i in range(freq_matrix.shape[0]):
        for j in range(freq_matrix.shape[1]):
            pct = freq_matrix[i, j] * 100
            color = "white" if freq_matrix[i, j] > vmax * 0.55 else "black"
            ax.text(j, i, f"{pct:.1f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    return im


def plot_learning_curve(ax, iters, rewards, title, color="b"):
    """Plot learning curve (mean eval reward vs training iteration)."""
    ax.plot(iters, rewards, color=color, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title(title, fontsize=11)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.3)


def plot_pnl_with_regimes(ax, rewards, regimes, title):
    """Plot cumulative PnL with colored background bands showing regime switches."""
    cum_pnl = np.cumsum(rewards)
    steps = np.arange(len(rewards))

    # Draw regime background bands
    regime_arr = np.asarray(regimes)
    i = 0
    while i < len(regime_arr):
        r = int(regime_arr[i])
        j = i + 1
        while j < len(regime_arr) and int(regime_arr[j]) == r:
            j += 1
        ax.axvspan(i, j, alpha=0.15, color=REGIME_COLORS_LIGHT[r], linewidth=0)
        i = j

    ax.plot(steps, cum_pnl, "k-", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)

    # Legend for regime colors
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=REGIME_COLORS_LIGHT[r], label=REGIME_NAMES[r])
               for r in range(3)]
    ax.legend(handles=patches, fontsize=8, loc="upper left")


def make_per_regime_figure(agents_by_regime, reward_histories, sim_config,
                           agent_name="PPO", n_eval_episodes=20):
    """Create per-regime analysis figure (3x2): learning curve + action heatmap per regime.

    Args:
        agents_by_regime: dict {regime_idx: trained_agent}
        reward_histories: dict {regime_idx: (iters_list, rewards_list)}
        sim_config: SimConfig
        agent_name: name for titles and filename
        n_eval_episodes: episodes for action distribution evaluation
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    for regime_idx in range(3):
        agent = agents_by_regime[regime_idx]
        iters, rewards = reward_histories[regime_idx]
        color = REGIME_COLORS[regime_idx]
        name = REGIME_NAMES[regime_idx]

        # Col 0: Learning curve
        plot_learning_curve(axes[regime_idx, 0], iters, rewards,
                           f"{name} — Learning Curve", color=color)

        # Col 1: Action distribution
        key = jax.random.PRNGKey(99 + regime_idx)
        data = collect_eval_data_batch(agent, sim_config, key,
                                       n_episodes=n_eval_episodes,
                                       n_steps=sim_config.max_steps,
                                       locked_regime=regime_idx)
        freq = action_freq_matrix(data.actions)
        plot_action_heatmap(axes[regime_idx, 1], freq, f"{name} — Actions")

    fig.suptitle(f"{agent_name} Trained Per-Regime", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, f"{agent_name.lower()}_per_regime.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_mixed_figure(agent, reward_history, sim_config,
                      agent_name="PPO", n_eval_episodes=20,
                      per_regime_freqs=None):
    """Create mixed-regime analysis figure (2x3).

    Top row: action distribution evaluated on Noise / Bull / Bear separately
    Bottom row: overall action distribution | learning curve | PnL with regime bands

    Args:
        agent: trained agent (trained on mixed regimes)
        reward_history: (iters_list, rewards_list)
        sim_config: SimConfig
        agent_name: name for titles and filename
        n_eval_episodes: episodes for action distribution evaluation
        per_regime_freqs: optional dict {regime_idx: freq_matrix} from per-regime training,
                          used to overlay stars marking per-regime optimal actions
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Top row: per-regime action distributions (using mixed-trained agent)
    all_actions = []
    for regime_idx in range(3):
        key = jax.random.PRNGKey(200 + regime_idx)
        data = collect_eval_data_batch(agent, sim_config, key,
                                       n_episodes=n_eval_episodes,
                                       n_steps=sim_config.max_steps,
                                       locked_regime=regime_idx)
        freq = action_freq_matrix(data.actions)
        plot_action_heatmap(axes[0, regime_idx], freq,
                           f"Actions in {REGIME_NAMES[regime_idx]}")
        all_actions.append(np.asarray(data.actions).reshape(-1))

        # Overlay per-regime optimal as star if available
        if per_regime_freqs is not None and regime_idx in per_regime_freqs:
            opt_flat = np.argmax(per_regime_freqs[regime_idx])
            opt_row, opt_col = divmod(opt_flat, len(ASK_TICKS))
            axes[0, regime_idx].plot(
                opt_col, opt_row, marker="*", markersize=18,
                color=REGIME_COLORS[regime_idx], markeredgecolor="black",
                markeredgewidth=1.0)

    # Bottom-left: overall action distribution
    all_actions_flat = np.concatenate(all_actions)
    overall_freq = action_freq_matrix(all_actions_flat)
    plot_action_heatmap(axes[1, 0], overall_freq, "Actions Overall (Mixed)")

    # Bottom-center: learning curve
    iters, rewards = reward_history
    plot_learning_curve(axes[1, 1], iters, rewards, "Learning Curve", color="black")

    # Bottom-right: PnL with regime bands (single eval episode, unlocked regime)
    key = jax.random.PRNGKey(42)
    eval_data = collect_eval_data(agent, sim_config, key,
                                  n_steps=sim_config.max_steps,
                                  locked_regime=-1)
    plot_pnl_with_regimes(axes[1, 2], np.asarray(eval_data.rewards),
                          np.asarray(eval_data.regimes),
                          "PnL with Regime Switches")

    fig.suptitle(f"{agent_name} on Mixed Regimes", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, f"{agent_name.lower()}_mixed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
