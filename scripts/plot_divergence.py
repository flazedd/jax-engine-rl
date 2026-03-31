"""Phase 4 — policy divergence heatmaps.

Runs full N=200, T=1000 Monte Carlo evaluation, then generates
plots/divergence.png with three 5x5 heatmaps (shared color scale),
optimal action per regime, cross-regime penalty table, and Cohen's d.
"""
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from lob_sim.actions import ACTION_TABLE, N_ACTIONS
from lob_sim.config import SimConfig
from lob_sim.regime import N_REGIMES
from lob_sim.step import run_episode

N_EPISODES = 200
T_STEPS = 1000
REGIME_NAMES = ["Noise", "Bull", "Bear"]

config = SimConfig(max_steps=T_STEPS)


def compute_reward_stats(config, action_idx, regime_idx, master_key):
    """Return (mean_total_reward, std_total_reward) over N_EPISODES."""
    keys = jax.random.split(master_key, N_EPISODES)
    actions_repeated = jnp.full((T_STEPS,), action_idx, dtype=jnp.int32)
    batched_run = jax.vmap(
        lambda k: run_episode(config, k, actions_repeated, locked_regime=regime_idx)
    )
    _, outputs = batched_run(keys)
    total_rewards = outputs["reward"].sum(axis=1)
    return total_rewards.mean(), total_rewards.std()


def main():
    master_key = jax.random.PRNGKey(42)

    mean_matrix = np.zeros((N_ACTIONS, N_REGIMES))
    std_matrix = np.zeros((N_ACTIONS, N_REGIMES))

    jit_compute = jax.jit(compute_reward_stats, static_argnums=(0, 1, 2))

    print("Computing reward matrix (25 actions x 3 regimes)...")
    for regime_idx in range(N_REGIMES):
        for action_idx in range(N_ACTIONS):
            key = jax.random.fold_in(master_key, regime_idx * N_ACTIONS + action_idx)
            m, s = jit_compute(config, action_idx, regime_idx, key)
            mean_matrix[action_idx, regime_idx] = float(m)
            std_matrix[action_idx, regime_idx] = float(s)
            print(f"  regime={REGIME_NAMES[regime_idx]:5s} action={action_idx:2d} "
                  f"mean={float(m):+8.2f} std={float(s):6.2f}")

    # --- Optimal actions ---
    print("\n=== Optimal Actions ===")
    for r in range(N_REGIMES):
        opt = mean_matrix[:, r].argmax()
        bid, ask = ACTION_TABLE[opt]
        print(f"  {REGIME_NAMES[r]:5s}: action {opt:2d} (bid={int(bid)}, ask={int(ask)}) "
              f"reward={mean_matrix[opt, r]:+.2f}")

    # --- Cross-regime penalty table ---
    print("\n=== Cross-Regime Penalty Table ===")
    header = f"{'Apply↓ / In→':>16s}"
    for r in range(N_REGIMES):
        header += f"  {REGIME_NAMES[r]:>8s}"
    print(header)
    for r_apply in range(N_REGIMES):
        opt_apply = mean_matrix[:, r_apply].argmax()
        row = f"  {REGIME_NAMES[r_apply]:>14s}"
        for r_in in range(N_REGIMES):
            reward = mean_matrix[opt_apply, r_in]
            best = mean_matrix[:, r_in].max()
            pct = (reward - best) / (abs(best) + 1e-8) * 100
            row += f"  {pct:+7.1f}%"
        print(row)

    # --- Cohen's d ---
    print("\n=== Cohen's d (optimal vs second-best) ===")
    for r in range(N_REGIMES):
        sorted_idx = np.argsort(mean_matrix[:, r])[::-1]
        best_idx, second_idx = sorted_idx[0], sorted_idx[1]
        d = (mean_matrix[best_idx, r] - mean_matrix[second_idx, r]) / (
            (std_matrix[best_idx, r] + std_matrix[second_idx, r]) / 2 + 1e-8
        )
        print(f"  {REGIME_NAMES[r]:5s}: d = {d:.3f}")

    # --- Heatmaps ---
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    reshaped = mean_matrix.reshape(5, 5, N_REGIMES)

    vmin = mean_matrix.min()
    vmax = mean_matrix.max()

    for r in range(N_REGIMES):
        ax = axes[r]
        im = ax.imshow(reshaped[:, :, r], cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       origin="lower", aspect="equal")
        ax.set_title(f"{REGIME_NAMES[r]} Regime")
        ax.set_xlabel("Ask Ticks")
        ax.set_ylabel("Bid Ticks")
        ax.set_xticks(range(5))
        ax.set_xticklabels([1, 2, 3, 4, 5])
        ax.set_yticks(range(5))
        ax.set_yticklabels([1, 2, 3, 4, 5])

        # Annotate cells
        for i in range(5):
            for j in range(5):
                val = reshaped[i, j, r]
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color="black")

        # Mark optimal
        opt = mean_matrix[:, r].argmax()
        opt_bid = opt // 5
        opt_ask = opt % 5
        ax.add_patch(plt.Rectangle(
            (opt_ask - 0.5, opt_bid - 0.5), 1, 1,
            fill=False, edgecolor="blue", linewidth=3
        ))

    fig.colorbar(im, ax=axes, label="Mean Total Reward", shrink=0.8)
    fig.suptitle("Policy Divergence: Mean Reward by Action & Regime", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(plot_dir, "divergence.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
