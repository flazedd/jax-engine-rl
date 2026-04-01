"""Phase 4 — policy divergence heatmaps with convergence diagnostics.

Runs Monte Carlo evaluation, generates plots/divergence.png with heatmaps,
and prints convergence analysis (bootstrap confidence + multi-seed check).
"""
import os
import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from lob_sim.actions import ACTION_TABLE, N_ACTIONS, BID_TICKS, ASK_TICKS
from lob_sim.config import SimConfig
from lob_sim.regime import N_REGIMES
from lob_sim.step import run_episode

N_BID = len(BID_TICKS)
N_ASK = len(ASK_TICKS)

_parser = argparse.ArgumentParser()
_parser.add_argument("--fast", action="store_true",
                     help="Quick smoke test with fewer episodes/steps")
_args = _parser.parse_args()

N_EPISODES = 10 if _args.fast else 200
T_STEPS = 200 if _args.fast else 1000
N_SEEDS = 2 if _args.fast else 3
N_BOOTSTRAP = 100 if _args.fast else 1000
REGIME_NAMES = ["Noise", "Bull", "Bear"]


def compute_episode_rewards(config, action_idx, regime_idx, master_key, n_episodes):
    """Return per-episode total rewards: shape (n_episodes,)."""
    t_steps = config.max_steps
    keys = jax.random.split(master_key, n_episodes)
    actions_repeated = jnp.full((t_steps,), action_idx, dtype=jnp.int32)
    batched_run = jax.vmap(
        lambda k: run_episode(config, k, actions_repeated, locked_regime=regime_idx)
    )
    _, outputs = batched_run(keys)
    return outputs["reward"].sum(axis=1)


def action_str(idx):
    bid, ask = int(ACTION_TABLE[idx][0]), int(ACTION_TABLE[idx][1])
    return f"({bid},{ask})"


def main():
    config = SimConfig(max_steps=T_STEPS)

    # ── Step 1: Compute per-episode rewards for all (action, regime, seed) ──
    # Shape: (N_SEEDS, N_ACTIONS, N_REGIMES, N_EPISODES)
    all_rewards = np.zeros((N_SEEDS, N_ACTIONS, N_REGIMES, N_EPISODES))

    jit_compute = jax.jit(compute_episode_rewards, static_argnums=(0, 1, 2, 4))

    for seed_idx in range(N_SEEDS):
        master_key = jax.random.PRNGKey(42 + seed_idx)
        print(f"Computing reward matrix (seed={42 + seed_idx}, "
              f"{N_ACTIONS} actions x {N_REGIMES} regimes x {N_EPISODES} episodes)...")
        for regime_idx in range(N_REGIMES):
            for action_idx in range(N_ACTIONS):
                key = jax.random.fold_in(master_key, regime_idx * N_ACTIONS + action_idx)
                rewards = jit_compute(config, action_idx, regime_idx, key, N_EPISODES)
                all_rewards[seed_idx, action_idx, regime_idx] = np.array(rewards)

    # Use first seed as the primary result
    primary_rewards = all_rewards[0]  # (N_ACTIONS, N_REGIMES, N_EPISODES)
    mean_matrix = primary_rewards.mean(axis=2)  # (N_ACTIONS, N_REGIMES)
    std_matrix = primary_rewards.std(axis=2)
    se_matrix = std_matrix / np.sqrt(N_EPISODES)

    # ── Step 2: Print primary results ──
    print("\n=== Optimal Actions ===")
    for r in range(N_REGIMES):
        opt = mean_matrix[:, r].argmax()
        print(f"  {REGIME_NAMES[r]:5s}: action {opt:2d} {action_str(opt)} "
              f"reward={mean_matrix[opt, r]:+.2f} ± {se_matrix[opt, r]:.2f} (SE)")

    # Cross-regime penalty table
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

    # Cohen's d
    print("\n=== Cohen's d (optimal vs second-best) ===")
    for r in range(N_REGIMES):
        sorted_idx = np.argsort(mean_matrix[:, r])[::-1]
        best_idx, second_idx = sorted_idx[0], sorted_idx[1]
        d = (mean_matrix[best_idx, r] - mean_matrix[second_idx, r]) / (
            (std_matrix[best_idx, r] + std_matrix[second_idx, r]) / 2 + 1e-8
        )
        print(f"  {REGIME_NAMES[r]:5s}: d = {d:.3f}")

    # ── Step 3: Bootstrap convergence ──
    print(f"\n=== Bootstrap Convergence ({N_BOOTSTRAP} resamples) ===")
    rng = np.random.default_rng(0)

    for r in range(N_REGIMES):
        bootstrap_optima = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(N_EPISODES, size=N_EPISODES, replace=True)
            resampled_means = primary_rewards[:, r, idx].mean(axis=1)
            bootstrap_optima.append(resampled_means.argmax())

        bootstrap_optima = np.array(bootstrap_optima)
        primary_opt = mean_matrix[:, r].argmax()
        agreement = (bootstrap_optima == primary_opt).mean() * 100

        # Find all actions that ever won
        unique, counts = np.unique(bootstrap_optima, return_counts=True)
        winners = sorted(zip(counts, unique), reverse=True)

        parts = [f"{action_str(int(a))} {c / N_BOOTSTRAP * 100:.1f}%"
                 for c, a in winners]
        print(f"  {REGIME_NAMES[r]:5s}: {action_str(primary_opt)} wins {agreement:.1f}% — "
              f"[{', '.join(parts)}]")

    # ── Step 4: Multi-seed check ──
    print(f"\n=== Multi-Seed Check ({N_SEEDS} seeds) ===")
    all_agree = True
    for r in range(N_REGIMES):
        seed_optima = []
        for s in range(N_SEEDS):
            seed_mean = all_rewards[s, :, r, :].mean(axis=1)
            opt = seed_mean.argmax()
            seed_optima.append(opt)

        actions_str = [f"seed {42+s}: {action_str(o)}" for s, o in enumerate(seed_optima)]
        agree = len(set(seed_optima)) == 1
        if not agree:
            all_agree = False
        status = "AGREE" if agree else "DISAGREE"
        print(f"  {REGIME_NAMES[r]:5s}: {status} — {', '.join(actions_str)}")

    if all_agree:
        print("\n  All seeds agree on optimal actions — results are stable.")
    else:
        print("\n  WARNING: Seeds disagree — consider increasing N_EPISODES or T_STEPS.")

    # ── Step 5: Reward estimate stability (SE as % of gap to second-best) ──
    print("\n=== Estimate Precision ===")
    for r in range(N_REGIMES):
        sorted_idx = np.argsort(mean_matrix[:, r])[::-1]
        best_idx, second_idx = sorted_idx[0], sorted_idx[1]
        gap = mean_matrix[best_idx, r] - mean_matrix[second_idx, r]
        se = se_matrix[best_idx, r]
        ratio = se / (gap + 1e-8) * 100
        print(f"  {REGIME_NAMES[r]:5s}: best {action_str(best_idx)} = {mean_matrix[best_idx, r]:+.2f}, "
              f"2nd {action_str(int(second_idx))} = {mean_matrix[second_idx, r]:+.2f}, "
              f"gap = {gap:.2f}, SE = {se:.2f} ({ratio:.1f}% of gap)")

    # ── Step 6: Heatmaps ──
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    reshaped = mean_matrix.reshape(N_BID, N_ASK, N_REGIMES)

    vmin = mean_matrix.min()
    vmax = mean_matrix.max()

    for r in range(N_REGIMES):
        ax = axes[r]
        im = ax.imshow(reshaped[:, :, r], cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       origin="lower", aspect="equal")
        ax.set_title(f"{REGIME_NAMES[r]} Regime")
        ax.set_xlabel("Ask Ticks")
        ax.set_ylabel("Bid Ticks")
        ax.set_xticks(range(N_ASK))
        ax.set_xticklabels(ASK_TICKS)
        ax.set_yticks(range(N_BID))
        ax.set_yticklabels(BID_TICKS)

        # Annotate cells with mean ± SE
        se_reshaped = se_matrix.reshape(N_BID, N_ASK, N_REGIMES)
        for i in range(N_BID):
            for j in range(N_ASK):
                val = reshaped[i, j, r]
                se = se_reshaped[i, j, r]
                ax.text(j, i, f"{val:.0f}±{se:.0f}", ha="center", va="center",
                        fontsize=8, color="black")

        # Mark optimal
        opt = mean_matrix[:, r].argmax()
        opt_bid = opt // N_ASK
        opt_ask = opt % N_ASK
        ax.add_patch(plt.Rectangle(
            (opt_ask - 0.5, opt_bid - 0.5), 1, 1,
            fill=False, edgecolor="blue", linewidth=3
        ))

    fig.suptitle(f"Policy Divergence: Mean Reward by Action & Regime "
                 f"(N={N_EPISODES}, T={T_STEPS}, {N_SEEDS} seeds)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Mean Total Reward")
    out_path = os.path.join(plot_dir, "divergence.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
