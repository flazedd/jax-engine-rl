"""Train PPO on noise regime and analyze the learned policy."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from lob_sim.config import SimConfig
from lob_sim.actions import ACTION_TABLE, BID_TICKS, ASK_TICKS
from lob_sim.agents.ppo import PPOAgent, PPOConfig
from lob_sim.training.rollout import collect_rollout, collect_rollout_batch
from lob_sim.training.trainer import compute_gae, ppo_update, create_optimizer
from lob_sim.training.eval import evaluate_agent


def train(n_iterations=300, seed=42):
    key = jax.random.PRNGKey(seed)
    sim_cfg = SimConfig()
    ppo_cfg = PPOConfig()

    key, k0 = jax.random.split(key)
    agent = PPOAgent(ppo_config=ppo_cfg, key=k0)
    optimizer, opt_state = create_optimizer(ppo_cfg, agent)

    reward_history = []
    entropy_history = []

    for i in range(n_iterations):
        key, k_roll, k_upd, k_eval = jax.random.split(key, 4)

        batch = collect_rollout_batch(
            agent, sim_cfg, k_roll,
            n_envs=ppo_cfg.n_envs, n_steps=ppo_cfg.n_steps, locked_regime=0,
        )
        adv, ret = compute_gae(batch, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        agent, opt_state, metrics = ppo_update(
            agent, optimizer, opt_state, batch, adv, ret, ppo_cfg, k_upd,
        )

        if i % 10 == 0:
            stats = evaluate_agent(agent, sim_cfg, k_eval, n_episodes=20, locked_regime=0)
            mr = float(stats["mean_reward"])
            ent = float(metrics["entropy"])
            reward_history.append((i, mr))
            entropy_history.append((i, ent))
            print(f"iter {i:>4d}  reward {mr:>8.1f}  entropy {ent:.3f}")

    return agent, sim_cfg, reward_history, entropy_history


def analyze(agent, sim_cfg, reward_history, entropy_history):
    key = jax.random.PRNGKey(123)

    # --- 1. Collect a long evaluation rollout ---
    _, traj, _ = collect_rollout(agent, sim_cfg, key, n_steps=sim_cfg.max_steps, locked_regime=0)

    actions = np.array(traj.action)
    rewards = np.array(traj.reward)
    obs = np.array(traj.obs)
    inventory = obs[:, 31]  # inv_norm is obs index 31 (3*10 + 1)

    # --- 2. Action distribution heatmap ---
    n_bid = len(BID_TICKS)
    n_ask = len(ASK_TICKS)
    n_actions = n_bid * n_ask
    action_counts = np.bincount(actions, minlength=n_actions).reshape(n_bid, n_ask)
    action_freq = action_counts / action_counts.sum()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im = axes[0, 0].imshow(action_freq, cmap="YlOrRd", origin="lower")
    axes[0, 0].set_xticks(range(n_ask), [str(a) for a in ASK_TICKS])
    axes[0, 0].set_yticks(range(n_bid), [str(b) for b in BID_TICKS])
    axes[0, 0].set_xlabel("Ask offset (ticks)")
    axes[0, 0].set_ylabel("Bid offset (ticks)")
    axes[0, 0].set_title("Action Frequency (noise regime)")
    for i in range(n_bid):
        for j in range(n_ask):
            axes[0, 0].text(j, i, f"{action_freq[i, j]:.1%}",
                            ha="center", va="center", fontsize=10,
                            color="white" if action_freq[i, j] > 0.15 else "black")
    fig.colorbar(im, ax=axes[0, 0])

    # --- 3. Reward curve ---
    iters, rews = zip(*reward_history)
    axes[0, 1].plot(iters, rews, "b-o", markersize=3)
    axes[0, 1].set_xlabel("Training iteration")
    axes[0, 1].set_ylabel("Mean eval reward")
    axes[0, 1].set_title("Training reward curve")
    axes[0, 1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)

    # --- 4. Inventory over time ---
    axes[1, 0].plot(inventory * sim_cfg.max_inventory, "g-", alpha=0.7, linewidth=0.5)
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Inventory")
    axes[1, 0].set_title("Inventory trajectory (eval episode)")
    axes[1, 0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    # --- 5. Cumulative reward ---
    axes[1, 1].plot(np.cumsum(rewards), "r-", linewidth=0.8)
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Cumulative reward")
    axes[1, 1].set_title("Cumulative PnL (eval episode)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/ppo_noise_analysis.png", dpi=150)
    print("\nSaved plots/ppo_noise_analysis.png")
    plt.show()

    # --- 6. Print summary stats ---
    print("\n=== Policy Summary (noise regime) ===")
    top5 = np.argsort(action_counts.ravel())[::-1][:5]
    print("\nTop 5 actions:")
    for idx in top5:
        bid, ask = int(ACTION_TABLE[idx][0]), int(ACTION_TABLE[idx][1])
        print(f"  bid={bid} ask={ask}  freq={action_freq.ravel()[idx]:.1%}")

    mean_bid = np.average([ACTION_TABLE[a][0] for a in actions])
    mean_ask = np.average([ACTION_TABLE[a][1] for a in actions])
    print(f"\nMean bid offset: {mean_bid:.2f} ticks")
    print(f"Mean ask offset: {mean_ask:.2f} ticks")
    print(f"Bid-ask symmetry (should be ~0 for noise): {mean_bid - mean_ask:.2f}")
    print(f"Mean inventory: {np.mean(inventory) * sim_cfg.max_inventory:.2f}")
    print(f"Final cumulative reward: {np.sum(rewards):.1f}")


if __name__ == "__main__":
    agent, sim_cfg, rh, eh = train(n_iterations=300)
    analyze(agent, sim_cfg, rh, eh)
