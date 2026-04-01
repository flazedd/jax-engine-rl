"""Train PPO separately on each locked regime, compare learned policies."""
import jax
import jax.numpy as jnp
import numpy as np

from lob_sim.config import SimConfig
from lob_sim.actions import ACTION_TABLE, BID_TICKS, ASK_TICKS, N_ACTIONS
from lob_sim.agents.ppo import PPOAgent, PPOConfig
from lob_sim.training.rollout import collect_rollout, collect_rollout_batch
from lob_sim.training.trainer import compute_gae, ppo_update, create_optimizer

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--fast", action="store_true",
                     help="Quick smoke test with minimal iterations")
_args = _parser.parse_args()

REGIME_NAMES = ["noise", "bull", "bear"]
N_ITER = 10 if _args.fast else 150
sim_cfg = SimConfig()
ppo_cfg = PPOConfig(n_envs=16, n_steps=128)


def train_on_regime(regime_idx, seed=42):
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    agent = PPOAgent(ppo_config=ppo_cfg, key=k0)
    optimizer, opt_state = create_optimizer(ppo_cfg, agent)

    for i in range(N_ITER):
        key, k_roll, k_upd = jax.random.split(key, 3)
        batch = collect_rollout_batch(
            agent, sim_cfg, k_roll,
            n_envs=ppo_cfg.n_envs, n_steps=ppo_cfg.n_steps,
            locked_regime=regime_idx,
        )
        adv, ret = compute_gae(batch, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        agent, opt_state, metrics = ppo_update(
            agent, optimizer, opt_state, batch, adv, ret, ppo_cfg, k_upd,
        )

    return agent


def evaluate_policy(agent, regime_idx):
    key = jax.random.PRNGKey(123)
    _, traj, _ = collect_rollout(agent, sim_cfg, key, n_steps=sim_cfg.max_steps, locked_regime=regime_idx)
    actions = np.array(traj.action)
    rewards = np.array(traj.reward)
    obs = np.array(traj.obs)
    inv = obs[:, 31] * sim_cfg.max_inventory

    n_bid, n_ask = len(BID_TICKS), len(ASK_TICKS)
    action_freq = np.bincount(actions, minlength=N_ACTIONS) / len(actions)

    mean_bid = np.mean([float(ACTION_TABLE[a][0]) for a in actions])
    mean_ask = np.mean([float(ACTION_TABLE[a][1]) for a in actions])

    return {
        "reward": rewards.sum(),
        "action_freq": action_freq,
        "mean_bid": mean_bid,
        "mean_ask": mean_ask,
        "mean_inv": np.mean(inv),
        "top_action": int(np.argmax(action_freq)),
    }


print(f"Training PPO on each regime ({N_ITER} iters, {ppo_cfg.n_envs} envs)...\n")

results = {}
for r in range(3):
    print(f"--- Training on {REGIME_NAMES[r]} regime ---")
    agent = train_on_regime(r)
    stats = evaluate_policy(agent, r)
    results[r] = stats

    top_idx = stats["top_action"]
    top_bid, top_ask = int(ACTION_TABLE[top_idx][0]), int(ACTION_TABLE[top_idx][1])
    print(f"  reward={stats['reward']:.1f}  top_action=({top_bid},{top_ask}) "
          f"mean_bid={stats['mean_bid']:.2f}  mean_ask={stats['mean_ask']:.2f}  "
          f"mean_inv={stats['mean_inv']:.1f}")
    print()

# Summary
print("=" * 60)
print(f"{'Regime':<8} {'Reward':>8} {'Top Action':>12} {'Mean Bid':>10} {'Mean Ask':>10} {'Bid-Ask':>8}")
print("-" * 60)
for r in range(3):
    s = results[r]
    top_idx = s["top_action"]
    top_bid, top_ask = int(ACTION_TABLE[top_idx][0]), int(ACTION_TABLE[top_idx][1])
    print(f"{REGIME_NAMES[r]:<8} {s['reward']:>8.1f} {f'({top_bid},{top_ask})':>12} "
          f"{s['mean_bid']:>10.2f} {s['mean_ask']:>10.2f} {s['mean_bid']-s['mean_ask']:>8.2f}")

print()
# Check divergence
top_actions = [results[r]["top_action"] for r in range(3)]
if len(set(top_actions)) == 3:
    print("All 3 regimes learned DIFFERENT optimal actions.")
elif len(set(top_actions)) >= 2:
    print("At least 2 different optimal actions across regimes.")
else:
    print("WARNING: All regimes converged to the SAME action — no divergence.")

# Check expected asymmetry
bull_bias = results[1]["mean_bid"] - results[1]["mean_ask"]
bear_bias = results[2]["mean_bid"] - results[2]["mean_ask"]
print(f"\nBull bid-ask bias: {bull_bias:+.2f} (expect positive: wide bid, tight ask)")
print(f"Bear bid-ask bias: {bear_bias:+.2f} (expect negative: tight bid, wide ask)")
