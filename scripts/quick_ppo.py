"""Quick PPO smoke test with diagnostics."""
import argparse
import jax
import jax.numpy as jnp
import numpy as np

from lob_sim.config import SimConfig
from lob_sim.agents.ppo import PPOAgent, PPOConfig
from lob_sim.training.rollout import collect_rollout, collect_rollout_batch
from lob_sim.training.trainer import compute_gae, ppo_update, create_optimizer
from lob_sim.training.eval import evaluate_agent

_parser = argparse.ArgumentParser()
_parser.add_argument("--fast", action="store_true",
                     help="Quick smoke test with minimal iterations")
_args = _parser.parse_args()

# --- First: diagnose reward components with a random agent ---
sim_cfg = SimConfig()
ppo_cfg = PPOConfig(n_envs=16, n_steps=128)

key = jax.random.PRNGKey(42)
key, k0 = jax.random.split(key)
agent = PPOAgent(ppo_config=ppo_cfg, key=k0)

# Run one rollout and inspect
key, k_diag = jax.random.split(key)
_, traj, _ = collect_rollout(agent, sim_cfg, k_diag, n_steps=500, locked_regime=0)

rewards = np.array(traj.reward)
obs = np.array(traj.obs)
inv = obs[:, 31] * sim_cfg.max_inventory

print("=== Diagnostics (untrained agent, 500 steps) ===")
print(f"Per-step reward:  mean={rewards.mean():.4f}  std={rewards.std():.4f}")
print(f"Inventory:        mean={np.mean(np.abs(inv)):.1f}  max={np.max(np.abs(inv)):.1f}")
print(f"Cumulative reward: {rewards.sum():.1f}")
print(f"Reward range:      [{rewards.min():.3f}, {rewards.max():.3f}]")
print()

# Estimate spread capture vs penalty
tick = sim_cfg.tick_size
# The spread capture part is always >= 0, penalty part is always <= 0
# reward = spread_capture - inv_penalty * inv^2
# At each step, spread_capture = reward + penalty
penalty_per_step = sim_cfg.inventory_penalty * inv**2
spread_capture = rewards + penalty_per_step[:-1] if len(penalty_per_step) > len(rewards) else rewards + penalty_per_step[:len(rewards)]
print(f"Spread capture:   mean={np.mean(spread_capture):.4f}/step  total={np.sum(spread_capture):.1f}")
print(f"Inv penalty:      mean={np.mean(penalty_per_step):.4f}/step  total={np.sum(penalty_per_step):.1f}")
print()

# --- Now train ---
N_ITER = 10 if _args.fast else 100
EVAL_EVERY = 5 if _args.fast else 10

print(f"Config: lr={ppo_cfg.lr} ent={ppo_cfg.entropy_coef} inv_pen={sim_cfg.inventory_penalty}")
print(f"        max_inv={sim_cfg.max_inventory} max_steps={sim_cfg.max_steps} actions=9")
print()

optimizer, opt_state = create_optimizer(ppo_cfg, agent)

for i in range(N_ITER):
    key, k_roll, k_upd, k_eval = jax.random.split(key, 4)

    batch = collect_rollout_batch(
        agent, sim_cfg, k_roll,
        n_envs=ppo_cfg.n_envs, n_steps=ppo_cfg.n_steps, locked_regime=0,
    )
    adv, ret = compute_gae(batch, ppo_cfg.gamma, ppo_cfg.gae_lambda)
    agent, opt_state, metrics = ppo_update(
        agent, optimizer, opt_state, batch, adv, ret, ppo_cfg, k_upd,
    )

    if i % EVAL_EVERY == 0:
        stats = evaluate_agent(agent, sim_cfg, k_eval, n_episodes=5, locked_regime=0)
        mr = float(stats["mean_reward"])
        ent = float(metrics["entropy"])
        print(f"iter {i:>3d}  reward {mr:>8.1f}  entropy {ent:.3f}")
