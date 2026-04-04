"""Train an RL agent and save results for comparison with VI oracle.

Usage:
    uv run python scripts/compute_agent.py                        # ppo, all regimes
    uv run python scripts/compute_agent.py --agent rl2 --fast
    uv run python scripts/compute_agent.py --agent ppo --regime bull
    uv run python scripts/compute_agent.py --agent varibad --regime mixed

Saves per regime:
    plots/{agent}_{regime}.json — action-inventory histogram, rewards, training curve
"""
import argparse
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np

from lob_sim.agents import available_agents, make_agent
from lob_sim.actions import N_ACTIONS
from lob_sim.config import SimConfig
from lob_sim.obs import observe
from lob_sim.state import init_state
from lob_sim.step import make_step_fn
from lob_sim.training.eval import evaluate_agent
from lob_sim.training.rollout import collect_rollout_batch
from lob_sim.training.trainer import compute_gae, create_optimizer, ppo_update

REGIME_NAMES = ["noise", "bull", "bear"]
REGIME_MAP = {"noise": 0, "bull": 1, "bear": 2, "mixed": -1}
PLOTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "plots"))


# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--agent", default="ppo", choices=available_agents())
parser.add_argument("--regime", default=None,
                    choices=["noise", "bull", "bear", "mixed"],
                    help="Single regime (default: all)")
parser.add_argument("--fast", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

SIM_CFG = SimConfig()
MAX_ITERS = 50 if args.fast else 200
EVAL_EVERY = 5 if args.fast else 10
N_EVAL = 10 if args.fast else 100
PATIENCE = 10
REL_THRESHOLD = 0.02
ABS_FLOOR = 0.1

regimes_to_run = [args.regime] if args.regime else REGIME_NAMES + ["mixed"]


# ── Collect (inventory, action) pairs from a trained agent ──────
def collect_action_inventory(agent, sim_config, rng_key, n_episodes,
                             locked_regime):
    """Run episodes, return (actions, inventories, rewards, mask) per step."""
    step_fn = make_step_fn(sim_config, locked_regime=locked_regime)
    n_steps = sim_config.max_steps

    def run_one(key):
        k_init, k_agent, k_run = jax.random.split(key, 3)
        sim_state = init_state(sim_config, k_init)
        agent_state = agent.initial_agent_state(k_agent)

        def step(carry, _):
            s, a_st, rng = carry
            rng, rng_act = jax.random.split(rng)
            obs = observe(s, sim_config)
            action, new_a_st, _ = agent.get_action(obs, a_st, rng_act)
            new_s, sim_out = step_fn(s, action)
            new_a_st = new_a_st._replace(
                prev_reward=sim_out["reward"],
                prev_done=sim_out["done"].astype(jnp.float32),
            )
            was_live = ~s.done
            return (new_s, new_a_st, rng), (action, s.inventory,
                                             sim_out["reward"], was_live)

        _, (actions, inventories, rewards, mask) = jax.lax.scan(
            step, (sim_state, agent_state, k_run), None, length=n_steps
        )
        return actions, inventories, rewards, mask

    keys = jax.random.split(rng_key, n_episodes)
    return jax.vmap(run_one)(keys)  # each (n_episodes, n_steps)


def build_action_inventory_hist(actions, inventories, mask, max_inv):
    """Build (N_ACTIONS, n_inv_bins) histogram from rollout data."""
    n_inv = 2 * max_inv + 1
    hist = np.zeros((N_ACTIONS, n_inv))
    flat_a = np.array(actions).reshape(-1)
    flat_inv = np.array(inventories).reshape(-1)
    flat_mask = np.array(mask).reshape(-1).astype(bool)

    flat_a = flat_a[flat_mask]
    flat_inv = flat_inv[flat_mask]

    # Bin inventory to nearest integer, clip to grid
    inv_idx = np.clip(np.round(flat_inv).astype(int) + max_inv, 0, n_inv - 1)

    for a in range(N_ACTIONS):
        for i in range(n_inv):
            hist[a, i] = np.sum((flat_a == a) & (inv_idx == i))

    # Normalize per inventory level (columns sum to 1)
    col_sums = hist.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    hist_freq = hist / col_sums
    return hist_freq, hist


# ── Training loop ───────────────────────────────────────────────
def train_agent(agent_name, locked_regime, seed=42):
    """Train until convergence or MAX_ITERS.

    Returns (agent, eval_rewards, eval_iters, entropies).
    """
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)

    fast_overrides = dict(n_envs=16, n_steps=128) if args.fast else {}
    agent, cfg = make_agent(agent_name, key=k0, **fast_overrides)
    optimizer, opt_state = create_optimizer(cfg, agent)

    regime_str = REGIME_NAMES[locked_regime] if locked_regime >= 0 else "mixed"

    eval_rewards, eval_iters, entropies = [], [], []
    for i in range(MAX_ITERS):
        key, k_roll, k_upd = jax.random.split(key, 3)
        batch = collect_rollout_batch(
            agent, SIM_CFG, k_roll,
            n_envs=cfg.n_envs, n_steps=cfg.n_steps,
            locked_regime=locked_regime,
            meta_episode=True,
        )
        adv, ret = compute_gae(batch, cfg.gamma, cfg.gae_lambda)
        agent, opt_state, metrics = ppo_update(
            agent, optimizer, opt_state, batch, adv, ret, cfg, k_upd,
        )

        if i % EVAL_EVERY == 0 or i == MAX_ITERS - 1:
            key, k_eval = jax.random.split(key)
            stats = evaluate_agent(
                agent, SIM_CFG, k_eval,
                n_episodes=N_EVAL,
                locked_regime=locked_regime,
            )
            eval_rewards.append(float(stats["mean_reward"]))
            eval_iters.append(i)
            entropies.append(float(metrics["entropy"]))

            print(f"  [{regime_str}] iter {i:3d} | "
                  f"reward: {eval_rewards[-1]:7.3f} | "
                  f"entropy: {entropies[-1]:.3f}")

            # Convergence check
            n = len(eval_rewards)
            if n >= 2 * PATIENCE:
                prev_mean = sum(eval_rewards[n-2*PATIENCE:n-PATIENCE]) / PATIENCE
                curr_mean = sum(eval_rewards[n-PATIENCE:]) / PATIENCE
                delta = curr_mean - prev_mean
                threshold = max(abs(curr_mean) * REL_THRESHOLD, ABS_FLOOR)
                if delta < threshold:
                    print(f"  [{regime_str}] converged at iter {i}")
                    break

    return agent, eval_rewards, eval_iters, entropies


# ── Main ────────────────────────────────────────────────────────
print("=" * 60)
print(f"  compute_agent.py")
print("=" * 60)
print(f"  agent:      {args.agent}")
print(f"  regimes:    {regimes_to_run}")
print(f"  fast:       {args.fast}")
print(f"  seed:       {args.seed}")
print(f"  max_iters:  {MAX_ITERS}")
print("=" * 60)

max_inv = SIM_CFG.max_inventory

for regime_name in regimes_to_run:
    locked = REGIME_MAP[regime_name]
    print(f"\n{'─'*60}")
    print(f"  Training {args.agent.upper()} on {regime_name} regime")
    print(f"{'─'*60}")

    agent, eval_rewards, eval_iters, entropies = train_agent(
        args.agent, locked, seed=args.seed)

    # Collect action-inventory data for evaluation
    # For mixed: evaluate separately on each locked regime
    eval_regimes = [0, 1, 2] if locked == -1 else [locked]

    result = {
        "agent": args.agent,
        "regime": regime_name,
        "seed": args.seed,
        "max_inventory": max_inv,
        "training": {
            "iters": eval_iters,
            "rewards": eval_rewards,
            "entropies": entropies,
        },
        "eval": {},
    }

    for eval_r in eval_regimes:
        eval_name = REGIME_NAMES[eval_r]
        print(f"\n  Evaluating on {eval_name} regime ({N_EVAL} episodes)...")
        key_eval = jax.random.PRNGKey(args.seed + 1000 + eval_r)

        actions, inventories, rewards, mask = collect_action_inventory(
            agent, SIM_CFG, key_eval, n_episodes=N_EVAL,
            locked_regime=eval_r)

        # Action-inventory histogram
        hist_freq, hist_counts = build_action_inventory_hist(
            actions, inventories, mask, max_inv)

        # Per-episode total rewards
        ep_rewards = np.array(jnp.sum(rewards * mask, axis=1)).tolist()

        # Cumulative reward trajectory (mean across episodes)
        cum_rewards = np.array(jnp.cumsum(rewards * mask, axis=1))
        cum_mean = cum_rewards.mean(axis=0).tolist()
        cum_std = cum_rewards.std(axis=0).tolist()

        result["eval"][eval_name] = {
            "action_inventory_freq": hist_freq.tolist(),
            "action_inventory_counts": hist_counts.tolist(),
            "episode_rewards": ep_rewards,
            "cumulative_reward_mean": cum_mean,
            "cumulative_reward_std": cum_std,
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
        }

        print(f"    mean reward: {np.mean(ep_rewards):.2f} "
              f"(std: {np.std(ep_rewards):.2f})")

    # Save
    filename = f"{args.agent}_{regime_name}.json"
    path = os.path.join(PLOTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved {path}")

print("\nDone.")
