"""Training script for RL agents."""
import argparse

import jax
import jax.numpy as jnp

from lob_sim.config import SimConfig
from lob_sim.agents.ppo import PPOAgent, PPOConfig
from lob_sim.training.rollout import collect_rollout_batch
from lob_sim.training.trainer import compute_gae, ppo_update, create_optimizer
from lob_sim.training.eval import evaluate_agent
from lob_sim.training.logger import Logger

REGIME_MAP = {"noise": 0, "bull": 1, "bear": 2, "free": -1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="ppo", choices=["ppo"])
    parser.add_argument("--regime", default="noise", choices=list(REGIME_MAP.keys()))
    parser.add_argument("--n_iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke test with minimal iterations")
    args = parser.parse_args()

    if args.fast:
        args.n_iterations = 10

    locked_regime = REGIME_MAP[args.regime]
    key = jax.random.PRNGKey(args.seed)
    sim_config = SimConfig()
    ppo_config = PPOConfig()
    logger = Logger()

    key, k0 = jax.random.split(key)
    agent = PPOAgent(ppo_config=ppo_config, key=k0)
    optimizer, opt_state = create_optimizer(ppo_config, agent)

    for iteration in range(args.n_iterations):
        key, k_roll, k_upd = jax.random.split(key, 3)

        batch = collect_rollout_batch(
            agent, sim_config, k_roll,
            n_envs=ppo_config.n_envs,
            n_steps=ppo_config.n_steps,
            locked_regime=locked_regime,
        )
        advantages, returns = compute_gae(batch, ppo_config.gamma, ppo_config.gae_lambda)
        agent, opt_state, metrics = ppo_update(
            agent, optimizer, opt_state, batch, advantages, returns, ppo_config, k_upd
        )

        if iteration % 10 == 0:
            key, k_eval = jax.random.split(key)
            eval_stats = evaluate_agent(
                agent, sim_config, k_eval, n_episodes=20, locked_regime=locked_regime
            )
            combined = {**{k: float(v) for k, v in metrics.items()}, **{k: float(v) for k, v in eval_stats.items() if k != "rewards"}}
            logger.log(iteration, combined)
            logger.print_summary(last_n=5)


if __name__ == "__main__":
    main()
