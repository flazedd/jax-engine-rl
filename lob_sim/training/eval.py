"""Evaluation harness."""
import jax
import jax.numpy as jnp

from lob_sim.config import SimConfig
from lob_sim.state import init_state
from lob_sim.obs import observe
from lob_sim.step import make_step_fn


def evaluate_agent(agent, sim_config, rng_key, n_episodes=50, locked_regime=-1):
    """Run n_episodes episodes, return mean total reward and per-episode stats.

    Returns dict:
        'mean_reward':  scalar
        'std_reward':   scalar
        'mean_episode_length': scalar
        'rewards':      (n_episodes,) array of per-episode total rewards
    """
    step_fn = make_step_fn(sim_config, locked_regime=locked_regime)
    n_steps = sim_config.max_steps

    def run_one_episode(key):
        k_init, k_agent, k_run = jax.random.split(key, 3)
        sim_state = init_state(sim_config, k_init)
        agent_state = agent.initial_agent_state(k_agent)

        def step(carry, _):
            sim_state, agent_state, rng, total_reward, ep_len = carry
            rng, rng_action = jax.random.split(rng)

            obs = observe(sim_state, sim_config)
            action, new_agent_state, _ = agent.get_action(obs, agent_state, rng_action)
            new_sim_state, sim_out = step_fn(sim_state, action)

            # Update agent state with reward/done for recurrent agents
            new_agent_state = new_agent_state._replace(
                prev_reward=sim_out["reward"],
                prev_done=sim_out["done"].astype(jnp.float32),
            )

            total_reward = total_reward + sim_out["reward"]
            ep_len = ep_len + jnp.where(sim_state.done, 0, 1)

            return (new_sim_state, new_agent_state, rng, total_reward, ep_len), None

        init_carry = (sim_state, agent_state, k_run, jnp.float32(0.0), jnp.int32(0))
        (_, _, _, total_reward, ep_len), _ = jax.lax.scan(step, init_carry, None, length=n_steps)
        return total_reward, ep_len

    keys = jax.random.split(rng_key, n_episodes)
    rewards, lengths = jax.vmap(run_one_episode)(keys)

    return {
        "mean_reward": jnp.mean(rewards),
        "std_reward": jnp.std(rewards),
        "mean_episode_length": jnp.mean(lengths.astype(jnp.float32)),
        "rewards": rewards,
    }
