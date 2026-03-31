"""Closed-loop rollout collection via lax.scan."""
import typing

import jax
import jax.numpy as jnp

from lob_sim.config import SimConfig
from lob_sim.state import OrderBookState, init_state
from lob_sim.obs import observe
from lob_sim.step import make_step_fn
from lob_sim.agents.base import AgentState, RolloutBatch


class RolloutCarry(typing.NamedTuple):
    sim_state: OrderBookState
    agent_state: AgentState
    rng_key: jnp.ndarray


class StepOutput(typing.NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def collect_rollout(agent, sim_config, rng_key, n_steps, locked_regime=-1, meta_episode=False):
    """Collect n_steps of experience from a single environment.

    Args:
        agent: Agent instance.
        sim_config: SimConfig.
        rng_key: JAX PRNGKey.
        n_steps: Number of steps to collect.
        locked_regime: Lock regime (-1 for free transitions).
        meta_episode: If True, reset sim on done but keep agent hidden state.

    Returns (final_carry, StepOutput, last_value) where StepOutput has leading dim n_steps.
    """
    step_fn = make_step_fn(sim_config, locked_regime=locked_regime)

    if meta_episode:
        def scan_body(carry: RolloutCarry, _):
            rng, rng_action, rng_step, rng_reset = jax.random.split(carry.rng_key, 4)

            obs = observe(carry.sim_state, sim_config)
            action, new_agent_state, info = agent.get_action(obs, carry.agent_state, rng_action)
            new_sim_state, sim_out = step_fn(carry.sim_state, action)

            output = StepOutput(
                obs=obs,
                action=action,
                log_prob=info["log_prob"],
                value=info["value"],
                reward=sim_out["reward"],
                done=sim_out["done"],
            )

            # Update agent state with reward and done from this step
            new_agent_state = new_agent_state._replace(
                prev_reward=sim_out["reward"],
                prev_done=sim_out["done"].astype(jnp.float32),
            )

            # Meta-episode: reset sim on done, keep agent hidden
            fresh_sim = init_state(sim_config, rng_reset)
            new_sim_state = jax.tree.map(
                lambda fresh, curr: jnp.where(new_sim_state.done, fresh, curr),
                fresh_sim, new_sim_state,
            )

            new_carry = RolloutCarry(
                sim_state=new_sim_state,
                agent_state=new_agent_state,
                rng_key=rng,
            )
            return new_carry, output
    else:
        def scan_body(carry: RolloutCarry, _):
            rng, rng_action, rng_step = jax.random.split(carry.rng_key, 3)

            obs = observe(carry.sim_state, sim_config)
            action, new_agent_state, info = agent.get_action(obs, carry.agent_state, rng_action)
            new_sim_state, sim_out = step_fn(carry.sim_state, action)

            output = StepOutput(
                obs=obs,
                action=action,
                log_prob=info["log_prob"],
                value=info["value"],
                reward=sim_out["reward"],
                done=sim_out["done"],
            )

            # Update agent state with reward and done from this step
            new_agent_state = new_agent_state._replace(
                prev_reward=sim_out["reward"],
                prev_done=sim_out["done"].astype(jnp.float32),
            )

            new_carry = RolloutCarry(
                sim_state=new_sim_state,
                agent_state=new_agent_state,
                rng_key=rng,
            )
            return new_carry, output

    rng_init, rng_agent, rng_scan = jax.random.split(rng_key, 3)
    init_sim_state = init_state(sim_config, rng_init)
    init_agent_state = agent.initial_agent_state(rng_agent)
    init_carry = RolloutCarry(init_sim_state, init_agent_state, rng_scan)

    final_carry, trajectory = jax.lax.scan(scan_body, init_carry, None, length=n_steps)

    last_obs = observe(final_carry.sim_state, sim_config)
    _, _, last_info = agent.get_action(last_obs, final_carry.agent_state, rng_key)
    last_value = last_info["value"]

    return final_carry, trajectory, last_value


def collect_rollout_batch(agent, sim_config, rng_key, n_envs, n_steps, locked_regime=-1, meta_episode=False):
    """vmap collect_rollout over n_envs independent environments.
    Returns RolloutBatch with shapes (n_envs, n_steps, ...).
    """
    keys = jax.random.split(rng_key, n_envs)
    vmapped = jax.vmap(
        lambda k: collect_rollout(agent, sim_config, k, n_steps, locked_regime, meta_episode)
    )
    _, trajectories, last_values = vmapped(keys)

    return RolloutBatch(
        obs=trajectories.obs,
        actions=trajectories.action,
        log_probs=trajectories.log_prob,
        values=trajectories.value,
        rewards=trajectories.reward,
        dones=trajectories.done,
        last_value=last_values,
    )
