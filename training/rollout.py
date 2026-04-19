"""lax.scan-based rollout.

The rollout is the inner-loop building block: `parallel_envs` copies of an env
(via vmap) run for `rollout_length` steps (via scan). The compiled rollout is
called once per training iteration; its signature is shape-stable so the whole
per-iteration (rollout + update) step is a single JIT.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp


def _vmapped_reset(env, keys):
    return jax.vmap(env.reset)(keys)


def _vmapped_step(env, states, actions, keys):
    return jax.vmap(env.step)(states, actions, keys)


def _vmapped_act(agent, agent_state, obs, keys):
    def act_one(obs_i, key_i):
        return agent.act(agent_state, obs_i, key_i)
    return jax.vmap(act_one)(obs, keys)


@partial(jax.jit, static_argnames=("env", "agent", "parallel_envs", "rollout_length"))
def rollout(
    env,
    agent,
    agent_state: chex.ArrayTree,
    key: chex.PRNGKey,
    parallel_envs: int,
    rollout_length: int,
) -> dict[str, chex.Array]:
    """Run `parallel_envs` envs for `rollout_length` steps.

    Returns a dict of stacked per-step arrays with shape [rollout_length, parallel_envs, ...].
    For the M0 dummy agent the dummy-agent path returns action-only; no policy
    outputs are captured. The schema is intentionally minimal; richer agents
    will extend it in later milestones.
    """
    reset_key, key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, parallel_envs)
    env_states, obs = _vmapped_reset(env, reset_keys)

    def scan_step(carry, _):
        env_states, obs, key = carry
        act_key, step_key, key = jax.random.split(key, 3)
        act_keys = jax.random.split(act_key, parallel_envs)
        # act returns per-env actions; agent_state is shared (static across envs)
        actions, _ = _vmapped_act(agent, agent_state, obs, act_keys)
        step_keys = jax.random.split(step_key, parallel_envs)
        new_states, next_obs, rewards, dones, _info = _vmapped_step(
            env, env_states, actions, step_keys
        )
        out = {
            "obs": obs,
            "action": actions,
            "reward": rewards,
            "done": dones,
        }
        return (new_states, next_obs, key), out

    (final_states, final_obs, _), trajectory = jax.lax.scan(
        scan_step, (env_states, obs, key), xs=None, length=rollout_length
    )
    return trajectory
