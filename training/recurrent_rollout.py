"""Recurrent variant of the lax.scan rollout.

Like `training/rollout.py`, but threads a per-env recurrent carry (e.g. a GRU
hidden state) through the scan. The carry resets to zeros on episode boundaries
(`done=True`) so the next episode begins with a fresh hidden state — matching
the canonical RL² / VariBAD setup where one episode = one task.

Returns the same `(trajectory, final_obs)` as the MLP rollout, plus:
  - `init_carry`: the per-env carry at the start of the rollout (needed for
    BPTT during the PPO update so the agent can replay the recurrent forward
    pass from the same h_0 it acted from).
  - `final_carry`: the per-env carry after the last step, returned for the
    next iteration's rollout (so hidden state propagates across iterations
    when the rollout doesn't fully cover an episode).

The agent must implement:
  - `init_carry(n_parallel) -> per_env_carry`  (zeros-tree for recurrent agents)
  - `act(state, carry, obs, key) -> (action, extras, new_carry)`
"""
from __future__ import annotations

from functools import partial

import chex
import jax
import jax.numpy as jnp


def _vmapped_reset(env, keys):
    return jax.vmap(env.reset)(keys)


def _vmapped_step(env, states, actions, keys):
    return jax.vmap(env.step)(states, actions, keys)


def _vmapped_act_recurrent(agent, agent_state, carry, obs, keys):
    """Vmap agent.act across the env axis. `carry` is per-env (leading dim N)."""
    def act_one(carry_i, obs_i, key_i):
        return agent.act(agent_state, carry_i, obs_i, key_i)
    return jax.vmap(act_one)(carry, obs, keys)


def _reset_carry_on_done(carry, dones, init_carry):
    """For each env i where dones[i] is True, reset carry[i] to init_carry[i]."""
    def _select(c, c0):
        # c, c0 have leading dim N (parallel envs). dones has shape [N].
        broadcast_shape = (dones.shape[0],) + (1,) * (c.ndim - 1)
        mask = dones.reshape(broadcast_shape).astype(c.dtype)
        return mask * c0 + (1.0 - mask) * c
    return jax.tree_util.tree_map(_select, carry, init_carry)


@partial(jax.jit, static_argnames=("env", "agent", "parallel_envs", "rollout_length"))
def recurrent_rollout(
    env,
    agent,
    agent_state: chex.ArrayTree,
    initial_carry: chex.ArrayTree,
    key: chex.PRNGKey,
    parallel_envs: int,
    rollout_length: int,
) -> tuple[dict[str, chex.Array], chex.Array, chex.ArrayTree, chex.ArrayTree]:
    reset_key, key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, parallel_envs)
    env_states, obs = _vmapped_reset(env, reset_keys)

    init_carry = initial_carry  # zeros pytree, per-env [N, ...]

    def scan_step(scan_carry, _):
        env_states, obs, agent_carry, key = scan_carry
        act_key, step_key, key = jax.random.split(key, 3)
        act_keys = jax.random.split(act_key, parallel_envs)
        actions, extras, new_agent_carry = _vmapped_act_recurrent(
            agent, agent_state, agent_carry, obs, act_keys
        )
        step_keys = jax.random.split(step_key, parallel_envs)
        new_states, next_obs, rewards, dones, _info = _vmapped_step(
            env, env_states, actions, step_keys
        )
        # Reset agent carry where the env just ended an episode.
        new_agent_carry = _reset_carry_on_done(new_agent_carry, dones, init_carry)
        out = {
            "obs": obs,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "carry_in": agent_carry,
        }
        out.update({k: v for k, v in extras.items()})
        return (new_states, next_obs, new_agent_carry, key), out

    (final_states, final_obs, final_carry, _), trajectory = jax.lax.scan(
        scan_step,
        (env_states, obs, init_carry, key),
        xs=None,
        length=rollout_length,
    )
    return trajectory, final_obs, init_carry, final_carry
